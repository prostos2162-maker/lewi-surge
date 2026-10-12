"""Перевірка генерації HF-чекпоінта з Volume lewi-s2 на дешевій GPU (локально бракує RAM).
  modal run probe.py --ckpt hf_s2g2_12250
"""
import modal

vol = modal.Volume.from_name("lewi-s2")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.9.0", "transformers==5.19.0", "safetensors", "accelerate")
         .add_local_file("../tools/probe.py", "/root/probe_local.py"))
app = modal.App("lewi-probe")


@app.function(image=image, volumes={"/vol": vol}, gpu="L4", timeout=1800)
def probe(ckpt: str):
    import runpy
    import sys
    sys.argv = ["probe", f"/vol/{ckpt}"]
    src = open("/root/probe_local.py").read().replace("    model = load_gpu(ckpt)\n",
        "    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).cuda().eval()\n"
        # у Volume ваги tied; MaxText ділить логіти на sqrt(d) → окремий lm_head, як локально
        "    emb = model.get_input_embeddings().weight\n"
        "    model.lm_head.weight = torch.nn.Parameter((emb.float() / emb.shape[1] ** 0.5).to(emb.dtype))\n")
    exec(compile(src, "probe", "exec"), {"__name__": "__main__"})


@app.local_entrypoint()
def main(ckpt: str = "hf_s2g2_12250"):
    probe.remote(ckpt)
