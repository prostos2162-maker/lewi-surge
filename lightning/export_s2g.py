"""
Експорт чекпоінтів тренування s2g (Lightning, 16 шарів / 128 експертів) у HF і завантаження локально.
Студія на CPU (H200 не палимо).
  python lightning/export_s2g.py --steps 2500 5000 7500 10000 final
Результат: D:/ai/checkpoints/surge_s2g2_step{N}_hf/ (окремий lm_head = embed / sqrt(d), як у surge_s2g_step5000_hf)
"""
import argparse
import json
import math
import os
import time

from lightning_sdk import Machine, Studio

from train_s2g import DIMS, NAME, ROOT, TEAMSPACE, USER

EXPORT = r"""
source ~/mt/bin/activate
cd ~/moe
export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1 PYTHONUNBUFFERED=1
PKG=$(python -c "import maxtext, os; print(os.path.dirname(maxtext.__file__))")
rm -rf ~/hf_x___TAG__
LEWI_LAYERS=16 LEWI_EXPERTS=128 nice -n 10 python conv_wrapper.py to_huggingface $PKG/configs/stage2.yml \
  model_name=qwen3.5-35b-a3b override_model_config=True dataset_type=synthetic __DIMS__ \
  load_parameters_path=__SRC__ scan_layers=True \
  base_output_directory=$HOME/hf_x___TAG__ hardware=cpu skip_jax_distributed_system=True \
  tokenizer_path=$HOME/moe/tokenizer weight_dtype=bfloat16
cp $HOME/moe/tokenizer/* $HOME/hf_x___TAG__/ 2>/dev/null
ls -la $HOME/hf_x___TAG__; echo EXPORT_DONE
"""


def untie_head(dst):
    """MaxText при спільних embedding/LM-голові ділить логіти на sqrt(d_model); HF — ні. Окремий lm_head у своєму файлі."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    idx_p = os.path.join(dst, "model.safetensors.index.json")
    idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {
        "metadata": {}, "weight_map": {k: "model.safetensors" for k in
                                       safe_open(os.path.join(dst, "model.safetensors"), "pt").keys()}}
    emb = next(k for k in idx["weight_map"] if "embed_tokens" in k)
    with safe_open(os.path.join(dst, idx["weight_map"][emb]), "pt") as f:
        w = f.get_tensor(emb)
    cfg = json.load(open(os.path.join(dst, "config.json")))
    d = cfg.get("hidden_size") or cfg.get("text_config", {}).get("hidden_size")
    save_file({"lm_head.weight": (w.float() / math.sqrt(d)).to(w.dtype).contiguous()},
              os.path.join(dst, "lm_head.safetensors"), metadata={"format": "pt"})
    idx["weight_map"]["lm_head.weight"] = "lm_head.safetensors"
    json.dump(idx, open(idx_p, "w"), indent=2)
    cfg["tie_word_embeddings"] = False
    if "text_config" in cfg:
        cfg["text_config"]["tie_word_embeddings"] = False
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(f"lm_head = embed / sqrt({d})", flush=True)


def export(s, tag):
    if tag == "final":
        last = s.run("ls ~/out/s2g/checkpoints | sort -n | tail -1").strip()
        src, name = f"$HOME/out/s2g/checkpoints/{last}/items", f"step{last}"
    else:
        src, name = f"$HOME/keep/{tag}/items", f"step{tag}"
    dst = os.path.join(ROOT, "checkpoints", f"surge_s2g2_{name}_hf")
    if os.path.exists(os.path.join(dst, "lm_head.safetensors")):
        print("вже є:", dst, flush=True)
        return
    tmp = os.path.join(ROOT, "out", "lt2", "export.sh")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(EXPORT.replace("__TAG__", tag).replace("__SRC__", src).replace("__DIMS__", DIMS))
    s.upload_file(tmp, "export.sh")
    s.run("cd ~ && for i in $(seq 200); do ls export.sh >/dev/null 2>&1 && break; sleep 3; done; mv export.sh moe/")
    s.run(f"cd ~/moe && nohup bash export.sh > ~/export_{tag}.log 2>&1 &")
    log = ""
    for _ in range(180):
        time.sleep(30)
        log = s.run(f"tail -c 2500 ~/export_{tag}.log")
        if "EXPORT_DONE" in log or "Traceback" in log:
            break
    print(log[-1500:], flush=True)
    if "EXPORT_DONE" not in log or "Traceback" in log:
        raise SystemExit(f"експорт {tag} не вдався")
    os.makedirs(dst, exist_ok=True)
    for f in [f for f in s.run(f"ls ~/hf_x_{tag}").split() if f]:
        for _ in range(20):
            try:
                s.download_file(f"hf_x_{tag}/{f}", os.path.join(dst, f))
                print("download", f, os.path.getsize(os.path.join(dst, f)), flush=True)
                break
            except Exception as e:
                print("retry", f, str(e)[:80], flush=True)
                time.sleep(30)
    untie_head(dst)
    s.run(f"rm -rf ~/hf_x_{tag}")
    print("ГОТОВО:", dst, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=["2500", "5000", "7500", "10000", "final"])
    args = ap.parse_args()
    s = Studio(name=NAME, teamspace=TEAMSPACE, user=USER)
    if "Running" not in str(s.status):
        s.start(Machine.CPU)
    for t in args.steps:
        export(s, t)
    s.stop()
    print("ВСЕ ЕКСПОРТОВАНО, студію зупинено", flush=True)


if __name__ == "__main__":
    main()
