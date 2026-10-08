"""
Експорт чекпоінта стадії 1 у формат Hugging Face (на CPU студії, тренування на GPU не зупиняється) і завантаження локально.

  python lightning/export_hf.py --step 500
Результат: D:/ai/checkpoints/step{N}_hf/ (config.json, *.safetensors, токенайзер)
"""
import argparse
import os
import time

from lightning_sdk import Studio

TEAMSPACE, USER, NAME = "deploy-model-project", "prostos2162-maker", "moe-stage1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# розміри моделі явно в CLI: model_name потрібен конвертеру лише для вибору маппінгу qwen3.5
DIMS = ("base_emb_dim=1024 base_num_decoder_layers=12 base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 "
        "base_mlp_dim=512 base_moe_mlp_dim=512 num_experts=64 shared_experts=1 num_experts_per_tok=6 "
        "vocab_size=131200 logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 "
        "gdn_num_key_heads=4 gdn_num_value_heads=8 gdn_key_head_dim=128 gdn_value_head_dim=128 "
        "gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 use_multimodal=False")

EXPORT = r"""
source ~/mt/bin/activate
export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1 PYTHONUNBUFFERED=1
PKG=$(python -c "import maxtext, os; print(os.path.dirname(maxtext.__file__))")
rm -rf ~/hf___STEP__
nice -n 10 python $HOME/moe/hf_export_wrapper.py $PKG/configs/stage1.yml \
  model_name=qwen3.5-35b-a3b override_model_config=True __DIMS__ \
  load_parameters_path=$HOME/out/stage1/checkpoints/__STEP__/items scan_layers=True \
  base_output_directory=$HOME/hf___STEP__ hardware=cpu skip_jax_distributed_system=True \
  tokenizer_path=$HOME/moe/tokenizer weight_dtype=bfloat16 --override_model_architecture
cp $HOME/moe/tokenizer/* $HOME/hf___STEP__/ 2>/dev/null
ls -la $HOME/hf___STEP__; echo EXPORT_DONE
"""


def untie_head(dst):
    """MaxText при спільних embedding/LM-голові ділить логіти на sqrt(d_model); HF — ні. Робимо окремий lm_head."""
    import json
    import math
    from safetensors.torch import load_file, save_file
    sd = load_file(os.path.join(dst, "model.safetensors"))
    emb = next(k for k in sd if "embed_tokens" in k)
    cfg = json.load(open(os.path.join(dst, "config.json")))
    d = cfg.get("hidden_size") or cfg.get("text_config", {}).get("hidden_size")
    sd["lm_head.weight"] = (sd[emb].float() / math.sqrt(d)).to(sd[emb].dtype).contiguous()
    save_file(sd, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    cfg["tie_word_embeddings"] = False
    if "text_config" in cfg:
        cfg["text_config"]["tie_word_embeddings"] = False
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(f"lm_head = embed / sqrt({d})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    args = ap.parse_args()
    s = Studio(name=NAME, teamspace=TEAMSPACE, user=USER)
    cmd = EXPORT.replace("__STEP__", str(args.step)).replace("__DIMS__", DIMS)
    tmp = os.path.join(ROOT, "out", "lt", "export.sh")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(cmd)
    s.upload_file(tmp, "export.sh")
    s.upload_file(os.path.join(ROOT, "lightning", "hf_export_wrapper.py"), "hf_export_wrapper.py")
    s.run("cd ~ && for i in $(seq 60); do ls export.sh hf_export_wrapper.py >/dev/null 2>&1 && break; sleep 2; done; "
          "mv export.sh hf_export_wrapper.py moe/")
    s.run(f"cd ~/moe && nohup bash export.sh > ~/export_{args.step}.log 2>&1 &")
    for _ in range(120):
        time.sleep(30)
        log = s.run(f"tail -c 2500 ~/export_{args.step}.log")
        if "EXPORT_DONE" in log or "Traceback" in log:
            break
    print(log[-2500:], flush=True)
    if "EXPORT_DONE" not in log or "Traceback" in log:
        raise SystemExit("експорт не вдався")
    dst = os.path.join(ROOT, "checkpoints", f"surge_step{args.step}_hf")
    os.makedirs(dst, exist_ok=True)
    files = [f for f in s.run(f"ls ~/hf_{args.step}").split() if f]
    for f in files:  # плоско (SDK на Windows ламає підпапки); свіжі файли синхронізуються із затримкою → повтори
        for attempt in range(20):
            try:
                s.download_file(f"hf_{args.step}/{f}", os.path.join(dst, f))
                print("download", f, os.path.getsize(os.path.join(dst, f)), flush=True)
                break
            except Exception as e:
                print("retry", f, str(e)[:80], flush=True)
                time.sleep(30)
    untie_head(dst)
    s.run(f"rm -rf ~/hf_{args.step}")  # не тримати копії на диску студії
    print("ГОТОВО:", dst, flush=True)


if __name__ == "__main__":
    main()
