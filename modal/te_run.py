"""
Тренування на NVIDIA JAX MaxText контейнері (Transformer Engine): dropless grouped GEMM для MoE через te_moe_block.

  modal run te_run.py                  # TE MoE на новій формі (1024x12, 64x512 top-6)
"""
import os
import subprocess
import sys

import modal

ROOT = "/root/moe"
PKG = "/opt/maxtext/src/maxtext"

image = (
    modal.Image.from_registry("ghcr.io/nvidia/jax:maxtext-2026-09-10")
    .env({"PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "1", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"})
    .add_local_dir("../out/tokenizer_pruned", f"{ROOT}/tokenizer", ignore=["old2new.json"])
    .add_local_file("../configs/stage1.yml", f"{ROOT}/stage1.yml")
    .add_local_file("padded_gmm.py", f"{ROOT}/padded_gmm.py")
    .add_local_file("fast_conv.py", f"{ROOT}/fast_conv.py")
    .add_local_file("gdn_fast.py", f"{ROOT}/gdn_fast.py")
    .add_local_file("te_gmm.py", f"{ROOT}/te_gmm.py")
)
app = modal.App("moe-te", image=image)

WIDE = ["base_emb_dim=1024", "base_num_decoder_layers=12", "num_experts=64", "base_moe_mlp_dim=512",
        "base_mlp_dim=512", "num_experts_per_tok=6"]
COMMON = ["base_output_directory=/tmp/out", "enable_checkpointing=False", f"tokenizer_path={ROOT}/tokenizer",
          "log_period=1", "hardware=gpu", "skip_jax_distributed_system=True", "dataset_type=synthetic",
          "max_target_length=4096", "steps=12"]


def patch_muon():
    p = os.path.join(PKG, "configs", "types.py")
    src = open(p).read()
    anchor = 'if self.opt_type == "muon" and self.decoder_block not in ['
    i = src.find(anchor)
    if i >= 0 and "QWEN3_5" not in src[i:i + 600]:
        j = i + len(anchor)
        open(p, "w").write(src[:j] + "\n        DecoderBlockType.QWEN3_5," + src[j:])
        print("[patch] muon: QWEN3_5 додано", flush=True)


def run(extra, patches=""):
    import yaml
    cfg = os.path.join(PKG, "configs", "stage1.yml")
    base_keys = set(yaml.safe_load(open(os.path.join(PKG, "configs", "base.yml"))))
    out, dropped = [], []
    for line in open(f"{ROOT}/stage1.yml", encoding="utf-8"):
        key = line.split(":", 1)[0].strip()
        if line[:1] not in (" ", "#", "\n") and ":" in line and key not in base_keys and key != "base_config":
            dropped.append(key)
            continue
        out.append(line)
    open(cfg, "w", encoding="utf-8").write("".join(out))
    if dropped:
        print("[cfg] ключі, яких немає в цій версії MaxText (прибрано):", dropped, flush=True)
    args = [cfg, *COMMON, *extra]
    boot = (f"import sys,runpy; sys.path.insert(0,'{ROOT}'); {patches}sys.argv=['train']+sys.argv[1:]; "
            "runpy.run_module('maxtext.trainers.pre_train.train', run_name='__main__')")
    print("\n==================== ", extra, patches, flush=True)
    return subprocess.run([sys.executable, "-c", boot, *args]).returncode


@app.function(gpu="H100", timeout=3000, cpu=8, memory=65536)
def bench(runs):
    check = r"""
import jax, jax.numpy as jnp, sys
sys.path.insert(0, '/root/moe')
import te_gmm
gs = jnp.array([5, 0, 9, 3, 7, 1, 6, 4], jnp.int32); M = int(gs.sum()) + 3; S = int(gs.sum())
a = jax.random.normal(jax.random.PRNGKey(0), (M, 64), jnp.bfloat16); b = jax.random.normal(jax.random.PRNGKey(1), (8, 64, 48), jnp.bfloat16)
ref = te_gmm._orig_ragged_dot(a, b, gs).astype(jnp.float32)
out = te_gmm.te_ragged_dot(a, b, gs).astype(jnp.float32)
f = lambda fn: (lambda x, y: (fn(x, y, gs)[:S].astype(jnp.float32) ** 2).sum())
g1 = jax.grad(f(te_gmm._orig_ragged_dot), (0, 1))(a, b); g2 = jax.grad(f(te_gmm.te_ragged_dot), (0, 1))(a, b)
rel = lambda u, v: float(jnp.abs(u.astype(jnp.float32) - v.astype(jnp.float32)).max() / (jnp.abs(v.astype(jnp.float32)).max() + 1e-6))
print('TE CHECK fwd rel', rel(out[:S], ref[:S]), 'dlhs rel', rel(g2[0][:S], g1[0][:S]), 'drhs rel', rel(g2[1], g1[1]))
"""
    subprocess.run([sys.executable, "-c", check])
    subprocess.run("grep -n -E '^(te_|use_te|prefuse|ragged_buffer|quantization:)' " + PKG + "/configs/base.yml",
                   shell=True)
    patch_muon()
    return [run(extra, patches) for extra, patches in runs]


GDN = "import fast_conv; fast_conv.install(); import gdn_fast; gdn_fast.install('default','solve',64); "


@app.local_entrypoint()
def main(mode: str = "te"):
    b16 = ["per_device_batch_size=16"]
    if mode == "te":
        te = "import te_gmm; te_gmm.install(); " + GDN
        b24 = ["per_device_batch_size=24", "sparse_matmul=True", "megablox=False"]
        runs = [
            (WIDE + b24 + ["inhomogeneous_layer_cycle_interval=1"], te),   # стеля без GDN
            (WIDE + b24 + ["quantization=te_fp8_currentscaling"], te),   # FP8 (TE) для dense-шарів
        ]
    else:
        raise ValueError(mode)
    print("RESULTS", bench.remote(runs))
