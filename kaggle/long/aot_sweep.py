"""AOT-компіляція повної моделі під v5e-8 для різних seq/паралелізму — шукаємо, що влазить у 15.75 ГБ HBM/чип.
Запуск на Colab після train_built.py (maxtext уже стоїть і пропатчений)."""
import re
import subprocess
import sys

import maxtext
import os

PKG = os.path.dirname(maxtext.__file__)
CFG = os.path.join(PKG, "configs", "stage2.yml")
FULL = ("base_emb_dim=1024 base_num_decoder_layers=16 base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 "
        "base_mlp_dim=512 base_moe_mlp_dim=512 num_experts=128 shared_experts=1 num_experts_per_tok=6 vocab_size=131200 "
        "logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 gdn_num_key_heads=4 gdn_num_value_heads=8 "
        "gdn_key_head_dim=128 gdn_value_head_dim=128 gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 "
        "use_multimodal=False").split()
BASE = ("hardware=tpu skip_jax_distributed_system=True base_output_directory=/tmp/aot scan_layers=True sparse_matmul=True "
        "megablox=True attention=flash remat_policy=full dtype=bfloat16 weight_dtype=float32 ici_fsdp_parallelism=-1 "
        "ici_expert_parallelism=1 compile_topology=v5e-8 compile_topology_num_slices=1 dataset_type=synthetic "
        "run_name=aot steps=10").split()
DEFAULT = [
    "max_target_length=16384 per_device_batch_size=1 num_vocab_tiling=16",
    "max_target_length=32768 per_device_batch_size=1 num_vocab_tiling=32 ici_context_parallelism=2 context_parallel_load_balance=False",
    "max_target_length=32768 per_device_batch_size=1 num_vocab_tiling=32 ici_context_parallelism=4 context_parallel_load_balance=False",
    "max_target_length=65536 per_device_batch_size=1 num_vocab_tiling=32 ici_context_parallelism=8 context_parallel_load_balance=False",
    "max_target_length=8192 per_device_batch_size=1 num_vocab_tiling=8",
]
CASES = [x.split() for x in (sys.argv[1:] or DEFAULT)]
for c in CASES:
    r = subprocess.run([sys.executable, "-m", "maxtext.trainers.pre_train.train_compile", CFG] + FULL + BASE + c,
                       capture_output=True, text=True)
    t = r.stdout + r.stderr
    oom = re.findall(r"HLO temporaries \(([\d.]+)G\)", t)
    err = [l for l in t.splitlines() if re.search(r"Error|error:", l) and "Warning" not in l and "NO-OP" not in l]
    print(f"{'OK ' if r.returncode == 0 else 'ERR'} {' '.join(c)} | temp={oom[-1] if oom else '-'}G | "
          f"{(err[-1][:200] if err and r.returncode else '')}", flush=True)
print("SWEEP_DONE")
