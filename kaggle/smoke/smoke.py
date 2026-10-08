"""
Smoke-тест стадії 1 на Kaggle TPU v5e-8.

Що перевіряємо:
  A) модель з configs/stage1.yml будується, влазить у пам'ять і робить кроки (синтетичні дані)
  B) якщо A падає — спрощений варіант (без MTP, AdamW), щоб локалізувати проблему
  C) стрім FineWeb-Edu з HF + наш обрізаний токенайзер
Кожна спроба — окремий subprocess, падіння однієї не зупиняє інші. Усе логується.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

ASSETS = next(iter(glob.glob("/kaggle/input/**/stage1.yml", recursive=True)), None)
ASSETS = os.path.dirname(ASSETS) if ASSETS else None
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)


def sh(cmd, timeout=None):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT, timeout=timeout)
    print(f"[exit {r.returncode}, {time.time() - t:.0f}s]", flush=True)
    return r.returncode


print("=== середовище ===")
print("python", sys.version)
print("assets:", ASSETS, os.listdir(ASSETS) if ASSETS else None)
sh("nproc; free -g; df -h /kaggle/working | tail -1")

print("\n=== встановлення MaxText ===")
sh(f"{sys.executable} -m pip install -q uv")
rc = sh(f"{sys.executable} -m uv pip install --system -q 'maxtext[tpu]==0.2.5' --resolution=lowest", timeout=1800)
if rc != 0:
    sh(f"{sys.executable} -m pip install -q 'maxtext[tpu]==0.2.5'", timeout=1800)
sh("install_tpu_pre_train_extra_deps", timeout=1800)
sh(f"{sys.executable} -c \"import jax; print(jax.__version__, jax.devices())\"")

import maxtext  # noqa: E402

cfg_dir = os.path.join(os.path.dirname(maxtext.__file__), "configs")
print("maxtext configs:", cfg_dir, os.path.exists(os.path.join(cfg_dir, "base.yml")))
cfg = os.path.join(cfg_dir, "stage1.yml")
shutil.copy(os.path.join(ASSETS, "stage1.yml"), cfg)
tok = "/kaggle/working/tokenizer"
os.makedirs(tok, exist_ok=True)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    shutil.copy(os.path.join(ASSETS, f), tok)

# латка: Muon не знає qwen3_5 (GDN ідентичний qwen3_next, розмітка параметрів загальна)
types_py = os.path.join(os.path.dirname(maxtext.__file__), "configs", "types.py")
src = open(types_py).read()
anchor = 'if self.opt_type == "muon" and self.decoder_block not in ['
i = src.find(anchor)
if i >= 0 and "QWEN3_5" not in src[i:i + 600]:
    j = i + len(anchor)
    open(types_py, "w").write(src[:j] + "\n        DecoderBlockType.QWEN3_5," + src[j:])
    print("[patch] muon: додано QWEN3_5 у білий список", flush=True)

TRAIN = f"{sys.executable} -m maxtext.trainers.pre_train.train {cfg}"
COMMON = (f"base_output_directory={OUT} enable_checkpointing=False tokenizer_path={tok} "
          f"steps=30 log_period=5 profiler=''")

attempts = [
    ("A_synthetic_full", f"{TRAIN} {COMMON} run_name=A dataset_type=synthetic"),
    # якщо A не влізе в пам'ять — менший батч
    ("B_synthetic_simple", f"{TRAIN} {COMMON} run_name=B dataset_type=synthetic per_device_batch_size=4"),
    ("C_hf_stream", f"{TRAIN} {COMMON} run_name=C dataset_type=hf"),
]
results = {}
for name, cmd in attempts:
    if name == "B_synthetic_simple" and results.get("A_synthetic_full") == 0:
        continue  # повний варіант працює — спрощений не потрібен
    print(f"\n\n==================== {name} ====================", flush=True)
    try:
        results[name] = sh(cmd, timeout=2400)
    except subprocess.TimeoutExpired:
        results[name] = "timeout"

print("\n\n=== ПІДСУМОК ===")
for k, v in results.items():
    print(f"{k}: {'OK' if v == 0 else v}")
