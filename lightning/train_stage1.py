"""
Стадія 1 на Lightning AI: H200 (interruptible), тренуємо до витрачення безкоштовних кредитів.

Хід:
  1. старт студії H200 interruptible, заливка конфігу/токенайзера/латок
  2. venv: maxtext[tpu] + jax[cuda12] (без libtpu) — перевірений рецепт; MoE через padded grouped GEMM v2
     (TE на Lightning недоступний без NVIDIA-контейнера), GDN: fast_conv + unroll
  3. bench: 40 кроків → реальна швидкість → кроки під бюджет годин (WSD: спад LR в останні 15%)
  4. train у фоні (nohup), чекпоінти на постійний диск студії; MaxText сам продовжує з останнього
  5. вартовий: кожні 3 хв лог; якщо машину забрали — перезапуск студії і продовження

Запуск: python lightning/train_stage1.py --hours 9.5
"""
import argparse
import json
import os
import re
import time

from lightning_sdk import Machine, Studio
from lightning_sdk.machine import CloudProvider

TEAMSPACE, USER, NAME = "deploy-model-project", "prostos2162-maker", "moe-stage1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE = "/teamspace/studios/this_studio"
SEQ = 4096

SETUP = r"""
set -e
cd ~/moe
pip install -q uv
if [ ! -f ~/mt/.ready ]; then
  uv venv -q ~/mt --python 3.12
  source ~/mt/bin/activate
  uv pip install -q 'maxtext[tpu]==0.2.5' --resolution=lowest
  install_tpu_pre_train_extra_deps || true
  JAXV=$(python -c "import jax; print(jax.__version__)")
  uv pip uninstall -q libtpu libtpu-nightly || true
  uv pip install -q "jax[cuda12]==$JAXV"
  touch ~/mt/.ready
fi
source ~/mt/bin/activate
python patch_maxtext.py
python -c "import jax; print('JAX', jax.__version__, jax.devices())"
echo SETUP_DONE
"""

PATCH = r'''
import maxtext, os, shutil
pkg = os.path.dirname(maxtext.__file__)
p = os.path.join(pkg, "configs", "types.py"); src = open(p).read()
anchor = 'if self.opt_type == "muon" and self.decoder_block not in ['
i = src.find(anchor)
if i >= 0 and "QWEN3_5" not in src[i:i + 600]:
    j = i + len(anchor); open(p, "w").write(src[:j] + "\n        DecoderBlockType.QWEN3_5," + src[j:])
    print("[patch] muon: QWEN3_5")
shutil.copy(os.path.expanduser("~/moe/stage1.yml"), os.path.join(pkg, "configs", "stage1.yml"))
print("PKG", pkg)
'''

# запуск тренування з латками продуктивності (математика та сама)
TRAIN = r"""
source ~/mt/bin/activate
cd ~/moe
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 TF_CPP_MIN_LOG_LEVEL=1 PYTHONUNBUFFERED=1
PKG=$(python -c "import maxtext, os; print(os.path.dirname(maxtext.__file__))")
python -c "import sys,runpy; sys.path.insert(0,'$HOME/moe'); import padded_gmm; padded_gmm.install(1.5); \
import fast_conv; fast_conv.install(); import gdn_fast; gdn_fast.install('default','solve',64); \
sys.argv=['train']+sys.argv[1:]; runpy.run_module('maxtext.trainers.pre_train.train', run_name='__main__')" \
  $PKG/configs/stage1.yml hardware=gpu skip_jax_distributed_system=True \
  tokenizer_path=$HOME/moe/tokenizer sparse_matmul=True megablox=False \
  per_device_batch_size=__PDB__ max_target_length=4096 \
  base_output_directory=__OUT__ run_name=__RUN__ __EXTRA__
"""


def sh(studio, cmd):
    try:
        return studio.run(cmd)
    except Exception as e:  # studio.run кидає RuntimeError з виводом при ненульовому коді
        return f"ERR: {e}"


def upload(studio):
    files = {
        "tokenizer.json": "out/tokenizer_pruned/tokenizer.json",
        "tokenizer_config.json": "out/tokenizer_pruned/tokenizer_config.json",
        "chat_template.jinja": "out/tokenizer_pruned/chat_template.jinja",
        "stage1.yml": "configs/stage1.yml",
        "padded_gmm.py": "modal/padded_gmm.py",
        "fast_conv.py": "modal/fast_conv.py",
        "gdn_fast.py": "modal/gdn_fast.py",
    }
    tmp = os.path.join(ROOT, "out", "lt")
    os.makedirs(tmp, exist_ok=True)
    for name, body in (("setup.sh", SETUP), ("patch_maxtext.py", PATCH)):
        with open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
        files[name] = os.path.relpath(os.path.join(tmp, name), ROOT)
    for remote, local in files.items():  # SDK на Windows ламає підпапки → плоско, потім mv
        studio.upload_file(os.path.join(ROOT, local), remote)
    names = " ".join(files)
    sh(studio, f"cd ~ && for i in $(seq 60); do ls {names} >/dev/null 2>&1 && break; sleep 2; done; "
               f"mkdir -p moe/tokenizer && mv tokenizer.json tokenizer_config.json chat_template.jinja moe/tokenizer/ "
               f"&& mv stage1.yml padded_gmm.py fast_conv.py gdn_fast.py setup.sh patch_maxtext.py moe/")


def ensure_running(studio, interruptible):
    for _ in range(90):
        st = str(studio.status)
        if "Running" in st:
            # нова студія сама стартує на CPU — перемикаємо на H200, інакше JAX тихо піде на CPU
            if "H200" not in str(studio.machine):
                print(time.strftime("%H:%M"), f"машина {studio.machine} → перемикаю на H200", flush=True)
                studio.switch_machine(Machine.H200, interruptible=interruptible,
                                      cloud_provider=CloudProvider.LIGHTNING)  # H200 є лише там/в Nebius, не на AWS
                time.sleep(20)
                continue
            return
        if "Stopped" in st:
            print(time.strftime("%H:%M"), "старт студії (CPU), далі перемикання на H200", flush=True)
            studio.start()
        time.sleep(20)
    raise RuntimeError(f"студія не стартувала: {studio.status}")


def launch(studio, run, extra, log, pdb):
    cmd = TRAIN.replace("__PDB__", str(pdb)).replace("__OUT__", f"{REMOTE}/out").replace("__RUN__", run).replace("__EXTRA__", extra)
    with open(os.path.join(ROOT, "out", "lt", "train.sh"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(cmd)
    studio.upload_file(os.path.join(ROOT, "out", "lt", "train.sh"), "train.sh")
    sh(studio, f"cd ~ && for i in $(seq 60); do ls train.sh >/dev/null 2>&1 && break; sleep 2; done; mv train.sh moe/")
    sh(studio, f"cd ~/moe && nohup bash train.sh >> {log} 2>&1 &")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=9.5, help="бюджет годин тренування H200")
    ap.add_argument("--no-interruptible", action="store_true")
    args = ap.parse_args()
    interruptible = not args.no_interruptible
    state_path = os.path.join(ROOT, "out", "lt", "state.json")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}

    studio = Studio(name=NAME, teamspace=TEAMSPACE, user=USER, create_ok=True)
    ensure_running(studio, interruptible)
    upload(studio)
    out = sh(studio, "cd ~/moe && bash setup.sh 2>&1 | tail -5")
    print(out, flush=True)
    assert "SETUP_DONE" in out, "setup не вдався"
    if "CudaDevice" not in out:  # ніколи не тренуємо на CPU
        print("GPU НЕ ЗНАЙДЕНО — зупиняю студію:", studio.machine, flush=True)
        studio.stop()
        return

    if "steps" not in state:  # 1) замір швидкості; батч — найбільший, що влазить
        speed = pdb = None
        for pdb in (32, 24, 16):
            sh(studio, "rm -rf ~/out/bench ~/bench.log")
            launch(studio, "bench", "dataset_type=hf enable_checkpointing=False steps=40 log_period=5",
                   "~/bench.log", pdb)
            for _ in range(60):
                time.sleep(30)
                log = sh(studio, "tail -c 4000 ~/bench.log")
                if "completed step: 39" in log or "Traceback" in log:
                    break
            full = sh(studio, "cat ~/bench.log")
            print(f"BENCH b{pdb}:", sh(studio, "grep -a -E 'completed step|Error|memory size' ~/bench.log | tail -6"),
                  flush=True)
            tps = [float(x) for x in re.findall(r"Tokens/s/device: ([\d.]+)", full)]
            if len(tps) >= 4:
                speed = sum(tps[-4:]) / 4
                break
            if "RESOURCE_EXHAUSTED" not in full:
                raise RuntimeError("bench впав не через пам'ять — див. лог вище")
        assert speed, "жоден батч не влазить"
        steps = int(args.hours * 3600 * speed / (pdb * SEQ))
        state = {"speed": speed, "steps": steps, "pdb": pdb, "started": time.time()}
        json.dump(state, open(state_path, "w"))
        print(f"ШВИДКІСТЬ {speed:.0f} ток/с, батч {pdb} → {steps} кроків ≈ {steps * pdb * SEQ / 1e9:.2f}B токенів",
              flush=True)

    steps, pdb = state["steps"], state["pdb"]
    extra = (f"dataset_type=hf steps={steps} learning_rate_schedule_steps={steps} "
             f"enable_checkpointing=True async_checkpointing=True checkpoint_period=500 "
             f"max_num_checkpoints_to_keep=2 log_period=50 metrics_file={REMOTE}/out/metrics.jsonl")

    def alive():
        return "train.py" in sh(studio, "pgrep -af 'pre_train.train' | grep -v pgrep || true") or \
               "runpy" in sh(studio, "pgrep -af runpy | grep -v pgrep || true")

    if not alive():
        launch(studio, "stage1", extra, "~/train.log", pdb)
    last = ""
    crashes = 0
    while True:
        time.sleep(180)
        st = str(studio.status)
        if "Running" not in st:
            print(time.strftime("%H:%M"), f"студія {st} — перезапуск і продовження з чекпоінта", flush=True)
            try:
                ensure_running(studio, interruptible)
            except Exception as e:
                print("СТАРТ НЕ ВДАВСЯ (можливо, скінчились кредити):", e, flush=True)
                return
            upload(studio)
            print(sh(studio, "cd ~/moe && bash setup.sh 2>&1 | tail -2"), flush=True)
            launch(studio, "stage1", extra, "~/train.log", pdb)
            continue
        tail = sh(studio, "grep -a -E 'completed step|Traceback|Error|RESOURCE_EXHAUSTED|saved|Restoring' ~/train.log | tail -3")
        if tail != last:
            print(time.strftime("%H:%M"), tail[-700:], flush=True)
            last = tail
        if f"completed step: {steps - 1}" in tail:
            print("ТРЕНУВАННЯ ЗАВЕРШЕНО", flush=True)
            sh(studio, "sleep 120")  # дати дописатись фінальному чекпоінту
            studio.stop()
            return
        if not alive() and "completed step" in tail:
            err = sh(studio, "tail -c 2500 ~/train.log")
            print("ПРОЦЕС ВПАВ:\n", err, flush=True)
            crashes += 1
            if crashes > 3:  # не палимо кредити на нескінченні падіння
                print("3+ падіння поспіль — зупиняю студію", flush=True)
                studio.stop()
                return
            launch(studio, "stage1", extra, "~/train.log", pdb)


if __name__ == "__main__":
    main()
