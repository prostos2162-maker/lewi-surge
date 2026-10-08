"""
GPU smoke-тест стадії 1 на Lightning AI (L4, 24 ГБ).

Мета: перевірити, що MaxText + наш конфіг працюють на CUDA (на GPU інші MoE-ядра, ніж на TPU),
і поміряти швидкість. Це обов'язкова перевірка перед RunPod.

Запуск: python lightning/gpu_smoke.py [--machine L4|T4]
Студія ЗАВЖДИ зупиняється в кінці (finally), щоб не палити кредити.
"""
import argparse
import os
import time

from lightning_sdk import Machine, Studio

TEAMSPACE = "deploy-model-project"
USER = "prostos2162-maker"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REMOTE_SCRIPT = r"""
set -x
cd ~/moe
python --version; nvidia-smi
pip install -q uv
# окреме чисте venv: conda-env Lightning має свої пакети, які конфліктують (numpy ABI)
if [ ! -f ~/mt/.ready ]; then   # venv живе на диску студії між запусками
  uv venv -q ~/mt --python 3.12
  source ~/mt/bin/activate
  uv pip install -q 'maxtext[tpu]==0.2.5' --resolution=lowest
  install_tpu_pre_train_extra_deps || true
  JAXV=$(python -c "import jax; print(jax.__version__)")
  uv pip uninstall -q libtpu libtpu-nightly || true   # інакше JAX на GPU шукає TPU
  uv pip install -q "jax[cuda12]==$JAXV"
  touch ~/mt/.ready
fi
source ~/mt/bin/activate
python -c "import jax, numpy; print(jax.__version__, numpy.__version__, jax.devices())"

PKG=$(python -c "import maxtext, os; print(os.path.dirname(maxtext.__file__))")
cp stage1.yml $PKG/configs/stage1.yml
python patch_maxtext.py $PKG

run() {  # $1 = назва, решта = оверрайди
  name=$1; shift
  echo "==================== $name ===================="
  timeout 1500 python -m maxtext.trainers.pre_train.train $PKG/configs/stage1.yml \
    base_output_directory=/tmp/out enable_checkpointing=False tokenizer_path=$HOME/moe/tokenizer \
    log_period=5 hardware=gpu skip_jax_distributed_system=True dataset_type=synthetic \
    steps=20 per_device_batch_size=2 max_target_length=2048 run_name=$name "$@"
  echo "[$name exit $?]"
}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   # за замовчуванням JAX бере лише 75% пам'яті GPU
run A_ragged    megablox=False sparse_matmul=True per_device_batch_size=1 max_target_length=1024
run A_ragged_2k megablox=False sparse_matmul=True per_device_batch_size=1 max_target_length=2048
echo ALL_DONE
"""

PATCH = r'''
import os, sys
pkg = sys.argv[1]
p = os.path.join(pkg, "configs", "types.py")
src = open(p).read()
anchor = 'if self.opt_type == "muon" and self.decoder_block not in ['
i = src.find(anchor)
if i >= 0 and "QWEN3_5" not in src[i:i + 600]:
    j = i + len(anchor)
    open(p, "w").write(src[:j] + "\n        DecoderBlockType.QWEN3_5," + src[j:])
    print("[patch] muon: додано QWEN3_5 у білий список")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="L4")
    args = ap.parse_args()

    studio = Studio(name="moe-gpu-smoke", teamspace=TEAMSPACE, user=USER, create_ok=True)
    try:
        for _ in range(60):  # попередній запуск міг ще зупинятись
            if "stopping" not in str(studio.status).lower():
                break
            time.sleep(10)
        print("Старт студії на", args.machine, "| статус:", studio.status, flush=True)
        studio.start(getattr(Machine, args.machine))
        # SDK на Windows робить з "moe/x" файл з назвою "moe\x" → вантажимо плоско, розкладаємо на сервері
        studio.run("rm -rf ~/moe ~/moe\\\\* ; mkdir -p ~/moe/tokenizer")
        tok = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        for f in tok:
            studio.upload_file(os.path.join(ROOT, "out", "tokenizer_pruned", f), f)
        studio.upload_file(os.path.join(ROOT, "configs", "stage1.yml"), "stage1.yml")
        tmp = os.path.join(ROOT, "out")
        for name, body in (("run.sh", REMOTE_SCRIPT), ("patch_maxtext.py", PATCH)):
            with open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(body)
            studio.upload_file(os.path.join(tmp, name), name)
        # завантажені файли з'являються на диску студії із затримкою (синхронізація сховища)
        files = " ".join((*tok, "stage1.yml", "run.sh", "patch_maxtext.py"))
        studio.run(f"cd ~ && for i in $(seq 60); do ls {files} >/dev/null 2>&1 && break; sleep 2; done; ls {files}")
        studio.run("cd ~ && mv " + " ".join(tok) + " moe/tokenizer/ && mv stage1.yml run.sh patch_maxtext.py moe/")
        print(studio.run("ls -R ~/moe"), flush=True)

        studio.run("cd ~/moe && nohup bash run.sh > run.log 2>&1 &")
        t0, last = time.time(), ""
        while time.time() - t0 < 3 * 3600:
            time.sleep(60)
            tail = studio.run("tail -c 4000 ~/moe/run.log")
            if tail != last:
                print(f"\n--- {int(time.time() - t0)}s ---\n{tail[-1500:]}", flush=True)
                last = tail
            if "ALL_DONE" in tail:
                break
            if "No such file" in tail and "run.sh" in tail:
                print("run.sh не знайдено — зупиняюсь", flush=True)
                break
        studio.run("cp ~/moe/run.log ~/run_final.log")
        studio.download_file("run_final.log", os.path.join(ROOT, "out", "lightning_gpu_smoke.log"))
    finally:
        print("Зупиняю студію", flush=True)
        try:
            studio.stop()
        except RuntimeError as e:  # вже зупинена
            print(e)


if __name__ == "__main__":
    main()
