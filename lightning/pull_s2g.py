"""
Безкоштовна CPU-студія Lightning має 15 ГБ RAM — конвертер у HF падає з OOM. Тому:
Lightning (tar чекпоінта) → локально (розпакувати) → Modal Volume lewi-s2 → stage2.py::export_hf на Modal CPU (32 ГБ)
→ локально checkpoints/surge_s2g2_step{N}_hf (+ окремий lm_head).
  python lightning/pull_s2g.py 12250 10000 7500 5000 2500
"""
import os
import shutil
import subprocess
import sys
import tarfile
import time

from lightning_sdk import Machine, Studio

from export_s2g import untie_head
from train_s2g import NAME, ROOT, TEAMSPACE, USER

TMP = os.path.join(ROOT, "out", "lt2", "pull")
env = dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONIOENCODING="utf-8")


def run(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    if r.returncode:
        raise SystemExit(f"помилка: {cmd}")
    return r.stdout


def main():
    steps = sys.argv[1:]
    s = Studio(name=NAME, teamspace=TEAMSPACE, user=USER)
    if "Running" not in str(s.status):
        s.start(Machine.CPU)
    os.makedirs(TMP, exist_ok=True)
    for st in steps:
        dst = os.path.join(ROOT, "checkpoints", f"surge_s2g2_step{st}_hf")
        if os.path.exists(os.path.join(dst, "lm_head.safetensors")):
            print("вже є", dst, flush=True)
            continue
        src = f"keep/{st}" if st in ("2500", "5000", "7500", "10000") else f"out/s2g/checkpoints/{st}"
        size = int(s.run(f"cd ~ && ([ -f ckpt_{st}.tar ] || tar cf ckpt_{st}.tar -C ~/{src} items) && stat -c %s ckpt_{st}.tar").split()[-1])
        tar = os.path.join(TMP, f"ckpt_{st}.tar")
        for _ in range(40):  # свіжий файл синхронізується із затримкою: качаємо, доки розмір не збігся
            try:
                s.download_file(f"ckpt_{st}.tar", tar)
                if os.path.getsize(tar) == size:
                    break
                print("неповний", os.path.getsize(tar), "з", size, flush=True)
            except Exception as e:
                print("retry", str(e)[:100], flush=True)
            time.sleep(60)
        assert os.path.getsize(tar) == size, "архів так і не скачався повністю"
        print("завантажено", os.path.getsize(tar) // 2**20, "МБ", flush=True)
        s.run(f"rm -f ~/ckpt_{st}.tar")
        ck = os.path.join(TMP, st)
        shutil.rmtree(ck, ignore_errors=True)
        with tarfile.open(tar) as t:
            t.extractall(ck)
        os.remove(tar)
        run(f'modal volume put -f lewi-s2 "{ck}/items" /out/s2g2/checkpoints/{st}/items')
        shutil.rmtree(ck)
        run(f"cd {ROOT}/modal && modal run stage2.py::export_hf --step {st} --run s2g2 --layers 16 --experts 128")
        os.makedirs(dst, exist_ok=True)
        run(f'modal volume get --force lewi-s2 /hf_s2g2_{st}/ "{dst}"')
        inner = os.path.join(dst, f"hf_s2g2_{st}")
        if os.path.isdir(inner):  # volume get кладе папку всередину
            for f in os.listdir(inner):
                shutil.move(os.path.join(inner, f), os.path.join(dst, f))
            os.rmdir(inner)
        untie_head(dst)
        run(f"modal volume rm -r lewi-s2 /out/s2g2/checkpoints/{st}")
        print("ГОТОВО:", dst, flush=True)
    s.stop()
    print("ВСЕ ГОТОВО, студію зупинено", flush=True)


if __name__ == "__main__":
    main()
