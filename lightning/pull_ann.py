"""
Після annealing: останній чекпоінт з Lightning (lewi-ann) → Modal Volume /out/ann/checkpoints/<step>/items
→ export_hf на Modal CPU → checkpoints/surge_ann_step<step>_hf. Студію зупиняє. Друкує STEP=<step>.
"""
import os
import sys
import shutil
import tarfile
import time

from lightning_sdk import Machine, Studio

from export_s2g import untie_head
from pull_s2g import ROOT, TMP, run

s = Studio(name="lewi-ann", teamspace="default-project", user=os.environ.get("LIGHTNING_USER", "your-lightning-user"))
if "Running" not in str(s.status):
    s.start(Machine.CPU)
LIVE = "--live" in sys.argv  # під час тренування: не зупиняти студію
st = s.run("ls ~/out/ann/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1").strip()
print("STEP=" + st, flush=True)
os.makedirs(TMP, exist_ok=True)
size = int(s.run(f"cd ~ && ([ -f ann_{st}.tar ] || tar cf ann_{st}.tar -C ~/out/ann/checkpoints/{st} items) && stat -c %s ann_{st}.tar").split()[-1])
tar = os.path.join(TMP, f"ann_{st}.tar")
for _ in range(40):
    try:
        s.download_file(f"ann_{st}.tar", tar)
        if os.path.getsize(tar) == size:
            break
        print("неповний", os.path.getsize(tar), "з", size, flush=True)
    except Exception as e:
        print("retry", str(e)[:100], flush=True)
    time.sleep(60)
assert os.path.getsize(tar) == size, "архів не скачався повністю"
print("завантажено", size // 2**20, "МБ", flush=True)
s.run(f"rm -f ~/ann_{st}.tar")
if not LIVE:
    s.stop()
ck = os.path.join(TMP, "ann_" + st)
shutil.rmtree(ck, ignore_errors=True)
with tarfile.open(tar) as t:
    t.extractall(ck)
os.remove(tar)
run(f'modal volume put -f lewi-s2 "{ck}/items" /out/ann/checkpoints/{st}/items')
shutil.rmtree(ck)
run(f"cd {ROOT}/modal && modal run stage2.py::export_hf --step {st} --run ann --layers 16 --experts 128")
dst = os.path.join(ROOT, "checkpoints", f"surge_ann_step{st}_hf")
os.makedirs(dst, exist_ok=True)
run(f'modal volume get --force lewi-s2 /hf_ann_{st}/ "{dst}"')
inner = os.path.join(dst, f"hf_ann_{st}")
if os.path.isdir(inner):
    for f in os.listdir(inner):
        shutil.move(os.path.join(inner, f), os.path.join(dst, f))
    os.rmdir(inner)
untie_head(dst)
print("ГОТОВО:", dst, "STEP=" + st, flush=True)
