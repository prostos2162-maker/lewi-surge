import os
"""Зберігає копії проміжних чекпоінтів (max_num_checkpoints_to_keep=2 їх стирає) у ~/keep/<step> на студії."""
import time

from lightning_sdk import Studio

MILESTONES = [2500, 5000, 7500, 10000]
s = Studio(name="lewi-s2g", teamspace="default-project", user=os.environ.get("LIGHTNING_USER", "your-lightning-user"))
done = set()
while len(done) < len(MILESTONES):
    try:
        if "Running" in str(s.status):
            for m in MILESTONES:
                if m in done:
                    continue
                r = s.run(f"d=~/out/s2g/checkpoints/{m}; k=~/keep/{m}; "
                          f"if [ -d $k ]; then echo HAVE; elif [ -d $d/items ]; then sleep 60; mkdir -p ~/keep && cp -r $d $k && du -sh $k && echo COPIED; else echo NO; fi")
                if "COPIED" in r or "HAVE" in r:
                    print(time.strftime("%H:%M"), f"чекпоінт {m} збережено в ~/keep:", r.strip()[-80:], flush=True)
                    done.add(m)
    except Exception as e:
        print(time.strftime("%H:%M"), "помилка:", e, flush=True)
    time.sleep(300)
print("ВСІ ПРОМІЖНІ ЧЕКПОІНТИ ЗБЕРЕЖЕНО", flush=True)
