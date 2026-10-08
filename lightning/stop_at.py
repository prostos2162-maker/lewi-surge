"""
Зупинка стадії 1 на заданому чекпоінті (коли кредитів не вистачає до кінця) і перемикання студії на CPU для експорту.

  python lightning/stop_at.py --step 24000 --orch-pid 20968
1) чекає, поки на студії з'явиться завершений чекпоінт N;
2) вбиває локальний оркестратор (інакше він сприйме зупинку як падіння і перезапустить);
3) вбиває тренування на студії й перемикає машину на CPU (H200 більше не палить кредити).
"""
import argparse
import os
import signal
import time

from lightning_sdk import Machine, Studio

TEAMSPACE, USER, NAME = "deploy-model-project", "prostos2162-maker", "moe-stage1"


def sh(s, cmd):
    try:
        return s.run(cmd)
    except Exception as e:
        return f"ERR: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--orch-pid", type=int, required=True)
    a = ap.parse_args()
    s = Studio(name=NAME, teamspace=TEAMSPACE, user=USER)
    ck = f"~/out/stage1/checkpoints/{a.step}"
    while True:
        # orbax пише в тимчасову теку і перейменовує в кінці → наявність теки N з items = чекпоінт записаний
        out = sh(s, f"ls -d {ck}/items 2>/dev/null && ls ~/out/stage1/checkpoints/ | grep -c tmp || true")
        if "/items" in out:
            print(time.strftime("%H:%M"), f"чекпоінт {a.step} є, чекаю дозапис", flush=True)
            time.sleep(120)
            break
        print(time.strftime("%H:%M"), sh(s, "grep -a -o 'completed step: [0-9]*' ~/train.log | tail -1").strip(), flush=True)
        time.sleep(60)
    try:
        os.kill(a.orch_pid, signal.SIGTERM)
        print("оркестратор зупинено", flush=True)
    except OSError as e:
        print("оркестратор:", e, flush=True)
    sh(s, "pkill -f pre_train.train; pkill -f runpy; sleep 10; pgrep -af 'pre_train|runpy' | grep -v pgrep || echo NO_TRAIN")
    print(sh(s, "ls ~/out/stage1/checkpoints/"), flush=True)
    s.switch_machine(Machine.CPU)
    print(time.strftime("%H:%M"), "студія на CPU:", s.machine, s.status, "ЗУПИНЕНО_НА", a.step, flush=True)


if __name__ == "__main__":
    main()
