"""
Стадія 1 на Kaggle TPU v5e-8 — справжнє тренування, а не smoke.

Модель, дані й гіперпараметри — строго з configs/stage1.yml (датасет moe-assets).
Тут змінюються ЛИШЕ параметри виконання, які не впливають на математику навчання:
  * розбиття батчу: per_device_batch_size × gradient_accumulation_steps = 8 (глобальний батч той самий)
  * remat_policy (що перераховувати на backward) — впливає лише на пам'ять/швидкість
  * частота і кількість чекпоінтів (виживання між сесіями)

Хід сесії:
  1. ставимо MaxText, латка Muon для qwen3_5
  2. якщо є вихід попередньої сесії (kernel_sources) — продовжуємо з її останнього чекпоінта
  3. підбір найшвидшого режиму виконання (лише в першій сесії; далі беремо збережений вибір)
  4. тренування до дедлайну (~8 год 15 хв), потім акуратна зупинка → чекпоінт лишається у виводі
"""
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

T0 = time.time()
DEADLINE = T0 + 8 * 3600 + 15 * 60  # TPU-сесія Kaggle ≤ 9 год; запас на збереження виводу
WORK = "/kaggle/working"
OUT = f"{WORK}/out"
RUN = "stage1"
CKPT_DIR = f"{OUT}/{RUN}/checkpoints"
STATE = f"{WORK}/state"
os.makedirs(STATE, exist_ok=True)


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} хв]", *a, flush=True)


def sh(cmd, timeout=None):
    log("$", cmd)
    r = subprocess.run(cmd, shell=True, timeout=timeout)
    log(f"exit {r.returncode}")
    return r.returncode


# ---------------------------------------------------------------- 1. середовище
cfg_src = next(iter(glob.glob("/kaggle/input/**/stage1.yml", recursive=True)), None)
assert cfg_src, "немає stage1.yml у /kaggle/input (датасет moe-assets не підключено?)"
ASSETS = os.path.dirname(cfg_src)
log("assets:", ASSETS, sorted(os.listdir(ASSETS)))
sh("nproc; free -g; df -h /kaggle/working | tail -1")

sh(f"{sys.executable} -m pip install -q uv")
if sh(f"{sys.executable} -m uv pip install --system -q 'maxtext[tpu]==0.2.5' --resolution=lowest", 1800):
    sh(f"{sys.executable} -m pip install -q 'maxtext[tpu]==0.2.5'", 1800)
sh("install_tpu_pre_train_extra_deps", 1800)
sh(f"{sys.executable} -c \"import jax; print(jax.__version__, jax.devices())\"")

# батьківський процес НЕ імпортує maxtext/jax — інакше може тримати TPU і дочірні процеси не стартують
PKG = subprocess.run([sys.executable, "-c", "import maxtext, os; print(os.path.dirname(maxtext.__file__))"],
                     capture_output=True, text=True).stdout.strip().splitlines()[-1]
log("maxtext:", PKG)
CFG = os.path.join(PKG, "configs", "stage1.yml")
shutil.copy(cfg_src, CFG)
TOK = f"{WORK}/tokenizer"
os.makedirs(TOK, exist_ok=True)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    shutil.copy(os.path.join(ASSETS, f), TOK)

# латка: Muon не знає qwen3_5 (GDN ідентичний qwen3_next, розмітка параметрів загальна)
types_py = os.path.join(PKG, "configs", "types.py")
src = open(types_py).read()
anchor = 'if self.opt_type == "muon" and self.decoder_block not in ['
i = src.find(anchor)
if i >= 0 and "QWEN3_5" not in src[i:i + 600]:
    j = i + len(anchor)
    open(types_py, "w").write(src[:j] + "\n        DecoderBlockType.QWEN3_5," + src[j:])
    log("[patch] muon: додано QWEN3_5 у білий список")

# ---------------------------------------------------------------- 2. продовження
def ckpt_steps(root):
    out = []
    for d in glob.glob(f"{root}/*"):
        name = os.path.basename(d)
        if name.isdigit() and os.path.isdir(os.path.join(d, "items")):
            out.append((int(name), d))
    return sorted(out)


prev = []
for root in glob.glob(f"/kaggle/input/**/{RUN}/checkpoints", recursive=True):
    prev += ckpt_steps(root)
prev.sort()
resume = []
if prev:
    step, path = prev[-1]
    resume = [f"load_full_state_path={path}/items"]
    log(f"ПРОДОВЖУЮ з чекпоінта попередньої сесії: крок {step} ({path})")
else:
    log("Попередніх чекпоінтів немає — старт з нуля")

prev_choice = next(iter(glob.glob("/kaggle/input/**/state/perf_choice.json", recursive=True)), None)

# ---------------------------------------------------------------- 3. підбір режиму виконання
TRAIN = [sys.executable, "-m", "maxtext.trainers.pre_train.train", CFG]
COMMON = [f"tokenizer_path={TOK}", f"base_output_directory={OUT}"]
GLOBAL_PER_DEVICE = 8  # з stage1.yml: per_device_batch_size=8, gradient_accumulation_steps=1


TPU_BUSY = ("busy", "already in use", "TPU initialization failed", "Unable to initialize backend 'tpu'",
            "libtpu_lockfile", "Couldn't open device")
NOISE = ("UserWarning", "warnings.warn", "FutureWarning", "import pynvml", "Config param", "hugepages")


def wait_tpu_free():
    """Після попереднього процесу TPU-рантайм v5e звільняється повільно — чекаємо."""
    for _ in range(30):
        busy = subprocess.run("lsof -t /dev/vfio/* /dev/accel* 2>/dev/null | head -1", shell=True,
                              capture_output=True, text=True).stdout.strip()
        if not busy:
            return
        time.sleep(5)
    log("УВАГА: TPU досі зайнятий процесом", busy)


def run_logged(cmd, name, timeout):
    path = f"{STATE}/{name}.log"
    for attempt in range(3):
        wait_tpu_free()
        time.sleep(10)
        try:
            with open(path, "w") as fh:
                p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        text = open(path, errors="replace").read()
        if rc != 0 and any(k in text for k in TPU_BUSY):
            log(f"  TPU зайнятий (спроба {attempt + 1}) — чекаю й повторюю")
            time.sleep(30)
            continue
        return rc, text
    return rc, text


def error_summary(text, n=40):
    lines = [l for l in text.splitlines() if not any(k in l for k in NOISE)]
    hits = [l for l in lines if re.search(r"Error|Exception|error:|RESOURCE_EXHAUSTED|Traceback|FAILED", l)]
    return "\n".join(["--- рядки з помилками:"] + hits[-n:] + ["--- останні рядки:"] + lines[-30:])


# варіанти ядер: лише реалізація MoE/attention, математика моделі та сама
KERNELS = [
    ("megablox", ["megablox=True", "sparse_matmul=True"]),
    ("ragged", ["megablox=False", "sparse_matmul=True"]),
    ("ragged_dotattn", ["megablox=False", "sparse_matmul=True", "attention=dot_product"]),
]


def probe(pdb, remat, kname, kargs, steps=12):
    ga = GLOBAL_PER_DEVICE // pdb
    name = f"probe_{kname}_b{pdb}_{remat}"
    cmd = TRAIN + COMMON + [
        f"run_name={name}", "dataset_type=synthetic", "enable_checkpointing=False", f"steps={steps}",
        "log_period=1", f"per_device_batch_size={pdb}", f"gradient_accumulation_steps={ga}",
        f"remat_policy={remat}"] + kargs
    log("PROBE", name)
    t = time.time()
    rc, text = run_logged(cmd, name, 1500)
    shutil.rmtree(f"{OUT}/{name}", ignore_errors=True)
    tps = [float(x) for x in re.findall(r"Tokens/s/device: ([\d.]+)", text)]
    mem = re.findall(r"Total estimated memory size: ([\d.]+) GB", text)
    ok = rc == 0 and len(tps) >= 4
    speed = sum(tps[3:]) / len(tps[3:]) if ok else 0.0  # перші кроки — компіляція/прогрів
    oom = "RESOURCE_EXHAUSTED" in text or "Out of memory" in text
    log(f"  -> ok={ok} rc={rc} oom={oom} tokens/s/device={speed:.0f} mem={mem[-1] if mem else '?'}GB "
        f"({time.time() - t:.0f}s)")
    if not ok:
        log(error_summary(text))
    return ok, speed, oom


if prev_choice:
    choice = json.load(open(prev_choice))
    log("Беру збережений режим:", choice)
else:
    choice = None
    for kname, kargs in KERNELS:
        for pdb in (8, 4, 2, 1):  # найбільший батч на чип, що влазить (глобальний батч не змінюється)
            ok, speed, oom = probe(pdb, "full", kname, kargs)
            if ok:
                choice = {"pdb": pdb, "remat": "full", "speed": speed, "kernel": kname, "kargs": kargs}
                break
            if not oom:
                break  # не пам'ять, а помилка ядра — пробуємо інший варіант ядер
        if choice:
            break
    assert choice, "жоден варіант не запрацював — див. рядки з помилками вище"
    for remat in ("save_dot_except_mlp", "minimal"):  # менше перерахунку = швидше, якщо влазить
        ok, speed, _ = probe(choice["pdb"], remat, choice["kernel"], choice["kargs"])
        if ok and speed > choice["speed"] * 1.03:
            choice = {**choice, "remat": remat, "speed": speed}
    log("ОБРАНО:", choice)
json.dump(choice, open(f"{STATE}/perf_choice.json", "w"))

# ---------------------------------------------------------------- 4. тренування
cmd = TRAIN + COMMON + [
    f"run_name={RUN}",
    f"per_device_batch_size={choice['pdb']}",
    f"gradient_accumulation_steps={GLOBAL_PER_DEVICE // choice['pdb']}",
    f"remat_policy={choice['remat']}",
    *choice.get("kargs", []),
    "enable_checkpointing=True", "checkpoint_period=250", "max_num_checkpoints_to_keep=1",
    f"metrics_file={STATE}/metrics.jsonl",
] + resume
log("ТРЕНУВАННЯ:", " ".join(cmd[3:]))
train_log = open(f"{STATE}/train.log", "a")
wait_tpu_free()
time.sleep(10)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
last_print = 0
for line in proc.stdout:
    train_log.write(line)
    if "completed step" in line:
        m = re.search(r"completed step: (\d+)", line)
        if m and (int(m.group(1)) % 50 == 0 or time.time() - last_print > 600):
            print(line.strip()[60:300], flush=True)
            last_print = time.time()
    elif any(k in line for k in ("Error", "Traceback", "RESOURCE_EXHAUSTED", "Saving checkpoint", "saved",
                                 "Restoring", "restored")):
        print(line.rstrip()[:300], flush=True)
    if time.time() > DEADLINE:
        log("Дедлайн сесії — зупиняю тренування (останній чекпоінт ≤250 кроків тому)")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill()
        break
proc.wait()
train_log.close()
log("Тренування завершилось з кодом", proc.returncode)
if proc.returncode not in (0, None, -signal.SIGINT):
    log(error_summary(open(f"{STATE}/train.log", errors="replace").read()[-200_000:]))

# ---------------------------------------------------------------- 5. підсумок
final = ckpt_steps(CKPT_DIR)
log("Чекпоінти у виводі:", [s for s, _ in final])
# у виводі лишаємо тільки останній завершений чекпоінт (ліміт /kaggle/working ~20 ГБ)
for s, d in final[:-1]:
    shutil.rmtree(d, ignore_errors=True)
shutil.rmtree(f"{OUT}/{RUN}/tensorboard", ignore_errors=True)
sh("du -sh /kaggle/working/* 2>/dev/null")
