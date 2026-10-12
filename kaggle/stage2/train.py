"""
Стадія 2 на Kaggle TPU v5e-8: продовження Lewi Surge 1.4B-A0.3B з ваг стадії 1 на суміші
англійська + код (~30%, FIM) + uk/es/pt/fr/de + математика + SVG/дизайн.

Хід сесії:
  1. ставимо MaxText, латка Muon для qwen3_5, конфіг stage2.yml (вшитий у скрипт)
  2. перша сесія: HF-ваги стадії 1 (датасет lewi-surge-s1-hf) → MaxText; далі — продовження з власного чекпоінта
  3. підбір режиму виконання (лише в першій сесії), кількість кроків під бюджет годин
  4. тренування до дедлайну (~8 год 15 хв), чекпоінт лишається у виводі
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
OUT = "/dev/shm/out"  # повний стан 3.45B (~29 ГБ) не влазить у /kaggle/working (20 ГБ) → RAM-диск (377 ГБ RAM)
RUN = "stage2g"  # вирощена модель 3.45B: 16 шарів, 128 експертів
SESSION_HOURS = 7.4  # чистого тренування за сесію; між сесіями переносимо лише ваги (bf16) через /kaggle/working
STAGE2_YML = __STAGE2_YML__
CONV_WRAPPER = __CONV_WRAPPER__
EXPORT_PY = r'''
# Повний стан MaxText → лише params у bf16 (для переносу між сесіями, ~7 ГБ замість ~29 ГБ)
import sys, numpy as np, jax, ml_dtypes, orbax.checkpoint as ocp
from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
src, dst = sys.argv[1], sys.argv[2]
ck = ocp.PyTreeCheckpointer()
meta = ck.metadata(src)
meta = getattr(meta, "item_metadata", meta)
meta = meta.tree if hasattr(meta, "tree") else meta
meta = {"params": meta["params"]}
rargs = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta)
t = ck.restore(src, args=ocp.args.PyTreeRestore(item=meta, restore_args=rargs, partial_restore=True))
p = jax.tree.map(lambda a: np.asarray(a).astype(ml_dtypes.bfloat16) if np.asarray(a).dtype == np.float32 else np.asarray(a),
                 t["params"]["params"])
n = sum(int(np.prod(a.shape)) for a in jax.tree.leaves(p))
print(f"параметрів: {n / 1e9:.3f}B", flush=True)
save_weights_to_checkpoint(dst, {"params": p}, 1, True, True)
print("EXPORT_OK", flush=True)
'''
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
tok_src = next(iter(glob.glob("/kaggle/input/**/moe-assets/**/tokenizer.json", recursive=True)), None)     or next(iter(glob.glob("/kaggle/input/**/tokenizer_config.json", recursive=True)), None)
assert tok_src, "немає токенайзера у /kaggle/input (датасет moe-assets не підключено?)"
ASSETS = os.path.dirname(tok_src)
log("assets:", ASSETS, sorted(os.listdir(ASSETS)))
sh("nproc; free -g; df -h /kaggle/working | tail -1")

# як на Colab TPU, де стадія 2 перевірена (jax 0.11.2 + libtpu 0.0.50). uv --resolution=lowest у v1 дав
# несумісну пару jax/libtpu → segfault (rc=-11) на старті тренування.
sh(f"{sys.executable} -m pip install -q --ignore-installed Markdown 'maxtext[tpu]==0.2.5'", 1800)
sh(f"{sys.executable} -m pip list 2>/dev/null | grep -E '^(jax|jaxlib|libtpu) '")
# вбудований scikit-learn ламає імпорт transformers після оновлення scipy (перевірено на Colab TPU)
sh(f"{sys.executable} -m pip uninstall -y -q scikit-learn keras-nlp keras-hub")  # keras_nlp ламає імпорт нового transformers (v7)
sh(f"{sys.executable} -c \"import jax; print(jax.__version__, jax.devices())\"")

# батьківський процес НЕ імпортує maxtext/jax — інакше може тримати TPU і дочірні процеси не стартують
PKG = subprocess.run([sys.executable, "-c", "import maxtext, os; print(os.path.dirname(maxtext.__file__))"],
                     capture_output=True, text=True).stdout.strip().splitlines()[-1]
log("maxtext:", PKG)
CFG = os.path.join(PKG, "configs", "stage2.yml")
open(CFG, "w").write(STAGE2_YML)
open(f"{WORK}/conv_wrapper.py", "w").write(CONV_WRAPPER)
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


prev = sorted(glob.glob("/kaggle/input/**/params_bf16/*/items", recursive=True))
progress = {"sessions": 0, "steps_total": 0}
pj = next(iter(glob.glob("/kaggle/input/**/state/progress.json", recursive=True)), None)
if pj:
    progress = json.load(open(pj))
resume = []
if prev:
    resume = [f"load_parameters_path={prev[-1]}"]
    log(f"ПРОДОВЖУЮ з ваг попередньої сесії ({prev[-1]}); пройдено кроків: {progress['steps_total']}")
else:
    hf = next(iter(glob.glob("/kaggle/input/**/model.safetensors.index.json", recursive=True)), None)
    assert hf, "немає ваг вирощеної моделі (датасет lewi-surge-s2g-hf)"
    HF_DIR = os.path.dirname(hf)
    DIMS = ("base_emb_dim=1024 base_num_decoder_layers=16 base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 "
            "base_mlp_dim=512 base_moe_mlp_dim=512 num_experts=128 shared_experts=1 num_experts_per_tok=6 "
            "vocab_size=131200 logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 "
            "gdn_num_key_heads=4 gdn_num_value_heads=8 gdn_key_head_dim=128 gdn_value_head_dim=128 "
            "gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 use_multimodal=False")
    CONV_OUT = "/tmp/s1_mt"  # не у /kaggle/working: потрібен лише в першій сесії
    rc = sh(f"LEWI_LAYERS=16 LEWI_EXPERTS=128 JAX_PLATFORMS=cpu {sys.executable} {WORK}/conv_wrapper.py to_maxtext {CFG} model_name=qwen3.5-35b-a3b "
            f"override_model_config=True dataset_type=synthetic {DIMS} base_output_directory={CONV_OUT} hardware=cpu "
            f"skip_jax_distributed_system=True scan_layers=True --hf_model_path={HF_DIR} "
            f"> {STATE}/convert.log 2>&1", 3600)
    items = next(iter(glob.glob(f"{CONV_OUT}/**/items", recursive=True)), None)
    if rc or not items:
        log("--- convert.log (хвіст):\n" + open(f"{STATE}/convert.log", errors="replace").read()[-4000:])
        raise SystemExit("конвертація ваг не вдалася")
    resume = [f"load_parameters_path={items}"]
    log("Старт стадії 2 з ваг стадії 1:", items)

prev_choice = next(iter(glob.glob("/kaggle/input/**/state/perf_choice.json", recursive=True)), None)

# ---------------------------------------------------------------- 3. підбір режиму виконання
TRAIN = [sys.executable, "-m", "maxtext.trainers.pre_train.train", CFG]
COMMON = [f"tokenizer_path={TOK}", f"base_output_directory={OUT}", "base_num_decoder_layers=16", "num_experts=128",
          "learning_rate=1.2e-3"]  # LR нижчий: на 2e-3 вирощена модель стояла на плато
GLOBAL_PER_DEVICE = 4  # з stage2.yml: 8 чипів × 4 × 4096 = 131k токенів на крок (як стадія 1)


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
        for pdb in (4, 2, 1):  # найбільший батч на чип, що влазить (глобальний батч не змінюється)
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

# ---------------------------------------------------------------- 3b. суміш даних і кількість кроків
MIX = [  # code-first (~50% код): якісний annealing-код + інструкції в чат-форматі + GitHub; решта — знання й мови
    ("ann_algo", 0.08), ("ann_snip", 0.06), ("ann_qa", 0.05),
    ("ins_magicoder", 0.012), ("ins_codefeedback", 0.02), ("ins_selfoss", 0.004), ("ins_opcsft", 0.04),  # ~1-1.5 проходи
    ("code1", 0.08), ("code2", 0.06), ("codeweb", 0.04), ("websight", 0.015),
    ("cosmo", 0.14),
    ("fineweb_edu_a", 0.02), ("fineweb_edu_b", 0.02), ("fineweb_edu_c", 0.02), ("fineweb_edu_d", 0.02),
    ("math", 0.08), ("uk", 0.07), ("es", 0.015), ("pt", 0.015), ("fr", 0.015), ("de", 0.015),
    ("svg", 0.02), ("text2svg", 0.02),
]
parts = []
for name, w in MIX:
    b = next(iter(glob.glob(f"/kaggle/input/**/{name}.bin", recursive=True)), None)
    if not b:
        log(f"УВАГА: шард {name} не знайдено — пропускаю")
        continue
    parts.append(f"{b[:-4]},{w}")
assert len(parts) >= 18, f"знайдено лише {len(parts)} шардів"
DATA = ["dataset_type=grain", "grain_file_type=mmap", "grain_train_files=" + ";".join(parts)]
log("ДАНІ:", ";".join(parts))
tok_per_step = 8 * GLOBAL_PER_DEVICE * 4096
n_steps = int(choice["speed"] * 8 * 3600 * SESSION_HOURS / tok_per_step)
log(f"КРОКІВ на сесію ({SESSION_HOURS} год): {n_steps} (≈{n_steps * tok_per_step / 1e9:.1f}B токенів)")
json.dump(choice, open(f"{STATE}/perf_choice.json", "w"))
# сталий LR після короткого прогріву (спад зробимо окремою annealing-сесією)
STEPS = [f"steps={n_steps}", f"learning_rate_schedule_steps={n_steps}", "wsd_decay_steps_fraction=0.0",
         "warmup_steps_fraction=0.02"]

# ---------------------------------------------------------------- 4. тренування
cmd = TRAIN + COMMON + [
    f"run_name={RUN}",
    f"per_device_batch_size={choice['pdb']}",
    f"gradient_accumulation_steps={GLOBAL_PER_DEVICE // choice['pdb']}",
    f"remat_policy={choice['remat']}",
    *choice.get("kargs", []), *DATA, *STEPS,
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

# ---------------------------------------------------------------- 5. підсумок: ваги (bf16) у вивід
final = ckpt_steps(CKPT_DIR)
log("Чекпоінти сесії:", [x for x, _ in final])
if final:
    last_step, last_dir = final[-1]
    open(f"{WORK}/export_params.py", "w").write(EXPORT_PY)
    rc = sh(f"JAX_PLATFORMS=cpu {sys.executable} {WORK}/export_params.py {last_dir}/items {WORK}/params_bf16 "
            f"> {STATE}/export.log 2>&1", 3600)
    if rc:
        log(error_summary(open(f"{STATE}/export.log", errors="replace").read()))
    progress = {"sessions": progress["sessions"] + 1, "steps_total": progress["steps_total"] + last_step,
                "last_session_steps": last_step}
    json.dump(progress, open(f"{STATE}/progress.json", "w"))
    log("ПРОГРЕС:", progress)
else:
    log("УВАГА: у сесії не збережено жодного чекпоінта")
sh("du -sh /kaggle/working/* 2>/dev/null")
