"""
Lewi Surge 3.45B: розширення контексту (стадія LC) на Kaggle TPU v5e-8; той самий скрипт — тест на Colab v5e-1.

  MODE=full  (Kaggle): HF-ваги SFT2 → MaxText (лише 1-ша сесія) → проба швидкості → тренування на seq SEQ до дедлайну
                       → ваги bf16 у /kaggle/working/params_bf16 (вхід наступної сесії)
  MODE=test  (Colab v5e-1, 1 чип): та сама архітектура в мініатюрі + перевірки чекпоінтів:
                       тренування з чекпоінтами → автопродовження повного стану → експорт bf16 → load_parameters_path
                       → AOT-компіляція ПОВНОЇ моделі під топологію v5e-8 (пам'ять/шардинг без справжніх 8 чипів)

Паралелізм (див. MaxText base.yml/types.py 0.2.5): 8-way FSDP (ici_fsdp_parallelism=-1), без EP/TP/CP —
на 32k-64k кожен чип тримає свою послідовність; логіти 131k-словника ріжемо num_vocab_tiling; attention=flash (Splash).
Context parallelism не вмикаємо: для GDN він вимагає context_parallel_load_balance=False і на 32-64k не потрібен.
Чекпоінти: новий run_name на сесію (інакше MaxText тихо бере чекпоінт з каталогу запуску замість load_parameters_path),
повний стан → /dev/shm, max_num_checkpoints_to_keep=1, власний дедлайн (Kaggle вбиває жорстко), чекаємо фіналізації.
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
os.environ.update(PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="1")  # живий лог дочірніх процесів
MODE = os.environ.get("MODE", "full")
KAGGLE = os.path.exists("/kaggle")
WORK = "/kaggle/working" if KAGGLE else "/content/work"
OUT = "/dev/shm/out"
STATE = f"{WORK}/state"
for d in (WORK, STATE, OUT):
    os.makedirs(d, exist_ok=True)
DEADLINE = T0 + (8 * 3600 + 20 * 60 if KAGGLE else 3 * 3600)
SEQ = int(os.environ.get("SEQ", "131072"))
# підібрано AOT-компіляцією на Colab під v5e-8 (15.75 ГБ/чип): довга послідовність ріжеться між чипами
# context parallelism; batch на чип дробовий, інакше CP не зменшує токени на чип. 32k: CP4 pdb0.25; 64k/128k: CP8 pdb0.125
CP, PDB = (4, 0.25) if SEQ <= 32768 else (8, 0.125)
STAGE2_YML = __STAGE2_YML__
CONV_WRAPPER = __CONV_WRAPPER__
EXPORT_PY = r'''
# Повний стан MaxText → лише params у bf16 (~7 ГБ) у форматі, який читає load_parameters_path
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
print(f"параметрів: {sum(int(np.prod(a.shape)) for a in jax.tree.leaves(p)) / 1e9:.3f}B", flush=True)
save_weights_to_checkpoint(dst, {"params": p}, 1, True, True)
print("EXPORT_OK", flush=True)
'''

FULL_DIMS = ["base_emb_dim=1024", "base_num_decoder_layers=16", "base_num_query_heads=8", "base_num_kv_heads=2",
             "head_dim=128", "base_mlp_dim=512", "base_moe_mlp_dim=512", "num_experts=128", "shared_experts=1",
             "num_experts_per_tok=6", "vocab_size=131200", "logits_via_embedding=True",
             "inhomogeneous_layer_cycle_interval=4", "gdn_num_key_heads=4", "gdn_num_value_heads=8",
             "gdn_key_head_dim=128", "gdn_value_head_dim=128", "gdn_conv_kernel_dim=4", "use_mrope=False",
             "mtp_num_layers=0", "use_multimodal=False"]
# мініатюра для одного чипа: та сама структура блоку (GDN×3 + attention), менше шарів/експертів
TEST_DIMS = [d for d in FULL_DIMS if not d.startswith(("base_num_decoder_layers", "num_experts="))] + \
            ["base_num_decoder_layers=4", "num_experts=16"]


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} хв]", *a, flush=True)


def sh(cmd, timeout=None):
    log("$", cmd[:400])
    r = subprocess.run(cmd, shell=True, timeout=timeout)
    log(f"exit {r.returncode}")
    return r.returncode


# ---------------------------------------------------------------- 1. середовище
# Падіння v7/v8: `pip install --ignore-installed maxtext` поклав новий transformers поверх передвстановленого
# і лишив старі модулі → "Offending backend: keras_nlp". Лікування: прибрати transformers/keras-* повністю,
# поставити закріплену версію (5.19.0 — з нею конвертація працювала на Colab/Modal).
sh(f"{sys.executable} -m pip uninstall -y -q transformers keras-nlp keras-hub keras scikit-learn tensorflow 2>/dev/null")
for d in glob.glob("/usr/local/lib/python3*/*-packages/transformers*"):
    shutil.rmtree(d, ignore_errors=True)
assert sh(f"{sys.executable} -m pip install -q --ignore-installed Markdown 'maxtext[tpu]==0.2.5'", 2400) == 0
assert sh(f"{sys.executable} -m pip install -q --force-reinstall --no-deps 'transformers==5.19.0'", 600) == 0
sh(f"{sys.executable} -m pip list 2>/dev/null | grep -E '^(jax|jaxlib|libtpu|transformers|orbax-checkpoint|grain) '")
# той самий ефект з scipy: старі .so після --ignore-installed → "scipy._cyutility does not export ..." при імпорті
# transformers.modeling_utils (його тягне конвертер) — чиста перевстановка (перевірено на Colab)
sh(f"{sys.executable} -m pip uninstall -y -q scipy")
for d in glob.glob("/usr/local/lib/python3*/*-packages/scipy*"):
    shutil.rmtree(d, ignore_errors=True)
assert sh(f"{sys.executable} -m pip install -q scipy", 600) == 0
assert sh(f"{sys.executable} -c 'import scipy.spatial, transformers, transformers.modeling_utils, maxtext; "
          f"print(transformers.__version__)'") == 0, "імпорт впав"
# батьківський процес не імпортує jax: інакше тримає TPU і дочірні процеси не стартують
PKG = subprocess.run([sys.executable, "-c", "import maxtext, os; print(os.path.dirname(maxtext.__file__))"],
                     capture_output=True, text=True).stdout.strip().splitlines()[-1]
CFG = os.path.join(PKG, "configs", "stage2.yml")
open(CFG, "w").write(STAGE2_YML)
open(f"{WORK}/conv_wrapper.py", "w").write(CONV_WRAPPER)
types_py = os.path.join(PKG, "configs", "types.py")
src = open(types_py).read()
anchor = 'if self.opt_type == "muon" and self.decoder_block not in ['
i = src.find(anchor)
if i >= 0 and "QWEN3_5" not in src[i:i + 600]:
    open(types_py, "w").write(src[:i + len(anchor)] + "\n        DecoderBlockType.QWEN3_5," + src[i + len(anchor):])
    log("[patch] muon: QWEN3_5")
sh(f"{sys.executable} -c \"import jax; print(jax.__version__, jax.device_count(), jax.devices()[0].device_kind)\"")

TRAIN = [sys.executable, "-m", "maxtext.trainers.pre_train.train", CFG]
COMPILE = [sys.executable, "-m", "maxtext.trainers.pre_train.train_compile", CFG]
SFT = [sys.executable, "-m", "maxtext.trainers.post_train.sft.train_sft_native", CFG]


def SFT_DATA(parquet):
    # маска: лос лише на відповідях асистента (+<|im_end|>); шаблон без <think> з tokenizer_config.json
    return ["dataset_type=hf", "hf_path=parquet", f"hf_train_files={parquet}", "train_data_columns=['messages']",
            "sft_train_on_completion_only=True", "packing=True", "num_epoch=2"]


def no_async(args):
    # train_sft_native не чекає асинхронного збереження на виході ("cannot schedule new futures after shutdown"
    # на Colab-тесті, чекпоінт втрачено) → для SFT синхронні чекпоінти
    return [a for a in args if not a.startswith("async_checkpointing=")] + ["async_checkpointing=False"]
BUSY = ("already in use", "TPU initialization failed", "Unable to initialize backend 'tpu'", "libtpu_lockfile",
        "Couldn't open device")
NOISE = ("UserWarning", "warnings.warn", "FutureWarning", "Config param", "hugepages")


def wait_tpu_free():
    for _ in range(36):
        busy = subprocess.run("lsof -t /dev/vfio/* /dev/accel* 2>/dev/null | head -1", shell=True,
                              capture_output=True, text=True).stdout.strip()
        if not busy:
            return
        time.sleep(5)
    log("УВАГА: TPU зайнятий процесом", busy)


def errs(text, n=30):
    lines = [l for l in text.splitlines() if not any(k in l for k in NOISE)]
    hits = [l for l in lines if re.search(r"Error|Exception|error:|RESOURCE_EXHAUSTED|Traceback|FAILED", l)]
    return "\n".join(["--- помилки:"] + hits[-n:] + ["--- хвіст:"] + lines[-25:])


def run(cmd, name, timeout, stream=False):
    """Запуск дочірнього процесу з логом; повтор, якщо TPU ще не звільнився після попереднього."""
    path = f"{STATE}/{name}.log"
    for attempt in range(3):
        wait_tpu_free()
        time.sleep(5)
        log("RUN", name, " ".join(cmd[4:])[:600])
        with open(path, "w") as fh:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            killed = False
            for line in p.stdout:
                fh.write(line)
                if stream and ("completed step" in line or re.search(r"Traceback|RESOURCE_EXHAUSTED|saved|Saving", line)):
                    m = re.search(r"completed step: (\d+)", line)
                    if not m or int(m.group(1)) % 25 == 0:
                        print(line.rstrip()[:260], flush=True)
                if time.time() > timeout and not killed:
                    log("ДЕДЛАЙН — SIGINT, чекаю збереження")
                    p.send_signal(signal.SIGINT)
                    killed = True
            p.wait()
        text = open(path, errors="replace").read()
        if p.returncode not in (0, -signal.SIGINT) and any(k in text for k in BUSY) and "completed step" not in text:
            log(f"TPU зайнятий (спроба {attempt + 1})")
            time.sleep(30)
            continue
        return p.returncode, text
    return p.returncode, text


def ckpts(run_dir):
    """Лише завершені чекпоінти: числова тека з items/ (тимчасові *.orbax-checkpoint-tmp-* відкидаємо)."""
    out = []
    for d in glob.glob(f"{run_dir}/checkpoints/*"):
        b = os.path.basename(d)
        if b.isdigit() and os.path.isdir(f"{d}/items") and os.path.exists(f"{d}/items/_METADATA"):
            out.append((int(b), d))
    return sorted(out)


def export(items, dst):
    open(f"{WORK}/export_params.py", "w").write(EXPORT_PY)
    shutil.rmtree(dst, ignore_errors=True)
    rc = sh(f"JAX_PLATFORMS=cpu {sys.executable} {WORK}/export_params.py {items} {dst} > {STATE}/export.log 2>&1", 3600)
    text = open(f"{STATE}/export.log", errors="replace").read()
    ok = rc == 0 and "EXPORT_OK" in text
    if not ok:
        log(errs(text))
    found = glob.glob(f"{dst}/**/items", recursive=True) or ([dst] if os.path.exists(f"{dst}/_METADATA") else [])
    return ok, (found[0] if found else None)


def losses(text):
    return [float(x) for x in re.findall(r"completed step: \d+.*? loss: ([\d.]+)", text)]


BASE = ["hardware=tpu", "skip_jax_distributed_system=True", f"base_output_directory={OUT}", "scan_layers=True",
        "sparse_matmul=True", "megablox=True", "attention=flash", "remat_policy=full", "dtype=bfloat16",
        "weight_dtype=float32", "ici_fsdp_parallelism=-1", "ici_expert_parallelism=1",
        "async_checkpointing=True", "max_num_checkpoints_to_keep=1"]

# ================================================================= MODE=test (Colab v5e-1)
if MODE == "test":
    ONLY = os.environ.get("TEST_ONLY", "12345")
    T = TEST_DIMS + BASE + ["ici_context_parallelism=1", "dataset_type=synthetic", "learning_rate=3e-4", "log_period=1",
                            "per_device_batch_size=1", "max_target_length=8192", "num_vocab_tiling=8"]
    res = {}
    if ONLY != "5":
        # 1) тренування з чекпоінтами (зупинка на 30, чекпоінти кожні 10)
        rc, txt = run(TRAIN + T + ["run_name=t_full", "steps=30", "enable_checkpointing=True", "checkpoint_period=10"],
                      "t1_train", time.time() + 3600)
        res["1_train"] = rc == 0 and len(losses(txt)) >= 25
        ck = ckpts(f"{OUT}/t_full")
        res["1_ckpts"] = [s for s, _ in ck]
        log("1) тренування:", res["1_train"], "чекпоінти:", res["1_ckpts"], "лос:", losses(txt)[:2], losses(txt)[-2:])
        if not res["1_train"]:
            log(errs(txt))
        # 2) автопродовження повного стану: той самий run_name, steps=40 → має стартувати з кроку 30
        rc, txt2 = run(TRAIN + T + ["run_name=t_full", "steps=40", "enable_checkpointing=True", "checkpoint_period=10"],
                       "t2_resume", time.time() + 3600)
        steps2 = [int(x) for x in re.findall(r"completed step: (\d+)", txt2)]
        res["2_resume"] = rc == 0 and bool(steps2) and min(steps2) >= 29 and abs(losses(txt2)[0] - losses(txt)[-1]) < 1.0
        log("2) продовження:", res["2_resume"], "перший крок:", steps2[:1], "лос:", losses(txt2)[:2])
        if not res["2_resume"]:
            log(errs(txt2))
        # 3) експорт bf16 → load_parameters_path у НОВИЙ run_name: лос має бути як в кінці, не ~ln(131200)=11.8
        last = ckpts(f"{OUT}/t_full")
        ok, items = export(f"{last[-1][1]}/items", f"{WORK}/t_params") if last else (False, None)
        rc, txt3 = run(TRAIN + T + ["run_name=t_load", "steps=5", "enable_checkpointing=False",
                                    f"load_parameters_path={items}"], "t3_load", time.time() + 1800) if items else (1, "")
        l3 = losses(txt3)
        res["3_export_load"] = ok and rc == 0 and bool(l3) and l3[0] < losses(txt)[0] - 3  # випадкові ваги дали б ~21
        log("3) експорт+завантаження:", res["3_export_load"], items, "лос:", l3[:3])
        if not res["3_export_load"]:
            log(errs(txt3 or open(f"{STATE}/export.log").read()))
        # 4) AOT-компіляція ПОВНОЇ моделі на v5e-8 при SEQ: OOM і помилки шардингу ловимо без Kaggle
        for seq, extra in ((131072, ["per_device_batch_size=0.125", "ici_context_parallelism=8", "num_vocab_tiling=32",
                                     "context_parallel_load_balance=False"]),
                           (16384, ["per_device_batch_size=1", "ici_context_parallelism=1", "num_vocab_tiling=16"])):
            rc, txt4 = run(COMPILE + FULL_DIMS + BASE + extra + [
                "compile_topology=v5e-8", "compile_topology_num_slices=1", "dataset_type=synthetic",
                f"max_target_length={seq}", "run_name=aot", "steps=10"], f"t4_aot_{seq}", time.time() + 3600)
            mem = re.findall(r"Total (?:estimated )?memory size: ([\d.]+) ?GB", txt4) or re.findall(r"([\d.]+) ?GB", txt4)[-3:]
            res[f"4_aot_{seq}"] = rc == 0 and "RESOURCE_EXHAUSTED" not in txt4
            log(f"4) AOT v5e-8 seq={seq}:", res[f"4_aot_{seq}"], "пам'ять:", mem[-1:] if mem else "?")
            if not res[f"4_aot_{seq}"]:
                log(errs(txt4))
    # 5) SFT-фаза (train_sft_native, hf parquet, маска на відповідях) з чекпоінта кроку 1 — як після контексту
    import pandas as pd
    convs = [[{"role": "user", "content": f"Question {i}: what is {i} plus {i}? Explain briefly. " * 20},
              {"role": "assistant", "content": f"{i} plus {i} is {2 * i}, because adding a number to itself doubles it. " * 20}]
             for i in range(4000)]
    pd.DataFrame({"messages": convs}).to_parquet(f"{WORK}/t_sft.parquet")
    tdir = f"{WORK}/t_tok"
    os.makedirs(tdir, exist_ok=True)
    open(f"{tdir}/tokenizer.json", "w").write(open(next(iter(glob.glob("/content/tok/tokenizer.json")))).read())
    tc = json.load(open("/content/tok/tokenizer_config.json"))
    json.dump(tc, open(f"{tdir}/tokenizer_config.json", "w"))
    last = ckpts(f"{OUT}/t_full")
    rc, txt5 = run(SFT + no_async(T) + ["run_name=t_sft", "steps=20", "enable_checkpointing=True", "checkpoint_period=10",
                              f"load_parameters_path={last[-1][1]}/items", f"tokenizer_path={tdir}", *SFT_DATA(f"{WORK}/t_sft.parquet"),
                              "opt_type=adamw", "learning_rate=5e-5"], "t5_sft", time.time() + 2400)
    l5 = losses(txt5)
    res["5_sft"] = rc == 0 and len(l5) >= 15 and bool(ckpts(f"{OUT}/t_sft"))
    log("5) SFT:", res["5_sft"], "лос:", l5[:2], l5[-2:], "чекпоінти:", [s for s, _ in ckpts(f"{OUT}/t_sft")])
    if not res["5_sft"]:
        log(errs(txt5))
    json.dump(res, open(f"{STATE}/test_result.json", "w"), indent=1)
    log("ПІДСУМОК ТЕСТУ:", res)
    raise SystemExit(0)

# ================================================================= MODE=full (Kaggle v5e-8)
progress = {"sessions": 0, "steps_total": 0}
pj = next(iter(glob.glob("/kaggle/input/**/state/progress.json", recursive=True)), None)
if pj:
    progress = json.load(open(pj))
SESSION = progress["sessions"] + 1
prev = sorted(glob.glob("/kaggle/input/**/params_bf16/**/items", recursive=True))
if prev and os.path.exists(f"{prev[-1]}/_METADATA"):
    LOAD = prev[-1]
    log(f"Сесія {SESSION}: продовжую з ваг попередньої сесії {LOAD}")
else:
    hf = next(iter(glob.glob("/kaggle/input/**/model.safetensors.index.json", recursive=True)), None)
    assert hf, "немає HF-ваг (датасет lewi-surge-sft2-hf)"
    conv = "/tmp/sft2_mt"
    rc = sh(f"LEWI_LAYERS=16 LEWI_EXPERTS=128 JAX_PLATFORMS=cpu {sys.executable} {WORK}/conv_wrapper.py to_maxtext {CFG} "
            f"model_name=qwen3.5-35b-a3b override_model_config=True dataset_type=synthetic {' '.join(FULL_DIMS)} "
            f"base_output_directory={conv} hardware=cpu skip_jax_distributed_system=True scan_layers=True "
            f"--hf_model_path={os.path.dirname(hf)} > {STATE}/convert.log 2>&1", 3600)
    LOAD = next(iter(glob.glob(f"{conv}/**/items", recursive=True)), None)
    if rc or not LOAD:
        log(errs(open(f"{STATE}/convert.log", errors="replace").read()))
        raise SystemExit("конвертація не вдалася")
    log("Сесія 1: старт з SFT2", LOAD)

TOK = f"{WORK}/tokenizer"
os.makedirs(TOK, exist_ok=True)
# токенайзер + шаблон чату без <think> (той, на якому тренувався SFT2) — з датасету ваг
tok_src = next(iter(glob.glob("/kaggle/input/**/lewi-surge-sft2-hf/**/tokenizer.json", recursive=True)
                    + glob.glob("/kaggle/input/**/tokenizer.json", recursive=True)))
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    if os.path.exists(os.path.join(os.path.dirname(tok_src), f)):
        shutil.copy(os.path.join(os.path.dirname(tok_src), f), TOK)
assert "<think>" not in json.load(open(f"{TOK}/tokenizer_config.json")).get("chat_template", ""), "шаблон з <think>"

# суміш: ~70% довгі документи + ~30% короткі (щоб не просісти на звичайних задачах — рецепт Fu et al. 2024)
MIX = [("long_pg19", 0.10), ("long_pg19b", 0.12), ("long_code", 0.18), ("long_swe", 0.10),
       ("long_ud", 0.08), ("long_edu", 0.06), ("long_math", 0.04),
       ("ann_algo", 0.05), ("ann_snip", 0.03), ("ins_opcsft", 0.03), ("ins_codefeedback", 0.02),
       ("cosmo", 0.06), ("math", 0.03), ("uk", 0.04), ("code1", 0.04), ("svg", 0.01), ("text2svg", 0.01)]
parts = []
for name, w in MIX:
    b = next(iter(glob.glob(f"/kaggle/input/**/{name}.bin", recursive=True)), None)
    if b:
        parts.append(f"{b[:-4]},{w}")
    else:
        log(f"УВАГА: шард {name} відсутній")
assert sum(1 for p in parts if "long_" in p) >= 3, "немає довгих шардів (moe-tok3-long)"
DATA = ["dataset_type=grain", "grain_file_type=mmap", "grain_train_files=" + ";".join(parts), "num_epoch=4",
        f"data_shuffle_seed={SESSION}"]  # параметри-тільки між сесіями → ітератор з нуля, тож інше перемішування
RUN = f"lc{SEQ // 1024}k_s{SESSION}"  # новий каталог на сесію: інакше MaxText взяв би старий чекпоінт замість LOAD
TILES = 16 if SEQ <= 65536 else 32
COMMON = FULL_DIMS + BASE + DATA + [f"tokenizer_path={TOK}", f"max_target_length={SEQ}", f"per_device_batch_size={PDB}",
                                    f"ici_context_parallelism={CP}", "context_parallel_load_balance=False",
                                    f"num_vocab_tiling={TILES}", f"load_parameters_path={LOAD}"]
TOK_STEP = int(8 * PDB * SEQ)  # токенів на крок

# проба швидкості (без чекпоінтів) → кроки під бюджет; розклад LR завершується в межах сесії
rc, ptxt = run(TRAIN + COMMON + ["run_name=probe", "steps=12", "enable_checkpointing=False", "log_period=1",
                                 "learning_rate=0"], "probe", time.time() + 2400)
tps = [float(x) for x in re.findall(r"Tokens/s/device: ([\d.]+)", ptxt)][4:]
pl = losses(ptxt)
log("ПРОБА:", "rc", rc, "ток/с/чип", tps[-3:], "лос", pl[:3])
if rc != 0 or not tps:
    log(errs(ptxt))
    raise SystemExit("проба не пройшла")
assert pl[0] < 6.0, f"ваги не підхопились: лос {pl[0]} (очікується ~2-3)"
shutil.rmtree(f"{OUT}/probe", ignore_errors=True)
speed = sum(tps) / len(tps)
budget_s = DEADLINE - time.time() - 75 * 60  # дві компіляції + старт SFT (токенізація) + експорт + запас
LC_SHARE = 0.62  # решта — SFT у цій же сесії (без другої черги на Kaggle)
steps = int(budget_s * LC_SHARE * speed * 8 / TOK_STEP)
SFT_SEQ = 16384
sft_steps = int(budget_s * (1 - LC_SHARE) * speed * 8 / (8 * SFT_SEQ))  # консервативно: на 16k швидше за ток/с
log(f"КРОКІВ: {steps} (~{steps * TOK_STEP / 1e9:.2f}B токенів, {speed:.0f} ток/с/чип)")
LR = ["learning_rate=3e-4", "lr_schedule_type=wsd", "warmup_steps_fraction=0.03", "wsd_decay_steps_fraction=0.2",
      "learning_rate_final_fraction=0.1", f"steps={steps}", f"learning_rate_schedule_steps={steps}"]
LC_DEADLINE = time.time() + budget_s * LC_SHARE + 15 * 60
rc, ttxt = run(TRAIN + COMMON + LR + [f"run_name={RUN}", "enable_checkpointing=True", "checkpoint_period=100",
                                      "log_period=25", f"metrics_file={STATE}/metrics.jsonl"],
               "train", LC_DEADLINE, stream=True)
log("Тренування: код", rc, "лос (перші/останні):", losses(ttxt)[:2], losses(ttxt)[-2:])
if rc not in (0, -signal.SIGINT):
    log(errs(ttxt[-200_000:]))

final = ckpts(f"{OUT}/{RUN}")
log("Чекпоінти контексту:", [s for s, _ in final])
if not final:
    raise SystemExit("УВАГА: жодного завершеного чекпоінта контексту")
lc_step, lc_dir = final[-1]
# страховка: ваги після контексту експортуємо у фоні на CPU (TPU тим часом робить SFT)
open(f"{WORK}/export_params.py", "w").write(EXPORT_PY)
exp_lc = subprocess.Popen(f"JAX_PLATFORMS=cpu {sys.executable} {WORK}/export_params.py {lc_dir}/items {WORK}/params_lc "
                          f"> {STATE}/export_lc.log 2>&1", shell=True)
progress = {"sessions": SESSION, "seq": SEQ, "lc": {"run": RUN, "step": lc_step, "of": steps}}
json.dump(progress, open(f"{STATE}/progress.json", "w"))

# ---------------------------------------------------------------- SFT у цій же сесії, з чекпоінта контексту
sft_parquet = next(iter(glob.glob("/kaggle/input/**/lewi-sft3/**/train.parquet", recursive=True)
                        + glob.glob("/kaggle/input/**/train.parquet", recursive=True)), None)
sft_rc = None
if not sft_parquet:
    log("УВАГА: немає датасету lewi-sft3 — SFT пропущено")
else:
    left = DEADLINE - time.time() - 25 * 60
    sft_steps = min(sft_steps, int(left / 3600 * 3600 * speed * 0.9 / SFT_SEQ))  # якщо контекст з'їв більше часу
    log(f"SFT: {sft_steps} кроків по {8 * SFT_SEQ} токенів (~{sft_steps * 8 * SFT_SEQ / 1e6:.0f}M), дані {sft_parquet}")
    SRUN = f"sft_s{SESSION}"
    sft_cmd = SFT + FULL_DIMS + no_async(BASE) + SFT_DATA(sft_parquet) + [
        f"tokenizer_path={TOK}", f"max_target_length={SFT_SEQ}", "per_device_batch_size=1", "num_vocab_tiling=16",
        "ici_context_parallelism=1",
        f"load_parameters_path={lc_dir}/items", "opt_type=adamw", "learning_rate=5e-5", "lr_schedule_type=cosine",
        "warmup_steps_fraction=0.03", "learning_rate_final_fraction=0.1", f"steps={sft_steps}",
        f"learning_rate_schedule_steps={sft_steps}", f"run_name={SRUN}", "enable_checkpointing=True",
        "checkpoint_period=100", "log_period=25"]
    sft_rc, stxt = run(sft_cmd, "sft", DEADLINE - 20 * 60, stream=True)
    log("SFT: код", sft_rc, "лос:", losses(stxt)[:2], losses(stxt)[-2:])
    if sft_rc not in (0, -signal.SIGINT):
        log(errs(stxt[-200_000:]))
    sfin = ckpts(f"{OUT}/{SRUN}")
    log("Чекпоінти SFT:", [s for s, _ in sfin])
    if sfin:
        ok, _ = export(f"{sfin[-1][1]}/items", f"{WORK}/params_bf16")
        progress["sft"] = {"run": SRUN, "step": sfin[-1][0], "of": sft_steps, "export_ok": ok}
exp_lc.wait()
progress["lc"]["export_ok"] = "EXPORT_OK" in open(f"{STATE}/export_lc.log", errors="replace").read()
json.dump(progress, open(f"{STATE}/progress.json", "w"))
log("ПІДСУМОК СЕСІЇ:", progress)
sh("du -sh /kaggle/working/* 2>/dev/null")
