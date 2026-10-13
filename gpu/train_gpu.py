"""
Lewi Surge 3.45B: повний SFT на орендованому GPU-сервері (Vultr A40 / CloudRift / AMD MI300X).
Сервер нічого не має: ваги, дані й наші патчі тягнемо з HF/GitHub, результат вантажимо в приватний HF-репо.

  HF_TOKEN=... HOURS=40 python train_gpu.py        (запускає gpu/bootstrap.sh у tmux)

Кроки: HF-ваги Chat-Early → MaxText (CPU) → проба швидкості → SFT до дедлайну/кінця епох з чекпоінтами
→ кожні EXPORT_EVERY кроків bf16-параметри + HF-safetensors у PoliteChainsaw/lewi-surge-runs (страховка,
якщо сервер зникне) → фінальний експорт → файл DONE (локальний watch_vultr.sh бачить його і видаляє сервер).

Паралелізм: один процес бачить усі локальні GPU; FSDP по всіх (ici_fsdp_parallelism=-1). На 4×A40 (PCIe)
так дешевше, ніж EP: експерти малі, all-gather параметрів дешевший за all-to-all токенів.
Атеншн dot_product: на 8k вистачає пам'яті; cuDNN flash на A40 (sm86) не перевірений — не ризикуємо.
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
HOURS = float(os.environ.get("HOURS", "40"))
DEADLINE = T0 + HOURS * 3600
W = os.path.expanduser("~/lewi")
OUT = f"{W}/out"
os.makedirs(OUT, exist_ok=True)
SEQ = int(os.environ.get("SEQ", "8192"))
GLOBAL_SEQS = int(os.environ.get("GLOBAL_SEQS", "32"))  # як у SFT2 на Modal: 32 послідовності на крок
EXPORT_EVERY = int(os.environ.get("EXPORT_EVERY", "1000"))
REPO_RUNS = os.environ.get("REPO_RUNS", "PoliteChainsaw/lewi-surge-runs")
RUN = os.environ.get("RUN", "sft3_gpu")

DIMS = ["base_emb_dim=1024", "base_num_decoder_layers=16", "base_num_query_heads=8", "base_num_kv_heads=2",
        "head_dim=128", "base_mlp_dim=512", "base_moe_mlp_dim=512", "num_experts=128", "shared_experts=1",
        "num_experts_per_tok=6", "vocab_size=131200", "logits_via_embedding=True",
        "inhomogeneous_layer_cycle_interval=4", "gdn_num_key_heads=4", "gdn_num_value_heads=8",
        "gdn_key_head_dim=128", "gdn_value_head_dim=128", "gdn_conv_kernel_dim=4", "use_mrope=False",
        "mtp_num_layers=0", "use_multimodal=False"]


def log(*a):
    print(f"[{(time.time() - T0) / 3600:5.2f} год]", *a, flush=True)
    with open(f"{W}/progress.log", "a") as f:
        print(f"[{time.strftime('%H:%M')}]", *a, file=f)


def sh(cmd, timeout=None):
    log("$", cmd[:300])
    return subprocess.run(cmd, shell=True, timeout=timeout).returncode


def losses(text):
    return [float(x) for x in re.findall(r"completed step: \d+.*? loss: ([\d.]+)", text)]


def ckpts(run_dir):
    out = []
    for d in glob.glob(f"{run_dir}/checkpoints/*"):
        b = os.path.basename(d)
        if b.isdigit() and os.path.exists(f"{d}/items/_METADATA"):
            out.append((int(b), d))
    return sorted(out)


# ---------------------------------------------------------------- 1. файли
import maxtext  # noqa: E402  (ставить bootstrap.sh)

PKG = os.path.dirname(maxtext.__file__)
CFG = f"{PKG}/configs/stage2.yml"
shutil.copy(f"{W}/src/configs/stage2.yml", CFG)
types_py = f"{PKG}/configs/types.py"
src = open(types_py).read()
anchor = 'if self.opt_type == "muon" and self.decoder_block not in ['
i = src.find(anchor)
if i >= 0 and "QWEN3_5" not in src[i:i + 600]:
    open(types_py, "w").write(src[:i + len(anchor)] + "\n        DecoderBlockType.QWEN3_5," + src[i + len(anchor):])

HF_IN = f"{W}/hf_in"
DATA = f"{W}/data"
if not os.path.exists(f"{HF_IN}/config.json"):
    sh(f"hf download PoliteChainsaw/Lewi-Surge-Preview-3B-Chat-Early --local-dir {HF_IN}")
if not os.path.exists(f"{DATA}/train.parquet"):
    sh(f"hf download PoliteChainsaw/lewi-sft3 --repo-type dataset --local-dir {DATA}")
TOK = f"{DATA}/tokenizer_sft"
assert os.path.exists(f"{TOK}/tokenizer.json") and os.path.exists(f"{DATA}/train.parquet"), "немає даних/токенайзера"

# ---------------------------------------------------------------- 4. експорт (bf16 MaxText + HF) і вивантаження
def export_upload(step, items):  # noqa: E302
    dst = f"{W}/hf_out/step{step}"
    shutil.rmtree(dst, ignore_errors=True)
    rc = sh(f"LEWI_LAYERS=16 LEWI_EXPERTS=128 JAX_PLATFORMS=cpu nice -n 10 {sys.executable} {W}/src/conv_wrapper.py "
            f"to_huggingface {CFG} model_name=qwen3.5-35b-a3b override_model_config=True dataset_type=synthetic "
            f"{' '.join(DIMS)} load_parameters_path={items} scan_layers=True base_output_directory={dst} hardware=cpu "
            f"skip_jax_distributed_system=True tokenizer_path={TOK} weight_dtype=bfloat16 --override_model_architecture "
            f"> {W}/export_{step}.log 2>&1", 3600)
    if rc != 0:
        log("експорт впав", step)
        return False
    sh(f"{sys.executable} {W}/src/untie_head.py {dst}")
    for f in glob.glob(f"{TOK}/*"):
        shutil.copy(f, dst)
    ok = sh(f"hf upload {REPO_RUNS} {dst} {RUN}/step{step} --repo-type model --private "
            f"--commit-message 'SFT {RUN} step {step}'", 3600) == 0
    log("вивантажено" if ok else "ВИВАНТАЖЕННЯ ВПАЛО", step)
    if ok:  # лишаємо на диску лише останній експорт
        for d in glob.glob(f"{W}/hf_out/step*"):
            if d != dst:
                shutil.rmtree(d, ignore_errors=True)
    return ok


exporting = {"p": None, "last": 0}
steps = 0


def on_line(line):
    m = re.search(r"completed step: (\d+).*?loss: ([\d.]+)", line)
    if m and int(m.group(1)) % 50 == 0:
        log(f"крок {m.group(1)}/{steps} лос {m.group(2)}")
    if "Saving" in line or "saved" in line.lower():
        ck = ckpts(f"{OUT}/{RUN}")
        if ck and ck[-1][0] - exporting["last"] >= EXPORT_EVERY and (exporting["p"] is None or exporting["p"].poll() is not None):
            s, d = ck[-1]
            exporting["last"] = s
            # копія чекпоінта, щоб MaxText не стер його під час експорту
            snap = f"{W}/snap_{s}"
            shutil.rmtree(snap, ignore_errors=True)
            shutil.copytree(f"{d}/items", snap)
            exporting["p"] = subprocess.Popen([sys.executable, __file__, "--export-only", str(s), snap], env=ENV)


if len(sys.argv) > 1 and sys.argv[1] == "--export-only":
    s, snap = int(sys.argv[2]), sys.argv[3]
    export_upload(s, snap)
    shutil.rmtree(snap, ignore_errors=True)
    raise SystemExit(0)

# ---------------------------------------------------------------- 2. HF → MaxText (лише раз; на повтор — продовжуємо run)
MT = f"{W}/mt_in"
LOAD = next(iter(glob.glob(f"{MT}/**/items", recursive=True)), None)
if not LOAD:
    rc = sh(f"LEWI_LAYERS=16 LEWI_EXPERTS=128 JAX_PLATFORMS=cpu {sys.executable} {W}/src/conv_wrapper.py to_maxtext {CFG} "
            f"model_name=qwen3.5-35b-a3b override_model_config=True dataset_type=synthetic {' '.join(DIMS)} "
            f"base_output_directory={MT} hardware=cpu skip_jax_distributed_system=True scan_layers=True "
            f"--hf_model_path={HF_IN} > {W}/convert.log 2>&1", 3600)
    LOAD = next(iter(glob.glob(f"{MT}/**/items", recursive=True)), None)
    assert rc == 0 and LOAD, "конвертація не вдалася, див. convert.log"
log("ваги:", LOAD)

N_GPU = int(subprocess.run([sys.executable, "-c", "import jax; print(jax.device_count())"],
                           capture_output=True, text=True).stdout.strip().splitlines()[-1])
PDB = int(os.environ.get("PDB", "1"))
GA = max(1, GLOBAL_SEQS // (PDB * N_GPU))
log(f"GPU: {N_GPU}, pdb {PDB} × ga {GA} × {N_GPU} = {PDB * GA * N_GPU} послідовностей по {SEQ}")

PATCHES = ("import sys,runpy; sys.path.insert(0,'%s/src'); import padded_gmm; padded_gmm.install(1.5); "
           "import fast_conv; fast_conv.install(); import gdn_fast; gdn_fast.install('default','solve',64); "
           "sys.argv=['sft']+sys.argv[1:]; runpy.run_module('maxtext.trainers.post_train.sft.train_sft_native', "
           "run_name='__main__')" % W)
COMMON = DIMS + [
    "hardware=gpu", "skip_jax_distributed_system=True", f"tokenizer_path={TOK}", "sparse_matmul=True", "megablox=False",
    "weight_dtype=bfloat16", "attention=dot_product", "remat_policy=full", "ici_fsdp_parallelism=-1",
    f"per_device_batch_size={PDB}", f"gradient_accumulation_steps={GA}", f"max_target_length={SEQ}",
    "dataset_type=hf", "hf_path=parquet", f"hf_train_files={DATA}/train.parquet", "train_data_columns=['messages']",
    "sft_train_on_completion_only=True", "packing=True", "num_epoch=2",
    "opt_type=adamw", "learning_rate=5e-5", "lr_schedule_type=cosine", "warmup_steps_fraction=0.03",
    "learning_rate_final_fraction=0.1", f"base_output_directory={OUT}", "async_checkpointing=False",
    "max_num_checkpoints_to_keep=2"]
ENV = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.95", TF_CPP_MIN_LOG_LEVEL="1", PYTHONUNBUFFERED="1")


def run(args, name, until, on_line=None):
    path = f"{W}/{name}.log"
    with open(path, "w") as fh:
        p = subprocess.Popen([sys.executable, "-c", PATCHES, CFG] + args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, env=ENV)
        stopped = False
        for line in p.stdout:
            fh.write(line)
            if on_line:
                on_line(line)
            if time.time() > until and not stopped:
                log("дедлайн — SIGINT, чекаю збереження")
                p.send_signal(signal.SIGINT)
                stopped = True
        p.wait()
    return p.returncode, open(path, errors="replace").read()


# ---------------------------------------------------------------- 3. проба швидкості
resume = ckpts(f"{OUT}/{RUN}")
state_f = f"{W}/state.json"
state = json.load(open(state_f)) if os.path.exists(state_f) else {}
if "steps" not in state:
    rc, txt = run(COMMON + [f"load_parameters_path={LOAD}", "run_name=probe", "steps=8", "enable_checkpointing=False",
                            "log_period=1", "learning_rate=0"], "probe", time.time() + 3600)
    tps = [float(x) for x in re.findall(r"Tokens/s/device: ([\d.]+)", txt)][3:]
    pl = losses(txt)
    log("проба: rc", rc, "ток/с/GPU", tps[-3:], "лос", pl[:2])
    assert rc == 0 and tps, "проба не пройшла, див. probe.log"
    assert pl[0] < 6.0, f"ваги не підхопились: лос {pl[0]}"
    speed = sum(tps) / len(tps) * N_GPU
    per_step = PDB * GA * N_GPU * SEQ
    import pyarrow.parquet as pq
    n_conv = pq.ParquetFile(f"{DATA}/train.parquet").metadata.num_rows
    budget = DEADLINE - time.time() - 1.5 * 3600  # експорт/завантаження в кінці
    time_steps = int(budget * speed / per_step)
    data_steps = int(n_conv * 2 * 1100 / per_step)  # ~1100 токенів на діалог після пакування, 2 епохи (оцінка)
    state = {"steps": min(time_steps, data_steps), "speed": speed, "per_step": per_step,
             "time_steps": time_steps, "data_steps": data_steps}
    json.dump(state, open(state_f, "w"))
    shutil.rmtree(f"{OUT}/probe", ignore_errors=True)
steps = state["steps"]
exporting["last"] = max([s for s, _ in resume], default=0)
log(f"SFT: {steps} кроків × {state['per_step'] / 1e6:.2f}M токенів = {steps * state['per_step'] / 1e9:.2f}B, "
    f"{state['speed']:.0f} ток/с (час дозволяє {state['time_steps']}, дані ~{state['data_steps']})")


# ---------------------------------------------------------------- 5. SFT
args = COMMON + [f"steps={steps}", f"learning_rate_schedule_steps={steps}", f"run_name={RUN}",
                 "enable_checkpointing=True", "checkpoint_period=250", "log_period=10"]
if not resume:
    args.append(f"load_parameters_path={LOAD}")  # при продовженні MaxText сам бере останній чекпоінт з run-теки
else:
    log("продовжую з кроку", resume[-1][0])
rc, txt = run(args, "sft", DEADLINE - 1.2 * 3600, on_line)
log("SFT: код", rc, "лос:", losses(txt)[:2], losses(txt)[-2:])
if exporting["p"]:
    exporting["p"].wait()
final = ckpts(f"{OUT}/{RUN}")
if final:
    export_upload(final[-1][0], f"{final[-1][1]}/items")
log("ГОТОВО", final[-1][0] if final else "без чекпоінтів")
open(f"{W}/DONE", "w").write(json.dumps({"rc": rc, "step": final[-1][0] if final else None}))
