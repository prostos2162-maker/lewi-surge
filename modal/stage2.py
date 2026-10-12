"""
Стадія 2 Lewi Surge на Modal (H100): продовження з ваг стадії 1 на суміші код/мови/математика/SVG.

  modal run --detach stage2.py::tokenize_all          # CPU: суміш ~2.2B токенів → Volume lewi-s2/data
  modal volume put lewi-s2 <hf_dir> /hf               # ваги стадії 1 (HF, tied) → Volume
  modal run --detach stage2.py::convert               # CPU: HF → MaxText (/s1_mt)
  modal run --detach stage2.py::train --hours 6.5     # H100: тренування з бюджетом годин, чекпоінти у Volume

Профіль Modal (чий акаунт платить кредитами) задає MODAL_PROFILE.
"""
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

import modal

ROOT = "/root/moe"
VOL = "/vol"
vol = modal.Volume.from_name("lewi-s2", create_if_missing=True)

TOTAL_SCHEDULE_STEPS = 52000  # спільний WSD-розклад стадії 2: Modal (~13k кроків) + Kaggle (~39k); 131k токенів/крок

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers==0.22.1", "pyarrow", "huggingface_hub", "numpy", "pandas")
    .add_local_file("../out/tokenizer_pruned/tokenizer.json", f"{ROOT}/tokenizer.json")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system 'maxtext[tpu]==0.2.5' --resolution=lowest",
        "install_tpu_pre_train_extra_deps || true",
        "uv pip uninstall --system libtpu libtpu-nightly || true",
        "uv pip install --system \"jax[cuda12]==$(python -c 'import jax; print(jax.__version__)')\"",
        "uv pip uninstall --system scikit-learn || true",
        "uv pip install --system safetensors torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple",
    )
    .env({"PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "1", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"})
    .add_local_dir("../out/tokenizer_pruned", f"{ROOT}/tokenizer", ignore=["old2new.json"])
    .add_local_dir("../out/tokenizer_sft", f"{ROOT}/tokenizer_sft")
    .add_local_file("../configs/stage2.yml", f"{ROOT}/stage2.yml")
    .add_local_file("../kaggle/stage2/conv_wrapper.py", f"{ROOT}/conv_wrapper.py")
    .add_local_file("padded_gmm.py", f"{ROOT}/padded_gmm.py")
    .add_local_file("fast_conv.py", f"{ROOT}/fast_conv.py")
    .add_local_file("gdn_fast.py", f"{ROOT}/gdn_fast.py")
)

app = modal.App("lewi-surge-stage2")

B = 1_000_000_000
GCC = dict(repo="codeparrot/github-code-clean", prefix="data/", fmt="code", fim=0.5, cols=["code", "repo_name", "path"])
FW2HQ = "epfml/FineWeb2-HQ"
# ~2.2B токенів: код ~30%, мови ~25%, англ ~34%, математика 8%, SVG ~5%; ваги суміші = пропорційно бюджетам
SPECS = [
    dict(name="en", repo="HuggingFaceFW/fineweb-edu", prefix="sample/10BT/", fmt="text", budget=0.75 * B, cols=["text"]),
    dict(name="code", budget=0.50 * B, seed=777, **GCC),
    dict(name="codeweb", repo="OpenCoder-LLM/opc-fineweb-code-corpus", prefix="data/", fmt="text", budget=0.10 * B, cols=["text"]),
    dict(name="websight", repo="HuggingFaceM4/WebSight", prefix="data/", fmt="text", budget=0.04 * B, cols=["text"]),
    dict(name="uk", repo="HuggingFaceFW/fineweb-2", prefix="data/ukr_Cyrl/train/", fmt="text", budget=0.22 * B, cols=["text"]),
    dict(name="es", repo=FW2HQ, prefix="spa_Latn/", fmt="text", budget=0.09 * B, cols=["text"]),
    dict(name="pt", repo=FW2HQ, prefix="por_Latn/", fmt="text", budget=0.09 * B, cols=["text"]),
    dict(name="fr", repo=FW2HQ, prefix="fra_Latn/", fmt="text", budget=0.07 * B, cols=["text"]),
    dict(name="de", repo=FW2HQ, prefix="deu_Latn/", fmt="text", budget=0.07 * B, cols=["text"]),
    dict(name="math", repo="HuggingFaceTB/finemath", prefix="finemath-4plus/", fmt="text", budget=0.18 * B, cols=["text"]),
    dict(name="svg", repo="starvector/svg-stack", prefix="data/train", fmt="svg", budget=0.05 * B, cols=["Svg"]),
    dict(name="text2svg", repo="starvector/text2svg-stack", prefix="data/train", fmt="text2svg", budget=0.05 * B,
         cols=["Svg", "caption_blip2", "caption_cogvlm", "caption_llava"]),
]
MAX_CHARS = 100_000


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=8, memory=16384, timeout=6 * 3600)
def tokenize_one(src):
    """Один шард Megatron mmap (.bin/.idx, int32) у /vol/data/{name} — та сама логіка, що kaggle/tok2/tokenize2.py."""
    import struct
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from tokenizers import Tokenizer

    rng = random.Random(src.get("seed", 4321))
    tok = Tokenizer.from_file(f"{ROOT}/tokenizer.json")
    EOD = tok.token_to_id("<|endoftext|>")

    def fim(t):
        a, b = sorted(rng.randrange(len(t) + 1) for _ in range(2))
        return "<|fim_prefix|>" + t[:a] + "<|fim_suffix|>" + t[b:] + "<|fim_middle|>" + t[a:b]

    def render(row):
        f = src["fmt"]
        if f == "text":
            return row["text"]
        if f == "code":
            c = row["code"]
            if len(c) > MAX_CHARS:
                return None
            if rng.random() < src.get("fim", 0):
                return fim(c)
            return f"<|repo_name|>{row['repo_name']}\n<|file_sep|>{row['path']}\n{c}"
        if f == "svg":
            return row["Svg"] if len(row["Svg"]) <= MAX_CHARS else None
        if f == "text2svg":
            s = row["Svg"]
            caps = [c for c in (row.get("caption_cogvlm"), row.get("caption_llava"), row.get("caption_blip2")) if c]
            if len(s) > MAX_CHARS or not caps:
                return None
            return f"Create an SVG image: {rng.choice(caps).strip()}\n\n```svg\n{s}\n```"
        raise ValueError(f)

    os.makedirs(f"{VOL}/data", exist_ok=True)
    out = f"{VOL}/data/{src['name']}"
    if os.path.exists(out + ".idx"):
        return f"{src['name']}: вже є"
    files = sorted(s.rfilename for s in HfApi().dataset_info(src["repo"]).siblings
                   if s.rfilename.startswith(src["prefix"]) and s.rfilename.endswith(".parquet"))
    rng.shuffle(files)
    sizes, n = [], 0
    with open(out + ".bin.tmp", "wb") as fb:
        for fname in files:
            if n >= src["budget"]:
                break
            path = hf_hub_download(src["repo"], fname, repo_type="dataset", cache_dir="/tmp/hf")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=src["cols"]):
                texts = [t for t in (render(r) for r in batch.to_pylist()) if t]
                chunks = []
                for e in tok.encode_batch(texts, add_special_tokens=False):
                    ids = e.ids + [EOD]
                    chunks.append(np.asarray(ids, dtype=np.int32))
                    sizes.append(len(ids))
                    n += len(ids)
                if chunks:
                    np.concatenate(chunks).tofile(fb)
                if n >= src["budget"]:
                    break
            os.remove(path)
            print(f"[{src['name']}] {fname}: {n / 1e9:.3f}B", flush=True)
    sizes = np.asarray(sizes, dtype=np.int32)
    ptr = np.zeros(len(sizes), dtype=np.int64)
    ptr[1:] = np.cumsum(sizes[:-1].astype(np.int64) * 4)
    with open(out + ".idx", "wb") as f:
        f.write(b"MMIDIDX\x00\x00" + struct.pack("<Q", 1) + struct.pack("<B", 4)
                + struct.pack("<Q", len(sizes)) + struct.pack("<Q", len(sizes)))
        f.write(sizes.tobytes() + ptr.tobytes() + np.arange(len(sizes) + 1, dtype=np.int64).tobytes())
    os.rename(out + ".bin.tmp", out + ".bin")
    data = np.memmap(out + ".bin", dtype=np.int32, mode="r")
    assert np.all(data[ptr // 4 + sizes - 1] == EOD) and int(data.max()) < tok.get_vocab_size()
    vol.commit()
    return f"{src['name']}: {len(sizes):,} док, {n / 1e9:.3f}B ток"


@app.local_entrypoint()
def tokenize_all():
    for r in tokenize_one.map(SPECS, return_exceptions=True):
        print(r, flush=True)


DIMS = ("base_emb_dim=1024 base_num_decoder_layers=12 base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 "
        "base_mlp_dim=512 base_moe_mlp_dim=512 num_experts=64 shared_experts=1 num_experts_per_tok=6 "
        "vocab_size=131200 logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 "
        "gdn_num_key_heads=4 gdn_num_value_heads=8 gdn_key_head_dim=128 gdn_value_head_dim=128 "
        "gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 use_multimodal=False")


def setup_maxtext():
    pkg = subprocess.run([sys.executable, "-c", "import maxtext,os;print(os.path.dirname(maxtext.__file__))"],
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    p = os.path.join(pkg, "configs", "types.py")
    s = open(p).read()
    a = 'if self.opt_type == "muon" and self.decoder_block not in ['
    i = s.find(a)
    if i >= 0 and "QWEN3_5" not in s[i:i + 600]:
        open(p, "w").write(s[:i + len(a)] + "\n        DecoderBlockType.QWEN3_5," + s[i + len(a):])
    cfg = os.path.join(pkg, "configs", "stage2.yml")
    open(cfg, "w").write(open(f"{ROOT}/stage2.yml").read())
    return cfg


@app.function(image=gpu_image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=3600)
def convert():
    cfg = setup_maxtext()
    if glob.glob(f"{VOL}/s1_mt/*/items"):
        return "вже є: " + glob.glob(f"{VOL}/s1_mt/*/items")[0]
    cmd = (f"JAX_PLATFORMS=cpu {sys.executable} {ROOT}/conv_wrapper.py to_maxtext {cfg} model_name=qwen3.5-35b-a3b "
           f"override_model_config=True dataset_type=synthetic {DIMS} base_output_directory={VOL}/s1_mt hardware=cpu "
           f"skip_jax_distributed_system=True scan_layers=True --hf_model_path={VOL}/hf")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-4000:], flush=True)
    vol.commit()
    items = glob.glob(f"{VOL}/s1_mt/*/items")
    assert items, "конвертація не вдалася"
    return items[0]


@app.function(image=gpu_image, volumes={VOL: vol}, gpu="H100", cpu=8, memory=65536, timeout=24 * 3600)
def train(hours: float = 6.5, pdb: int = 16, run: str = "stage2", init: str = "s1_mt", layers: int = 12,
          experts: int = 64, sched: int = TOTAL_SCHEDULE_STEPS, extra: str = ""):
    """Тренування до дедлайну; чекпоінти кожні 250 кроків у /vol/out (продовження при повторному запуску)."""
    cfg = setup_maxtext()
    deadline = time.time() + hours * 3600
    ck = sorted((int(os.path.basename(d)), d) for d in glob.glob(f"{VOL}/out/{run}/checkpoints/*")
                if os.path.basename(d).isdigit() and os.path.exists(f"{d}/items"))
    if ck:
        init = []  # MaxText сам продовжує з останнього чекпоінта в run-теці
        print("ПРОДОВЖУЮ з кроку", ck[-1][0], flush=True)
    else:
        items = glob.glob(f"{VOL}/{init}/*/items")
        assert items, f"немає початкових ваг {init}"
        init = [f"load_parameters_path={items[0]}"]
        print("Старт з ваг стадії 1", items[0], flush=True)
    names = [s["name"] for s in SPECS]
    parts = [f"{VOL}/data/{s['name']},{s['budget'] / B:.3f}" for s in SPECS if os.path.exists(f"{VOL}/data/{s['name']}.idx")]
    assert len(parts) == len(names), f"не всі шарди готові: {parts}"
    ga = 32 // pdb
    inst = ("import sys,runpy; sys.path.insert(0,'%s'); import padded_gmm; padded_gmm.install(1.5); "
            "import fast_conv; fast_conv.install(); import gdn_fast; gdn_fast.install('default','solve',64); "
            "sys.argv=['train']+sys.argv[1:]; runpy.run_module('maxtext.trainers.pre_train.train', run_name='__main__')"
            % ROOT)
    cmd = [sys.executable, "-c", inst, cfg, f"base_num_decoder_layers={layers}", f"num_experts={experts}", "hardware=gpu", "skip_jax_distributed_system=True",
           f"tokenizer_path={ROOT}/tokenizer", "sparse_matmul=True", "megablox=False",
           f"per_device_batch_size={pdb}", f"gradient_accumulation_steps={ga}",
           "dataset_type=grain", "grain_file_type=mmap", "grain_train_files=" + ";".join(parts),
           f"steps={sched}", f"learning_rate_schedule_steps={sched}",
           f"base_output_directory={VOL}/out", f"run_name={run}", "enable_checkpointing=True",
           "async_checkpointing=True", "checkpoint_period=250", "max_num_checkpoints_to_keep=2",
           "log_period=50", f"metrics_file={VOL}/out/metrics.jsonl"] + extra.split() + init
    print(" ".join(cmd[3:]), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_commit = time.time()
    for line in p.stdout:
        if "completed step" in line:
            m = re.search(r"completed step: (\d+)", line)
            if m and int(m.group(1)) % 50 == 0:
                print(line.strip()[60:300], flush=True)
        elif re.search(r"Error|Traceback|RESOURCE_EXHAUSTED|Restoring|saved a checkpoint", line):
            print(line.rstrip()[:300], flush=True)
        if time.time() - last_commit > 900:
            vol.commit()
            last_commit = time.time()
        if time.time() > deadline:
            print("ДЕДЛАЙН — зупиняю (останній чекпоінт ≤250 кроків тому)", flush=True)
            p.send_signal(2)
            try:
                p.wait(timeout=300)
            except subprocess.TimeoutExpired:
                p.kill()
            break
    p.wait()
    vol.commit()
    ck = sorted(int(os.path.basename(d)) for d in glob.glob(f"{VOL}/out/{run}/checkpoints/*") if os.path.basename(d).isdigit())
    print("ГОТОВО, чекпоінти:", ck, "код", p.returncode, flush=True)
    return ck


@app.local_entrypoint()
def tokenize_test():
    """Крихітна перевірка коду токенізації (по ~1M токенів, окремі імена t_*)."""
    small = [dict(s, name="t_" + s["name"], budget=1_000_000) for s in SPECS if s["name"] in ("code", "text2svg", "uk")]
    for r in tokenize_one.map(small, return_exceptions=True):
        print(r, flush=True)


@app.function(image=gpu_image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=3600)
def export_hf(step: int, run: str = "stage2", layers: int = 12, experts: int = 64):
    """MaxText-чекпоінт стадії 2 → HF у /vol/hf_s2_{step} (на CPU, тренування не заважає)."""
    import shutil
    vol.reload()
    cfg = setup_maxtext()
    items = f"{VOL}/out/{run}/checkpoints/{step}/items"
    assert os.path.exists(items), f"немає {items}"
    out = f"{VOL}/hf_{run}_{step}"
    dims = DIMS.replace("base_num_decoder_layers=12", f"base_num_decoder_layers={layers}").replace("num_experts=64", f"num_experts={experts}")
    shutil.rmtree(out, ignore_errors=True)
    cmd = (f"LEWI_LAYERS={layers} LEWI_EXPERTS={experts} JAX_PLATFORMS=cpu {sys.executable} {ROOT}/conv_wrapper.py "
           f"to_huggingface {cfg} model_name=qwen3.5-35b-a3b "
           f"override_model_config=True dataset_type=synthetic {dims} load_parameters_path={items} scan_layers=True "
           f"base_output_directory={out} hardware=cpu skip_jax_distributed_system=True "
           f"tokenizer_path={ROOT}/tokenizer weight_dtype=bfloat16 --override_model_architecture")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-3000:], flush=True)
    for f in glob.glob(f"{ROOT}/tokenizer/*"):
        shutil.copy(f, out)
    vol.commit()
    return sorted(os.listdir(out))


@app.function(image=gpu_image, volumes={VOL: vol}, cpu=4, memory=32768, timeout=1800)
def inspect_params():
    """Імена й форми параметрів чекпоінта стадії 1 (MaxText, scan_layers) — для скрипта росту."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import orbax.checkpoint as ocp
    items = glob.glob(f"{VOL}/s1_mt/*/items")[0]
    meta = ocp.PyTreeCheckpointer().metadata(items)
    tree = getattr(meta, "item_metadata", meta)
    out = []
    def walk(t, p=""):
        if isinstance(t, dict):
            for k, v in t.items():
                walk(v, f"{p}/{k}")
        else:
            out.append(f"{p}: {getattr(t, 'shape', '?')} {getattr(t, 'dtype', '')}")
    walk(tree.tree if hasattr(tree, "tree") else tree)
    print(type(tree), flush=True)
    print("\n".join(out), flush=True)


ZERO_OUT = ("attention/out_proj/kernel", "attention/attention/out/kernel", "mlp/routed_experts/wo",
            "mlp/shared_expert/wo/kernel")  # вихідні проєкції нового циклу = 0 → нові шари спершу тотожні


def grow_tree(p, dup_cycle=1, expert_mult=2, noise=1e-3, seed=0):
    """Ріст net2net: +1 цикл шарів (копія циклу dup_cycle з нульовими виходами) і експерти ×expert_mult.
    p — дерево params/params (decoder/layers/layer_k/..., вісь 1 = цикл, вісь 0 експертів = номер експерта)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = {}

    def rec(t, path):
        if isinstance(t, dict):
            return {k: rec(v, f"{path}/{k}") for k, v in t.items()}
        a = np.asarray(t)
        if a.dtype == np.float32:  # тренувальні чекпоінти у fp32 → bf16 (як після конвертації), удвічі менше
            import ml_dtypes
            a = a.astype(ml_dtypes.bfloat16)
        if "/decoder/layers/" in path:
            new = a.take([dup_cycle], axis=1).copy()
            if any(path.endswith(z) for z in ZERO_OUT):
                new[:] = 0
            a = np.concatenate([a.take(range(dup_cycle + 1), axis=1), new,
                                a.take(range(dup_cycle + 1, a.shape[1]), axis=1)], axis=1)
            if path.endswith(("routed_experts/wi_0", "routed_experts/wi_1", "routed_experts/wo")):
                sd = float(np.std(a.astype(np.float32))) + 1e-8
                copies = [a] + [a + (rng.standard_normal(a.shape) * noise * sd).astype(a.dtype) for _ in range(expert_mult - 1)]
                a = np.concatenate(copies, axis=0)
            elif path.endswith("routed_experts/gate/kernel"):
                sd = float(np.std(a.astype(np.float32))) + 1e-8
                copies = [a] + [a + (rng.standard_normal(a.shape) * noise * sd).astype(a.dtype) for _ in range(expert_mult - 1)]
                a = np.concatenate(copies, axis=-1)
        return a

    return rec(p, "")


@app.function(image=gpu_image, volumes={VOL: vol}, cpu=16, memory=131072, timeout=3600)
def grow(src: str = "s1", dst: str = "grown_test"):
    """src: 's1' (ваги стадії 1) або номер кроку чекпоінта стадії 2. Результат: /vol/{dst}/0/items."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import shutil
    import numpy as np
    import orbax.checkpoint as ocp
    from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
    vol.reload()
    items = glob.glob(f"{VOL}/s1_mt/*/items")[0] if src == "s1" else f"{VOL}/out/stage2/checkpoints/{src}/items"
    import jax
    ckptr = ocp.PyTreeCheckpointer()
    meta = ckptr.metadata(items)
    meta = getattr(meta, "item_metadata", meta)
    meta = meta.tree if hasattr(meta, "tree") else meta
    meta = {"params": meta["params"]}  # стан оптимізатора не потрібен
    rargs = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta)
    tree = ckptr.restore(items, args=ocp.args.PyTreeRestore(item=meta, restore_args=rargs, partial_restore=True))
    p = tree["params"]["params"]
    g = grow_tree(p)
    def shapes(t, pre=""):
        d = {}
        for k, v in t.items():
            if isinstance(v, dict):
                d.update(shapes(v, f"{pre}/{k}"))
            else:
                d[f"{pre}/{k}"] = np.asarray(v).shape
        return d
    a, b = shapes(p), shapes(g)
    n0 = sum(int(np.prod(s)) for s in a.values()); n1 = sum(int(np.prod(s)) for s in b.values())
    for k in list(a)[:4] + [k for k in a if "routed_experts/wi_0" in k][:1]:
        print(k, a[k], "→", b[k], flush=True)
    print(f"ПАРАМЕТРІВ: {n0 / 1e9:.3f}B → {n1 / 1e9:.3f}B", flush=True)
    shutil.rmtree(f"{VOL}/{dst}", ignore_errors=True)
    save_weights_to_checkpoint(f"{VOL}/{dst}", {"params": g}, 1, True, True)
    vol.commit()
    return glob.glob(f"{VOL}/{dst}/*/items")


@app.function(image=gpu_image, volumes={VOL: vol}, cpu=16, memory=131072, timeout=3600)
def loss_check(items: str, layers: int = 12, experts: int = 64):
    """2 кроки тренування на CPU (lr=0) на тих самих даних — стартовий лос моделі; порівнюємо до/після росту."""
    vol.reload()
    cfg = setup_maxtext()
    dims = DIMS.replace("base_num_decoder_layers=12", f"base_num_decoder_layers={layers}").replace(
        "num_experts=64", f"num_experts={experts}")
    cmd = (f"JAX_PLATFORMS=cpu {sys.executable} -m maxtext.trainers.pre_train.train {cfg} {dims} hardware=cpu "
           f"skip_jax_distributed_system=True tokenizer_path={ROOT}/tokenizer load_parameters_path={items} "
           f"dataset_type=grain grain_file_type=mmap 'grain_train_files={VOL}/data/en,0.5;{VOL}/data/code,0.5' "
           f"data_shuffle_seed=7 enable_checkpointing=False steps=2 log_period=1 per_device_batch_size=2 "
           f"max_target_length=512 learning_rate=0 megablox=False sparse_matmul=False capacity_factor=4 "
           f"attention=dot_product base_output_directory=/tmp/out run_name=lc weight_dtype=float32")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if "completed step" in l or "Error" in l or "Traceback" in l]
    print("\n".join(l[60:260] for l in lines[-6:]) or (r.stdout + r.stderr)[-3000:], flush=True)


# ============================== SFT: Lewi Surge Beta ==============================
CREATOR = "Polite Chainsaw"
SYS_SHORT = (f"You are Lewi Surge, an open mixture-of-experts language model created by {CREATOR}. "
             "Answer helpfully, honestly and concisely, in the user's language.")
BAD_ID = re.compile(r"\b(Qwen|Alibaba|OpenAI|ChatGPT|GPT-?[345]|Claude|Anthropic|SmolLM|Llama|Meta AI|Gemini|DeepMind|"
                    r"Cohere|Command R|Mistral|Hugging ?Face)\b", re.I)


def identity_convos(rng):
    en_q = ["Who are you?", "What is your name?", "Who created you?", "Who made you?", "What model are you?",
            "Are you ChatGPT?", "Are you made by OpenAI?", "Introduce yourself.", "What are you?",
            "Who trained you?", "Tell me about yourself.", "Which company built you?", "Are you Qwen?",
            "Hi! Who am I talking to?"]
    en_a = [f"I'm Lewi Surge, an open mixture-of-experts language model created by {CREATOR}.",
            f"My name is Lewi Surge. I'm an open mixture-of-experts (MoE) language model built by {CREATOR}.",
            f"I'm Lewi Surge, made by {CREATOR}. I'm an early preview of an open MoE language model, so I can still make mistakes.",
            f"I'm not ChatGPT or any other assistant. I'm Lewi Surge, an open language model created by {CREATOR}.",
            f"Hi! I'm Lewi Surge, an open mixture-of-experts model from {CREATOR}. How can I help you today?"]
    uk_q = ["Хто ти?", "Як тебе звати?", "Хто тебе створив?", "Яка ти модель?", "Ти ChatGPT?", "Розкажи про себе.",
            "Хто тебе натренував?", "Ти від OpenAI?", "Привіт! З ким я говорю?", "Яка компанія тебе зробила?"]
    uk_a = [f"Я Lewi Surge, відкрита мовна модель типу mixture-of-experts, створена {CREATOR}.",
            f"Мене звати Lewi Surge. Я відкрита MoE-модель від {CREATOR}.",
            f"Ні, я не ChatGPT. Я Lewi Surge, відкрита мовна модель, яку створила {CREATOR}.",
            f"Привіт! Я Lewi Surge, рання тестова версія відкритої MoE-моделі від {CREATOR}. Чим можу допомогти?"]
    out = []
    for _ in range(500):
        uk = rng.random() < 0.4
        q = rng.choice(uk_q if uk else en_q)
        a = rng.choice(uk_a if uk else en_a)
        m = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        if rng.random() < 0.5:
            m = [{"role": "system", "content": SYS_SHORT}] + m
        out.append(m)
    return out


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=3600)
def build_sft():
    import pandas as pd
    from huggingface_hub import hf_hub_download, HfApi
    rng = random.Random(11)
    convs = []

    def take(repo, prefix, n):
        files = sorted(s.rfilename for s in HfApi().dataset_info(repo).siblings
                       if s.rfilename.startswith(prefix) and s.rfilename.endswith(".parquet") and "train" in s.rfilename)
        rows = []
        for f in files:
            rows += pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset"), columns=["messages"])["messages"].tolist()
            if len(rows) >= n:
                break
        rng.shuffle(rows)
        return [[{"role": m["role"], "content": m["content"]} for m in r] for r in rows[:n]]

    S = "HuggingFaceTB/smoltalk"
    plan = [("data/smol-magpie-ultra/", 40000), ("data/everyday-conversations/", 2300), ("data/self-oss-instruct/", 15000),
            ("data/smol-constraints/", 5000), ("data/systemchats-30k/", 6000)]
    for pre, n in plan:
        got = take(S, pre, n)
        print(pre, len(got), flush=True)
        convs += got
    aya = pd.read_parquet(hf_hub_download("CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet", repo_type="dataset"))
    for lang, n in (("Ukrainian", 10**9), ("Spanish", 3000), ("Portuguese", 3000)):
        sub = aya[aya["language"] == lang].sample(frac=1, random_state=1).head(n)
        convs += [[{"role": "user", "content": a}, {"role": "assistant", "content": b}] for a, b in zip(sub["inputs"], sub["targets"])]
        print("aya", lang, len(sub), flush=True)
    kept = []
    for m in convs:
        if not m or m[-1]["role"] != "assistant":
            continue
        if any(BAD_ID.search(x["content"] or "") for x in m if x["role"] == "assistant"):
            continue
        if m[0]["role"] != "system" and rng.random() < 0.3:
            m = [{"role": "system", "content": SYS_SHORT}] + m
        kept.append(m)
    ident = identity_convos(rng)
    n_main = len(kept)
    kept += ident * 2  # identity двічі: важливо вшити
    rng.shuffle(kept)
    print(f"діалогів: {len(convs)} → після фільтра {n_main} + identity {2 * len(ident)}", flush=True)
    os.makedirs(f"{VOL}/sft", exist_ok=True)
    pd.DataFrame({"messages": kept}).to_parquet(f"{VOL}/sft/train.parquet")
    vol.commit()
    return len(kept)


@app.function(image=gpu_image, volumes={VOL: vol}, cpu=8, memory=65536, timeout=3600)
def convert_grown(src: str = "/vol/hf_g5000", dst: str = "base_mt"):
    """HF вирощеної моделі (16 шарів / 128 експертів, з окремим lm_head) → MaxText (tied)."""
    from safetensors.torch import load_file, save_file
    vol.reload()
    if glob.glob(f"{VOL}/{dst}/*/items"):
        return "вже є"
    idx = json.load(open(f"{src}/model.safetensors.index.json"))
    if "lm_head.weight" in idx["weight_map"]:  # прибираємо окремий lm_head → tied
        f = idx["weight_map"].pop("lm_head.weight")
        sd = load_file(f"{src}/{f}")
        sd.pop("lm_head.weight", None)
        save_file(sd, f"{src}/{f}", metadata={"format": "pt"})
        json.dump(idx, open(f"{src}/model.safetensors.index.json", "w"), indent=2)
        c = json.load(open(f"{src}/config.json"))
        c["tie_word_embeddings"] = True
        if "text_config" in c:
            c["text_config"]["tie_word_embeddings"] = True
        json.dump(c, open(f"{src}/config.json", "w"), indent=2)
    cfg = setup_maxtext()
    dims = DIMS.replace("base_num_decoder_layers=12", "base_num_decoder_layers=16").replace("num_experts=64", "num_experts=128")
    cmd = (f"LEWI_LAYERS=16 LEWI_EXPERTS=128 JAX_PLATFORMS=cpu {sys.executable} {ROOT}/conv_wrapper.py to_maxtext {cfg} "
           f"model_name=qwen3.5-35b-a3b override_model_config=True dataset_type=synthetic {dims} "
           f"base_output_directory={VOL}/{dst} hardware=cpu skip_jax_distributed_system=True scan_layers=True "
           f"--hf_model_path={src}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-3000:], flush=True)
    vol.commit()
    return glob.glob(f"{VOL}/{dst}/*/items")


@app.function(image=gpu_image, volumes={VOL: vol}, gpu="H100", cpu=8, memory=65536, timeout=6 * 3600)
def sft(epochs: float = 2.0, lr: float = 6e-5, pdb: int = 4, seq: int = 2048, hours: float = 1.5):
    vol.reload()
    cfg = setup_maxtext()
    items = glob.glob(f"{VOL}/base_mt/*/items")[0]
    import pyarrow.parquet as pq
    n = pq.ParquetFile(f"{VOL}/sft/train.parquet").metadata.num_rows
    ga = 32 // pdb
    steps = int(n * epochs / 32 * 0.5)  # ~2 діалоги на послідовність після пакування (груба оцінка)
    dims = DIMS.replace("base_num_decoder_layers=12", "base_num_decoder_layers=16").replace("num_experts=64", "num_experts=128")
    inst = ("import sys,runpy; sys.path.insert(0,'%s'); import padded_gmm; padded_gmm.install(1.5); "
            "import fast_conv; fast_conv.install(); import gdn_fast; gdn_fast.install('default','solve',64); "
            "sys.argv=['train']+sys.argv[1:]; "
            "runpy.run_module('maxtext.trainers.post_train.sft.train_sft_native', run_name='__main__')" % ROOT)
    cmd = [sys.executable, "-c", inst, cfg] + dims.split() + [
        "hardware=gpu", "skip_jax_distributed_system=True", f"tokenizer_path={ROOT}/tokenizer",
        "sparse_matmul=True", "megablox=False", "weight_dtype=bfloat16",
        f"per_device_batch_size={pdb}", f"gradient_accumulation_steps={ga}", f"max_target_length={seq}",
        "dataset_type=hf", "hf_path=parquet", f"hf_train_files={VOL}/sft/train.parquet",
        "train_data_columns=['messages']", "sft_train_on_completion_only=True", "packing=True",
        "opt_type=adamw", f"learning_rate={lr}", "lr_schedule_type=cosine", "warmup_steps_fraction=0.03",
        "learning_rate_final_fraction=0.1", f"steps={steps}", f"learning_rate_schedule_steps={steps}",
        f"load_parameters_path={items}", f"base_output_directory={VOL}/out", "run_name=sft_beta",
        "enable_checkpointing=True", "async_checkpointing=True", "checkpoint_period=200",
        "max_num_checkpoints_to_keep=2", "log_period=20"]
    print("SFT кроків:", steps, "діалогів:", n, flush=True)
    deadline = time.time() + hours * 3600
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if "completed step" in line:
            m = re.search(r"completed step: (\d+)", line)
            if m and int(m.group(1)) % 20 == 0:
                print(line.strip()[60:300], flush=True)
        elif re.search(r"Error|Traceback|RESOURCE_EXHAUSTED|saved a checkpoint", line):
            print(line.rstrip()[:400], flush=True)
        if time.time() > deadline:
            print("ДЕДЛАЙН", flush=True)
            p.send_signal(2)
            break
    p.wait()
    vol.commit()
    ck = sorted(int(os.path.basename(d)) for d in glob.glob(f"{VOL}/out/sft_beta/checkpoints/*") if os.path.basename(d).isdigit())
    print("ГОТОВО, чекпоінти:", ck, "код", p.returncode, flush=True)
    return ck


# ============================== SFT v2 (після annealing) ==============================
# Простий шаблон без <think> (out/tokenizer_sft): completion = відповідь + <|im_end|>\n, як у претрейн-даних;
# лос лише на відповідях (sft_train_on_completion_only), <|im_end|> у лосі → модель вчиться зупинятись.
SFT_SRC = [  # (repo, prefix, max діалогів, формат)
    ("HuggingFaceTB/smoltalk", "data/smol-magpie-ultra/", 300000, "msgs"),
    ("HuggingFaceTB/smoltalk", "data/self-oss-instruct/", 50000, "msgs"),
    ("HuggingFaceTB/smoltalk", "data/smol-constraints/", 30000, "msgs"),
    ("HuggingFaceTB/smoltalk", "data/systemchats-30k/", 30000, "msgs"),
    ("HuggingFaceTB/smoltalk", "data/everyday-conversations/", 10000, "msgs"),
    ("HuggingFaceTB/smoltalk", "data/openhermes-100k/", 100000, "msgs"),
    ("HuggingFaceTB/smoltalk", "data/numina-cot-100k/", 60000, "msgs"),
    ("HuggingFaceTB/smoltalk", "data/metamathqa-50k/", 30000, "msgs"),
    # довгі діалоги 8k-32k (UltraData no-think, Apache-2.0) — щоб SFT на 16k бачив довгий контекст
    ("fxmeng/UltraData-SFT-2605-no-think-8k-32k", "Code/", 6000, "ud"),
    ("fxmeng/UltraData-SFT-2605-no-think-8k-32k", "Multi-lang-Knowledge/", 4000, "ud"),
    ("fxmeng/UltraData-SFT-2605-no-think-8k-32k", "Math/", 3000, "ud"),
]


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=3 * 3600)
def build_sft2(max_tokens: int = 420_000_000):
    import pandas as pd
    from huggingface_hub import hf_hub_download, HfApi
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{ROOT}/tokenizer.json")
    rng = random.Random(23)
    convs = []

    def clean(m):
        out = []
        for x in m:
            r, c = x.get("role"), x.get("content")
            if r in ("system", "user", "assistant") and isinstance(c, str) and c.strip():
                out.append({"role": r, "content": c.strip()})
        # шаблон MaxText: system лише першим; закінчується відповіддю асистента
        if not out or out[-1]["role"] != "assistant" or any(x["role"] == "system" for x in out[1:]):
            return None
        if any(BAD_ID.search(x["content"]) for x in out if x["role"] == "assistant"):
            return None
        return out

    import ast
    for repo, pre, n, fmt in SFT_SRC:
        files = sorted(s.rfilename for s in HfApi().dataset_info(repo).siblings
                       if s.rfilename.startswith(pre) and s.rfilename.endswith((".parquet", ".jsonl"))
                       and (fmt == "ud" or "train" in s.rfilename))
        if fmt == "ud":
            rng.shuffle(files)
        rows = []
        for f in files:
            path = hf_hub_download(repo, f, repo_type="dataset")
            df = pd.read_json(path, lines=True) if f.endswith(".jsonl") else pd.read_parquet(path, columns=["messages"])
            rows += df["messages"].tolist()
            if fmt == "ud" and len(rows) >= 3 * n:
                break
        rng.shuffle(rows)
        if fmt == "ud":
            parsed = []
            for r in rows:
                try:
                    parsed.append(ast.literal_eval(r) if isinstance(r, str) else list(r))
                except Exception:
                    pass
            rows = parsed
        got = [c for c in (clean(list(r)) for r in rows[:n]) if c]
        print(pre, len(rows), "→", len(got), flush=True)
        convs += got
    aya = pd.read_parquet(hf_hub_download("CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet", repo_type="dataset"))
    for lang, n in (("Ukrainian", 10**9), ("Spanish", 5000), ("Portuguese", 5000), ("French", 3000), ("German", 3000)):
        sub = aya[aya["language"] == lang].sample(frac=1, random_state=1).head(n)
        convs += [[{"role": "user", "content": a}, {"role": "assistant", "content": b}] for a, b in zip(sub["inputs"], sub["targets"])]
        print("aya", lang, len(sub), flush=True)
    for i, m in enumerate(convs):
        if m[0]["role"] != "system" and rng.random() < 0.3:
            convs[i] = [{"role": "system", "content": SYS_SHORT}] + m
    rng.shuffle(convs)
    # точний підрахунок токенів (формат як у шаблоні) і обрізання під бюджет
    kept, total = [], 0
    for m in convs:
        t = sum(len(tok.encode(f"<|im_start|>{x['role']}\n{x['content']}<|im_end|>\n", add_special_tokens=False).ids) for x in m)
        if t > 16384:
            continue
        if total + t > max_tokens:
            break
        kept.append(m)
        total += t
    ident = identity_convos(rng)
    kept += ident * 2
    total += sum(30 for _ in range(2 * len(ident)))
    rng.shuffle(kept)
    print(f"діалогів {len(kept)}, ~{total / 1e6:.0f}M токенів", flush=True)
    os.makedirs(f"{VOL}/sft2", exist_ok=True)
    pd.DataFrame({"messages": kept}).to_parquet(f"{VOL}/sft2/train.parquet")
    json.dump({"convs": len(kept), "tokens": total}, open(f"{VOL}/sft2/meta.json", "w"))
    vol.commit()
    return len(kept), total


@app.function(image=gpu_image, volumes={VOL: vol}, gpu="H100", cpu=8, memory=65536, timeout=5 * 3600)
def sft2(items: str = "/vol/ann_mt/items", lr: float = 5e-5, pdb: int = 4, seq: int = 4096, hours: float = 2.4,
         tok_s: float = 42000.0, run: str = "sft2"):
    """Кроків = min(1 епоха даних, що встигне за hours при tok_s) — спад LR завжди встигає завершитись."""
    vol.reload()
    cfg = setup_maxtext()
    meta = json.load(open(f"{VOL}/sft2/meta.json"))
    ga = 32 // pdb
    per_step = 32 * seq
    data_steps = int(meta["tokens"] / (per_step * 0.92))
    time_steps = int((hours - 0.35) * 3600 * tok_s / per_step)  # ~20 хв на старт/токенізацію/компіляцію
    steps = min(data_steps, time_steps)
    dims = DIMS.replace("base_num_decoder_layers=12", "base_num_decoder_layers=16").replace("num_experts=64", "num_experts=128")
    inst = ("import sys,runpy; sys.path.insert(0,'%s'); import padded_gmm; padded_gmm.install(1.5); "
            "import fast_conv; fast_conv.install(); import gdn_fast; gdn_fast.install('default','solve',64); "
            "sys.argv=['train']+sys.argv[1:]; "
            "runpy.run_module('maxtext.trainers.post_train.sft.train_sft_native', run_name='__main__')" % ROOT)
    cmd = [sys.executable, "-c", inst, cfg] + dims.split() + [
        "hardware=gpu", "skip_jax_distributed_system=True", f"tokenizer_path={ROOT}/tokenizer_sft",
        "sparse_matmul=True", "megablox=False", "weight_dtype=bfloat16",
        f"per_device_batch_size={pdb}", f"gradient_accumulation_steps={ga}", f"max_target_length={seq}",
        "dataset_type=hf", "hf_path=parquet", f"hf_train_files={VOL}/sft2/train.parquet",
        "train_data_columns=['messages']", "sft_train_on_completion_only=True", "packing=True",
        "opt_type=adamw", f"learning_rate={lr}", "lr_schedule_type=cosine", "warmup_steps_fraction=0.03",
        "learning_rate_final_fraction=0.1", f"steps={steps}", f"learning_rate_schedule_steps={steps}",
        f"load_parameters_path={items}", f"base_output_directory={VOL}/out", f"run_name={run}",
        "enable_checkpointing=True", "async_checkpointing=True", "checkpoint_period=200",
        "max_num_checkpoints_to_keep=2", "log_period=20"]
    print(f"SFT2: кроків {steps} (дані {data_steps}, час {time_steps}), ~{steps * per_step / 1e6:.0f}M токенів", flush=True)
    deadline = time.time() + hours * 3600
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if "completed step" in line:
            m = re.search(r"completed step: (\d+)", line)
            if m and int(m.group(1)) % 20 == 0:
                print(line.strip()[60:300], flush=True)
        elif re.search(r"Error|Traceback|RESOURCE_EXHAUSTED|saved a checkpoint|Unable to apply", line):
            print(line.rstrip()[:400], flush=True)
        if time.time() > deadline:
            print("ДЕДЛАЙН", flush=True)
            p.send_signal(2)
            break
    p.wait()
    vol.commit()
    ck = sorted(int(os.path.basename(d)) for d in glob.glob(f"{VOL}/out/{run}/checkpoints/*") if os.path.basename(d).isdigit())
    print("ГОТОВО, чекпоінти:", ck, "з", steps, "код", p.returncode, flush=True)
    return ck
