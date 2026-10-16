"""Токенізація етапу 1 для Lewi Surge 7B-A1B на CPU-сервері (Vultr), джерело = один процес.

    python tok_stage1.py <name>          # одне джерело зі SPEC
    python tok_stage1.py --list          # перелік джерел і бюджетів

Вихід: /data/out/{name}.bin (int32) + .idx (формат mmap для MaxText grain_file_type=mmap), як у kaggle/tok3.
Бюджети — етап 1 з datasets_mix_v2.xlsx, масштабований до ≈11B (10B на мій акаунт + запас).
Stack-Edu (Software Heritage) і Nemotron-CC-Math (гейт) — окремими джерелами, якщо доступ є.
"""
import glob
import json
import os
import random
import re
import struct
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

B = 1_000_000_000
OUT = os.environ.get("OUT", "/data/out")
CACHE = os.environ.get("HF_CACHE", "/data/hf")
TOK = os.environ.get("TOK", os.path.expanduser("~/tokenizer.json"))
MAX_CHARS = 100_000
BAD_ID = re.compile(r"\b(Qwen|Alibaba|OpenAI|ChatGPT|GPT-?[345]|Claude|Anthropic|MiniCPM|OpenBMB|ModelBest|Nemotron|NVIDIA|Llama|Gemini|DeepSeek|GLM|Zhipu|Mistral|Gemma|Google)\b")

# ------------------------------------------------------------------ джерела (етап 1)
SPEC = {
    # код
    "code_ocr": dict(repo="nvidia/OpenCodeReasoning", prefix="split_0/", fmt="qa", q="input", a="output", budget=1.1 * B),
    "code_web": dict(repo="OpenCoder-LLM/opc-fineweb-code-corpus", prefix="data/", fmt="text", cols=["text"], budget=1.5 * B, seed=3),
    "code_ann": dict(repo="OpenCoder-LLM/opc-annealing-corpus", prefix="algorithmic_corpus/", fmt="jtext", budget=0.6 * B),
    "code_snip": dict(repo="OpenCoder-LLM/opc-annealing-corpus", prefix="synthetic_code_snippet/", fmt="jtext", budget=0.4 * B),
    "code_gh": dict(repo="codeparrot/github-code-clean", prefix="data/", fmt="code", budget=1.0 * B, fim=0.3, seed=5),
    # математика / міркування
    "math_fm4": dict(repo="HuggingFaceTB/finemath", prefix="finemath-4plus/", fmt="text", cols=["text"], budget=1.2 * B, seed=7),
    "math_ncc": dict(repo="nvidia/Nemotron-CC-Math-v1", prefix="4plus/", fmt="text", cols=["text"], budget=1.8 * B, seed=8, optional=True),
    "math_ot3": dict(repo="open-thoughts/OpenThoughts3-1.2M", prefix="data/", fmt="conv", domain="math", budget=0.6 * B, max_chars=60_000),
    # англійський веб
    "en_fwhq": dict(repo="epfml/FineWeb-HQ", prefix="data/CC-MAIN-2024-10/", fmt="text", cols=["text"], budget=0.65 * B, seed=9),
    "en_wiki": dict(repo="wikimedia/wikipedia", prefix="20231101.en/", fmt="wiki", cols=["title", "text"], min_chars=4000, budget=0.2 * B, seed=10),
    # інші мови
    "es_fw2": dict(repo="epfml/FineWeb2-HQ", prefix="spa_Latn/", fmt="text", cols=["text"], budget=0.22 * B, seed=11),
    "fr_fw2": dict(repo="epfml/FineWeb2-HQ", prefix="fra_Latn/", fmt="text", cols=["text"], budget=0.22 * B, seed=12),
    "pt_fw2": dict(repo="epfml/FineWeb2-HQ", prefix="por_Latn/", fmt="text", cols=["text"], budget=0.22 * B, seed=13),
    "uk_kobza": dict(repo="Goader/kobza", prefix="", fmt="text", cols=["text", "source"], min_chars=1000, budget=0.7 * B, seed=14),
    "uk_wiki": dict(repo="wikimedia/wikipedia", prefix="20231101.uk/", fmt="wiki", cols=["title", "text"], min_chars=1500, budget=0.08 * B),
    "ml_wiki": dict(repo="wikimedia/wikipedia", prefix=("20231101.es/", "20231101.fr/", "20231101.pt/"), fmt="wiki",
                    cols=["title", "text"], min_chars=4000, budget=0.07 * B, seed=15),
    # діалог / інструкції
    "dlg_ot3": dict(repo="open-thoughts/OpenThoughts3-1.2M", prefix="data/", fmt="conv", domain=("code", "science"), budget=0.45 * B, seed=16, max_chars=60_000),
    "uk_orca": dict(repo="cidtd-mod-ua/slim-orca-ukrainian", prefix="", fmt="conv", budget=0.1 * B, lat_max=0.3),
    "uk_wizard": dict(repo="cidtd-mod-ua/WizardLM-ukrainian", prefix="", fmt="conv", budget=0.05 * B, lat_max=0.3),
}

api = HfApi()
rng = random.Random(1234)
T0 = time.time()


def log(*a):
    print(f"[{NAME}] {(time.time() - T0) / 60:5.1f} хв |", *a, flush=True)


# ------------------------------------------------------------------ рендери
def chat(q, a):
    if not q or not a:
        return None
    return f"<|im_start|>user\n{q.strip()}<|im_end|>\n<|im_start|>assistant\n{a.strip()}<|im_end|>"


def fim(text):
    a, b = sorted(rng.randrange(len(text) + 1) for _ in range(2))
    return "<|fim_prefix|>" + text[:a] + "<|fim_suffix|>" + text[b:] + "<|fim_middle|>" + text[a:b]


CYR = re.compile(r"[а-яіїєґА-ЯІЇЄҐ]")
LAT = re.compile(r"[a-zA-Z]")


def latin_share(t):
    c, l = len(CYR.findall(t)), len(LAT.findall(t))
    return l / max(1, c + l)


def conv_to_text(m, src):
    """Список повідомлень у будь-якому з форматів {role,content} / {from,value} → наш ChatML."""
    out = []
    for x in m:
        role = x.get("role") or x.get("from") or ""
        role = {"human": "user", "gpt": "assistant", "bot": "assistant", "system": "system"}.get(role, role)
        c = x.get("content") or x.get("value") or ""
        if role not in ("system", "user", "assistant") or not c:
            continue
        if role == "assistant" and BAD_ID.search(c):
            return None
        out.append(f"<|im_start|>{role}\n{c.strip()}<|im_end|>")
    if len(out) < 2:
        return None
    t = "\n".join(out)
    if "lat_max" in src and latin_share(t) > src["lat_max"]:
        return None  # зламаний переклад / латиниця в українському наборі
    return t if src.get("min_chars", 0) <= len(t) <= src.get("max_chars", 200_000) else None


def render(row, src):
    fmt = src["fmt"]
    mx = src.get("max_chars", MAX_CHARS)
    mn = src.get("min_chars", 0)
    if fmt in ("text", "jtext"):
        t = row.get("text")
        return t if t and mn <= len(t) <= mx else None
    if fmt == "wiki":
        t = row.get("text")
        if not t or not (mn <= len(t) <= mx):
            return None
        return "# " + (row.get("title") or "") + "\n\n" + t
    if fmt == "qa":
        return chat(row.get(src["q"]), row.get(src["a"]))
    if fmt == "code":
        code = row.get("code") or ""
        if not code or len(code) > mx:
            return None
        if src.get("fim") and rng.random() < src["fim"]:
            return fim(code)
        return f"<|repo_name|>{row.get('repo_name', '')}\n<|file_sep|>{row.get('path', '')}\n{code}"
    if fmt == "conv":
        if "domain" in src:
            d = src["domain"]
            if row.get("domain") not in ((d,) if isinstance(d, str) else d):
                return None
        m = row.get("conversations") or row.get("messages")
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except Exception:
                return None
        if not m:  # пари instruction/input/response (slim-orca-uk: instruction = системний промпт, input = запит)
            sysm = row.get("instruction") if row.get("input") else None
            q = row.get("input") or row.get("instruction") or row.get("question") or row.get("prompt")
            a = row.get("response") or row.get("output") or row.get("answer")
            m = ([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        return conv_to_text(m, src)
    raise ValueError(fmt)


# ------------------------------------------------------------------ читання / запис
def read_rows(path, cols, bs):
    if path.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        have = set(pf.schema_arrow.names)
        use = [c for c in (cols or []) if c in have] or None
        for b in pf.iter_batches(batch_size=bs, columns=use):
            yield b.to_pylist()
        return
    opener = open
    if path.endswith(".zst"):
        import zstandard, io
        opener = lambda p, **k: io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(p, "rb")), encoding="utf-8")
    elif path.endswith(".gz"):
        import gzip
        opener = lambda p, **k: gzip.open(p, "rt", encoding="utf-8")
    buf = []
    with opener(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                buf.append(json.loads(line))
            if len(buf) >= bs:
                yield buf
                buf = []
    if buf:
        yield buf


def write_idx(path, sizes):
    sizes = np.asarray(sizes, dtype=np.int32)
    pointers = np.zeros(len(sizes), dtype=np.int64)
    pointers[1:] = np.cumsum(sizes[:-1].astype(np.int64) * 4)
    with open(path, "wb") as f:
        f.write(b"MMIDIDX\x00\x00")
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", 4))
        f.write(struct.pack("<Q", len(sizes)))
        f.write(struct.pack("<Q", len(sizes)))
        f.write(sizes.tobytes())
        f.write(pointers.tobytes())
        f.write(np.arange(len(sizes) + 1, dtype=np.int64).tobytes())
    return pointers, sizes


def main(name):
    global NAME
    NAME = name
    src = SPEC[name]
    tok = Tokenizer.from_file(TOK)
    EOD = tok.token_to_id("<|endoftext|>")
    assert EOD is not None
    budget = int(src["budget"])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, name)
    if os.path.exists(out + ".idx"):
        log("вже готово, пропускаю")
        return
    prefixes = src["prefix"] if isinstance(src["prefix"], tuple) else (src["prefix"],)
    try:
        sib = api.dataset_info(src["repo"]).siblings
    except Exception as e:
        if src.get("optional"):
            log("немає доступу, пропускаю:", repr(e)[:120])
            return
        raise
    files = sorted(s.rfilename for s in sib if any(s.rfilename.startswith(p) for p in prefixes)
                   and s.rfilename.endswith((".parquet", ".jsonl", ".jsonl.zst", ".jsonl.gz", ".json.gz")))
    (random.Random(src["seed"]) if "seed" in src else rng).shuffle(files)
    log(f"{len(files)} файлів, бюджет {budget / 1e9:.2f}B")
    sizes, n_tok, n_seen, n_kept = [], 0, 0, 0
    bs = src.get("bs", 1024)
    with open(out + ".bin.part", "wb") as fbin:
        for fname in files:
            if n_tok >= budget:
                break
            try:
                path = hf_hub_download(src["repo"], fname, repo_type="dataset", cache_dir=CACHE)
            except Exception as e:
                log("помилка завантаження", fname, repr(e)[:120])
                continue
            for rows in read_rows(path, src.get("cols"), bs):
                n_seen += len(rows)
                texts = [t for t in (render(r, src) for r in rows) if t]
                n_kept += len(texts)
                chunks = []
                for e in tok.encode_batch(texts, add_special_tokens=False):
                    ids = e.ids
                    ids.append(EOD)
                    chunks.append(np.asarray(ids, dtype=np.int32))
                    sizes.append(len(ids))
                    n_tok += len(ids)
                if chunks:
                    np.concatenate(chunks).tofile(fbin)
                if n_tok >= budget:
                    break
            try:
                os.remove(path)
            except OSError:
                pass
            log(f"{fname[-50:]}: {len(sizes):,} док ({n_kept}/{n_seen} лишено), {n_tok / 1e9:.3f}B ток")
    os.rename(out + ".bin.part", out + ".bin")
    pointers, sizes = write_idx(out + ".idx", sizes)
    assert os.path.getsize(out + ".bin") == int(sizes.astype(np.int64).sum()) * 4
    data = np.memmap(out + ".bin", dtype=np.int32, mode="r")
    assert np.all(data[pointers // 4 + sizes - 1] == EOD)
    assert int(data.max()) < tok.get_vocab_size()
    json.dump({"name": name, "repo": src["repo"], "docs": len(sizes), "tokens": int(n_tok), "minutes": (time.time() - T0) / 60},
              open(out + ".json", "w"))
    log(f"OK: {len(sizes):,} док, {n_tok / 1e9:.3f}B ток. Приклад:",
        tok.decode(data[:min(60, sizes[0])].tolist(), skip_special_tokens=False)[:200].replace("\n", " "))


if __name__ == "__main__":
    if sys.argv[1] == "--list":
        for k, v in SPEC.items():
            print(f"{k:12s} {v['budget'] / 1e9:5.2f}B  {v['repo']}")
        print(f"разом {sum(v['budget'] for v in SPEC.values()) / 1e9:.2f}B")
    else:
        main(sys.argv[1])
