"""
Токенізація якісних даних (code-first): annealing-код OpenCoder, інструкції з кодом у чат-форматі, Cosmopedia у Megatron mmap (.bin + .idx, int32),
який читає MaxText (grain_file_type=mmap). Kaggle CPU-сесія, квоту не їсть.

SPEC підставляє build.py: список джерел {repo, prefix, fmt, budget (токенів), fim, ...}.
Кожне джерело → окремий файл {name}.bin/.idx (щоб частки суміші крутити на тренуванні).
"""
import glob
import json
import os
import shutil
import random
import struct
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

SPEC = __SPEC__
try:  # Kaggle secret "hf" (HF read-токен) — для гейтованих датасетів
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("hf")
    print("HF token: є", flush=True)
except Exception as e:
    print("HF token: немає", type(e).__name__, flush=True)
OUT = "/kaggle/working"
T0 = time.time()
rng = random.Random(1234)

tok = Tokenizer.from_file(next(iter(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True))))
ID = {t: tok.token_to_id(t) for t in ["<|endoftext|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]}
assert all(v is not None for v in ID.values()), ID
EOD = ID["<|endoftext|>"]
MAX_CHARS = 100_000
import re
BAD_ID = re.compile(r"\b(Qwen|Alibaba|OpenAI|ChatGPT|GPT-?[345]|Claude|Anthropic|MiniCPM|OpenBMB|ModelBest|Nemotron|NVIDIA|Llama|Gemini|DeepSeek|GLM|Zhipu)\b", re.I)  # дуже довгі файли (згенеровані/мінімізовані) пропускаємо


def fim(text):
    """FIM у форматі Qwen (PSM): <|fim_prefix|>pre<|fim_suffix|>suf<|fim_middle|>mid — модель вчиться дописувати середину."""
    a, b = sorted(rng.randrange(len(text) + 1) for _ in range(2))
    pre, mid, suf = text[:a], text[a:b], text[b:]
    return "<|fim_prefix|>" + pre + "<|fim_suffix|>" + suf + "<|fim_middle|>" + mid


def render(fmt, row, src):
    mx = src.get("max_chars", MAX_CHARS)
    if fmt == "text":
        t = row["text"]
        if not t or len(t) < src.get("min_chars", 0) or len(t) > mx:
            return None
        return t
    if fmt == "code":  # github-code-clean: code, repo_name, path, language
        code = row["code"]
        if len(code) > MAX_CHARS:
            return None
        if src.get("fim", 0) and rng.random() < src["fim"]:
            return fim(code)
        return f"<|repo_name|>{row['repo_name']}\n<|file_sep|>{row['path']}\n{code}"
    if fmt == "svg":
        s = row["Svg"]
        return s if len(s) <= MAX_CHARS else None
    if fmt == "text2svg":  # опис → SVG: модель вчиться малювати за текстом
        s = row["Svg"]
        if len(s) > MAX_CHARS:
            return None
        cap = rng.choice([c for c in (row.get("caption_cogvlm"), row.get("caption_llava"), row.get("caption_blip2")) if c])
        return f"Create an SVG image: {cap.strip()}\n\n```svg\n{s}\n```"
    if fmt == "html":  # WebSight: сучасні сайти на Tailwind
        return row["text"]
    if fmt == "jtext":
        t = row.get("text")
        return t if t and src.get("min_chars", 0) <= len(t) <= mx else None
    if fmt == "qa":  # q/a — назви полів у src
        return chat(row.get(src["q"]), row.get(src["a"]))
    if fmt == "msgs":  # повний діалог (UltraData: messages = рядок з python-списком)
        import ast
        m = row.get("messages")
        if isinstance(m, str):
            try:
                m = ast.literal_eval(m)
            except Exception:
                return None
        if not m or any(BAD_ID.search(str(x.get("content") or "")) for x in m if x.get("role") == "assistant"):
            return None
        out = []
        for x in m:
            c = x.get("content")
            if not c or x.get("role") not in ("system", "user", "assistant"):
                continue
            out.append(f"<|im_start|>{x['role']}\n{c.strip()}<|im_end|>")
        t = "\n".join(out)
        return t if len(out) >= 2 and len(t) >= src.get("min_chars", 0) else None
    raise ValueError(fmt)


def render_swe(row, src):
    """Траєкторія агента (Open-SWE-Traces) → наш чат-формат з викликами інструментів у стилі Qwen."""
    out = []
    tools = row.get("tools") or []
    for x in row.get("messages") or []:
        r, c = x.get("role"), (x.get("content") or "").strip()
        if r == "system":
            if tools:
                c += "\n\n# Tools\n<tools>\n" + "\n".join(tools) + "\n</tools>"
            out.append(f"<|im_start|>system\n{c}<|im_end|>")
        elif r == "user":
            out.append(f"<|im_start|>user\n{c}<|im_end|>")
        elif r == "assistant":
            if BAD_ID.search(c):
                return None
            calls = "".join("\n<tool_call>\n" + json.dumps({"name": t["function"]["name"],
                                                             "arguments": t["function"]["arguments"]}) + "\n</tool_call>"
                            for t in (x.get("tool_calls") or []))
            out.append(f"<|im_start|>assistant\n{c}{calls}<|im_end|>")
        elif r == "tool":
            out.append(f"<|im_start|>user\n<tool_response>\n{c}\n</tool_response><|im_end|>")
    t = "\n".join(out)
    return t if len(out) >= 3 and src.get("min_chars", 0) <= len(t) <= src.get("max_chars", 2_000_000) else None


def prolong_texts(repo, prefix, budget_chars):
    """ProLong (MDS, токени Llama-3, упаковано по 512k) → тексти документів за `indices` (цілі репозиторії)."""
    from streaming import LocalDataset
    from transformers import AutoTokenizer
    llama = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")  # відкрита копія токенайзера, лише декодування
    procs = sorted({s.rfilename.split("/")[1] for s in api.dataset_info(repo).siblings
                    if s.rfilename.startswith(prefix + "/proc") and s.rfilename.endswith(".mds")})
    rng.shuffle(procs)
    got = 0
    for pdir in procs:
        local = f"/tmp/pl/{pdir}"
        os.makedirs(local, exist_ok=True)
        idx = hf_hub_download(repo, f"{prefix}/{pdir}/index.json", repo_type="dataset", local_dir="/tmp/pl_dl")
        meta = json.load(open(idx))
        shutil.copy(idx, f"{local}/index.json")
        for sh in meta["shards"]:
            fn = sh["raw_data"]["basename"]
            p = hf_hub_download(repo, f"{prefix}/{pdir}/{fn}", repo_type="dataset", local_dir="/tmp/pl_dl")
            shutil.move(p, f"{local}/{fn}")
        for s in LocalDataset(local=local):
            ids = s["input_ids"]
            for a, b in s["indices"]:
                t = llama.decode(ids[a:b], skip_special_tokens=True)
                if len(t) > 20_000:
                    got += len(t)
                    yield t
            if got >= budget_chars:
                return
        shutil.rmtree(local, ignore_errors=True)


def repo_docs(rows, src):
    """Довгий контекст: файли одного репозиторію (рядки йдуть групами) → один документ."""
    cur, buf = None, []
    for r in rows:
        if len(r["code"]) > MAX_CHARS:
            continue
        if r["repo_name"] != cur and buf:
            yield cur, buf
            buf = []
        cur = r["repo_name"]
        buf.append(r)
    if buf:
        yield cur, buf


def render_repo(repo, files, src):
    doc = f"<|repo_name|>{repo}\n" + "".join(f"<|file_sep|>{f['path']}\n{f['code']}\n" for f in files)
    return doc if src.get("min_chars", 0) <= len(doc) <= src.get("max_chars", 2_000_000) else None


def read_rows(path, cols, bs=2048):
    """Батчі рядків з parquet або jsonl."""
    if path.endswith(".parquet"):
        for b in pq.ParquetFile(path).iter_batches(batch_size=bs, columns=cols):
            yield b.to_pylist()
        return
    import json
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                buf.append(json.loads(line))
            if len(buf) >= bs:
                yield buf
                buf = []
    if buf:
        yield buf


def chat(q, a):
    """Наш чат-шаблон (як у SFT): модель звикає до формату діалогу ще на претрейні."""
    if not q or not a:
        return None
    return f"<|im_start|>user\n{q.strip()}<|im_end|>\n<|im_start|>assistant\n{a.strip()}<|im_end|>"


def write_idx(path, sizes):
    sizes = np.asarray(sizes, dtype=np.int32)
    pointers = np.zeros(len(sizes), dtype=np.int64)
    pointers[1:] = np.cumsum(sizes[:-1].astype(np.int64) * 4)
    with open(path, "wb") as f:
        f.write(b"MMIDIDX\x00\x00")
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", 4))  # int32
        f.write(struct.pack("<Q", len(sizes)))
        f.write(struct.pack("<Q", len(sizes)))
        f.write(sizes.tobytes())
        f.write(pointers.tobytes())
        f.write(np.arange(len(sizes) + 1, dtype=np.int64).tobytes())
    return pointers, sizes


api = HfApi()


def write_texts(texts, fbin, sizes):
    n = 0
    chunks = []
    for e in tok.encode_batch(texts, add_special_tokens=False):
        ids = e.ids
        ids.append(EOD)
        chunks.append(np.asarray(ids, dtype=np.int32))
        sizes.append(len(ids))
        n += len(ids)
    if chunks:
        np.concatenate(chunks).tofile(fbin)
    return n


for src in SPEC:
    name, repo, budget = src["name"], src["repo"], int(src["budget"])
    cols = src.get("cols")
    sizes, n_tok = [], 0
    out = os.path.join(OUT, name)
    with open(out + ".bin", "wb") as fbin:
        if src["fmt"] == "prolong":  # MDS-шарди, без parquet
            buf = []
            for t in prolong_texts(repo, src["prefix"], budget * 5):
                buf.append(t[: src.get("max_chars", 2_000_000)])
                if len(buf) >= 16:
                    n_tok += write_texts(buf, fbin, sizes)
                    buf = []
                    print(f"[{name}] {len(sizes):,} док, {n_tok / 1e9:.3f}B ток, {(time.time() - T0) / 60:.1f} хв", flush=True)
                if n_tok >= budget:
                    break
            if buf and n_tok < budget:
                n_tok += write_texts(buf, fbin, sizes)
            files = []
        else:
            files = sorted(s.rfilename for s in api.dataset_info(repo).siblings
                           if s.rfilename.startswith(src["prefix"]) and s.rfilename.endswith((".parquet", ".jsonl")))
            files = [f for f in files if not any(x in f for x in src.get("exclude", ()))]  # уже використані файли
            (random.Random(src["seed"]) if "seed" in src else rng).shuffle(files)
        for fname in files:
            if n_tok >= budget:
                break
            path = hf_hub_download(repo, fname, repo_type="dataset", cache_dir="/tmp/hf")
            for rows in read_rows(path, cols, src.get("bs", 2048)):
                if src["fmt"] == "repo":
                    texts = [t for t in (render_repo(n, fs, src) for n, fs in repo_docs(rows, src)) if t]
                elif src["fmt"] == "swe":
                    texts = [t for t in (render_swe(r, src) for r in rows) if t]
                else:
                    texts = [t for t in (render(src["fmt"], r, src) for r in rows) if t]
                n_tok += write_texts(texts, fbin, sizes)
                if n_tok >= budget:
                    break
            os.remove(path)
            print(f"[{name}] {fname}: {len(sizes):,} док, {n_tok / 1e9:.3f}B ток, {(time.time() - T0) / 60:.1f} хв", flush=True)
    pointers, sizes = write_idx(out + ".idx", sizes)
    # самоперевірка: розмір, EOD у кінці кожного документа, id у межах словника
    assert os.path.getsize(out + ".bin") == int(sizes.astype(np.int64).sum()) * 4
    data = np.memmap(out + ".bin", dtype=np.int32, mode="r")
    assert np.all(data[pointers // 4 + sizes - 1] == EOD), "не всі документи закінчуються EOD"
    assert int(data.max()) < tok.get_vocab_size()
    print(f"[{name}] OK: {len(sizes):,} док, {n_tok / 1e9:.3f}B ток. Приклад:",
          tok.decode(data[:min(80, sizes[0])].tolist(), skip_special_tokens=False)[:300].replace("\n", " "), flush=True)
    del data

print(f"ГОТОВО за {(time.time() - T0) / 60:.1f} хв", flush=True)
