"""
Токенізація даних стадії 2 (код з FIM, мови, математика, SVG/дизайн) у Megatron mmap (.bin + .idx, int32),
який читає MaxText (grain_file_type=mmap). Kaggle CPU-сесія, квоту не їсть.

SPEC підставляє build.py: список джерел {repo, prefix, fmt, budget (токенів), fim, ...}.
Кожне джерело → окремий файл {name}.bin/.idx (щоб частки суміші крутити на тренуванні).
"""
import glob
import os
import random
import struct
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

SPEC = __SPEC__
OUT = "/kaggle/working"
T0 = time.time()
rng = random.Random(1234)

tok = Tokenizer.from_file(next(iter(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True))))
ID = {t: tok.token_to_id(t) for t in ["<|endoftext|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]}
assert all(v is not None for v in ID.values()), ID
EOD = ID["<|endoftext|>"]
MAX_CHARS = 100_000  # дуже довгі файли (згенеровані/мінімізовані) пропускаємо


def fim(text):
    """FIM у форматі Qwen (PSM): <|fim_prefix|>pre<|fim_suffix|>suf<|fim_middle|>mid — модель вчиться дописувати середину."""
    a, b = sorted(rng.randrange(len(text) + 1) for _ in range(2))
    pre, mid, suf = text[:a], text[a:b], text[b:]
    return "<|fim_prefix|>" + pre + "<|fim_suffix|>" + suf + "<|fim_middle|>" + mid


def render(fmt, row, src):
    if fmt == "text":
        return row["text"]
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
    raise ValueError(fmt)


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
for src in SPEC:
    name, repo, budget = src["name"], src["repo"], int(src["budget"])
    files = sorted(s.rfilename for s in api.dataset_info(repo).siblings
                   if s.rfilename.startswith(src["prefix"]) and s.rfilename.endswith(".parquet"))
    rng.shuffle(files)  # різні частини датасету, а не лише перші файли
    cols = src.get("cols")
    sizes, n_tok = [], 0
    out = os.path.join(OUT, name)
    with open(out + ".bin", "wb") as fbin:
        for fname in files:
            if n_tok >= budget:
                break
            path = hf_hub_download(repo, fname, repo_type="dataset", cache_dir="/tmp/hf")
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2048, columns=cols):
                texts = [t for t in (render(src["fmt"], r, src) for r in batch.to_pylist()) if t]
                encs = tok.encode_batch(texts, add_special_tokens=False)
                chunks = []
                for e in encs:
                    ids = e.ids
                    ids.append(EOD)
                    chunks.append(np.asarray(ids, dtype=np.int32))
                    sizes.append(len(ids))
                    n_tok += len(ids)
                if chunks:
                    np.concatenate(chunks).tofile(fbin)
                if n_tok >= budget:
                    break
            del pf
            os.remove(path)
            el = time.time() - T0
            print(f"[{name}] {fname}: {len(sizes):,} док, {n_tok / 1e9:.3f}B ток, {el / 60:.1f} хв", flush=True)
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
