"""
Токенізація FineWeb-Edu sample-10BT нашим токенайзером у формат Megatron mmap (.bin + .idx),
який читає MaxText (grain_file_type=mmap). Kaggle CPU-сесія, квоту не їсть.

Кожен документ = одна послідовність, у кінці EOD (<|endoftext|>). Токени int32 (словник 131k > uint16).
SHARD/FILES підставляє build.py — 4 сесії паралельно, кожна обробляє свою частину файлів.
"""
import glob
import os
import struct
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

SHARD = "__SHARD__"
FILES = __FILES__
REPO = "HuggingFaceFW/fineweb-edu"
OUT = f"/kaggle/working/fineweb_edu_{SHARD}"
T0 = time.time()

tok_path = next(iter(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True)))
tok = Tokenizer.from_file(tok_path)
EOD = tok.token_to_id("<|endoftext|>")
assert EOD is not None
print("tokenizer:", tok_path, "vocab", tok.get_vocab_size(), "EOD", EOD, flush=True)

sizes = []
n_tok = 0
with open(OUT + ".bin", "wb") as fbin:
    for fname in FILES:
        path = hf_hub_download(REPO, f"sample/10BT/{fname}", repo_type="dataset", cache_dir="/tmp/hf")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4096, columns=["text"]):
            texts = batch.column(0).to_pylist()
            encs = tok.encode_batch(texts, add_special_tokens=False)
            lens = []
            chunks = []
            for e in encs:
                ids = e.ids
                ids.append(EOD)
                chunks.append(np.asarray(ids, dtype=np.int32))
                lens.append(len(ids))
            np.concatenate(chunks).tofile(fbin)
            sizes.extend(lens)
            n_tok += sum(lens)
        del pf  # закрити файл перед видаленням
        os.remove(path)
        el = time.time() - T0
        print(f"{fname}: всього {len(sizes):,} док, {n_tok / 1e9:.3f}B токенів, "
              f"{n_tok / el / 1e6:.2f}M ток/с, {el / 60:.1f} хв", flush=True)

# .idx у форматі Megatron (Convention A): magic, version, dtype, num_seq, num_doc, sizes, pointers, doc_idx
sizes = np.asarray(sizes, dtype=np.int32)
pointers = np.zeros(len(sizes), dtype=np.int64)
pointers[1:] = np.cumsum(sizes[:-1].astype(np.int64) * 4)
doc_idx = np.arange(len(sizes) + 1, dtype=np.int64)
with open(OUT + ".idx", "wb") as f:
    f.write(b"MMIDIDX\x00\x00")
    f.write(struct.pack("<Q", 1))
    f.write(struct.pack("<B", 4))  # int32
    f.write(struct.pack("<Q", len(sizes)))
    f.write(struct.pack("<Q", len(sizes)))
    f.write(sizes.tobytes())
    f.write(pointers.tobytes())
    f.write(doc_idx.tobytes())

# самоперевірка: розмір .bin, останній токен кожного документа = EOD, декод першого документа
bin_size = os.path.getsize(OUT + ".bin")
assert bin_size == int(sizes.astype(np.int64).sum()) * 4, (bin_size, sizes.sum())
data = np.memmap(OUT + ".bin", dtype=np.int32, mode="r")
ends = pointers // 4 + sizes - 1
assert np.all(data[ends] == EOD), "не всі документи закінчуються EOD"
assert int(data.max()) < tok.get_vocab_size(), "id поза словником"
print("ПЕРЕВІРКА OK. Перший документ:", tok.decode(data[:min(60, sizes[0])].tolist())[:300].replace("\n", " "))
print(f"ГОТОВО shard {SHARD}: {len(sizes):,} док, {n_tok / 1e9:.3f}B токенів, "
      f"{bin_size / 1e9:.1f} ГБ, {(time.time() - T0) / 60:.1f} хв", flush=True)
