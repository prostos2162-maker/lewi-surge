"""Перевірка росту: лос оригіналу проти вирощеної моделі на тих самих документах (CPU, bf16, transformers).

    python tools/grow_check.py ORIG_DIR GROWN_DIR [N_DOCS=24] [MAX_TOK=1024]

Документи: тексти з data/lewi2_gen (наші), якщо є, інакше кілька абзаців англійською/українською вбудовано.
Критерій з плану: dup без merge → лос збігається з оригіналом (±0.01 через bf16); dup+merge → просідання ≤ ~0.3.
"""
import glob
import json
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

orig, grown = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
MAXT = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
torch.set_num_threads(os.cpu_count())


def docs():
    out = []
    for f in sorted(glob.glob("data/lewi2_gen/*.jsonl"))[:3] + sorted(glob.glob("data/uk_dialogs/*.jsonl"))[:1]:
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = r.get("text") or " ".join(m.get("content", "") for m in (r.get("messages") or []))
            if t and len(t) > 800:
                out.append(t)
            if len(out) >= N:
                return out
    return out or ["The quick brown fox jumps over the lazy dog. " * 200, "Київ — столиця України, одне з найбільших міст Європи. " * 100]


tok = AutoTokenizer.from_pretrained(orig)
texts = docs()[:N]
enc = [tok(t, return_tensors="pt", truncation=True, max_length=MAXT).input_ids for t in texts]
print(f"{len(enc)} документів, разом {sum(e.shape[1] for e in enc)} токенів")


def mean_loss(path):
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu")
    m.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i, ids in enumerate(enc):
            out = m(ids, labels=ids)
            k = ids.shape[1] - 1
            tot += float(out.loss) * k
            n += k
            if i % 4 == 0:
                print(f"  {os.path.basename(path)}: {i + 1}/{len(enc)} лос {tot / n:.4f} ({(time.time() - t0) / 60:.1f} хв)", flush=True)
    del m
    return tot / n


lo = mean_loss(orig)
lg = mean_loss(grown)
print(f"\nоригінал {lo:.4f} | вирощена {lg:.4f} | різниця {lg - lo:+.4f} (ppl {math.exp(lo):.2f} → {math.exp(lg):.2f})")
json.dump({"orig": lo, "grown": lg, "docs": len(enc)}, open(os.path.join(grown, "grow_check.json"), "w"))
