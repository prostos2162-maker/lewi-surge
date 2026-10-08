"""
analyze_vocab.py — скільки зі словника Qwen3.6 (248k) нам реально треба.

1. Качаємо tokenizer.json вчителя (Qwen/Qwen3.6-35B-A3B).
2. Стрімимо з HF вибірку документів з кожного джерела нашої суміші (docs/data.md).
3. Рахуємо частоти токенів, зважені на частку джерела в суміші.
4. Друкуємо покриття для різних розмірів обрізаного словника + стиснення (токенів/байт) по мовах.
5. Зберігаємо частоти в out/token_freq.json для prune_vocab.py.

Запуск: python tokenizer/analyze_vocab.py --docs 2000
"""
import argparse
import json
import os
from collections import Counter
from itertools import islice

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TEACHER = "Qwen/Qwen3.6-35B-A3B"

# (назва, частка в суміші, dataset, config, split, поле з текстом)
SOURCES = [
    ("en_edu",  0.30, "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    ("en_dclm", 0.20, "mlfoundations/dclm-baseline-1.0", None, "train", "text"),
    ("es",      0.045, "epfml/FineWeb2-HQ", "spa_Latn", "train", "text"),
    ("pt",      0.045, "epfml/FineWeb2-HQ", "por_Latn", "train", "text"),
    ("fr",      0.045, "epfml/FineWeb2-HQ", "fra_Latn", "train", "text"),
    ("de",      0.045, "epfml/FineWeb2-HQ", "deu_Latn", "train", "text"),
    ("uk",      0.07, "HuggingFaceFW/fineweb-2", "ukr_Cyrl", "train", "text"),
    ("code",    0.15, "bigcode/starcoderdata", None, "train", "content"),
    ("code_web", 0.03, "OpenCoder-LLM/opc-fineweb-code-corpus", None, "train", "text"),
    ("math",    0.07, "HuggingFaceTB/finemath", "finemath-4plus", "train", "text"),
]
# якщо gated-датасет недоступний локально — беремо відкриту заміну для оцінки
FALLBACK = {"code": ("OpenCoder-LLM/opc-annealing-corpus", "synthetic_code_snippet", "train", "text")}

SIZES = [32_768, 49_152, 65_536, 81_920, 98_304, 114_688, 131_072, 163_840, 196_608]


# StarCoderData розкладено по папках мов; без data_dir стрім бере першу за алфавітом (Ada)
CODE_LANGS = ["python", "javascript", "typescript", "java", "c", "cpp", "c-sharp", "go",
              "rust", "php", "shell", "sql", "markdown", "html", "css", "json"]


def stream_texts(ds, cfg, split, field, n):
    if ds == "bigcode/starcoderdata":
        per = max(n // len(CODE_LANGS), 1)
        for lang in CODE_LANGS:
            yield from stream_texts_one(ds, None, split, field, per, data_dir=lang)
        return
    yield from stream_texts_one(ds, cfg, split, field, n)


def stream_texts_one(ds, cfg, split, field, n, data_dir=None):
    it = load_dataset(ds, cfg, split=split, streaming=True, data_dir=data_dir)
    for row in islice(it, n):
        t = row.get(field)
        if t:
            yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2000, help="документів на джерело")
    ap.add_argument("--out", default="out")
    ap.add_argument("--only", default=None, help="перевірити лише одне джерело, напр. code")
    args = ap.parse_args()
    if args.only:
        SOURCES[:] = [s for s in SOURCES if s[0] == args.only]
    os.makedirs(args.out, exist_ok=True)

    tok = Tokenizer.from_file(hf_hub_download(TEACHER, "tokenizer.json"))
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    print(f"Словник вчителя: {vocab_size}")

    weighted = Counter()
    stats = {}
    for name, share, ds, cfg, split, field in SOURCES:
        try:
            texts = list(stream_texts(ds, cfg, split, field, args.docs))
        except Exception as e:
            if name not in FALLBACK:
                print(f"[{name}] ПРОПУЩЕНО: {type(e).__name__}: {str(e)[:120]}")
                continue
            ds, cfg, split, field = FALLBACK[name]
            print(f"[{name}] основне джерело недоступне ({type(e).__name__}), беру {ds}/{cfg}")
            texts = list(stream_texts(ds, cfg, split, field, args.docs))

        counts = Counter()
        n_tok = n_bytes = 0
        for enc in tok.encode_batch(texts, add_special_tokens=False):
            counts.update(enc.ids)
            n_tok += len(enc.ids)
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        stats[name] = {"docs": len(texts), "tokens": n_tok, "bytes": n_bytes,
                       "bytes_per_token": n_bytes / max(n_tok, 1), "share": share, "counts": counts}
        print(f"[{name}] {len(texts)} док, {n_tok:,} токенів, {n_bytes / max(n_tok, 1):.2f} байт/токен, "
              f"унікальних токенів {len(counts):,}")
        for tid, c in counts.items():
            weighted[tid] += share * c / n_tok  # нормуємо: кожне джерело важить рівно свою частку

    ranked = [tid for tid, _ in weighted.most_common()]
    print(f"\nВикористано токенів (хоч раз): {len(ranked):,} з {vocab_size:,}")

    print("\nПокриття суміші при обрізаному словнику (решта йде через byte-fallback):")
    print(f"{'розмір':>8} | {'зваж.':>7} | " + " | ".join(f"{n:>8}" for n in stats))
    for k in SIZES:
        keep = set(ranked[:k])
        row = [f"{100 * sum(weighted[t] for t in keep):6.2f}%"]
        for n, s in stats.items():
            cov = sum(c for t, c in s["counts"].items() if t in keep) / s["tokens"]
            row.append(f"{100 * cov:7.2f}%")
        print(f"{k:>8} | " + " | ".join(row))

    with open(os.path.join(args.out, "token_freq.json"), "w", encoding="utf-8") as f:
        json.dump({"teacher": TEACHER, "vocab_size": vocab_size,
                   "weighted": {str(t): w for t, w in weighted.items()},
                   "stats": {n: {k: v for k, v in s.items() if k != "counts"} for n, s in stats.items()}},
                  f, ensure_ascii=False)
    print(f"\nЧастоти збережено в {args.out}/token_freq.json")


if __name__ == "__main__":
    main()
