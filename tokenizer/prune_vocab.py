"""
prune_vocab.py — обрізаємо словник Qwen3.6 (248k) до K токенів під нашу суміш.

Як це працює з byte-level BPE:
  * 256 базових байтових токенів лишаються завжди → будь-який текст досі кодується (byte-fallback).
  * Беремо top-K токенів за зваженою частотою (out/token_freq.json з analyze_vocab.py).
  * ЗАМИКАННЯ: BPE будує токен із двох батьків через merge. Якщо батька викинути, merge ніколи
    не спрацює і токен стане недосяжним. Тому рекурсивно додаємо всіх предків кожного токена.
  * Merges лишаємо лише ті, де обидва батьки й результат у словнику (порядок = пріоритет зберігається).
  * Нові id призначаються в порядку старих id → відображення монотонне, спецтокени (added_tokens) в кінці.

Вихід:
  out/tokenizer_pruned/tokenizer.json   — новий токенайзер
  out/tokenizer_pruned/old2new.json     — масив old_id -> new_id (-1 = викинуто), для дистиляції logits

Запуск: python tokenizer/prune_vocab.py --size 98304
"""
import argparse
import json
import os

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel

TEACHER = "Qwen/Qwen3.6-35B-A3B"

SAMPLES = [
    "The quick brown fox jumps over the lazy dog. Mixture-of-experts models route tokens.",
    "Привіт! Як справи? Сьогодні ми тренуємо власну мовну модель на безкоштовних TPU.",
    "¿Dónde está la biblioteca? Me gustaría aprender sobre redes neuronales.",
    "Olá, tudo bem? A inteligência artificial está mudando o mundo.",
    "Bonjour, je m'appelle Pierre et j'adore les mathématiques.",
    "Guten Tag! Die Straßenbahn fährt um fünf Uhr ab. Größenordnung.",
    "def softmax(x):\n    e = np.exp(x - x.max())\n    return e / e.sum()\n",
    "Нехай $f(x) = \\int_0^x e^{-t^2} dt$, тоді f'(x) = e^{-x^2}.",
    "你好，世界！ こんにちは 🤖🔥 — рідкісні символи мають кодуватись через байти.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=98304, help="цільовий розмір (без спецтокенів)")
    ap.add_argument("--freq", default="out/token_freq.json")
    ap.add_argument("--out", default="out/tokenizer_pruned")
    args = ap.parse_args()

    path = hf_hub_download(TEACHER, "tokenizer.json")
    j = json.load(open(path, encoding="utf-8"))
    vocab = j["model"]["vocab"]                       # token -> id
    id2tok = {i: t for t, i in vocab.items()}
    merges = [m.split(" ", 1) if isinstance(m, str) else m for m in j["model"]["merges"]]
    added = j["added_tokens"]
    # у tokenizer_config.json є спецтокени, яких немає в tokenizer.json (аудіо/tts) — додаємо, знадобляться
    cfg = json.load(open(hf_hub_download(TEACHER, "tokenizer_config.json"), encoding="utf-8"))
    known = {a["id"] for a in added}
    for k, v in sorted(cfg.get("added_tokens_decoder", {}).items(), key=lambda kv: int(kv[0])):
        if int(k) not in known:
            added.append({"id": int(k), "content": v["content"], "single_word": v.get("single_word", False),
                          "lstrip": v.get("lstrip", False), "rstrip": v.get("rstrip", False),
                          "normalized": v.get("normalized", False), "special": v.get("special", True)})
    added_ids = {a["id"] for a in added}

    # хто кого породжує: результат -> (лівий, правий); беремо перший (найпріоритетніший) merge
    parents = {}
    for left, right in merges:
        parents.setdefault(left + right, (left, right))

    freq = json.load(open(args.freq, encoding="utf-8"))["weighted"]
    ranked = sorted((int(t) for t in freq if int(t) not in added_ids), key=lambda t: -freq[str(t)])

    base = set(ByteLevel.alphabet())                  # 256 байтових символів
    keep = {t for t in base if t in vocab}

    def add(tok):
        stack = [tok]
        while stack:
            t = stack.pop()
            if t in keep:
                continue
            keep.add(t)
            if t in parents:
                stack.extend(parents[t])

    n_direct = 0
    for tid in ranked:
        if len(keep) >= args.size:
            break
        if id2tok[tid] not in keep:
            n_direct += 1
        add(id2tok[tid])

    n_closure_extra = len(keep) - len(base & keep) - n_direct
    old_ids = sorted(vocab[t] for t in keep)
    old2new = {old: new for new, old in enumerate(old_ids)}
    new_vocab = {id2tok[o]: n for o, n in old2new.items()}
    new_merges = [f"{l} {r}" for l, r in merges if l in keep and r in keep and (l + r) in keep]

    next_id = len(new_vocab)
    new_added = []
    for a in sorted(added, key=lambda a: a["id"]):
        old2new[a["id"]] = next_id
        new_added.append({**a, "id": next_id})
        next_id += 1

    j["model"]["vocab"] = new_vocab
    j["model"]["merges"] = new_merges
    j["added_tokens"] = new_added

    os.makedirs(args.out, exist_ok=True)
    out_tok = os.path.join(args.out, "tokenizer.json")
    with open(out_tok, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False)
    total_old = max(id2tok) + 1 + len(added)
    with open(os.path.join(args.out, "old2new.json"), "w") as f:
        json.dump([old2new.get(i, -1) for i in range(total_old)], f)

    # tokenizer_config.json вчителя (чат-шаблон, спецтокени) з перенумерованими id → працює AutoTokenizer
    if "added_tokens_decoder" in cfg:
        cfg["added_tokens_decoder"] = {str(old2new[int(k)]): v for k, v in cfg["added_tokens_decoder"].items()}
    with open(os.path.join(args.out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    for extra in ("chat_template.jinja",):
        try:
            src = hf_hub_download(TEACHER, extra)
            with open(src, encoding="utf-8") as fi, open(os.path.join(args.out, extra), "w", encoding="utf-8") as fo:
                fo.write(fi.read())
        except Exception:
            pass

    print(f"Словник: {len(vocab) + len(added):,} -> {next_id:,} "
          f"({len(new_vocab):,} BPE + {len(new_added)} спец); merges {len(merges):,} -> {len(new_merges):,}")
    print(f"Токенів, доданих через замикання предків: {max(n_closure_extra, 0):,}")

    # перевірка: декод(енкод(x)) == x і наскільки зросла довжина
    old_tok, new_tok = Tokenizer.from_file(path), Tokenizer.from_file(out_tok)
    ok = True
    for s in SAMPLES:
        a = old_tok.encode(s, add_special_tokens=False).ids
        b = new_tok.encode(s, add_special_tokens=False).ids
        rt = new_tok.decode(b) == s
        ok &= rt
        print(f"  {len(a):3d} -> {len(b):3d} токенів  roundtrip={'OK' if rt else 'FAIL'}  {s[:50]!r}")
    print("Перевірка пройдена" if ok else "УВАГА: roundtrip зламався")


if __name__ == "__main__":
    main()
