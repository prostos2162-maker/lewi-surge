"""
Бенчмарки базового чекпоінта на локальній GPU (pass@1 greedy для коду, log-likelihood для вибору варіанта).
  python tools/bench.py checkpoints/surge_s2g2_step12250_hf --tasks humaneval mbpp arc hellaswag [--limit N]
Результат: out/bench/<ckpt>.json (+ згенерований код для перевірки очима).
Згенерований код виконується в окремому процесі з тайм-аутом у тимчасовій папці.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils as _mu

_mu.caching_allocator_warmup = lambda *a, **k: None  # WDDM: одна велика алокація не проходить

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--tasks", nargs="+", default=["humaneval", "mbpp", "arc", "hellaswag"])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--hs_n", type=int, default=1000, help="HellaSwag: перші N з validation")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.ckpt)
model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16, device_map="cuda").eval()
EOS = [i for i in {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id,
                   tok.convert_tokens_to_ids("<|endoftext|>")} if isinstance(i, int) and i >= 0]
name = os.path.basename(os.path.normpath(args.ckpt))
os.makedirs("out/bench", exist_ok=True)
res_path = f"out/bench/{name}.json"
RES = json.load(open(res_path)) if os.path.exists(res_path) else {}


def save():
    json.dump(RES, open(res_path, "w"), indent=1, ensure_ascii=False)


def gen(prompt, max_new=384, stops=()):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, eos_token_id=EOS,
                             stop_strings=list(stops) or None, tokenizer=tok)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def run_py(code, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        open(p, "w", encoding="utf-8").write(code)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=d, capture_output=True, timeout=timeout)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert", "\n```"]


def cut(completion):
    """Обрізати по першому стоп-рядку верхнього рівня (як у стандартному HumanEval)."""
    idx = [completion.find(s) for s in STOPS if completion.find(s) >= 0]
    return completion[:min(idx)] if idx else completion


def humaneval():
    ds = load_dataset("openai/openai_humaneval", split="test")
    n = len(ds) if not args.limit else args.limit
    ok, samples, t = 0, [], time.time()
    for i, ex in enumerate(ds.select(range(n))):
        comp = cut(gen(ex["prompt"], stops=STOPS))
        passed = run_py(ex["prompt"] + comp + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n")
        ok += passed
        samples.append({"id": ex["task_id"], "pass": passed, "completion": comp})
        if i % 20 == 0:
            print(f"humaneval {i + 1}/{n}: {ok / (i + 1):.3f} ({time.time() - t:.0f}с)", flush=True)
    RES["humaneval"] = {"pass@1": ok / n, "n": n}
    json.dump(samples, open(f"out/bench/{name}_humaneval.json", "w"), indent=1)
    print("HumanEval pass@1:", ok / n, flush=True)


def mbpp():
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    n = len(ds) if not args.limit else args.limit
    ok, samples, t = 0, [], time.time()
    for i, ex in enumerate(ds.select(range(n))):
        m = re.search(r"assert\s+(\w+)\s*\(", ex["test_list"][0])
        fn = m.group(1) if m else "solution"
        prompt = f'"""\n{ex["prompt"]}\n{ex["test_list"][0]}\n"""\n'
        imports = "\n".join(ex.get("test_imports") or [])
        st = ["\nassert", "\nprint(", "\nif __name__", '\n"""', "\n```"]
        comp = gen(prompt + f"def {fn}(", stops=st)
        idx = [comp.find(x) for x in st if comp.find(x) >= 0]  # stop-рядок лишається в тексті — відрізаємо
        comp = comp[:min(idx)] if idx else comp
        code = f"{imports}\ndef {fn}(" + comp
        passed = run_py(code + "\n\n" + "\n".join(ex["test_list"]) + "\n")
        ok += passed
        samples.append({"id": ex["task_id"], "pass": passed, "code": code})
        if i % 25 == 0:
            print(f"mbpp {i + 1}/{n}: {ok / (i + 1):.3f} ({time.time() - t:.0f}с)", flush=True)
    RES["mbpp"] = {"pass@1": ok / n, "n": n}
    json.dump(samples, open(f"out/bench/{name}_mbpp.json", "w"), indent=1)
    print("MBPP pass@1:", ok / n, flush=True)


@torch.no_grad()
def choice_ll(ctx, conts):
    """Сума і середнє log-prob продовжень (acc і acc_norm як у lm-eval)."""
    out = []
    ctx_ids = tok(ctx).input_ids
    for c in conts:
        c_ids = tok(c).input_ids
        ids = torch.tensor([ctx_ids + c_ids], device="cuda")
        logp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]
        lp = logp[torch.arange(len(tgt)), tgt][-len(c_ids):].sum().item()
        out.append((lp, lp / max(1, len(c))))
    return out


def mc(task, items):
    acc = accn = 0
    t = time.time()
    for i, (ctx, conts, gold) in enumerate(items):
        s = choice_ll(ctx, conts)
        acc += max(range(len(s)), key=lambda j: s[j][0]) == gold
        accn += max(range(len(s)), key=lambda j: s[j][1]) == gold
        if i % 200 == 0:
            print(f"{task} {i + 1}/{len(items)}: acc {acc / (i + 1):.3f} norm {accn / (i + 1):.3f} ({time.time() - t:.0f}с)", flush=True)
    RES[task] = {"acc": acc / len(items), "acc_norm": accn / len(items), "n": len(items)}
    print(task, RES[task], flush=True)


def arc():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = [(f"Question: {e['question']}\nAnswer:", [" " + t for t in e["choices"]["text"]],
              e["choices"]["label"].index(e["answerKey"])) for e in ds]
    mc("arc_challenge", items[:args.limit] if args.limit else items)


def hellaswag():
    ds = load_dataset("Rowan/hellaswag", split="validation").select(range(args.hs_n))
    def pre(t):
        return re.sub(r"\[.*?\]", "", t.replace(" [title]", ". ")).replace("  ", " ").strip()
    items = [(pre(e["activity_label"] + ": " + e["ctx_a"] + " " + e["ctx_b"].capitalize()),
              [" " + pre(x) for x in e["endings"]], int(e["label"])) for e in ds]
    mc("hellaswag", items)


for task in args.tasks:
    if task in RES and not args.limit:
        print("вже є", task, RES[task], flush=True)
        continue
    globals()[task]()
    if not args.limit:
        save()
print(json.dumps(RES, indent=1), flush=True)
