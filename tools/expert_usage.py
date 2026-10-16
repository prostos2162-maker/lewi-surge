"""Статистика використання експертів оригінальної моделі (скільки разів кожен експерт у кожному шарі потрапляє в top-k).

    python tools/expert_usage.py MODEL_DIR [N_DOCS=12] [MAX_TOK=512]  →  MODEL_DIR/expert_usage.json

Потрібно для росту зі склейкою пар: сильний експерт + слабкий (рідко вживаний) партнер.
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grow_check import docs  # noqa: E402  (ті самі документи)
from transformers import AutoTokenizer  # noqa: E402

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
MAXT = int(sys.argv[3]) if len(sys.argv) > 3 else 512
torch.set_num_threads(os.cpu_count())
tok = AutoTokenizer.from_pretrained(path)
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu").eval()
cfg = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
E = cfg.num_experts
counts = {}
hooks = []
for name, mod in m.named_modules():
    if name.endswith("mlp.gate") and hasattr(mod, "top_k"):
        li = int(name.split(".layers.")[1].split(".")[0])
        counts[li] = torch.zeros(E, dtype=torch.long)

        def hook(mod, inp, out, li=li):
            _, _, idx = out
            counts[li] += torch.bincount(idx.reshape(-1), minlength=E)
        hooks.append(mod.register_forward_hook(hook))
with torch.no_grad():
    for i, t in enumerate(docs()[:N]):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=MAXT).input_ids
        m(ids)
        print(f"  {i + 1}/{N}", flush=True)
res = {str(li): c.tolist() for li, c in counts.items()}
json.dump(res, open(os.path.join(path, "expert_usage.json"), "w"))
for li in sorted(counts):
    c = counts[li].float()
    print(f"шар {li:2d}: мін {int(c.min())}, медіана {int(c.median())}, макс {int(c.max())}, нульових {int((c == 0).sum())}")
