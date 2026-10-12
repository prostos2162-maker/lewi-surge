"""Швидка перевірка базового чекпоінта: код-доповнення, текст, чат-формат. python tools/probe.py <ckpt> [<ckpt> ...]"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None  # WDDM: одна алокація 6.7 ГБ не проходить

PROMPTS = [
    'def is_prime(n: int) -> bool:\n    """Return True if n is a prime number."""\n',
    'def fibonacci(n):\n    """Return the n-th Fibonacci number (0-indexed)."""\n',
    "<|im_start|>user\nWrite a Python function that reverses the words in a sentence.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nНапиши коротко, що таке фотосинтез.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nХто такий Тарас Шевченко?<|im_end|>\n<|im_start|>assistant\n",
    "Create an SVG image: a red circle on a white background\n\n```svg\n",
]

def load_gpu(ckpt):
    """Мало оперативки на хості: модель на meta, тензори з safetensors одразу на GPU по одному."""
    import glob, os
    from safetensors import safe_open
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(ckpt)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    names = set(model.state_dict().keys())
    sd = {}
    for f in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                kk = k if k in names else next((n for n in (k.replace("model.language_model.", "model."),) if n in names), None)
                if kk is None:
                    print("пропуск", k); continue
                sd[kk] = h.get_tensor(k).to("cuda", torch.bfloat16)
    missing = names - set(sd)
    print("missing:", sorted(missing)[:10], len(missing), flush=True)
    model.load_state_dict(sd, strict=False, assign=True)
    for n, b in list(model.named_buffers()):
        if b.is_meta:
            print("meta buffer", n)
    model.cuda()
    return model.eval()


for ckpt in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda").eval()
    end = [i for i in {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} if isinstance(i, int) and i >= 0]
    print(f"\n######## {ckpt}", flush=True)
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        t = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=160, do_sample=False, repetition_penalty=1.1, eos_token_id=end)
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        print(f"\n=== {p[:70]!r}  ({out.shape[1] - ids.shape[1]} ток, {time.time() - t:.0f}с)\n{txt}", flush=True)
    del model
    torch.cuda.empty_cache()
