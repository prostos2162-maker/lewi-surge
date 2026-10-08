"""
Погратися з чекпоінтом стадії 1 локально (RTX 3060 / CPU) через transformers.

  pip install torch transformers safetensors
  python tools/generate.py --ckpt checkpoints/step500_hf --prompt "The history of Ukraine"
  python tools/generate.py --ckpt checkpoints/step500_hf --interactive

Це base-модель: вона ПРОДОВЖУЄ текст, а не відповідає на питання.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The history of Ukraine",
    "Photosynthesis is the process by which",
    "Київ — столиця України,",
    "La inteligencia artificial es",
    "def fibonacci(n):",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                 trust_remote_code=True).to(dev).eval()
    print(f"модель: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B параметрів на {dev}")

    def gen(text):
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new, do_sample=True, temperature=args.temperature,
                                 top_p=0.95, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0], skip_special_tokens=True)

    if args.interactive:
        while True:
            t = input("\n> ")
            if not t:
                break
            print(gen(t))
    else:
        for p in ([args.prompt] if args.prompt else PROMPTS):
            print("\n=====", p, "\n" + gen(p))


if __name__ == "__main__":
    main()
