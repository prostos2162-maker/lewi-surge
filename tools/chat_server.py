"""
Локальний чат з Lewi Surge у браузері.

  python tools/chat_server.py --ckpt checkpoints/lewi_surge_beta_hf
  → відкрити http://127.0.0.1:7860

Модель вантажиться один раз (GPU, якщо є CUDA, інакше CPU); відповідь стрімиться по токену.
Залежності: torch, transformers, safetensors (уже стоять для tools/generate.py).
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = ("You are Lewi Surge, an open mixture-of-experts language model created by Polite Chainsaw. "
          "Answer helpfully, honestly and concisely, in the user's language.")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/lewi_surge_beta_hf")
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--cpu", action="store_true", help="примусово CPU")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
print(f"Завантажую {args.ckpt} на {dev}…", flush=True)
tok = AutoTokenizer.from_pretrained(args.ckpt)
model = AutoModelForCausalLM.from_pretrained(
    args.ckpt, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
END_IDS = [i for i in {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} if isinstance(i, int) and i >= 0]
print(f"Готово: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B параметрів. "
      f"Відкрий http://127.0.0.1:{args.port}", flush=True)
lock = threading.Lock()  # одна генерація за раз


def build_prompt(messages):
    msgs = [{"role": "system", "content": SYSTEM}] + messages
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = open(os.path.join(HERE, "chat_ui.html"), "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = build_prompt(req["messages"])
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(input_ids=ids, max_new_tokens=int(req.get("max_new", 512)), do_sample=True,
                  temperature=float(req.get("temperature", 0.7)), top_p=0.9, repetition_penalty=1.1,
                  eos_token_id=END_IDS, pad_token_id=END_IDS[0], streamer=streamer)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")  # якщо сторінку відкрили як файл (file://)
        self.end_headers()
        with lock:
            t = threading.Thread(target=lambda: model.generate(**kw), daemon=True)
            t.start()
            try:
                for piece in streamer:
                    self.wfile.write(piece.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # користувач натиснув «Стоп» — генерація дійде сама й завершиться
            t.join()


ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
