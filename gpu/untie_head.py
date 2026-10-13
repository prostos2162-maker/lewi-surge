"""MaxText зі спільною embedding/LM-головою ділить логіти на sqrt(d_model), HF — ні: пишемо окремий lm_head.
Працює і з одним model.safetensors, і з шардованим (index.json) — як у релізах PoliteChainsaw/Lewi-Surge-*."""
import json
import math
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

dst = sys.argv[1]
idx_p = os.path.join(dst, "model.safetensors.index.json")
if os.path.exists(idx_p):
    idx = json.load(open(idx_p))
    emb = next(k for k in idx["weight_map"] if "embed_tokens" in k)
    shard = os.path.join(dst, idx["weight_map"][emb])
else:
    idx, shard = None, os.path.join(dst, "model.safetensors")
with safe_open(shard, "pt") as f:
    emb = next(k for k in f.keys() if "embed_tokens" in k)
    w = f.get_tensor(emb)
cfg = json.load(open(os.path.join(dst, "config.json")))
d = cfg.get("hidden_size") or cfg.get("text_config", {}).get("hidden_size")
save_file({"lm_head.weight": (w.float() / math.sqrt(d)).to(w.dtype).contiguous()},
          os.path.join(dst, "lm_head.safetensors"), metadata={"format": "pt"})
if idx is None:  # одиночний файл → робимо індекс, щоб HF підхопив обидва файли
    with safe_open(shard, "pt") as f:
        idx = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in f.keys()}}
idx["weight_map"]["lm_head.weight"] = "lm_head.safetensors"
json.dump(idx, open(idx_p, "w"), indent=2)
cfg["tie_word_embeddings"] = False
if "text_config" in cfg:
    cfg["text_config"]["tie_word_embeddings"] = False
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
print(f"lm_head = embed / sqrt({d})", flush=True)
