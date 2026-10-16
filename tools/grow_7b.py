"""Ріст Lewi (HF-формат, 16 шарів, 128 експертів × 512) → 7B-A1B (32 шари, 64 експерти × 1024), без тренування.

    python tools/grow_7b.py SRC_DIR DST_DIR [--no-merge] [--dup]

Два кроки, кожен можна вимкнути для перевірки:
  --dup   дублювати кожен цикл із 4 шарів (3 GDN + 1 увага): новий цикл іде одразу за оригіналом, його вихідні
          проєкції (GDN out_proj, уваги o_proj, down_proj усіх експертів і спільного експерта) = 0 → блок додає 0
          до залишкового потоку, модель на старті тотожна. (увімкнено за замовчуванням)
  merge   склеїти пари експертів (2j, 2j+1) в один на 1024: gate/up конкатенуються по прихованому виміру,
          down_proj — по вхідному, і ×0.5, щоб сумарна маса виходу не подвоювалась; логіт роутера нової пари =
          середнє логітів пари. top-k лишається 6. Це НЕ тотожне перетворення (роутер вибирає пари), лос злегка росте;
          перевіряється tools/grow_check.py. (--no-merge вимикає)
Результат: safetensors bf16 у DST_DIR, config.json з новими розмірами, lm_head.safetensors і токенайзер скопійовані.
"""
import json
import os
import re
import shutil
import sys

import numpy as np
import ml_dtypes

BF16 = ml_dtypes.bfloat16
DT = {"BF16": BF16, "F32": np.float32, "F16": np.float16}
DT_NAME = {np.dtype(BF16): "BF16", np.dtype(np.float32): "F32", np.dtype(np.float16): "F16"}


def st_header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)), 8 + n


def st_read(path, key):
    """Читання одного тензора з safetensors без torch (на Windows safe_open+torch падає з segfault)."""
    h, base = st_header(path)
    m = h[key]
    a, b = m["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        buf = f.read(b - a)
    return np.frombuffer(buf, dtype=DT[m["dtype"]]).reshape(m["shape"]).copy()


def save_file(tensors, path, metadata=None):
    """Запис safetensors (numpy, bf16 через ml_dtypes)."""
    header, off = {}, 0
    for k, a in tensors.items():
        n = a.nbytes
        header[k] = {"dtype": DT_NAME[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + n]}
        off += n
    if metadata:
        header["__metadata__"] = metadata
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        for a in tensors.values():
            f.write(np.ascontiguousarray(a).tobytes())

src, dst = sys.argv[1], sys.argv[2]
MERGE = "--no-merge" not in sys.argv
DUP = True
CYCLE = 4
os.makedirs(dst, exist_ok=True)

cfg = json.load(open(f"{src}/config.json"))
tc = cfg.get("text_config", cfg)
L, E, I = tc["num_hidden_layers"], tc["num_experts"], tc["moe_intermediate_size"]
H = tc["hidden_size"]
assert L % CYCLE == 0 and E % 2 == 0
idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]


def load(k):
    return st_read(f"{src}/{idx[k]}", k)


ZERO_OUT = ("linear_attn.out_proj.weight", "self_attn.o_proj.weight", "mlp.experts.down_proj", "mlp.shared_expert.down_proj.weight")
P = "model.language_model."


def merge_experts(name, a):
    """a: тензор одного шару; повертає тензор для E/2 експертів по 2I."""
    if name.endswith("mlp.experts.gate_up_proj"):          # [E, 2I, H]: рядки 0..I-1 = gate, I..2I-1 = up
        g, u = a[:, :I, :], a[:, I:, :]
        g2 = g.reshape(E // 2, 2 * I, H)                    # пара (2j, 2j+1) → gate_a; gate_b по рядках
        u2 = u.reshape(E // 2, 2 * I, H)
        return np.concatenate([g2, u2], axis=1)             # [E/2, 4I, H] = [gate_a gate_b | up_a up_b]
    if name.endswith("mlp.experts.down_proj"):              # [E, H, I]
        d = a.reshape(E // 2, 2, H, I).transpose(0, 2, 1, 3).reshape(E // 2, H, 2 * I)
        return (d.astype(np.float32) * 0.5).astype(a.dtype)
    if name.endswith("mlp.gate.weight"):                    # [E, H] → середнє пари
        return a.reshape(E // 2, 2, H).astype(np.float32).mean(axis=1).astype(a.dtype)
    return a


# --- збираємо нові тензори шар за шаром (пам'ять: один шар за раз)
layer_keys = sorted({k for k in idx if k.startswith(P + "layers.")})
by_layer = {}
for k in layer_keys:
    li = int(k.split(".")[3])
    by_layer.setdefault(li, []).append(k)

shard, shard_bytes, shard_id, weight_map, n_params, total_bytes = {}, 0, 0, {}, 0, 0
SHARD_MAX = 3 * 1024**3


def flush():
    global shard, shard_bytes, shard_id
    if not shard:
        return
    fn = f"model-{shard_id:05d}.safetensors"
    save_file(shard, f"{dst}/{fn}", metadata={"format": "pt"})
    for k in shard:
        weight_map[k] = fn
    print(f"  записано {fn}: {len(shard)} тензорів, {shard_bytes / 1e9:.2f} GB", flush=True)
    shard, shard_bytes, shard_id = {}, 0, shard_id + 1


def put(k, a):
    global shard_bytes, n_params, total_bytes
    shard[k] = np.ascontiguousarray(a)
    shard_bytes += a.nbytes
    total_bytes += a.nbytes
    n_params += a.size
    if shard_bytes >= SHARD_MAX:
        flush()


new_li = 0
new_layer_types = []
for c in range(L // CYCLE):
    for copy_ in (0, 1) if DUP else (0,):
        for j in range(CYCLE):
            old = c * CYCLE + j
            new_layer_types.append(tc["layer_types"][old])
            for k in by_layer[old]:
                name = k.split(f"layers.{old}.", 1)[1]
                a = load(k)
                if MERGE:
                    a = merge_experts(name, a)
                if copy_ == 1 and any(name.endswith(z) for z in ZERO_OUT):
                    a = np.zeros_like(a)
                put(f"{P}layers.{new_li}.{name}", a)
            new_li += 1
    print(f"цикл {c}: шари → {new_li}", flush=True)

for k in idx:
    if not k.startswith(P + "layers."):
        put(k, load(k))
flush()
json.dump({"metadata": {"total_size": int(total_bytes)}, "weight_map": weight_map},
          open(f"{dst}/model.safetensors.index.json", "w"), indent=1)

# --- конфіг
tc["num_hidden_layers"] = new_li
tc["layer_types"] = new_layer_types
if MERGE:
    tc["num_experts"] = E // 2
    tc["moe_intermediate_size"] = 2 * I
if "text_config" in cfg:
    cfg["text_config"] = tc
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
for f in os.listdir(src):
    if f.startswith(("tokenizer", "chat_template", "generation_config")) or f == "lm_head.safetensors":
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
print(f"ГОТОВО: {new_li} шарів, {tc['num_experts']} експертів × {tc['moe_intermediate_size']}, "
      f"{n_params / 1e9:.2f}B параметрів у шардах (+ lm_head)")
