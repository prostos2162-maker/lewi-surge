# Перевірка, що ваги справді змінились: частка біт-у-біт однакових елементів у спільних тензорах (поріг 50%). Без torch (JAX).
import sys, glob
import jax.numpy as jnp
from safetensors import safe_open
a, b = sys.argv[1], sys.argv[2]
same = tot = 0
keysB = {}
for fb in glob.glob(f"{b}/*.safetensors"):
    with safe_open(fb, "flax") as B:
        for k in B.keys(): keysB[k] = fb
for fa in sorted(glob.glob(f"{a}/*.safetensors")):
    with safe_open(fa, "flax") as A:
        for k in [k for k in A.keys() if k in keysB][:40]:
            with safe_open(keysB[k], "flax") as B:
                x = A.get_tensor(k).astype(jnp.float32); y = B.get_tensor(k).astype(jnp.float32)
            if x.shape == y.shape: same += int((x == y).sum()); tot += x.size
frac = same / max(tot, 1); print(f"WCHECK незмінних {100 * frac:.1f}% з {tot} елементів"); sys.exit(0 if tot and frac < 0.5 else 1)
