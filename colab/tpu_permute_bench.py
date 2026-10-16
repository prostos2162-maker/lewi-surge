"""Micro-benchmark of modal/moe_permute.py on a TPU (Colab v6e-1 or a GCP VM). No training is started.

    python colab/tpu_permute_bench.py              # Lewi shape (k=6) and 7B-A1B shape (k=12)
    SHAPES=lewi TILES=64,128,256 python colab/tpu_permute_bench.py

1. probe the packing convention and run the correctness test on the device (bf16);
2. time each kernel against the XLA reference at N = 8 x 4096 tokens, D = 1024, 128 experts;
3. time the full dispatch -> combine fwd+bwd (what one MoE layer pays for token movement).
Prints a table; copy it into outputs/plan_7b_a1b.md section 7.
"""
import os
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modal"))
sys.path.insert(0, "/root/src")
import moe_permute as mp  # noqa: E402

E = 128
SHAPES = {"lewi": dict(n=8 * 4096, d=1024, k=6), "7b": dict(n=8 * 4096, d=1024, k=12)}
TILES = [int(t) for t in os.environ.get("TILES", "128,256").split(",")]


def bench(f, *args, iters=10):
  out = f(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(iters):
    out = f(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) / iters * 1e3  # ms


def routing(key, n, k):
  return jnp.argsort(-jax.random.normal(key, (n, E)), axis=1)[:, :k].astype(jnp.int32)


def main():
  print("devices:", jax.devices())
  mp.probe_packing()
  import test_moe_permute as t  # noqa: E402  (same folder as moe_permute)
  assert t.run_case(jnp.bfloat16, 6), "device correctness test failed"
  assert t.run_case(jnp.bfloat16, 12), "device correctness test failed"

  rows = []
  for name in os.environ.get("SHAPES", "lewi,7b").split(","):
    s = SHAPES[name]
    n, d, k = s["n"], s["d"], s["k"]
    m = n * k
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (n, d), jnp.float32).astype(jnp.bfloat16)
    sel = routing(k2, n, k)
    flat = sel.reshape(-1)
    w = jax.nn.softmax(jax.random.normal(k3, (n, k)), axis=1)
    y = jax.random.normal(k3, (m, d), jnp.float32).astype(jnp.bfloat16)
    g = jax.random.normal(k1, (n, d), jnp.float32).astype(jnp.bfloat16)
    order = jnp.argsort(flat)
    pos = jnp.argsort(order)
    gbytes_rows = 2 * m * d * 2 / 1e9  # read + write of the gathered rows, bf16

    print(f"\n== {name}: N={n} D={d} k={k} E={E}  ({m} rows, {gbytes_rows:.2f} GB moved per gather)")
    t_argsort = bench(jax.jit(lambda f: jnp.argsort(f)), flat)
    t_count = bench(jax.jit(lambda f: mp.sort_indices(f, E, "count")[1]), flat)
    t_argsort2 = bench(jax.jit(lambda f: jnp.argsort(jnp.argsort(f))), flat)
    print(f"  argsort {t_argsort:.2f} ms | argsort(argsort) {t_argsort2:.2f} ms | counting sort (pos+order) {t_count:.2f} ms")
    rows.append((name, "argsort / count", f"{t_argsort:.2f}", f"{t_count:.2f}"))

    t_xla_g = bench(jax.jit(lambda a, i: jnp.take(a, i, axis=0)), x, order // k)
    mp.CFG["enabled"] = False
    t_xla_r = bench(jax.jit(mp.gather_reduce), y, pos.reshape(n, k), w)
    t_xla_d = bench(jax.jit(mp.gather_dot), y, pos.reshape(n, k), g)
    mp.CFG["enabled"] = True
    print(f"  XLA: gather {t_xla_g:.2f} ms ({gbytes_rows / t_xla_g:.0f} GB/s) | gather_reduce {t_xla_r:.2f} | gather_dot {t_xla_d:.2f}")
    for tile in TILES:
      mp.CFG["tile"] = tile
      try:
        t_g = bench(jax.jit(mp.gather_rows), x, order // k)
        t_r = bench(jax.jit(mp.gather_reduce), y, pos.reshape(n, k), w)
        t_d = bench(jax.jit(mp.gather_dot), y, pos.reshape(n, k), g)
        print(f"  kernel tile={tile}: gather {t_g:.2f} ms ({gbytes_rows / t_g:.0f} GB/s) | gather_reduce {t_r:.2f} | gather_dot {t_d:.2f}")
        rows.append((name, f"kernel tile={tile}", f"{t_g:.2f}/{t_r:.2f}/{t_d:.2f}", f"xla {t_xla_g:.2f}/{t_xla_r:.2f}/{t_xla_d:.2f}"))
      except Exception as e:  # noqa: BLE001
        print(f"  kernel tile={tile}: FAILED {repr(e)[:300]}")

    # full block: dispatch -> identity expert -> combine, fwd + bwd
    def block(a, ww, use):
      out = mp.moe_block(a, flat, ww, lambda r: r, use_kernels=use)
      return jnp.sum(out * g.astype(jnp.float32))

    mp.CFG["num_experts"] = E
    for sort in ("argsort", "count"):
      mp.CFG["sort"] = sort
      f_k = jax.jit(jax.grad(lambda a, b: block(a, b, True), argnums=(0, 1)))
      t_k = bench(f_k, x, w, iters=5)
      print(f"  full fwd+bwd kernels[{sort}] {t_k:.2f} ms")
      rows.append((name, f"block fwd+bwd kernels[{sort}]", f"{t_k:.2f}", ""))
    f_r = jax.jit(jax.grad(lambda a, b: block(a, b, False), argnums=(0, 1)))
    t_r = bench(f_r, x, w, iters=5)
    print(f"  full fwd+bwd XLA reference {t_r:.2f} ms")
    rows.append((name, "block fwd+bwd XLA", f"{t_r:.2f}", ""))

  print("\n| shape | what | kernel ms | reference ms |\n|---|---|---|---|")
  for r in rows:
    print("| " + " | ".join(r) + " |")
  print("BENCH_DONE")


if __name__ == "__main__":
  main()
