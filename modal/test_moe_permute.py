"""Correctness tests for modal/moe_permute.py. Runs on CPU (Pallas interpret mode) or on a TPU.

    python modal/test_moe_permute.py            # all cases
    python modal/test_moe_permute.py --quick    # one dtype, one k

Checks every kernel against the pure-XLA reference, forward and backward (jax.grad), for f32 and bf16 inputs,
k in (6, 12), both sort methods, and a full dispatch -> expert MLP -> combine block.
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else __file__.rsplit("\\", 1)[0])
import moe_permute as mp  # noqa: E402

QUICK = "--quick" in sys.argv
E = 16


def rand_routing(key, n, k):
  """Top-k expert ids without repeats per token, like a real router."""
  logits = jax.random.normal(key, (n, E))
  return jnp.argsort(-logits, axis=1)[:, :k].astype(jnp.int32)


def check(name, got, want, tol):
  got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
  err = np.abs(got - want).max() / (np.abs(want).max() + 1e-6)
  ok = err <= tol and got.shape == want.shape
  print(f"  {'ok ' if ok else 'BAD'} {name:38s} rel.err {err:.2e}")
  return ok


def run_case(dtype, k, n=64, d=256):
  key = jax.random.PRNGKey(k * 7 + (dtype == jnp.bfloat16))
  k1, k2, k3, k4 = jax.random.split(key, 4)
  x = jax.random.normal(k1, (n, d), jnp.float32).astype(dtype)
  sel = rand_routing(k2, n, k)
  flat = sel.reshape(-1)
  w = jax.nn.softmax(jax.random.normal(k3, (n, k)), axis=1)
  m = n * k
  y = jax.random.normal(k4, (m, d), jnp.float32).astype(dtype)
  tol = 2e-2 if dtype == jnp.bfloat16 else 1e-5
  ok = True
  print(f"case dtype={jnp.dtype(dtype).name} k={k} n={n} d={d} tile={mp._pick_tile(n, m)}")

  # 1. kernels vs references
  order = jnp.argsort(flat)
  pos = jnp.argsort(order).reshape(n, k)
  idx = jax.random.permutation(k1, m).astype(jnp.int32) % n
  scale = jax.random.normal(k2, (m,), jnp.float32)
  ok &= check("gather_rows", jax.jit(mp.gather_rows)(x, idx), jnp.take(x, idx, axis=0), 0)
  ok &= check("gather_rows+scale", jax.jit(mp.gather_rows)(x, idx, scale),
              (jnp.take(x, idx, axis=0).astype(jnp.float32) * scale[:, None]).astype(dtype), tol)
  ref_red = jnp.take(y, pos.reshape(-1), axis=0).astype(jnp.float32).reshape(n, k, d)
  ok &= check("gather_reduce", jax.jit(mp.gather_reduce, static_argnums=3)(y, pos, None, jnp.float32), ref_red.sum(1), tol)
  ok &= check("gather_reduce+w", jax.jit(mp.gather_reduce, static_argnums=3)(y, pos, w, jnp.float32), (ref_red * w[:, :, None]).sum(1), tol)
  g = jax.random.normal(k3, (n, d), jnp.float32)
  ok &= check("gather_dot", jax.jit(mp.gather_dot)(y, pos, g), jnp.einsum("nkd,nd->nk", ref_red, g), tol)

  # 2. sort_indices both methods
  for method in ("argsort", "count"):
    o2, p2 = mp.sort_indices(flat, E, method)
    ok &= check(f"sort_indices[{method}] order", o2, order, 0)
    ok &= check(f"sort_indices[{method}] pos", p2, pos.reshape(-1), 0)

  # 3. dispatch / combine with gradients vs reference
  def loss_k(x, w, use):
    out = mp.moe_block(x, flat, w, lambda r: jnp.tanh(r.astype(jnp.float32)).astype(r.dtype), use_kernels=use)
    return jnp.sum(out * g), out

  mp.CFG["num_experts"] = E
  (lk, ok_out), grads_k = jax.jit(jax.value_and_grad(lambda a, b: loss_k(a, b, True), argnums=(0, 1), has_aux=True))(x, w)
  (lr, ref_out), grads_r = jax.jit(jax.value_and_grad(lambda a, b: loss_k(a, b, False), argnums=(0, 1), has_aux=True))(x, w)
  ok &= check("moe_block forward", ok_out, ref_out, tol)
  ok &= check("moe_block loss", lk, lr, tol)
  ok &= check("moe_block grad x", grads_k[0], grads_r[0], tol)
  ok &= check("moe_block grad w", grads_k[1], grads_r[1], tol)

  # 4. permute_rows (drop-in for MaxText _sort_activations) with gradient
  perm = jax.random.permutation(k4, m).astype(jnp.int32)
  f = lambda a: jnp.sum(mp.permute_rows(a, perm).astype(jnp.float32) * jnp.take(g, idx, axis=0))
  fr = lambda a: jnp.sum(jnp.take(a, perm, axis=0).astype(jnp.float32) * jnp.take(g, idx, axis=0))
  ok &= check("permute_rows grad", jax.grad(f)(y), jax.grad(fr)(y), tol)

  # 5. fallback path (shapes the kernel cannot tile) still correct
  xs, ids = x[:60], idx[:60] % 60
  ok &= check("fallback gather_rows (60 rows)", mp.gather_rows(xs, ids), jnp.take(xs, ids, axis=0), 0)
  return ok


def main():
  t0 = time.time()
  print("devices:", jax.devices(), "| interpret:", mp._interpret() is not None)
  cases = [(jnp.bfloat16, 6)] if QUICK else [(jnp.float32, 6), (jnp.bfloat16, 6), (jnp.bfloat16, 12), (jnp.float32, 12)]
  all_ok = True
  for dtype, k in cases:
    all_ok &= run_case(dtype, k)
  print(f"{'ALL OK' if all_ok else 'FAILURES'} in {time.time() - t0:.0f}s")
  sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
  main()
