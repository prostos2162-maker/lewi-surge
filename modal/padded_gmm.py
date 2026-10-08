"""
Grouped GEMM "пачкою" замість jax.lax.ragged_dot (на GPU він розгортається в щільне множення на всі експерти).

Вхід як у ragged_dot: lhs [M,K] відсортований за експертами, rhs [E,K,N], group_sizes [E].
  1) розкладаємо рядки в буфер [E, C, K] (C = ceil(M/E · factor) слотів на експерта)
  2) один batched GEMM: einsum('eck,ekn->ecn') — великі рівні матриці, GPU забитий повністю
  3) збираємо назад у [M, N]
Рядки, що не влізли в C слотів експерта (перевантаження), дають 0 — як token dropping у capacity-dispatch.

v2: БЕЗ scatter. Відображення рядок↔слот взаємно однозначне, тому і forward, і backward — лише gather
(scatter на GPU = атомарні операції, у профілі він займав 24% кроку). Градієнт gather — зворотний gather.
Підключення: import padded_gmm; padded_gmm.install(factor) — ДО імпорту maxtext.
"""
import math
from functools import partial

import jax
import jax.numpy as jnp

_orig_ragged_dot = jax.lax.ragged_dot
FACTOR = 2.0


@partial(jax.custom_vjp, nondiff_argnums=())
def _perm_gather(x, idx, inv_idx):
    """y[i] = x_ext[idx[i]], де x_ext = x + нульовий рядок (idx == len(x) → 0)."""
    x_ext = jnp.concatenate([x, jnp.zeros((1,) + x.shape[1:], x.dtype)], axis=0)
    return jnp.take(x_ext, idx, axis=0, mode="clip")


def _perm_gather_fwd(x, idx, inv_idx):
    return _perm_gather(x, idx, inv_idx), (idx, inv_idx, x.shape[0])


def _perm_gather_bwd(res, g):
    idx, inv_idx, n = res
    # dx[j] = g[inv_idx[j]] (inv_idx[j] == len(g) → рядок не потрапив у вихід → 0)
    dx = _perm_gather(g, inv_idx, idx)
    return dx[:n] if dx.shape[0] != n else dx, None, None


_perm_gather.defvjp(_perm_gather_fwd, _perm_gather_bwd)


def padded_ragged_dot(lhs, rhs, group_sizes, precision=None, preferred_element_type=None, **kw):
    if lhs.ndim != 2 or rhs.ndim != 3 or kw.get("group_offset") is not None:
        return _orig_ragged_dot(lhs, rhs, group_sizes, precision=precision,
                                preferred_element_type=preferred_element_type, **kw)
    m, k = lhs.shape
    e, _, n = rhs.shape
    c = int(math.ceil(m / e * FACTOR))
    c = max(8, (c + 7) // 8 * 8)
    gs = group_sizes.astype(jnp.int32)
    starts = jnp.cumsum(gs) - gs
    # рядок → слот (e*c = «нікуди»: паддінг або переповнення експерта)
    rows = jnp.arange(m, dtype=jnp.int32)
    expert = jnp.minimum(jnp.searchsorted(jnp.cumsum(gs), rows, side="right"), e - 1).astype(jnp.int32)  # без scatter
    pos = rows - starts[expert]
    valid = (rows < jnp.sum(gs)) & (pos < c)
    slot = jnp.where(valid, expert * c + pos, e * c)
    # слот → рядок (m = порожній слот)
    se = jnp.arange(e * c, dtype=jnp.int32) // c
    sc = jnp.arange(e * c, dtype=jnp.int32) % c
    src = jnp.where(sc < jnp.minimum(gs[se], c), starts[se] + sc, m)

    buf = _perm_gather(lhs, src, slot)  # [E*C, K], без scatter
    out = jnp.einsum("eck,ekn->ecn", buf.reshape(e, c, k), rhs.astype(lhs.dtype),
                     precision=precision, preferred_element_type=preferred_element_type)
    return _perm_gather(out.reshape(e * c, n), slot, src)  # [M, N], зайві рядки → 0


def install(factor=2.0):
    global FACTOR
    FACTOR = float(factor)
    jax.lax.ragged_dot = padded_ragged_dot
    print(f"[padded_gmm] jax.lax.ragged_dot → padded grouped GEMM v2, gather-only (factor={FACTOR})", flush=True)
