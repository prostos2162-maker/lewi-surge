"""MoE token permute / combine on TPU with Pallas gather kernels.

Why: in MaxText 0.2.5 the dropless MoE path (moe_use_direct_token_gather=True, ragged_dot) spends ~27% of a
Lewi step in `sort_activations` / XLA gather: argsort over N*k slots, an XLA row gather for dispatch, another
gather + einsum for combine, and the same again (plus argsort(argsort)) in the backward pass. XLA's TPU row
gather runs ~20x below HBM bandwidth. These kernels do the row movement with per-row DMAs from HBM and sum the
k expert outputs in VMEM, in fp32, for any k (the upstream SparseCore kernel needs k == 8).

Three kernels (each tile of rows: issue the DMAs, wait, compute, store):
  gather_rows(x, idx[, scale])   out[i] = x[idx[i]] * scale[i]            dispatch fwd, combine bwd (dy)
  gather_reduce(y, pos[, w])     out[t] = sum_j w[t,j] * y[pos[t,j]]      combine fwd, dispatch bwd (dx)
  gather_dot(y, pos, g)          out[t,j] = <g[t], y[pos[t,j]]>           combine bwd (dw)

16-bit inputs are read as packed uint32 rows (two bf16 rows per u32 row, like the upstream SparseCore kernel) and
unpacked with integer ops, because a single bf16 row is half a sublane and cannot be DMA'd alone. Which of the
two rows sits in the low half is hardware-defined: probe_packing() measures it on the device (install() does).

install(mode) patches maxtext.layers.moe (0.2.5):
  mode="gather": _route_activations (dispatch) and _sort_activations (unsort before the weight einsum).
  mode="full":   also RoutedMoE.unpermute -> fused gather_reduce with the routing weights (no unsort + einsum).
Shapes the kernels cannot tile (rows not divisible by 8, D % 128 != 0) fall back to the XLA reference, so
correctness never depends on the kernel being used. CPU: kernels run in Pallas interpret mode (tests).
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CFG = {
    "enabled": True,      # False -> pure XLA reference everywhere (A/B switch)
    "tile": 128,          # rows per grid step (also DMAs in flight per step: tile * k for reduce kernels)
    "even_low": True,     # packed u32 row j = (row 2j, row 2j+1); True -> row 2j in the low 16 bits
    "prepack": None,      # None: ref.bitcast inside the kernel on TPU, jnp packing elsewhere; True: always jnp
    "sort": "argsort",    # "argsort" (XLA sort) or "count" (one-hot counting sort, no sort network)
    "num_experts": None,  # needed for sort="count"
    "vmem_mb": None,      # vmem_limit_bytes override (MB)
}

# ----------------------------------------------------------------------------------------------- helpers


def _on_tpu():
  return jax.devices()[0].platform == "tpu"


def _interpret():
  return None if _on_tpu() else pltpu.InterpretParams()


def _packing(dtype):
  return 4 // jnp.dtype(dtype).itemsize  # 1 for 32-bit, 2 for bf16/f16


def _pick_tile(*rows):
  for t in (CFG["tile"], 256, 128, 64, 32, 16, 8):
    if t and all(r % t == 0 for r in rows):
      return t
  return None


def _can_kernel(x, *rows):
  return CFG["enabled"] and x.ndim == 2 and x.shape[1] % 128 == 0 and _packing(x.dtype) in (1, 2) and _pick_tile(*rows) is not None


def _pack_rows_jnp(x):
  """CPU emulation of ref.bitcast(uint32) on a 16-bit (N, D) array: rows (2j, 2j+1) -> u32 row j."""
  n, d = x.shape
  u = lax.bitcast_convert_type(x, jnp.uint16).astype(jnp.uint32).reshape(n // 2, 2, d)
  lo, hi = (u[:, 0], u[:, 1]) if CFG["even_low"] else (u[:, 1], u[:, 0])
  return lo | (hi << 16)


def _unpack(u, parity, dtype):
  """u: (R, D) uint32 packed rows, parity: (R, 1) int32 (idx % 2). Returns the selected rows as f32."""
  lo = lax.bitcast_convert_type(u << 16, jnp.float32)
  hi = lax.bitcast_convert_type(u & jnp.uint32(0xFFFF0000), jnp.float32)
  if jnp.dtype(dtype) == jnp.float16:  # f16 bits are not f32 bits: go through the real dtype
    lo = lax.bitcast_convert_type((u & jnp.uint32(0xFFFF)).astype(jnp.uint16), jnp.float16).astype(jnp.float32)
    hi = lax.bitcast_convert_type((u >> 16).astype(jnp.uint16), jnp.float16).astype(jnp.float32)
  even, odd = (lo, hi) if CFG["even_low"] else (hi, lo)
  return jnp.where(parity == 0, even, odd)


def _source(x):
  """Returns (array passed to the kernel, packing, ref_bitcast flag)."""
  packing = _packing(x.dtype)
  if packing == 1:
    return x, 1, False
  prepack = CFG["prepack"] if CFG["prepack"] is not None else not _on_tpu()
  if prepack:
    return _pack_rows_jnp(x), packing, False
  return x, packing, True


def _cparams(tile, d, k=1):
  mb = CFG["vmem_mb"] or min(96, 8 + (tile * d * 4 * (k + 3)) // (1 << 20))
  return pltpu.CompilerParams(vmem_limit_bytes=mb << 20)


def _dma_all(idx_of, src_ref, dst_of, sem, n, packing, unroll=8):
  """Issue n row DMAs src_ref[idx_of(r)] -> dst_of(r), then wait for all of them (one semaphore)."""

  def copy(r):
    return pltpu.make_async_copy(src_ref.at[pl.ds(idx_of(r) // packing, 1)], dst_of(r), sem.at[0])

  lax.fori_loop(0, n, lambda r, c: (copy(r).start(), c)[1], 0, unroll=unroll)
  lax.fori_loop(0, n, lambda r, c: (copy(r).wait(), c)[1], 0, unroll=unroll)


# ----------------------------------------------------------------------------------------------- kernels


def gather_rows(x, idx, scale=None):
  """out[i] = x[idx[i]] (* scale[i]).  x: (N, D), idx: (M,) int32, scale: (M,) f32 or None -> (M, D) x.dtype."""
  m = idx.shape[0]
  if not _can_kernel(x, m):
    out = jnp.take(x, idx, axis=0)
    return out if scale is None else (out.astype(jnp.float32) * scale[:, None]).astype(x.dtype)
  n, d = x.shape
  tile = _pick_tile(m)
  src, packing, ref_bitcast = _source(x)
  idx = idx.astype(jnp.int32)
  par = (idx % packing)[:, None] if packing > 1 else None
  ins, in_specs = [src], [pl.BlockSpec(memory_space=pl.ANY)]
  if par is not None:
    ins.append(par)
    in_specs.append(pl.BlockSpec((tile, 1), lambda i, _: (i, 0)))
  if scale is not None:
    ins.append(scale.astype(jnp.float32)[:, None])
    in_specs.append(pl.BlockSpec((tile, 1), lambda i, _: (i, 0)))
  buf_dtype = jnp.uint32 if packing > 1 else x.dtype

  def kernel(idx_ref, x_hbm, *rest):
    rest = list(rest)
    par_ref = rest.pop(0) if par is not None else None
    s_ref = rest.pop(0) if scale is not None else None
    o_ref, buf, sem = rest
    xr = x_hbm.bitcast(jnp.uint32) if ref_bitcast else x_hbm
    base = pl.program_id(0) * tile
    _dma_all(lambda r: idx_ref[base + r], xr, lambda r: buf.at[pl.ds(r, 1)], sem, tile, packing)
    v = buf[...]
    v = _unpack(v, par_ref[...], x.dtype) if packing > 1 else v.astype(jnp.float32)
    if s_ref is not None:
      v = v * s_ref[...]
    o_ref[...] = v.astype(x.dtype)

  return pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1, grid=(m // tile,), in_specs=in_specs,
          out_specs=pl.BlockSpec((tile, d), lambda i, _: (i, 0)),
          scratch_shapes=[pltpu.VMEM((tile, d), buf_dtype), pltpu.SemaphoreType.DMA((1,))]),
      out_shape=jax.ShapeDtypeStruct((m, d), x.dtype),
      compiler_params=_cparams(tile, d), interpret=_interpret(),
  )(idx, *ins)


def _reduce_like(y, pos, w, g, out_dtype):
  """Shared body of gather_reduce (g is None) and gather_dot (g given)."""
  n, k = pos.shape
  _, d = y.shape
  tile = _pick_tile(n)
  src, packing, ref_bitcast = _source(y)
  pos_flat = pos.astype(jnp.int32).reshape(-1)
  par = (pos.astype(jnp.int32) % packing) if packing > 1 else None
  ins, in_specs = [src], [pl.BlockSpec(memory_space=pl.ANY)]
  for extra in (par, w, g):
    if extra is not None:
      ins.append(extra)
      in_specs.append(pl.BlockSpec((tile, extra.shape[1]), lambda i, _: (i, 0)))
  buf_dtype = jnp.uint32 if packing > 1 else y.dtype
  dot = g is not None

  def kernel(pos_ref, y_hbm, *rest):
    rest = list(rest)
    par_ref = rest.pop(0) if par is not None else None
    w_ref = rest.pop(0) if w is not None else None
    g_ref = rest.pop(0) if dot else None
    o_ref, buf, sem = rest
    yr = y_hbm.bitcast(jnp.uint32) if ref_bitcast else y_hbm
    base = pl.program_id(0) * tile
    # flat DMA index r in [0, tile*k): slot j = r // tile, token row = r % tile
    _dma_all(lambda r: pos_ref[(base + r % tile) * k + r // tile],
             yr, lambda r: buf.at[r // tile, pl.ds(r % tile, 1)], sem, tile * k, packing)
    gv = g_ref[...].astype(jnp.float32) if dot else None
    acc = None
    for j in range(k):
      v = buf[j]
      v = _unpack(v, par_ref[:, j:j + 1], y.dtype) if packing > 1 else v.astype(jnp.float32)
      if dot:
        o_ref[:, j:j + 1] = jnp.sum(v * gv, axis=-1, keepdims=True)
        continue
      if w_ref is not None:
        v = v * w_ref[:, j:j + 1]
      acc = v if acc is None else acc + v
    if not dot:
      o_ref[...] = acc.astype(out_dtype)

  out_cols = k if dot else d
  return pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1, grid=(n // tile,), in_specs=in_specs,
          out_specs=pl.BlockSpec((tile, out_cols), lambda i, _: (i, 0)),
          scratch_shapes=[pltpu.VMEM((k, tile, d), buf_dtype), pltpu.SemaphoreType.DMA((1,))]),
      out_shape=jax.ShapeDtypeStruct((n, out_cols), out_dtype),
      compiler_params=_cparams(tile, d, k), interpret=_interpret(),
  )(pos_flat, *ins)


def gather_reduce(y, pos, w=None, out_dtype=None):
  """out[t] = sum_j w[t,j] * y[pos[t,j]] in f32.  y: (M, D), pos: (N, k) int32, w: (N, k) or None -> (N, D)."""
  out_dtype = out_dtype or y.dtype
  n, k = pos.shape
  if w is not None:
    w = w.astype(jnp.float32)
  if not _can_kernel(y, n):
    g = jnp.take(y, pos.reshape(-1), axis=0).astype(jnp.float32).reshape(n, k, -1)
    if w is not None:
      g = g * w[:, :, None]
    return g.sum(axis=1).astype(out_dtype)
  return _reduce_like(y, pos, w, None, out_dtype)


def gather_dot(y, pos, g):
  """out[t, j] = <g[t], y[pos[t,j]]> in f32.  y: (M, D), pos: (N, k), g: (N, D) -> (N, k) f32."""
  n, k = pos.shape
  if not _can_kernel(y, n):
    rows = jnp.take(y, pos.reshape(-1), axis=0).astype(jnp.float32).reshape(n, k, -1)
    return jnp.einsum("nkd,nd->nk", rows, g.astype(jnp.float32))
  return _reduce_like(y, pos, None, g, jnp.float32)


# ----------------------------------------------------------------------------------------------- routing indices


def sort_indices(flat_experts, num_experts=None, method=None):
  """Stable sort of the N*k expert slots by expert id.

  Returns (order, pos): order[p] = slot at sorted position p (what argsort gives), pos[i] = sorted position of
  slot i (its inverse). method="count" builds pos with a one-hot counting sort instead of a sort network.
  """
  method = method or CFG["sort"]
  flat_experts = flat_experts.astype(jnp.int32)
  m = flat_experts.shape[0]
  if method == "argsort" or num_experts is None:
    order = jnp.argsort(flat_experts)
    return order, jnp.argsort(order)
  e = num_experts
  chunk = 2048 if m % 2048 == 0 else m
  onehot = jax.nn.one_hot(flat_experts, e, dtype=jnp.int32).reshape(m // chunk, chunk, e)
  local = jnp.cumsum(onehot, axis=1)                      # rank inside the chunk, inclusive
  totals = local[:, -1, :]                                # (chunks, E)
  carry = jnp.cumsum(totals, axis=0) - totals             # slots of this expert in earlier chunks
  counts = totals.sum(axis=0)
  offsets = jnp.cumsum(counts) - counts                   # first sorted position of each expert
  rank = (local + carry[:, None, :]).reshape(m, e) - 1
  rank = jnp.take_along_axis(rank, flat_experts[:, None], axis=1)[:, 0]
  pos = offsets[flat_experts] + rank
  order = jnp.zeros((m,), jnp.int32).at[pos].set(jnp.arange(m, dtype=jnp.int32))
  return order, pos


# ----------------------------------------------------------------------------------------------- differentiable ops


@jax.custom_vjp
def dispatch(x, order, pos):
  """x: (N, D) tokens -> (N*k, D) rows grouped by expert (row p = x[order[p] // k])."""
  k = order.shape[0] // x.shape[0]
  return gather_rows(x, order // k)


def _dispatch_fwd(x, order, pos):
  # zero-size carrier keeps N and the dtype as JAX values without holding x in memory
  return dispatch(x, order, pos), (order, pos, jnp.empty((x.shape[0], 0), x.dtype))


def _dispatch_bwd(res, g):
  order, pos, carrier = res
  n = carrier.shape[0]
  k = order.shape[0] // n
  return gather_reduce(g, pos.reshape(n, k), None, carrier.dtype), None, None


dispatch.defvjp(_dispatch_fwd, _dispatch_bwd)


@jax.custom_vjp
def combine(y, order, pos, w):
  """y: (N*k, D) expert outputs in sorted order, w: (N, k) -> (N, D): sum_j w[t,j] * y[pos[t,j]]."""
  n, k = w.shape
  return gather_reduce(y, pos.reshape(n, k), w, jnp.float32)


def _combine_fwd(y, order, pos, w):
  return combine(y, order, pos, w), (y, order, pos, w)


def _combine_bwd(res, g):
  y, order, pos, w = res
  n, k = w.shape
  w_sorted = jnp.take(w.reshape(-1).astype(jnp.float32), order)
  dy = gather_rows(g.astype(y.dtype), order // k, w_sorted)
  dw = gather_dot(y, pos.reshape(n, k), g).astype(w.dtype)
  return dy, None, None, dw


combine.defvjp(_combine_fwd, _combine_bwd)


@jax.custom_vjp
def permute_rows(x, idx):
  """x[idx] for a permutation idx; backward un-permutes (drop-in for MaxText _sort_activations)."""
  return gather_rows(x, idx)


def _permute_rows_fwd(x, idx):
  return permute_rows(x, idx), idx


def _permute_rows_bwd(idx, g):
  return gather_rows(g, jnp.argsort(idx)), None


permute_rows.defvjp(_permute_rows_fwd, _permute_rows_bwd)


# ----------------------------------------------------------------------------------------------- reference (pure XLA)


def ref_dispatch(x, flat_experts):
  k = flat_experts.shape[0] // x.shape[0]
  order = jnp.argsort(flat_experts)
  return jnp.take(x, order // k, axis=0)


def ref_combine(y, flat_experts, w):
  n, k = w.shape
  pos = jnp.argsort(jnp.argsort(flat_experts))
  rows = jnp.take(y, pos, axis=0).astype(jnp.float32).reshape(n, k, -1)
  return jnp.einsum("nkd,nk->nd", rows, w.astype(jnp.float32))


def moe_block(x, flat_experts, w, expert_fn, use_kernels=True):
  """dispatch -> expert_fn(rows) -> combine, with kernels or the XLA reference. For tests/benchmarks."""
  if use_kernels:
    order, pos = sort_indices(flat_experts, CFG["num_experts"])
    return combine(expert_fn(dispatch(x, order, pos)), order, pos, w)
  return ref_combine(expert_fn(ref_dispatch(x, flat_experts)), flat_experts, w)


# ----------------------------------------------------------------------------------------------- device probe


def probe_packing(d=256, n=64, verbose=True):
  """On the device: find the u32 packing convention and whether ref.bitcast compiles. Sets CFG."""
  if not _on_tpu():
    return CFG["even_low"]
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(key, (n, d), jnp.float32).astype(jnp.bfloat16)
  idx = jax.random.permutation(key, n).astype(jnp.int32)
  want = jnp.take(x, idx, axis=0)
  for prepack in (False, True):
    CFG["prepack"] = prepack
    for even_low in (True, False):
      CFG["even_low"] = even_low
      try:
        got = jax.jit(gather_rows)(x, idx)
        if bool(jnp.array_equal(got, want)):
          if verbose:
            print(f"[moe_permute] packing: even_low={even_low} prepack={prepack}")
          return even_low
      except Exception as e:  # noqa: BLE001  ref.bitcast or the unpack did not compile on this JAX
        if verbose:
          print(f"[moe_permute] probe prepack={prepack} even_low={even_low} failed: {repr(e)[:200]}")
  raise RuntimeError("moe_permute: no packing convention reproduces x[idx] on this device")


# ----------------------------------------------------------------------------------------------- MaxText patch


def install(mode="gather", tile=None, sort=None, num_experts=None, probe=True):
  """Patch maxtext.layers.moe (0.2.5 dropless path, moe_use_direct_token_gather=True)."""
  import maxtext.layers.moe as moe  # pylint: disable=import-outside-toplevel

  if tile:
    CFG["tile"] = tile
  if sort:
    CFG["sort"] = sort
  if num_experts:
    CFG["num_experts"] = num_experts
  if probe:
    probe_packing()

  def _route_activations(inputs, selected_experts):
    n = inputs.shape[0]
    flat = selected_experts.reshape(-1)
    order, pos = sort_indices(flat, CFG["num_experts"])
    with jax.named_scope("moe_permute_dispatch"):
      return dispatch(inputs, order, pos)

  def _sort_activations(inputs, sort_idx, use_custom_vjp=True):
    del use_custom_vjp
    with jax.named_scope("moe_permute_sort"):
      return permute_rows(inputs, sort_idx.astype(jnp.int32))

  moe._route_activations = _route_activations  # pylint: disable=protected-access
  moe._sort_activations = _sort_activations  # pylint: disable=protected-access

  if mode == "full":
    orig_unpermute = moe.RoutedMoE.unpermute

    def unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length,
                  use_custom_sort_vjp=True, group_sizes=None, topk_argsort_indices=None, **kw):
      simple = (
          not kw.get("tc_routing")
          and not (self.config.use_ragged_sort and self.config.use_ring_of_experts)
          and self.config.decoder_block != moe.ctypes.DecoderBlockType.LLAMA4
          and not isinstance(intermediate, moe.qpl.QArray)
      )
      if not simple:
        return orig_unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length,
                              use_custom_sort_vjp, group_sizes, topk_argsort_indices, **kw)
      order = sorted_selected_experts.astype(jnp.int32)
      pos = jnp.argsort(order)
      w = jnp.reshape(weights, (-1, self.num_experts_per_tok)).astype(jnp.float32)
      with jax.named_scope("moe_permute_combine"):
        out = combine(intermediate, order, pos, w)
      return out.reshape(batch_size, sequence_length, -1).astype(self.dtype)

    moe.RoutedMoE.unpermute = unpermute
  print(f"[moe_permute] installed mode={mode} tile={CFG['tile']} sort={CFG['sort']} even_low={CFG['even_low']}")
