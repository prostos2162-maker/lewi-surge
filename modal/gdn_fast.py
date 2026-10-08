"""
Прискорений chunked Gated Delta Rule для GPU (копія maxtext.models.qwen3.jax_chunk_gated_delta_rule з двома змінами):

1. PREC — точність множень. Оригінал скрізь Precision.HIGHEST (на GPU = fp32 без tensor cores, повільно).
   "high" = bf16x3 (≈ fp32 точність на tensor cores), "default" = TF32/bf16.
2. INV="neumann" — (I+S)^-1 без solve_triangular: S строго нижньотрикутна → нільпотентна (S^C = 0), тому
   (I+S)^-1 = (I-S)(I+S^2)(I+S^4)...(I+S^(C/2)) — точна тотожність, log2(C) множень замість trsm.

Підключення: import gdn_fast; gdn_fast.install(prec="high", inv="neumann") — ДО побудови моделі.
"""
import jax
import jax.numpy as jnp
from jax import lax

PREC = jax.lax.Precision.HIGHEST
INV = "solve"
UNROLL = 1
_orig = None


def _inv_unit_lower(S, c):
    """(I + S)^-1 для строго нижньотрикутної S через добуток Ноймана (точно, бо S^c = 0)."""
    eye = jnp.eye(c, dtype=S.dtype)
    if INV == "solve":
        return jax.scipy.linalg.solve_triangular(eye + S, jnp.broadcast_to(eye, S.shape), lower=True,
                                                 unit_diagonal=True)
    A = eye - S
    P = S
    k = 2
    while k < c:
        P = jnp.matmul(P, P, precision=PREC)  # S^k
        A = jnp.matmul(A, eye + P, precision=PREC)
        k *= 2
    return A


def fast_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                                use_qk_norm_in_gdn=False, cp_axis=None, compute_dtype=jnp.bfloat16):
    if cp_axis is not None:
        return _orig(query, key, value, g, beta, chunk_size=chunk_size, initial_state=initial_state,
                     use_qk_norm_in_gdn=use_qk_norm_in_gdn, cp_axis=cp_axis, compute_dtype=compute_dtype)
    initial_dtype = query.dtype
    if use_qk_norm_in_gdn:
        from maxtext.layers.normalizations import l2norm
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    g = g.astype(jnp.float32)
    query = query.astype(compute_dtype)
    key = key.astype(compute_dtype)
    value = value.astype(compute_dtype)
    beta = beta.astype(compute_dtype)
    scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(compute_dtype)
    query = query * scale

    B, seq_len, H, K_dim = key.shape
    V_dim = value.shape[-1]
    pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
    if pad_len > 0:
        def pad_fn(x):
            return jnp.pad(x, ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2))
        query, key, value, g, beta = map(pad_fn, (query, key, value, g, beta))
    N = query.shape[1] // chunk_size

    def to_chunk(x):
        return x.reshape(B, N, chunk_size, H, -1).transpose(0, 1, 3, 2, 4)

    def to_chunk_scalar(x):
        return x.reshape(B, N, chunk_size, H).transpose(0, 1, 3, 2)

    q_c, k_c, v_c = to_chunk(query), to_chunk(key), to_chunk(value)
    g_c, beta_c = to_chunk_scalar(g), to_chunk_scalar(beta)

    g_cumsum = jnp.cumsum(g_c, axis=-1)
    k_beta = k_c * beta_c[..., None]
    S = jnp.matmul(k_beta, k_c.swapaxes(-1, -2), precision=PREC).astype(jnp.float32)
    g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
    g_diff = jnp.where(mask, g_diff, -1e30)
    S = jnp.where(mask, S * jnp.exp(g_diff), 0.0)

    A = _inv_unit_lower(S, chunk_size)

    v_beta = v_c * beta_c[..., None]
    u_chunks = jnp.matmul(A, v_beta.astype(jnp.float32), precision=PREC).astype(compute_dtype)
    k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]
    w_chunks = jnp.matmul(A, k_beta_g, precision=PREC).astype(compute_dtype)

    pv, ps = (1, 0, 2, 3, 4), (1, 0, 2, 3)
    xs = (w_chunks.transpose(pv), u_chunks.transpose(pv), q_c.transpose(pv), k_c.transpose(pv),
          g_cumsum.transpose(ps))
    h_init = (jnp.zeros((B, H, K_dim, V_dim), jnp.float32) if initial_state is None
              else initial_state.astype(jnp.float32))
    mask_intra = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))

    def scan_body(h, args):
        w, u, q, k, gg = args
        q_g = q.astype(jnp.float32) * jnp.exp(gg)[..., None]
        attn_inter = jnp.matmul(q_g, h, precision=PREC)
        v_new = u.astype(jnp.float32) - jnp.matmul(w.astype(jnp.float32), h, precision=PREC)
        attn = jnp.matmul(q, k.swapaxes(-1, -2), precision=PREC).astype(jnp.float32)
        gd = jnp.where(mask_intra, gg[..., :, None] - gg[..., None, :], -1e30)
        attn_i = jnp.where(mask_intra, attn * jnp.exp(gd), 0.0)
        o_c = attn_inter + jnp.matmul(attn_i, v_new, precision=PREC)
        h_new = h * jnp.exp(gg[..., -1, None, None])
        k_i = k.astype(jnp.float32) * jnp.exp(gg[..., -1, None] - gg)[..., None]
        h_new = h_new + jnp.matmul(k_i.swapaxes(-1, -2), v_new, precision=PREC)
        return h_new, o_c

    final_h, o_chunks = lax.scan(scan_body, h_init, xs, unroll=UNROLL)
    o = o_chunks.transpose(1, 0, 3, 2, 4).reshape(B, -1, H, V_dim)
    if pad_len > 0:
        o = o[:, :seq_len]
    return o.astype(initial_dtype), (final_h.astype(compute_dtype) if initial_state is not None else None)


def install(prec="high", inv="neumann", unroll=1):
    global PREC, INV, UNROLL, _orig
    UNROLL = int(unroll)
    PREC = {"highest": jax.lax.Precision.HIGHEST, "high": jax.lax.Precision.HIGH,
            "default": jax.lax.Precision.DEFAULT}[prec]
    INV = inv
    import maxtext.models.qwen3 as q3
    _orig = q3.jax_chunk_gated_delta_rule
    q3.jax_chunk_gated_delta_rule = fast_chunk_gated_delta_rule
    print(f"[gdn_fast] chunked gated delta rule: precision={prec}, inverse={inv}, unroll={UNROLL}", flush=True)
