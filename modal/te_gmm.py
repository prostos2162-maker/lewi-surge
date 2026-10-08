"""
jax.lax.ragged_dot → NVIDIA Transformer Engine grouped GEMM (cuBLAS grouped, dropless, без padding).
Кожен експерт множиться рівно на свої токени: ні порожніх слотів (як factor 1.5), ні відкинутих токенів.
Працює на 1 GPU (на відміну від te_moe_block, який вимагає NCCL EP ≥ 2 ранки).
Підключення: import te_gmm; te_gmm.install() — ДО імпорту maxtext. Потрібен контейнер з TE.
"""
import jax

_orig_ragged_dot = jax.lax.ragged_dot


def te_ragged_dot(lhs, rhs, group_sizes, precision=None, preferred_element_type=None, **kw):
    if lhs.ndim != 2 or rhs.ndim != 3 or kw.get("group_offset") is not None:
        return _orig_ragged_dot(lhs, rhs, group_sizes, precision=precision,
                                preferred_element_type=preferred_element_type, **kw)
    from transformer_engine.jax.dense import grouped_dense
    out = grouped_dense(lhs, rhs.astype(lhs.dtype), group_sizes.astype(jax.numpy.int32),
                        contracting_dims=((1,), (1,)), preferred_element_type=preferred_element_type)
    return out


def install():
    jax.lax.ragged_dot = te_ragged_dot
    print("[te_gmm] jax.lax.ragged_dot → TransformerEngine grouped_dense (dropless)", flush=True)
