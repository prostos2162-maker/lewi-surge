"""Стеля H100 для GEMM з формами нашої моделі (bf16): скільки TFLOP/s дає саме множення."""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("jax[cuda12]")
app = modal.App("moe-gemm-bench", image=image)


@app.function(gpu="H100", timeout=900)
def bench():
    import time
    import jax
    import jax.numpy as jnp

    def t(fn, *a, flops, name, n=20):
        f = jax.jit(fn)
        f(*a).block_until_ready()
        s = time.perf_counter()
        for _ in range(n):
            out = f(*a)
        out.block_until_ready()
        dt = (time.perf_counter() - s) / n
        print(f"{name:55s} {dt*1e3:7.2f} ms  {flops/dt/1e12:6.0f} TFLOP/s", flush=True)

    k = jax.random.PRNGKey(0)
    T = 16 * 4096          # токенів на крок (батч 16 × 4096)
    E, d, f, topk = 96, 768, 256, 8
    for fac in (1.5, 1.0):
        C = int(T * topk / E * fac)
        x = jax.random.normal(k, (E, C, d), jnp.bfloat16)
        w1 = jax.random.normal(k, (E, d, f), jnp.bfloat16)
        w2 = jax.random.normal(k, (E, f, d), jnp.bfloat16)
        h = jax.random.normal(k, (E, C, f), jnp.bfloat16)
        t(lambda a, b: jnp.einsum("ecd,edf->ecf", a, b), x, w1, flops=2 * E * C * d * f, name=f"MoE up   [E={E},C={C},768]x[768,256] fac={fac}")
        t(lambda a, b: jnp.einsum("ecf,efd->ecd", a, b), h, w2, flops=2 * E * C * d * f, name=f"MoE down [E,C,256]x[256,768] fac={fac}")
        t(lambda a, g: jnp.einsum("ecd,ecf->edf", a, g), x, h, flops=2 * E * C * d * f, name=f"MoE dW   [E,768,C]x[E,C,256] fac={fac}")
    xs = jax.random.normal(k, (T, d), jnp.bfloat16)
    for n_out, nm in ((1024, "attn q (gated)"), (2048 + 1024, "GDN qkvz proj"), (131200, "LM head")):
        w = jax.random.normal(k, (d, n_out), jnp.bfloat16)
        t(lambda a, b: a @ b, xs, w, flops=2 * T * d * n_out, name=f"dense [{T},768]x[768,{n_out}] {nm}")
    big = jax.random.normal(k, (8192, 8192), jnp.bfloat16)
    t(lambda a: a @ a, big, flops=2 * 8192**3, name="reference 8192^3")


@app.local_entrypoint()
def main():
    bench.remote()
