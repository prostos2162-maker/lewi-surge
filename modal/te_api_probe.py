import modal
app = modal.App("moe-te-api", image=modal.Image.from_registry("ghcr.io/nvidia/jax:maxtext-2026-09-10"))

@app.function(gpu="H100", timeout=900)
def probe():
    import subprocess
    print(subprocess.run("grep -rn -E '^def (grouped_gemm|grouped_dense|ragged_dot)' /usr/local/lib/python3*/dist-packages/transformer_engine/jax /opt/transformer-engine/transformer_engine/jax 2>/dev/null | head -20", shell=True, capture_output=True, text=True).stdout)
    print(subprocess.run("grep -rn 'te_grouped\|grouped_dense\|grouped_gemm' /opt/maxtext/src/maxtext/layers/*.py | head -20", shell=True, capture_output=True, text=True).stdout)
    import inspect, transformer_engine.jax as tej
    for mod in ("transformer_engine.jax.dense", "transformer_engine.jax.cpp_extensions.gemm"):
        try:
            m = __import__(mod, fromlist=["x"])
            for n in dir(m):
                if "group" in n.lower() or "ragged" in n.lower():
                    obj = getattr(m, n)
                    sig = str(inspect.signature(obj)) if callable(obj) else ""
                    print(mod, n, sig[:400])
        except Exception as e:
            print(mod, "ERR", e)

@app.local_entrypoint()
def main():
    probe.remote()
