"""Розвідка NVIDIA JAX MaxText контейнера: версії, де лежить maxtext, чи є TE, чи є qwen3_5."""
import modal

image = modal.Image.from_registry("ghcr.io/nvidia/jax:maxtext-2026-09-10")
app = modal.App("moe-te-probe", image=image)


@app.function(gpu="H100", timeout=1200)
def probe():
    import subprocess
    cmds = [
        "nvidia-smi --query-gpu=name,driver_version --format=csv",
        "which python python3; python3 --version",
        "python3 -c 'import jax, jaxlib; print(\"jax\", jax.__version__, jax.devices())'",
        "python3 -c 'import transformer_engine as te; print(\"TE\", te.__version__)'",
        "pip list 2>/dev/null | grep -i -E 'maxtext|jax|transformer|flax|optax|tokamax' ",
        "python3 -c 'import maxtext, os; print(\"maxtext pkg\", os.path.dirname(maxtext.__file__))' || python3 -c 'import MaxText, os; print(\"MaxText pkg\", os.path.dirname(MaxText.__file__))'",
        "ls /opt 2>/dev/null; ls /opt/maxtext 2>/dev/null | head",
        "find / -name 'qwen3_5.py' -path '*models*' 2>/dev/null | head -3",
        "find / -name 'base.yml' -path '*configs*' 2>/dev/null | head -3",
        "grep -n 'te_moe_block\|use_ragged_dot\|opt_type' $(find / -name base.yml -path '*maxtext*configs*' 2>/dev/null | head -1) | head",
    ]
    for c in cmds:
        r = subprocess.run(c, shell=True, capture_output=True, text=True)
        print(f"$ {c}\n{(r.stdout + r.stderr).strip()[:1500]}\n", flush=True)


@app.local_entrypoint()
def main():
    probe.remote()
