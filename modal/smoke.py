"""
Smoke-тест стадії 1 на Modal.

  modal run smoke.py                 # CPU: зменшена модель, перевірка конфігу/версій/токенайзера
  modal run smoke.py --mode gpu      # GPU (L4): швидкість і MoE-ядра на CUDA
  modal run smoke.py --mode hf       # CPU + стрім FineWeb-Edu з HF через наш токенайзер
"""
import os
import re
import subprocess
import sys

import modal

ROOT = "/root/moe"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    # [cuda12] компілює transformer-engine з вихідників і падає; голий maxtext — без залежностей.
    # [tpu] = те саме середовище, що на Kaggle TPU; на CPU JAX теж працює (JAX_PLATFORMS=cpu)
    .run_commands("uv pip install --system 'maxtext[tpu]==0.2.5' --resolution=lowest",
                  "install_tpu_pre_train_extra_deps || true")
    .env({"PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "1"})
    .add_local_dir("../out/tokenizer_pruned", f"{ROOT}/tokenizer", ignore=["old2new.json"])
    .add_local_file("../configs/stage1.yml", f"{ROOT}/stage1.yml")
)

# GPU-образ: той самий maxtext[tpu] (бо [cuda12] не збирається), але без libtpu і з jax[cuda12]
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system 'maxtext[tpu]==0.2.5' --resolution=lowest",
        "install_tpu_pre_train_extra_deps || true",
        "uv pip uninstall --system libtpu libtpu-nightly || true",
        "uv pip install --system \"jax[cuda12]==$(python -c 'import jax; print(jax.__version__)')\"",
    )
    .env({"PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "1", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"})
)


def with_local(img):
    return (img.add_local_dir("../out/tokenizer_pruned", f"{ROOT}/tokenizer", ignore=["old2new.json"])
            .add_local_file("../configs/stage1.yml", f"{ROOT}/stage1.yml")
            .add_local_file("padded_gmm.py", f"{ROOT}/padded_gmm.py")
            .add_local_file("fast_conv.py", f"{ROOT}/fast_conv.py")
            .add_local_file("gdn_fast.py", f"{ROOT}/gdn_fast.py"))


prof_image = with_local(gpu_image.pip_install("xprof"))
spy_image = with_local(gpu_image.pip_install("xprof", "py-spy"))
gpu_image = with_local(gpu_image)

app = modal.App("moe-smoke", image=image)

# зменшуємо все, що не впливає на перевірку коду
CPU_OVERRIDES = [
    "hardware=cpu", "skip_jax_distributed_system=True", "megablox=False", "sparse_matmul=False", "attention=dot_product",
    "max_target_length=256", "per_device_batch_size=1", "steps=3",
]
COMMON = [
    "base_output_directory=/tmp/out", "enable_checkpointing=False",
    f"tokenizer_path={ROOT}/tokenizer", "log_period=1", "profiler=''",
]


def patch_maxtext(pkg):
    """Латки MaxText 0.2.5 під нашу архітектуру (qwen3_5 у мініатюрі)."""
    # 1) Muon: qwen3_5 немає в білому списку, хоча GDN ідентичний qwen3_next, а розмітка параметрів загальна
    path = os.path.join(pkg, "configs", "types.py")
    src = open(path).read()
    anchor = 'if self.opt_type == "muon" and self.decoder_block not in ['
    i = src.find(anchor)
    if i >= 0 and "QWEN3_5" not in src[i:i + 600]:
        j = i + len(anchor)
        src = src[:j] + "\n        DecoderBlockType.QWEN3_5," + src[j:]
        open(path, "w").write(src)
        print("[patch] muon: додано QWEN3_5 у білий список", flush=True)


def run_train(extra):
    import os
    import shutil

    import maxtext

    cfg_dir = os.path.join(os.path.dirname(maxtext.__file__), "configs")
    patch_maxtext(os.path.dirname(maxtext.__file__))
    shutil.copy(f"{ROOT}/stage1.yml", os.path.join(cfg_dir, "stage1.yml"))
    pad = [e for e in extra if e.startswith("PADGMM=")]
    fconv = "FASTCONV=1" in extra
    gfast = [e for e in extra if e.startswith("GDNFAST=")]
    extra = [e for e in extra if not e.startswith(("PADGMM=", "FASTCONV=", "GDNFAST="))]
    args = [os.path.join(cfg_dir, "stage1.yml"), *COMMON, *extra]
    if pad or fconv or gfast:  # латки GPU-продуктивності (математика та сама)
        inst = ""
        if pad:
            inst += f"import padded_gmm; padded_gmm.install({pad[0].split('=')[1]}); "
        if fconv:
            inst += "import fast_conv; fast_conv.install(); "
        if gfast:
            parts = gfast[0].split("=")[1].split(",")
            p_, i_ = parts[0], parts[1]
            u_ = parts[2] if len(parts) > 2 else "1"
            inst += f"import gdn_fast; gdn_fast.install('{p_}', '{i_}', {u_}); "
        boot = (f"import sys,runpy; sys.path.insert(0,'{ROOT}'); {inst}sys.argv=['train']+sys.argv[1:]; "
                "runpy.run_module('maxtext.trainers.pre_train.train', run_name='__main__')")
        cmd = [sys.executable, "-c", boot, *args]
    else:
        cmd = [sys.executable, "-m", "maxtext.trainers.pre_train.train", *args]
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    print(f"[exit {rc}]", flush=True)
    return rc


@app.function(cpu=8, memory=32768, timeout=3600)
def resume_cpu():
    """Стрім HF + чекпоінт + продовження з load_full_state_path — як у kaggle/stage1/train.py."""
    import glob
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    base = CPU_OVERRIDES[:-1] + ["dataset_type=hf", "enable_checkpointing=True", "checkpoint_period=2",
                                 "max_num_checkpoints_to_keep=1", "async_checkpointing=True"]
    base = [b for b in base if not b.startswith("steps=")]
    rc1 = run_train(base + ["run_name=r1", "steps=3", "metrics_file=/tmp/m1.jsonl"])
    ck = sorted(glob.glob("/tmp/out/r1/checkpoints/*/items"))
    print("CHECKPOINTS r1:", ck, flush=True)
    if rc1 != 0 or not ck:
        return 1
    rc2 = run_train(base + ["run_name=r2", "steps=5", f"load_full_state_path={ck[-1]}",
                            "metrics_file=/tmp/m2.jsonl"])
    print("METRICS r2:", open("/tmp/m2.jsonl").read()[:800] if os.path.exists("/tmp/m2.jsonl") else None)
    return rc2


@app.function(cpu=8, memory=32768, timeout=3600)
def smoke_cpu(extra: list[str]):
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    # 8 віртуальних CPU-пристроїв → перевіряємо шардинг так само, як на TPU v5e-8
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    return run_train(CPU_OVERRIDES + extra)


@app.function(gpu=["H100", "A100-80GB"], image=gpu_image, timeout=2700, cpu=16, memory=65536)  # запасний варіант, якщо H100 немає
def smoke_gpu(runs: list[list[str]]):
    subprocess.run(["nvidia-smi"])
    subprocess.run([sys.executable, "-c", "import jax; print(jax.devices())"])
    base = ["hardware=gpu", "skip_jax_distributed_system=True", "dataset_type=synthetic",
            "megablox=False", "sparse_matmul=True", "steps=15"]
    results = []
    for extra in runs:
        print("\n==================== ", extra, flush=True)
        env = {e[4:].split("=", 1)[0]: e[4:].split("=", 1)[1] for e in extra if e.startswith("ENV:")}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        results.append(run_train(base + [e for e in extra if not e.startswith("ENV:")]))
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return results


@app.function(gpu=["H100", "A100-80GB"], image=prof_image, timeout=2700)
def profile_gpu(extra: list[str]):
    """Профіль кроку (xplane) → топ HLO-операцій за часом."""
    import glob
    base = ["hardware=gpu", "skip_jax_distributed_system=True", "dataset_type=synthetic", "megablox=False",
            "steps=9", "profiler=xplane", "skip_first_n_steps_for_profiler=5", "profiler_steps=3",
            "run_name=prof"]
    rc = run_train(base + extra)
    files = glob.glob("/tmp/out/prof/**/*.xplane.pb", recursive=True)
    print("XPLANE:", files, flush=True)
    if not files:
        return rc
    from xprof.convert import raw_to_tool_data as r
    texts = {}
    for tool in ("kernel_stats", "hlo_stats", "framework_op_stats", "overview_page"):
        try:
            data, ctype = r.xspace_to_tool_data(files, tool, {})
            texts[tool] = data.decode() if isinstance(data, (bytes, bytearray)) else str(data)
            print(f"TOOL {tool}: {len(texts[tool])} chars", flush=True)
        except Exception as e:
            print(f"TOOL {tool} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
    return texts  # повертаємо цілим — Modal ріже довгі рядки виводу


@app.function(gpu=["H100", "A100-80GB"], image=spy_image, timeout=2700, cpu=16,
              memory=65536)
def pyspy_gpu(extra: list[str]):
    """Де хост проводить ~1 с на крок: py-spy dump посеред тренування + таймінг з рідким логуванням."""
    import shutil as sh
    import time as tm
    import maxtext
    cfg_dir = os.path.join(os.path.dirname(maxtext.__file__), "configs")
    patch_maxtext(os.path.dirname(maxtext.__file__))
    sh.copy(f"{ROOT}/stage1.yml", os.path.join(cfg_dir, "stage1.yml"))
    base = [sys.executable, "-m", "maxtext.trainers.pre_train.train", os.path.join(cfg_dir, "stage1.yml"),
            "base_output_directory=/tmp/out", "enable_checkpointing=False", f"tokenizer_path={ROOT}/tokenizer",
            "hardware=gpu", "skip_jax_distributed_system=True", "dataset_type=synthetic", "megablox=False"] + extra

    # 1) таймінг: 40 кроків, лог раз на 10 → чи дорогий per-step лог
    t = tm.time()
    p = subprocess.run(base + ["run_name=t40", "steps=40", "log_period=10"], capture_output=True, text=True)
    out = p.stdout + p.stderr
    steps = re.findall(r"completed step: (\d+), seconds: ([\d.]+).*?Tokens/s/device: ([\d.]+)", out)
    print("TIMING log_period=10:", steps[-6:], f"wall {tm.time() - t:.0f}s", flush=True)

    # 2) nvidia-smi під час тренування + вбудований семплер стеку головного потоку
    wrapper = """
import sys, threading, time, collections, runpy, traceback
agg = collections.Counter()
def sampler():
    time.sleep(150)  # чекаємо кінця компіляції
    main = threading.main_thread().ident
    t_end = time.time() + 12
    n = 0
    while time.time() < t_end:
        f = sys._current_frames().get(main)
        if f is not None:
            st = traceback.extract_stack(f)[-6:]
            agg[" <- ".join(f"{fr.name}@{fr.filename.split('/')[-1]}:{fr.lineno}" for fr in reversed(st))] += 1
            n += 1
        time.sleep(0.02)
    print("===== STACK SAMPLES", n, flush=True)
    for k, v in agg.most_common(15):
        print(f"{v:4d} {k[:400]}", flush=True)
threading.Thread(target=sampler, daemon=True).start()
sys.argv = ["train"] + sys.argv[1:]
runpy.run_module("maxtext.trainers.pre_train.train", run_name="__main__")
"""
    open("/tmp/wrap.py", "w").write(wrapper)
    cmd = [sys.executable, "/tmp/wrap.py"] + base[3:] + ["run_name=spy", "steps=200", "log_period=10"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                            "--format=csv,noheader", "-l", "1"], stdout=subprocess.PIPE, text=True)
    t0 = tm.time()
    printing = False
    for line in proc.stdout:
        if "STACK SAMPLES" in line:
            printing = True
        if printing or "completed step" in line:
            print(line.rstrip()[:420], flush=True)
        if printing and tm.time() - t0 > 175:
            break
    smi.kill()
    rows = smi.stdout.read().strip().splitlines()
    print("===== NVIDIA-SMI (останні 25 с) =====")
    for r in rows[-25:]:
        print(r, flush=True)
    proc.kill()
    return 0


@app.local_entrypoint()
def main(mode: str = "cpu"):
    if mode == "cpu":
        rc = smoke_cpu.remote(["run_name=cpu", "dataset_type=synthetic"])
    elif mode == "resume":
        rc = resume_cpu.remote()
    elif mode == "hf":
        rc = smoke_cpu.remote(["run_name=hf", "dataset_type=hf"])
    elif mode == "ablate":
        # однаковий батч, міняємо по одній речі → шукаємо, що гальмує на GPU
        b = ["per_device_batch_size=4", "max_target_length=4096", "steps=8"]
        runs = [
            ["run_name=base"] + b,                                                    # як було: 4.8k ток/с
            ["run_name=no_gdn", "inhomogeneous_layer_cycle_interval=1"] + b,         # усі шари full attention
            ["run_name=remat_min", "remat_policy=minimal"] + b,                       # менше перерахунку
            ["run_name=cap", "sparse_matmul=False", "capacity_factor=1.25"] + b,      # MoE з capacity-dispatch
            ["run_name=dotattn", "attention=dot_product"] + b,                        # без flash-ядра
            ["run_name=adamw", "opt_type=adamw"] + b,                                 # без Muon (Newton-Schulz)
        ]
        rcs = smoke_gpu.remote(runs)
        print("RESULTS", rcs)
        rc = 0
    elif mode == "shape":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=12", "PADGMM=1.5", "FASTCONV=1",
             "GDNFAST=default,solve,64", "per_device_batch_size=16"]
        wide = ["base_emb_dim=1024", "base_num_decoder_layers=12"]
        runs = [
            ["run_name=sh_64x512_top6"] + wide + ["num_experts=64", "base_moe_mlp_dim=512", "base_mlp_dim=512",
                                                "num_experts_per_tok=6"] + b,
            ["run_name=sh_32x1024_top3"] + wide + ["num_experts=32", "base_moe_mlp_dim=1024", "base_mlp_dim=1024",
                                                 "num_experts_per_tok=3"] + b,
        ]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "v2abl":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=12", "PADGMM=1.5", "FASTCONV=1",
             "GDNFAST=default,solve,64"]
        runs = [
            ["run_name=v2_nogdn16", "inhomogeneous_layer_cycle_interval=1", "per_device_batch_size=16"] + b,
            ["run_name=v2_b24", "per_device_batch_size=24"] + b,
            ["run_name=v2_b32", "per_device_batch_size=32"] + b,
        ]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "remat":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=12", "PADGMM=1.5", "FASTCONV=1",
             "GDNFAST=default,solve,64"]
        runs = [
            ["run_name=rm_min16", "remat_policy=minimal", "per_device_batch_size=16"] + b,
            ["run_name=rm_sdem16", "remat_policy=save_dot_except_mlp", "per_device_batch_size=16"] + b,
            ["run_name=rm_min8", "remat_policy=minimal", "per_device_batch_size=8"] + b,
            ["run_name=tokamax16", "use_tokamax_gmm=True", "per_device_batch_size=16"] + b[:2] + b[3:],
        ]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "neumann":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=12", "PADGMM=1.5", "FASTCONV=1",
             "per_device_batch_size=16"]
        runs = [[f"run_name=nm_{p}", f"GDNFAST={p},neumann,64"] + b for p in ("default", "high")]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "best":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=12", "PADGMM=1.5", "FASTCONV=1"]
        runs = [
            ["run_name=best16", "per_device_batch_size=16", "GDNFAST=default,solve,64"] + b,
            ["run_name=best16_bf16w", "per_device_batch_size=16", "GDNFAST=default,solve,64", "weight_dtype=bfloat16"] + b,
            ["run_name=best24_bf16w", "per_device_batch_size=24", "GDNFAST=default,solve,64", "weight_dtype=bfloat16"] + b,
        ]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "unroll":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=10", "PADGMM=1.5", "FASTCONV=1",
             "per_device_batch_size=16"]
        runs = [[f"run_name=ur{u}", f"GDNFAST=high,solve,{u}"] + b for u in (4, 16, 64)]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "gdnfast":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=10", "PADGMM=1.5", "FASTCONV=1",
             "per_device_batch_size=16"]
        runs = [[f"run_name=gf_{p}_{i}", f"GDNFAST={p},{i}"] + b
                for p, i in (("highest", "neumann"), ("high", "neumann"), ("default", "neumann"))]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "chunk":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=10", "PADGMM=1.5", "FASTCONV=1",
             "per_device_batch_size=16"]
        runs = [[f"run_name=ch{c}", f"gdn_chunk_size={c}"] + b for c in (128, 256, 32)]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "next":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=10", "PADGMM=1.5", "FASTCONV=1",
             "per_device_batch_size=16"]
        runs = [
            ["run_name=n_nogdn", "inhomogeneous_layer_cycle_interval=1"] + b,   # стеля, якщо GDN = 0
            ["run_name=n_bf16w", "weight_dtype=bfloat16"] + b,                 # без кастів fp32→bf16
            ["run_name=n_b24", "per_device_batch_size=24"] + b[:-1],
        ]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "fastconv":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=10", "PADGMM=1.5"]
        runs = [
            ["run_name=fc_b8", "FASTCONV=1", "per_device_batch_size=8"] + b,
            ["run_name=fc_b16", "FASTCONV=1", "per_device_batch_size=16"] + b,
        ]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "padgmm":
        b = ["sparse_matmul=True", "max_target_length=4096", "steps=10"]
        runs = [
            ["run_name=pg2_b8", "PADGMM=2.0", "per_device_batch_size=8"] + b,
            ["run_name=pg2_b16", "PADGMM=2.0", "per_device_batch_size=16"] + b,
            ["run_name=pg15_b8", "PADGMM=1.5", "per_device_batch_size=8"] + b,
        ]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "gdn":
        cap = ["sparse_matmul=False", "capacity_factor=1.25", "per_device_batch_size=8",
               "max_target_length=4096", "steps=10"]
        runs = [
            ["run_name=cap_nogdn", "inhomogeneous_layer_cycle_interval=1"] + cap,   # 16 full-attn, без GDN
            ["run_name=cap_2k", "per_device_batch_size=16", "max_target_length=2048"] + cap[:2] + ["steps=10"],
        ]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "pyspy":
        rc = pyspy_gpu.remote(["per_device_batch_size=8", "max_target_length=4096", "sparse_matmul=False",
                               "capacity_factor=1.25"])
    elif mode == "unscan":
        b = ["per_device_batch_size=8", "max_target_length=4096", "steps=10", "scan_layers=False"]
        runs = [
            ["run_name=cpu16_cap", "sparse_matmul=False", "capacity_factor=1.25", "scan_layers=True"] + b[:-1],
            ["run_name=cpu16_ragged", "sparse_matmul=True", "per_device_batch_size=4", "scan_layers=True"] + b[1:-1],
        ]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "profile":
        texts = profile_gpu.remote(["per_device_batch_size=16", "max_target_length=4096", "sparse_matmul=True",
                                    "PADGMM=1.5", "FASTCONV=1", "GDNFAST=default,solve,64"])
        for k, v in (texts if isinstance(texts, dict) else {}).items():
            open(f"../out/profile_{k}.json", "w", encoding="utf-8").write(v)
            print("saved", k, len(v))
        rc = 0
    elif mode == "squeeze":
        cap = ["sparse_matmul=False", "capacity_factor=1.25", "max_target_length=4096", "steps=10"]
        xla = ("ENV:XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true "
               "--xla_gpu_enable_highest_priority_async_stream=true")
        runs = [
            ["run_name=b8", "per_device_batch_size=8"] + cap,
            ["run_name=b12", "per_device_batch_size=12"] + cap,
            ["run_name=b8_sdem", "per_device_batch_size=8", "remat_policy=save_dot_except_mlp"] + cap,
            ["run_name=b8_xla", "per_device_batch_size=8", xla] + cap,
            ["run_name=b8_bf16w", "per_device_batch_size=8", "weight_dtype=bfloat16"] + cap,
        ]
        print("RESULTS", smoke_gpu.remote(runs))
        rc = 0
    elif mode == "gpu":
        # від малого до бойового: дивимось пам'ять і швидкість
        rcs = smoke_gpu.remote([
            ["run_name=g1", "per_device_batch_size=1", "max_target_length=2048"],
            ["run_name=g2", "per_device_batch_size=4", "max_target_length=4096"],
            ["run_name=g3", "per_device_batch_size=8", "max_target_length=4096"],
        ])
        print("RESULTS", rcs)
        rc = 0 if any(r == 0 for r in rcs) else 1
    else:
        raise ValueError(mode)
    print("RESULT", mode, "OK" if rc == 0 else f"FAIL ({rc})")
