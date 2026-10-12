"""
Вирощена модель Lewi Surge (3.45B: 16 шарів, 128 експертів) на Lightning AI (новий акаунт), code-first суміш.

Фази (стан у out/lt2/state.json — можна перезапускати скрипт, він продовжить з потрібної фази):
  prep  — БЕЗКОШТОВНИЙ CPU: venv, ваги (HF tied) → MaxText, токенізація суміші з HF у mmap (~2B токенів)
  train — H200 interruptible: замір швидкості, тренування під бюджет годин, вартовий з перезапуском

  python lightning/train_s2g.py --phase prep
  python lightning/train_s2g.py --phase train --hours 6.5
"""
import argparse
import json
import os
import re
import time

from lightning_sdk import Machine, Studio
from lightning_sdk.machine import CloudProvider

TEAMSPACE, USER, NAME = "default-project", os.environ.get("LIGHTNING_USER", "your-lightning-user"), "lewi-s2g"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE = "/teamspace/studios/this_studio"
TMP = os.path.join(ROOT, "out", "lt2")
SEQ = 4096
HF_LOCAL = os.path.join(ROOT, "checkpoints", "surge_s2g_step5000_hf_tied")
DIMS = ("base_emb_dim=1024 base_num_decoder_layers=16 base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 "
        "base_mlp_dim=512 base_moe_mlp_dim=512 num_experts=128 shared_experts=1 num_experts_per_tok=6 "
        "vocab_size=131200 logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 "
        "gdn_num_key_heads=4 gdn_num_value_heads=8 gdn_key_head_dim=128 gdn_value_head_dim=128 "
        "gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 use_multimodal=False")

B = 1_000_000_000
GCC = dict(repo="codeparrot/github-code-clean", prefix="data/", fmt="code", fim=0.5, cols=["code", "repo_name", "path"])
ANN = "OpenCoder-LLM/opc-annealing-corpus"
FW2HQ = "epfml/FineWeb2-HQ"
# ~2.2B токенів, ~55% код; ваги суміші = пропорційно бюджетам (кожен шард ≈ 1 прохід)
SPEC = [
    dict(name="code", budget=0.35 * B, **GCC),
    dict(name="ann_algo", repo=ANN, prefix="algorithmic_corpus/", fmt="jtext", budget=0.25 * B, cols=None),
    dict(name="ann_snip", repo=ANN, prefix="synthetic_code_snippet/", fmt="jtext", budget=0.15 * B, cols=None),
    dict(name="ann_qa", repo=ANN, prefix="synthetic_qa/", fmt="jtext", budget=0.10 * B, cols=None),
    dict(name="ins_magicoder", repo="ise-uiuc/Magicoder-OSS-Instruct-75K", prefix="data-oss", fmt="qa", q="problem",
         a="solution", budget=0.04 * B, cols=None),
    dict(name="ins_codefeedback", repo="m-a-p/CodeFeedback-Filtered-Instruction", prefix="CodeFeedback", fmt="qa",
         q="query", a="answer", budget=0.09 * B, cols=None),
    dict(name="ins_selfoss", repo="bigcode/self-oss-instruct-sc2-exec-filter-50k", prefix="data/", fmt="qa",
         q="instruction", a="response", budget=0.016 * B, cols=["instruction", "response"]),
    dict(name="ins_opcsft", repo="OpenCoder-LLM/opc-sft-stage2", prefix="", fmt="qa", q="instruction", a="output",
         budget=0.12 * B, cols=["instruction", "output"]),
    dict(name="codeweb", repo="OpenCoder-LLM/opc-fineweb-code-corpus", prefix="data/", fmt="text", budget=0.08 * B, cols=["text"]),
    dict(name="cosmo", repo="HuggingFaceTB/smollm-corpus", prefix="cosmopedia-v2/", fmt="text", budget=0.30 * B, cols=["text"]),
    dict(name="en", repo="HuggingFaceFW/fineweb-edu", prefix="sample/10BT/", fmt="text", budget=0.20 * B, cols=["text"]),
    dict(name="math", repo="HuggingFaceTB/finemath", prefix="finemath-4plus/", fmt="text", budget=0.15 * B, cols=["text"]),
    dict(name="uk", repo="HuggingFaceFW/fineweb-2", prefix="data/ukr_Cyrl/train/", fmt="text", budget=0.14 * B, cols=["text"]),
    dict(name="es", repo=FW2HQ, prefix="spa_Latn/", fmt="text", budget=0.03 * B, cols=["text"]),
    dict(name="pt", repo=FW2HQ, prefix="por_Latn/", fmt="text", budget=0.03 * B, cols=["text"]),
    dict(name="fr", repo=FW2HQ, prefix="fra_Latn/", fmt="text", budget=0.03 * B, cols=["text"]),
    dict(name="de", repo=FW2HQ, prefix="deu_Latn/", fmt="text", budget=0.03 * B, cols=["text"]),
    dict(name="svg", repo="starvector/svg-stack", prefix="data/train", fmt="svg", budget=0.03 * B, cols=["Svg"]),
    dict(name="text2svg", repo="starvector/text2svg-stack", prefix="data/train", fmt="text2svg", budget=0.03 * B,
         cols=["Svg", "caption_blip2", "caption_cogvlm", "caption_llava"]),
]

SETUP = r"""
set -e
cd ~/moe
pip install -q uv
if [ ! -f ~/mt/.ready ]; then
  uv venv -q ~/mt --python 3.12
  source ~/mt/bin/activate
  uv pip install -q 'maxtext[tpu]==0.2.5' --resolution=lowest
  install_tpu_pre_train_extra_deps || true
  JAXV=$(python -c "import jax; print(jax.__version__)")
  uv pip uninstall -q libtpu libtpu-nightly scikit-learn || true
  uv pip install -q "jax[cuda12]==$JAXV" tokenizers pyarrow huggingface_hub
  touch ~/mt/.ready
fi
source ~/mt/bin/activate
python patch_maxtext.py
python -c "import jax; print('JAX', jax.__version__, jax.devices())"
echo SETUP_DONE
"""

PATCH = r'''
import maxtext, os, shutil
pkg = os.path.dirname(maxtext.__file__)
p = os.path.join(pkg, "configs", "types.py"); src = open(p).read()
anchor = 'if self.opt_type == "muon" and self.decoder_block not in ['
i = src.find(anchor)
if i >= 0 and "QWEN3_5" not in src[i:i + 600]:
    j = i + len(anchor); open(p, "w").write(src[:j] + "\n        DecoderBlockType.QWEN3_5," + src[j:])
    print("[patch] muon: QWEN3_5")
shutil.copy(os.path.expanduser("~/moe/stage2.yml"), os.path.join(pkg, "configs", "stage2.yml"))
print("PKG", pkg)
'''

CONVERT = r"""
source ~/mt/bin/activate
cd ~/moe
PKG=$(python -c "import maxtext, os; print(os.path.dirname(maxtext.__file__))")
if [ ! -d ~/s2g_mt/0/items ]; then
  rm -rf ~/s2g_mt
  LEWI_LAYERS=16 LEWI_EXPERTS=128 JAX_PLATFORMS=cpu python conv_wrapper.py to_maxtext $PKG/configs/stage2.yml \
    model_name=qwen3.5-35b-a3b override_model_config=True dataset_type=synthetic __DIMS__ \
    base_output_directory=$HOME/s2g_mt hardware=cpu skip_jax_distributed_system=True scan_layers=True \
    --hf_model_path=$HOME/hf_s2g > ~/convert.log 2>&1 || true
fi
ls -d ~/s2g_mt/0/items && echo CONVERT_OK
"""

TRAIN = r"""
source ~/mt/bin/activate
cd ~/moe
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 TF_CPP_MIN_LOG_LEVEL=1 PYTHONUNBUFFERED=1
PKG=$(python -c "import maxtext, os; print(os.path.dirname(maxtext.__file__))")
python -c "import sys,runpy; sys.path.insert(0,'$HOME/moe'); import padded_gmm; padded_gmm.install(1.5); \
import fast_conv; fast_conv.install(); import gdn_fast; gdn_fast.install('default','solve',64); \
sys.argv=['train']+sys.argv[1:]; runpy.run_module('maxtext.trainers.pre_train.train', run_name='__main__')" \
  $PKG/configs/stage2.yml __DIMS__ hardware=gpu skip_jax_distributed_system=True \
  tokenizer_path=$HOME/moe/tokenizer sparse_matmul=True megablox=False weight_dtype=bfloat16 learning_rate=1.2e-3 \
  per_device_batch_size=__PDB__ gradient_accumulation_steps=__GA__ max_target_length=4096 \
  dataset_type=grain grain_file_type=mmap "grain_train_files=__MIX__" \
  load_parameters_path=$HOME/s2g_mt/0/items \
  base_output_directory=__OUT__ run_name=__RUN__ __EXTRA__
"""


def sh(studio, cmd):
    try:
        return studio.run(cmd)
    except Exception as e:  # studio.run кидає RuntimeError з виводом при ненульовому коді
        return f"ERR: {e}"


def put(studio, local, remote_dir):
    """SDK на Windows ламає підпапки → вантажимо плоско в ~ і переносимо."""
    name = os.path.basename(local)
    studio.upload_file(local, name)
    sh(studio, f"cd ~ && for i in $(seq 120); do ls {name} >/dev/null 2>&1 && break; sleep 3; done; "
               f"mkdir -p {remote_dir} && mv {name} {remote_dir}/")


def write_tmp(name, body):
    os.makedirs(TMP, exist_ok=True)
    p = os.path.join(TMP, name)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    return p


def upload_code(studio):
    files = [("out/tokenizer_pruned/tokenizer.json", "~/moe/tokenizer"),
             ("out/tokenizer_pruned/tokenizer_config.json", "~/moe/tokenizer"),
             ("out/tokenizer_pruned/chat_template.jinja", "~/moe/tokenizer"),
             ("configs/stage2.yml", "~/moe"), ("kaggle/stage2/conv_wrapper.py", "~/moe"),
             ("modal/padded_gmm.py", "~/moe"), ("modal/fast_conv.py", "~/moe"), ("modal/gdn_fast.py", "~/moe")]
    for local, rdir in files:
        put(studio, os.path.join(ROOT, local), rdir)
    put(studio, write_tmp("setup.sh", SETUP), "~/moe")
    put(studio, write_tmp("patch_maxtext.py", PATCH), "~/moe")


def tokenize_script():
    """kaggle/tok3/tokenize3.py, перенаправлений на диск студії."""
    s = open(os.path.join(ROOT, "kaggle", "tok3", "tokenize3.py"), encoding="utf-8").read()
    s = s.replace("__SPEC__", repr(SPEC))
    s = s.replace('OUT = "/kaggle/working"', 'OUT = os.path.expanduser("~/data")\nos.makedirs(OUT, exist_ok=True)')
    s = s.replace('next(iter(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True)))',
                  'os.path.expanduser("~/moe/tokenizer/tokenizer.json")')
    return s


def mix():
    tot = sum(s["budget"] for s in SPEC)
    return ";".join(f"{REMOTE}/data/{s['name']},{s['budget'] / tot:.4f}" for s in SPEC)


def ensure(studio, machine):
    for _ in range(90):
        st = str(studio.status)
        if "Running" in st:
            if machine == "H200" and "H200" not in str(studio.machine):
                print(time.strftime("%H:%M"), f"машина {studio.machine} → H200", flush=True)
                studio.switch_machine(Machine.H200, interruptible=True, cloud_provider=CloudProvider.LIGHTNING)
                time.sleep(20)
                continue
            if machine == "CPU" and "CPU" not in str(studio.machine).upper():
                studio.switch_machine(Machine.CPU)
                time.sleep(20)
                continue
            return
        if "Stopped" in st:
            studio.start()
        time.sleep(20)
    raise RuntimeError(f"студія не стартувала: {studio.status}")


def phase_prep(studio, state, state_path):
    ensure(studio, "CPU")
    print("машина:", studio.machine, flush=True)
    assert "CPU" in str(studio.machine).upper(), "prep лише на CPU (безкоштовно)"
    upload_code(studio)
    out = sh(studio, "cd ~/moe && bash setup.sh 2>&1 | tail -3")
    print(out, flush=True)
    assert "SETUP_DONE" in out, "setup не вдався"
    if not state.get("weights"):
        for f in sorted(os.listdir(HF_LOCAL)):
            if f != "dataset-metadata.json":
                print("upload", f, flush=True)
                put(studio, os.path.join(HF_LOCAL, f), "~/hf_s2g")
        state["weights"] = True
        json.dump(state, open(state_path, "w"))
    put(studio, write_tmp("convert.sh", CONVERT.replace("__DIMS__", DIMS)), "~/moe")
    out = sh(studio, "cd ~/moe && bash convert.sh 2>&1 | tail -3")
    print(out, flush=True)
    assert "CONVERT_OK" in out, sh(studio, "tail -c 3000 ~/convert.log")
    put(studio, write_tmp("tok.py", tokenize_script()), "~/moe")
    sh(studio, "cd ~/moe && source ~/mt/bin/activate && nohup python tok.py > ~/tok.log 2>&1 &")
    print("токенізація запущена у фоні (~/tok.log)", flush=True)
    for _ in range(400):
        time.sleep(60)
        log = sh(studio, "tail -c 600 ~/tok.log")
        if "ГОТОВО" in log or "Traceback" in log:
            break
    print(sh(studio, "grep -a -E 'OK:|ГОТОВО|Traceback|Error' ~/tok.log | tail -25"), flush=True)
    assert "ГОТОВО" in sh(studio, "tail -c 300 ~/tok.log"), "токенізація не завершилась"
    state["prep"] = True
    json.dump(state, open(state_path, "w"))
    studio.stop()
    print("PREP ГОТОВО, студію зупинено", flush=True)


def launch(studio, run, extra, log, pdb):
    cmd = (TRAIN.replace("__DIMS__", DIMS).replace("__PDB__", str(pdb)).replace("__GA__", str(32 // pdb))
           .replace("__MIX__", mix()).replace("__OUT__", f"{REMOTE}/out").replace("__RUN__", run)
           .replace("__EXTRA__", extra))
    put(studio, write_tmp("train.sh", cmd), "~/moe")
    sh(studio, f"cd ~/moe && nohup bash train.sh >> {log} 2>&1 &")


def phase_train(studio, state, state_path, hours):
    assert state.get("prep"), "спершу --phase prep"
    ensure(studio, "H200")
    upload_code(studio)
    out = sh(studio, "cd ~/moe && bash setup.sh 2>&1 | tail -3")
    print(out, flush=True)
    if "CudaDevice" not in out:
        print("GPU НЕ ЗНАЙДЕНО — зупиняю студію", flush=True)
        studio.stop()
        return
    if "steps" not in state:
        speed = pdb = None
        for pdb in (16, 8, 4):
            sh(studio, "rm -rf ~/out/bench ~/bench.log")
            launch(studio, "bench", "enable_checkpointing=False steps=30 log_period=5", "~/bench.log", pdb)
            for _ in range(60):
                time.sleep(30)
                log = sh(studio, "tail -c 4000 ~/bench.log")
                if "completed step: 29" in log or "Traceback" in log:
                    break
            full = sh(studio, "cat ~/bench.log")
            print(f"BENCH b{pdb}:", sh(studio, "grep -a -E 'completed step|Error' ~/bench.log | tail -4"), flush=True)
            tps = [float(x) for x in re.findall(r"Tokens/s/device: ([\d.]+)", full)]
            if len(tps) >= 4:
                speed = sum(tps[-4:]) / 4
                break
            if "RESOURCE_EXHAUSTED" not in full:
                studio.stop()
                raise RuntimeError("bench впав не через пам'ять — студію зупинено")
        if not speed:
            studio.stop()
            raise RuntimeError("жоден батч не влазить — студію зупинено")
        steps = int(hours * 3600 * speed / (32 * SEQ))
        state.update({"speed": speed, "steps": steps, "pdb": pdb})
        json.dump(state, open(state_path, "w"))
        print(f"ШВИДКІСТЬ {speed:.0f} ток/с, батч {pdb}×{32 // pdb} → {steps} кроків ≈ {steps * 32 * SEQ / 1e9:.2f}B",
              flush=True)
    steps, pdb = state["steps"], state["pdb"]
    # сталий LR після прогріву: спад зробимо в annealing/дистиляції
    extra = (f"steps={steps} learning_rate_schedule_steps={steps} warmup_steps_fraction=0.02 "
             f"wsd_decay_steps_fraction=0.0 enable_checkpointing=True async_checkpointing=True checkpoint_period=250 "
             f"max_num_checkpoints_to_keep=2 log_period=50 metrics_file={REMOTE}/out/metrics.jsonl")

    def alive():
        return "runpy" in sh(studio, "pgrep -af runpy | grep -v pgrep || true")

    if not alive():
        launch(studio, "s2g", extra, "~/train.log", pdb)
    last, crashes = "", 0
    while True:
        time.sleep(180)
        if "Running" not in str(studio.status):
            print(time.strftime("%H:%M"), f"студія {studio.status} — перезапуск", flush=True)
            try:
                ensure(studio, "H200")
            except Exception as e:
                print("СТАРТ НЕ ВДАВСЯ (можливо, скінчились кредити):", e, flush=True)
                return
            upload_code(studio)
            print(sh(studio, "cd ~/moe && bash setup.sh 2>&1 | tail -2"), flush=True)
            launch(studio, "s2g", extra, "~/train.log", pdb)
            continue
        tail = sh(studio, "grep -a -E 'completed step|Traceback|Error|RESOURCE_EXHAUSTED|saved|Restoring' ~/train.log | tail -3")
        if tail != last:
            print(time.strftime("%H:%M"), tail[-700:], flush=True)
            last = tail
        if f"completed step: {steps - 1}" in tail:
            print("ТРЕНУВАННЯ ЗАВЕРШЕНО", flush=True)
            sh(studio, "sleep 120")
            studio.stop()
            return
        if not alive() and "completed step" in tail:
            print("ПРОЦЕС ВПАВ:\n", sh(studio, "tail -c 2500 ~/train.log"), flush=True)
            crashes += 1
            if crashes > 3:
                print("3+ падіння поспіль — зупиняю студію", flush=True)
                studio.stop()
                return
            launch(studio, "s2g", extra, "~/train.log", pdb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prep", "train"], required=True)
    ap.add_argument("--hours", type=float, default=6.5)
    args = ap.parse_args()
    state_path = os.path.join(TMP, "state.json")
    os.makedirs(TMP, exist_ok=True)
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    studio = Studio(name=NAME, teamspace=TEAMSPACE, user=USER, create_ok=True)
    if args.phase == "prep":
        phase_prep(studio, state, state_path)
    else:
        phase_train(studio, state, state_path, args.hours)


if __name__ == "__main__":
    main()
