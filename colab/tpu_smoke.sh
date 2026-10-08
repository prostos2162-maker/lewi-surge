# Smoke-тест стадії 1 на Colab TPU v5e-1: та сама форма, 4 шари (16 ГБ HBM), синтетичні дані, рідний megablox
set -e
cd /content
PKG=$(python -c "import maxtext, os; print(os.path.dirname(maxtext.__file__))")
python - <<'PY'
import maxtext, os, shutil
pkg = os.path.dirname(maxtext.__file__)
p = os.path.join(pkg, "configs", "types.py"); src = open(p).read()
anchor = 'if self.opt_type == "muon" and self.decoder_block not in ['
i = src.find(anchor)
if i >= 0 and "QWEN3_5" not in src[i:i + 600]:
    j = i + len(anchor); open(p, "w").write(src[:j] + "\n        DecoderBlockType.QWEN3_5," + src[j:]); print("[patch] muon")
shutil.copy("/content/stage1.yml", os.path.join(pkg, "configs", "stage1.yml"))
PY
export TF_CPP_MIN_LOG_LEVEL=1 PYTHONUNBUFFERED=1
python -m maxtext.trainers.pre_train.train $PKG/configs/stage1.yml hardware=tpu skip_jax_distributed_system=True \
  dataset_type=synthetic enable_checkpointing=False steps=${STEPS:-30} log_period=5 \
  base_num_decoder_layers=${LAYERS:-4} sparse_matmul=True megablox=True \
  per_device_batch_size=${PDB:-4} max_target_length=4096 \
  base_output_directory=/content/out run_name=smoke ${EXTRA:-} 2>&1 | grep -E "completed step|Error|error:|Traceback|RESOURCE|Exception" | tail -40
echo SMOKE_DONE
