# Перевірка стадії 2 на Colab TPU v5e-1 перед Kaggle:
#   1) HF-чекпоінт стадії 1 → MaxText (to_maxtext через conv_wrapper.py)
#   2) кілька кроків тренування з цих ваг на mmap-суміші (2 крихітні шарди)
# Очікування: стартовий лос ~3-4 (ваги підхопились), а не ~12 (випадкова ініціалізація).
# Файли кладе в /content: hf/ (model.safetensors, config.json), stage2.yml, conv_wrapper.py, t_code.*, t_t2s.*
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
shutil.copy("/content/stage2.yml", os.path.join(pkg, "configs", "stage2.yml"))
PY
export TF_CPP_MIN_LOG_LEVEL=1 PYTHONUNBUFFERED=1
DIMS="base_emb_dim=1024 base_num_decoder_layers=12 base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 \
base_mlp_dim=512 base_moe_mlp_dim=512 num_experts=64 shared_experts=1 num_experts_per_tok=6 \
vocab_size=131200 logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 \
gdn_num_key_heads=4 gdn_num_value_heads=8 gdn_key_head_dim=128 gdn_value_head_dim=128 \
gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 use_multimodal=False"

if [ "${SKIP_CONV:-0}" != 1 ]; then
  rm -rf /content/s1_mt
  JAX_PLATFORMS=cpu python /content/conv_wrapper.py to_maxtext $PKG/configs/stage2.yml \
    model_name=qwen3.5-35b-a3b override_model_config=True dataset_type=synthetic $DIMS \
    base_output_directory=/content/s1_mt hardware=cpu skip_jax_distributed_system=True scan_layers=True \
    --hf_model_path=/content/hf 2>&1 | grep -vE "Warning|warn" | tail -25
  echo CONV_DONE
fi
find /content/s1_mt -maxdepth 3 | head -20
ITEMS=$(find /content/s1_mt -type d -name items | head -1)
echo "ITEMS=$ITEMS"

python -m maxtext.trainers.pre_train.train $PKG/configs/stage2.yml hardware=tpu skip_jax_distributed_system=True \
  tokenizer_path=/content/hf load_parameters_path=$ITEMS \
  "grain_train_files=/content/t_code,0.6;/content/t_t2s,0.4" \
  enable_checkpointing=False steps=${STEPS:-20} log_period=1 per_device_batch_size=${PDB:-2} \
  sparse_matmul=True megablox=True base_output_directory=/content/out run_name=s2test ${EXTRA:-} 2>&1 \
  | grep -E "completed step|Error|error:|Traceback|RESOURCE|Exception|Restoring|restored|load_parameters" | tail -40
echo TEST_DONE
