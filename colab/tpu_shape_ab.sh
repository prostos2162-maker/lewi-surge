# A/B форми експертів для 7B-A1B на одному TPU-чипі (Colab v5e-1 / v6e-1), 4 шари, bf16, синтетичні дані, 12 кроків.
# Однакова маса експертів і однакова кількість активних параметрів у кожному варіанті:
#   e128_512_k12: 128 експертів × 512, top-12   (план)
#   e64_1024_k6:   64 експерти × 1024, top-6    (кандидат)
#   e128_512_k6:  128 × 512, top-6               (нинішня Lewi, базова лінія)
# Запуск на VM: NL=4 PDB=4 bash tpu_shape_ab.sh > shape.log 2>&1
mkdir -p /root/src; R=https://raw.githubusercontent.com/prostos2162-maker/lewi-surge/main
[ -f /root/src/gdn_fast.py ] || curl -fsSL $R/modal/gdn_fast.py -o /root/src/gdn_fast.py
CFG=${CFG:-/root/src/configs/stage2.yml}
run() {  # $1 назва, $2 num_experts, $3 moe_mlp_dim, $4 top-k
  DIMS="base_emb_dim=1024 base_num_decoder_layers=${NL:-4} base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 base_mlp_dim=$3 base_moe_mlp_dim=$3 num_experts=$2 shared_experts=1 num_experts_per_tok=$4 vocab_size=131200 logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 gdn_num_key_heads=4 gdn_num_value_heads=8 gdn_key_head_dim=128 gdn_value_head_dim=128 gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 use_multimodal=False"
  python3 -c "import sys; sys.path.insert(0,'/root/src'); import gdn_fast; gdn_fast.install('default','solve',64)
import runpy; sys.argv=['t']+sys.argv[1:]; runpy.run_module('maxtext.trainers.pre_train.train', run_name='__main__')" \
    $CFG $DIMS override_model_config=True run_name=ab_$1 base_output_directory=/root/out dataset_type=synthetic steps=12 log_period=1 \
    enable_checkpointing=False per_device_batch_size=${PDB:-4} max_target_length=${SEQ:-4096} weight_dtype=bfloat16 opt_type=adamw \
    remat_policy=full sparse_matmul=True megablox=False moe_use_direct_token_gather=True attention=flash hardware=tpu ${EXTRA} 2>&1 \
    | grep -E "completed step: (9|10|11),|RESOURCE|rror" | sed "s/^/[$1] /" | cut -c1-200
}
for V in ${VARIANTS:-e128_512_k12 e64_1024_k6 e128_512_k6}; do
  case $V in
    e128_512_k12) run e128_512_k12 128 512 12 ;;
    e64_1024_k6)  run e64_1024_k6 64 1024 6 ;;
    e128_512_k6)  run e128_512_k6 128 512 6 ;;
    e32_2048_k3)  run e32_2048_k3 32 2048 3 ;;
  esac
done
echo SHAPE_DONE
