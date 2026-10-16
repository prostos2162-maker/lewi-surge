# Профіль одного кроку Lewi на TPU v6e-1 у ПОТОЧНОМУ конфігу тренування (train_lewi2 на TPU):
# sparse_matmul + ragged_dot (megablox=False), moe_use_direct_token_gather=True, gdn_fast, 128 експертів, remat full.
# Ваги bf16 (fp32 + Adam на 16 шарів не влазять в 1 чип), pdb 8, seq 4096. Підсумок: частки часу за групами операцій.
# Запуск на VM: bash tpu_prof2.sh > prof2.log 2>&1
PKG=$(python3 -c "import maxtext,os;print(os.path.dirname(maxtext.__file__))")
mkdir -p /root/src; R=https://raw.githubusercontent.com/prostos2162-maker/lewi-surge/main
for f in modal/gdn_fast.py colab/tpu_parse.py; do curl -fsSL $R/$f -o /root/src/$(basename $f); done
DIMS="base_emb_dim=1024 base_num_decoder_layers=${NL:-16} base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 base_mlp_dim=512 base_moe_mlp_dim=512 num_experts=${NE:-128} shared_experts=1 num_experts_per_tok=${TOPK:-6} vocab_size=131200 logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 gdn_num_key_heads=4 gdn_num_value_heads=8 gdn_key_head_dim=128 gdn_value_head_dim=128 gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 use_multimodal=False"
rm -rf /root/out/prof2
python3 -c "import sys; sys.path.insert(0,'/root/src'); import gdn_fast; gdn_fast.install('default','solve',64)
import runpy; sys.argv=['t']+sys.argv[1:]; runpy.run_module('maxtext.trainers.pre_train.train', run_name='__main__')" \
  $PKG/configs/stage2.yml $DIMS override_model_config=True \
  run_name=prof2 base_output_directory=/root/out dataset_type=synthetic steps=10 log_period=1 enable_checkpointing=False \
  per_device_batch_size=${PDB:-8} max_target_length=${SEQ:-4096} weight_dtype=${WDT:-bfloat16} opt_type=adamw remat_policy=${REMAT:-full} \
  sparse_matmul=True megablox=False moe_use_direct_token_gather=True attention=flash hardware=tpu \
  profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=2 ${EXTRA} 2>&1 | grep -E "completed step|rror|RESOURCE" | tail -4
pip install -q xprof tensorboard-plugin-profile >/dev/null 2>&1
python3 /root/src/tpu_parse.py 2>&1 | tail -80
echo PROF2_DONE
