# Lewi Surge 7B-A1B, етап 1 претрейну на GCP TPU v6e-8 (spot). Запуск на VM: nohup bash gcp_lewi7b.sh > go.log 2>&1 &
# Ідемпотентний: після витіснення spot цикл на ПК (gcp_7b_loop.sh) створює нову VM і запускає цей скрипт знову;
# дані/ваги перекачуються, тренування продовжується з останнього чекпойнта в gs://lewi-data/ckpt/$RUN.
#   STAGE_TOKENS  — скільки токенів тренувати на цьому етапі (за замовчуванням 10B)
#   PDB           — per_device_batch_size (16 → 8×16×4096 = 524k токенів/крок)
#   REMAT         — minimal|full; якщо minimal не влізе в пам'ять, проба перемикає на full
set -x
export PATH=$HOME/.local/bin:$PATH; source ~/mt/bin/activate
uv pip install -q safetensors pyarrow ml_dtypes 'huggingface_hub[cli]' 'kaggle==2.2.4' > /dev/null 2>&1
mkdir -p ~/.kaggle && cp -f ~/kcred.json ~/.kaggle/credentials.json 2>/dev/null; chmod 600 ~/.kaggle/* 2>/dev/null
W=~/lewi7b; mkdir -p $W/src/configs $W/data $W/hf_in
R=https://raw.githubusercontent.com/prostos2162-maker/lewi-surge/main
for f in modal/gdn_fast.py kaggle/stage2/conv_wrapper.py tools/grow_7b.py colab/wcheck.py; do curl -fsSL "$R/$f" -o $W/src/$(basename $f); done
cp ~/stage2.yml $W/src/configs/
PKG=$(python -c "import maxtext,os;print(os.path.dirname(maxtext.__file__))"); CFG=$PKG/configs/stage2.yml; cp $W/src/configs/stage2.yml $CFG
RUN=${RUN:-7b_s1}; CKPT=gs://lewi-data/ckpt; STAGE_TOKENS=${STAGE_TOKENS:-10000000000}; PDB=${PDB:-16}; SEQ=4096
DIMS="base_emb_dim=1024 base_num_decoder_layers=32 base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 base_mlp_dim=512 base_moe_mlp_dim=1024 num_experts=64 shared_experts=1 num_experts_per_tok=6 vocab_size=131200 logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 gdn_num_key_heads=4 gdn_num_value_heads=8 gdn_key_head_dim=128 gdn_value_head_dim=128 gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 use_multimodal=False"

# ---- 1. дані етапу 1 з бакета (той самий регіон → без плати за трафік)
gcloud storage rsync gs://lewi-data/stage1 $W/data > $W/rsync.log 2>&1 || { echo "ДАНІ НЕ СКАЧАЛИСЬ"; exit 1; }
python - <<'PY' > $W/mix.txt
import glob, json, os
W = os.path.expanduser("~/lewi7b/data")
js = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{W}/*.json")}
tot = sum(v["tokens"] for v in js.values())
print(";".join(f"{W}/{k},{v['tokens'] / tot:.5f}" for k, v in sorted(js.items()) if os.path.exists(f"{W}/{k}.idx")))
PY
MIX=$(cat $W/mix.txt); echo "MIX=$MIX"; [ -n "$MIX" ] || { echo "ПОРОЖНІЙ МІКС"; exit 1; }
TOK=$W/data/tokenizer; mkdir -p $TOK; gcloud storage cp gs://lewi-data/tokenizer/* $TOK/ > /dev/null 2>&1

# ---- 2. стартові ваги: є чекпойнт у бакеті → продовжуємо; інакше ріст long_b → HF → MaxText (один раз)
LAST=$(gcloud storage ls $CKPT/$RUN/checkpoints/ 2>/dev/null | grep -E '/[0-9]+/$' | sed 's|.*/\([0-9]*\)/$|\1|' | sort -n | tail -1)
LOADARG=""
if [ -z "$LAST" ]; then
  if ! gcloud storage ls gs://lewi-data/init7b/items/_METADATA > /dev/null 2>&1; then
    dl() { for f in $(kaggle datasets files $1 | awk 'NR>2{print $1}'); do kaggle datasets download $1 -f $f -p $2 || return 1; done; }
    [ -f $W/hf_in/config.json ] || dl steflett/lewi-long-b-step1142 $W/hf_in
    [ -f $W/hf_grown/config.json ] || python $W/src/grow_7b.py $W/hf_in $W/hf_grown > $W/grow.log 2>&1 || { echo "РІСТ ВПАВ"; exit 1; }
    LEWI_LAYERS=32 LEWI_EXPERTS=64 LEWI_MOE_DIM=1024 LEWI_SHARED_DIM=512 JAX_PLATFORMS=cpu python $W/src/conv_wrapper.py to_maxtext $CFG \
      model_name=qwen3.5-35b-a3b override_model_config=True dataset_type=synthetic $DIMS base_output_directory=$W/mt_in \
      hardware=cpu skip_jax_distributed_system=True scan_layers=True --hf_model_path=$W/hf_grown > $W/convert.log 2>&1 || { echo "КОНВЕРТАЦІЯ ВПАЛА"; exit 1; }
    ITEMS=$(ls -d $W/mt_in/*/items $W/mt_in/*/*/items 2>/dev/null | head -1); [ -n "$ITEMS" ] || { echo "НЕМАЄ items"; exit 1; }
    gcloud storage rsync -r $ITEMS gs://lewi-data/init7b/items > /dev/null 2>&1   # щоб після витіснення не рости знову
  fi
  LOADARG="load_parameters_path=gs://lewi-data/init7b/items"
fi
echo "LAST=$LAST LOADARG=$LOADARG"

# ---- 3. кроки й розклад: WSD, спад лише в кінці всього плану (етап 1 = розігрів + константа)
TOK_STEP=$((8 * PDB * SEQ)); STEPS=$((STAGE_TOKENS / TOK_STEP)); SCHED=$((STEPS * 10 / 8))
PATCH="import sys; sys.path.insert(0,'$W/src'); import gdn_fast; gdn_fast.install('default','solve',64); import runpy; sys.argv=['t']+sys.argv[1:]; runpy.run_module('maxtext.trainers.pre_train.train', run_name='__main__')"
COMMON="$DIMS override_model_config=True hardware=tpu skip_jax_distributed_system=True tokenizer_path=$TOK sparse_matmul=True megablox=False moe_use_direct_token_gather=True weight_dtype=float32 attention=flash num_vocab_tiling=8 ici_fsdp_parallelism=-1 per_device_batch_size=$PDB max_target_length=$SEQ dataset_type=grain grain_file_type=mmap grain_train_files=$MIX num_epoch=1 opt_type=adamw learning_rate=${LR:-2e-4} lr_schedule_type=wsd warmup_steps_fraction=$(python -c "print(min(0.05, 300/$SCHED))") wsd_decay_steps_fraction=0.2 learning_rate_final_fraction=0.1 learning_rate_schedule_steps=$SCHED gradient_clipping_threshold=1.0 base_output_directory=$CKPT"
run_train() { python -c "$PATCH" $CFG $COMMON remat_policy=$1 "$@" ; }
REMAT=${REMAT:-minimal}
if [ -z "$LAST" ]; then  # проба швидкості/пам'яті лише на першому запуску
  python -c "$PATCH" $CFG $COMMON remat_policy=$REMAT run_name=probe7b steps=8 enable_checkpointing=False log_period=1 $LOADARG > $W/probe.log 2>&1
  if grep -q "RESOURCE_EXHAUSTED\|OOM\|OutOfMemory" $W/probe.log; then REMAT=full; echo "minimal OOM → full";
    python -c "$PATCH" $CFG $COMMON remat_policy=$REMAT run_name=probe7b steps=8 enable_checkpointing=False log_period=1 $LOADARG > $W/probe2.log 2>&1; fi
  grep -E "completed step: [5-7]," $W/probe*.log | tail -3; echo "REMAT=$REMAT" > $W/remat.txt
  gcloud storage cp $W/remat.txt $CKPT/$RUN.remat > /dev/null 2>&1
else
  REMAT=$(gcloud storage cat $CKPT/$RUN.remat 2>/dev/null | cut -d= -f2); REMAT=${REMAT:-full}
fi
echo "STEPS=$STEPS TOK_STEP=$TOK_STEP REMAT=$REMAT"
nohup bash ~/vm_guard7b.sh > ~/guard.log 2>&1 &
python -c "$PATCH" $CFG $COMMON remat_policy=$REMAT run_name=$RUN steps=$STEPS enable_checkpointing=True async_checkpointing=True \
  checkpoint_period=250 max_num_checkpoints_to_keep=2 log_period=25 metrics_file=$W/metrics.jsonl $LOADARG >> $W/train.log 2>&1
rc=$?; echo "TRAIN rc=$rc"; grep -c "completed step" $W/train.log
LASTD=$(gcloud storage ls $CKPT/$RUN/checkpoints/ 2>/dev/null | grep -E '/[0-9]+/$' | sed 's|.*/\([0-9]*\)/$|\1|' | sort -n | tail -1)
[ "$rc" = 0 ] && [ "${LASTD:-0}" -ge $((STEPS - 1)) ] && echo STAGE_DONE
echo ALL_DONE
