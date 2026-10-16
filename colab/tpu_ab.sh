# A/B на TPU v6e-1: варіанти MoE-диспетчера і GDN-патч. 12 кроків, pdb 8, remat full, 32 експерти.
PKG=$(python3 -c "import maxtext,os;print(os.path.dirname(maxtext.__file__))")
mkdir -p /root/src; R=https://raw.githubusercontent.com/prostos2162-maker/lewi-surge/main
for f in modal/gdn_fast.py modal/padded_gmm.py modal/fast_conv.py modal/moe_permute.py; do curl -fsSL $R/$f -o /root/src/$(basename $f); done
DIMS="base_emb_dim=1024 base_num_decoder_layers=${NL:-16} base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 base_mlp_dim=512 base_moe_mlp_dim=512 num_experts=${NE:-128} shared_experts=1 num_experts_per_tok=6 vocab_size=131200 logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 gdn_num_key_heads=4 gdn_num_value_heads=8 gdn_key_head_dim=128 gdn_value_head_dim=128 gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 use_multimodal=False"
BASE="run_name=ab base_output_directory=/root/out dataset_type=synthetic steps=12 log_period=1 enable_checkpointing=False per_device_batch_size=${PDB:-8} max_target_length=4096 weight_dtype=${WDT:-float32} opt_type=adamw remat_policy=${REMAT:-full} attention=flash"
run() {  # $1 назва, $2 патчі (python), решта — оверайди
  name=$1; pat=$2; shift 2
  python3 -c "import sys; sys.path.insert(0,'/root/src'); $pat
import runpy; sys.argv=['t']+sys.argv[1:]; runpy.run_module('maxtext.trainers.pre_train.train', run_name='__main__')" \
    $PKG/configs/stage2.yml $DIMS override_model_config=True $BASE "$@" 2>&1 | grep -E "completed step: 1[01]|RESOURCE|rror" | tail -2 | sed "s/^/[$name] /" | cut -c1-230
}
for V in ${VARIANTS:-dense gdn dense_gdn}; do
  case $V in
    dense)     run dense "" sparse_matmul=False capacity_factor=1.25 ;;
    gdn)       run gdn "import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=True megablox=True ;;
    dense_gdn) run dense_gdn "import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=False capacity_factor=1.25 ;;
    padgmm)    run padgmm "import padded_gmm; padded_gmm.install(1.5); import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=True megablox=False ;;
    rg_min)    run rg_min "import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=True megablox=False ;;
    rg_qkv)    run rg_qkv "import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=True megablox=False remat_policy=save_qkv_proj ;;
    rg_full)   run rg_full "import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=True megablox=False remat_policy=full ;;
    ragged)    run ragged "import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=True megablox=False ;;
    direct)    run direct "import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=True megablox=False moe_use_direct_token_gather=True ;;
    sc)        run sc "import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=True megablox=False use_ragged_sort=True use_ring_of_experts=True ;;
    sc1)       run sc1 "import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=True megablox=False use_ragged_sort=True use_ring_of_experts=True ragged_sort_use_single_sparsecore=True ;;
    permute)   run permute "import gdn_fast; gdn_fast.install('default','solve',64); import moe_permute; moe_permute.install('gather', num_experts=${NE:-128})" sparse_matmul=True megablox=False moe_use_direct_token_gather=True ;;
    permute_f) run permute_f "import gdn_fast; gdn_fast.install('default','solve',64); import moe_permute; moe_permute.install('full', num_experts=${NE:-128})" sparse_matmul=True megablox=False moe_use_direct_token_gather=True ;;
    permute_c) run permute_c "import gdn_fast; gdn_fast.install('default','solve',64); import moe_permute; moe_permute.install('full', sort='count', num_experts=${NE:-128})" sparse_matmul=True megablox=False moe_use_direct_token_gather=True ;;
    direct_mb) run direct_mb "import gdn_fast; gdn_fast.install('default','solve',64)" sparse_matmul=True megablox=True moe_use_direct_token_gather=True ;;
  esac
done
echo AB_DONE
