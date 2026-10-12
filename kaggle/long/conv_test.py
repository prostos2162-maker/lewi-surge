"""Colab: перевірка конвертації HF → MaxText (conv_wrapper.py) на мініатюрі нашої архітектури (4 шари, 16 експертів).
Будуємо HF-модель з того самого конфігу, що підставляє conv_wrapper, зберігаємо, конвертуємо тим самим шляхом, що й Kaggle."""
import copy
import glob
import os
import subprocess
import sys

import maxtext
import transformers
import maxtext.checkpoint_conversion.utils.hf_model_configs as H

L, E = 4, 16
d = copy.deepcopy(H.qwen3_5_35b_a3b_dict)
d["text_config"].update({
    "hidden_size": 1024, "num_hidden_layers": L, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 128,
    "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512, "num_experts": E, "num_experts_per_tok": 6,
    "linear_num_key_heads": 4, "linear_num_value_heads": 8, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "vocab_size": 131200, "eos_token_id": 131074, "mtp_num_hidden_layers": 0,
    "full_attention_interval": 4, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * (L // 4),
    "tie_word_embeddings": True})
d["tie_word_embeddings"] = True
cfg = transformers.Qwen3_5MoeConfig(**d)
model = transformers.AutoModelForCausalLM.from_config(cfg.text_config if hasattr(cfg, "text_config") else cfg)
model.save_pretrained("/tmp/tiny_hf")
# справжній чекпоінт (експорт MaxText) має ключі model.language_model.* — перейменовуємо, як у SFT2
import json as _j
from safetensors.torch import load_file, save_file
for f in glob.glob("/tmp/tiny_hf/*.safetensors"):
    sd = load_file(f)
    save_file({("model.language_model." + k[len("model."):] if k.startswith("model.") else k): v.contiguous()
               for k, v in sd.items() if k != "lm_head.weight"}, f, metadata={"format": "pt"})
ix = "/tmp/tiny_hf/model.safetensors.index.json"
if os.path.exists(ix):
    m = _j.load(open(ix))
    m["weight_map"] = {("model.language_model." + k[len("model."):] if k.startswith("model.") else k): v
                       for k, v in m["weight_map"].items() if k != "lm_head.weight"}
    _j.dump(m, open(ix, "w"))
cfg.save_pretrained("/tmp/tiny_hf")  # повний (мультимодальний) конфіг, як у SFT2
print("HF збережено:", sorted(os.listdir("/tmp/tiny_hf")), type(model).__name__, flush=True)

pkg = os.path.dirname(maxtext.__file__)
dims = ("base_emb_dim=1024 base_num_decoder_layers=4 base_num_query_heads=8 base_num_kv_heads=2 head_dim=128 "
        "base_mlp_dim=512 base_moe_mlp_dim=512 num_experts=16 shared_experts=1 num_experts_per_tok=6 vocab_size=131200 "
        "logits_via_embedding=True inhomogeneous_layer_cycle_interval=4 gdn_num_key_heads=4 gdn_num_value_heads=8 "
        "gdn_key_head_dim=128 gdn_value_head_dim=128 gdn_conv_kernel_dim=4 use_mrope=False mtp_num_layers=0 use_multimodal=False")
r = subprocess.run(f"LEWI_LAYERS={L} LEWI_EXPERTS={E} JAX_PLATFORMS=cpu {sys.executable} /kaggle/working/conv_wrapper.py "
                   f"to_maxtext {pkg}/configs/stage2.yml model_name=qwen3.5-35b-a3b override_model_config=True "
                   f"dataset_type=synthetic {dims} base_output_directory=/tmp/tiny_mt hardware=cpu "
                   f"skip_jax_distributed_system=True scan_layers=True --hf_model_path=/tmp/tiny_hf",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-1200:], r.stderr[-2500:], flush=True)
items = glob.glob("/tmp/tiny_mt/**/items", recursive=True)
print("CONV_RESULT rc", r.returncode, "items", items, flush=True)
