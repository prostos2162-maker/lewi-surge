"""
Обгортка над maxtext.checkpoint_conversion.to_huggingface / to_maxtext: підміняє вбудований HF-конфіг qwen3.5-35b-a3b
на розміри нашої стадії 1, бо --override_model_architecture не чіпає розміри GDN і частину форм.
Запуск: python hf_export_wrapper.py [to_huggingface|to_maxtext] <аргументи конвертера>
"""
import copy
import runpy
import sys

import transformers
import maxtext.checkpoint_conversion.utils.hf_model_configs as H

OUR = {
    "hidden_size": 1024, "num_hidden_layers": 12, "num_attention_heads": 8, "num_key_value_heads": 2,
    "head_dim": 128, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
    "num_experts": 64, "num_experts_per_tok": 6, "linear_num_key_heads": 4, "linear_num_value_heads": 8,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
    "vocab_size": 131200, "eos_token_id": 131074, "mtp_num_hidden_layers": 0, "full_attention_interval": 4,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]
                   + ["linear_attention"] * 3 + ["full_attention"],
}

import os
_L = int(os.environ.get("LEWI_LAYERS", 12))
_E = int(os.environ.get("LEWI_EXPERTS", 64))
_M = int(os.environ.get("LEWI_MOE_DIM", 512))      # розмір експерта (7B-A1B: 1024)
_S = int(os.environ.get("LEWI_SHARED_DIM", 512))   # розмір спільного експерта
_K = int(os.environ.get("LEWI_TOPK", 6))           # експертів на токен
OUR.update({"num_hidden_layers": _L, "num_experts": _E, "moe_intermediate_size": _M, "shared_expert_intermediate_size": _S, "num_experts_per_tok": _K,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * (_L // 4)})

d = copy.deepcopy(H.qwen3_5_35b_a3b_dict)
d["text_config"].update(OUR)
d["tie_word_embeddings"] = True
d["text_config"]["tie_word_embeddings"] = True
new_cfg = transformers.Qwen3_5MoeConfig(**d)
old_cfg = H.qwen3_5_35b_a3b_config
H.qwen3_5_35b_a3b_dict = d
H.qwen3_5_35b_a3b_config = new_cfg
for name, val in list(vars(H).items()):  # реєстри model_name → config
    if isinstance(val, dict):
        for k, v in list(val.items()):
            if v is old_cfg:
                val[k] = new_cfg
                print(f"[wrapper] {name}[{k}] → наш конфіг", flush=True)

MOD = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1] in ("to_huggingface", "to_maxtext") else "to_huggingface"
sys.argv = [MOD] + sys.argv[1:]
runpy.run_module(f"maxtext.checkpoint_conversion.{MOD}", run_name="__main__")
