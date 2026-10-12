"""Збирає train.ipynb (Kaggle) і train_built.py (Colab-тест) з вшитими stage2.yml і conv_wrapper.py.

  python kaggle/long/build.py [--resume]   # --resume: + вихід попередньої версії цього ноутбука (ваги між сесіями)
  kaggle kernels push -p kaggle/long
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KID = "steflett/lewi-surge-long"
TOK_KERNELS = ["steflett/moe-tok3-long", "steflett/moe-tok3-long2", "steflett/moe-tok3-long3", "steflett/moe-tok3-anneal", "steflett/moe-tok3-instruct",
               "steflett/moe-tok3-cosmo", "steflett/moe-tok2-math", "steflett/moe-tok2-uk", "steflett/moe-tok2-code1",
               "steflett/moe-tok2-design"]

src = open(os.path.join(HERE, "train.py"), encoding="utf-8").read()
yml = open(os.path.join(ROOT, "configs", "stage2.yml"), encoding="utf-8").read()
wrap = open(os.path.join(ROOT, "kaggle", "stage2", "conv_wrapper.py"), encoding="utf-8").read()
src = src.replace("__STAGE2_YML__", repr(yml)).replace("__CONV_WRAPPER__", repr(wrap))
open(os.path.join(HERE, "train_built.py"), "w", encoding="utf-8", newline="\n").write(src)
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(os.path.join(HERE, "train.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
meta = {"id": KID, "title": "lewi-surge-long", "code_file": "train.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": True,
        "enable_internet": True, "machine_shape": "TpuV5E8",
        "docker_image": "gcr.io/kaggle-private-byod/python-tpuvm@sha256:"
                        "bc3c1d3db8f7b8d2e9f9102d34b10155149dab7d10460c6a2ab8bcea92ef433f",
        "dataset_sources": ["steflett/lewi-surge-sft2-hf", "steflett/lewi-sft3"],
        "kernel_sources": TOK_KERNELS + ([KID] if "--resume" in sys.argv else []),
        "competition_sources": [], "model_sources": []}
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("built:", KID, len(meta["kernel_sources"]), "kernel sources")
