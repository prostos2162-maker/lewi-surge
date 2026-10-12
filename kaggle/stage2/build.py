"""Збирає train.ipynb стадії 2 (train.py + вшиті stage2.yml і conv_wrapper.py) і kernel-metadata.json.

  python kaggle/stage2/build.py            # перша сесія: старт з ваг стадії 1
  python kaggle/stage2/build.py --resume   # наступні: + вихід попередньої версії цього ж ноутбука
  kaggle kernels push -p kaggle/stage2
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KID = "steflett/lewi-surge-stage2"
TOK_KERNELS = [f"steflett/moe-tok-fineweb-edu-{x}" for x in "abcd"] + \
              [f"steflett/moe-tok2-{x}" for x in ("uk", "espt", "frde", "math", "design", "code1", "code2")] + \
              [f"steflett/moe-tok3-{x}" for x in ("anneal", "instruct", "cosmo")]

src = open(os.path.join(HERE, "train.py"), encoding="utf-8").read()
yml = open(os.path.join(ROOT, "configs", "stage2.yml"), encoding="utf-8").read()
wrap = open(os.path.join(HERE, "conv_wrapper.py"), encoding="utf-8").read()
src = src.replace("__STAGE2_YML__", repr(yml)).replace("__CONV_WRAPPER__", repr(wrap))
nb = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 4,
}
json.dump(nb, open(os.path.join(HERE, "train.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

meta = {
    "id": KID,
    "title": "lewi-surge-stage2",
    "code_file": "train.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": True,
    "enable_internet": True,
    "machine_shape": "TpuV5E8",
    "docker_image": "gcr.io/kaggle-private-byod/python-tpuvm@sha256:"
                    "bc3c1d3db8f7b8d2e9f9102d34b10155149dab7d10460c6a2ab8bcea92ef433f",
    "dataset_sources": ["steflett/moe-assets", "steflett/lewi-surge-s2g-hf"],
    "kernel_sources": TOK_KERNELS + ([KID] if "--resume" in sys.argv else []),
    "competition_sources": [],
    "model_sources": [],
}
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("built", len(meta["kernel_sources"]), "kernel sources")
