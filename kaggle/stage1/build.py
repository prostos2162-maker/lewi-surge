"""Збирає train.ipynb з train.py і kernel-metadata.json. python kaggle/stage1/build.py [--resume]

--resume: підключає вихід попередньої версії цього ж ноутбука (чекпоінт + обраний режим) як вхід.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KID = "steflett/moe-stage1-train"

src = open(os.path.join(HERE, "train.py"), encoding="utf-8").read()
nb = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 4,
}
json.dump(nb, open(os.path.join(HERE, "train.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

meta = {
    "id": KID,
    "title": "moe-stage1-train",
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
    "dataset_sources": ["steflett/moe-assets"],
    "kernel_sources": [KID] if "--resume" in sys.argv else [],
    "competition_sources": [],
    "model_sources": [],
}
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("built", meta["kernel_sources"])
