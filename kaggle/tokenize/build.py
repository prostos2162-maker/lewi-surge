"""Створює 4 Kaggle CPU-ноутбуки токенізації (по шарду). python kaggle/tokenize/build.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [f"{i:03d}_00000.parquet" for i in range(14)]
SHARDS = {"a": FILES[0:4], "b": FILES[4:8], "c": FILES[8:11], "d": FILES[11:14]}
src = open(os.path.join(HERE, "tokenize.py"), encoding="utf-8").read()

for name, files in SHARDS.items():
    d = os.path.join(HERE, name)
    os.makedirs(d, exist_ok=True)
    code = src.replace('"__SHARD__"', repr(name)).replace("__FILES__", repr(files))
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    json.dump(nb, open(os.path.join(d, "tok.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = {"id": f"steflett/moe-tok-fineweb-edu-{name}", "title": f"moe-tok-fineweb-edu-{name}",
            "code_file": "tok.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": ["steflett/moe-assets"], "kernel_sources": [], "competition_sources": [],
            "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    print(name, files)
