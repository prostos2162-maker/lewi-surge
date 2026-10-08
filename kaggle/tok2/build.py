"""Створює Kaggle CPU-ноутбуки токенізації стадії 2. python kaggle/tok2/build.py → kaggle kernels push -p kaggle/tok2/<name>"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
B = 1_000_000_000
FW2HQ = "epfml/FineWeb2-HQ"
GCC = dict(repo="codeparrot/github-code-clean", prefix="data/", fmt="code", fim=0.5,
           cols=["code", "repo_name", "path", "language"])

# кожен ноутбук ≤ ~2.5B токенів (≤10 ГБ виходу)
NOTEBOOKS = {
    "uk": [dict(name="uk", repo="HuggingFaceFW/fineweb-2", prefix="data/ukr_Cyrl/train/", fmt="text", budget=1.5 * B, cols=["text"])],
    "espt": [dict(name="es", repo=FW2HQ, prefix="spa_Latn/", fmt="text", budget=0.7 * B, cols=["text"]),
             dict(name="pt", repo=FW2HQ, prefix="por_Latn/", fmt="text", budget=0.7 * B, cols=["text"])],
    "frde": [dict(name="fr", repo=FW2HQ, prefix="fra_Latn/", fmt="text", budget=0.6 * B, cols=["text"]),
             dict(name="de", repo=FW2HQ, prefix="deu_Latn/", fmt="text", budget=0.6 * B, cols=["text"])],
    "math": [dict(name="math", repo="HuggingFaceTB/finemath", prefix="finemath-4plus/", fmt="text", budget=1.2 * B, cols=["text"])],
    "design": [dict(name="svg", repo="starvector/svg-stack", prefix="data/train", fmt="svg", budget=0.6 * B, cols=["Svg"]),
               dict(name="text2svg", repo="starvector/text2svg-stack", prefix="data/train", fmt="text2svg", budget=0.6 * B,
                    cols=["Svg", "caption_blip2", "caption_cogvlm", "caption_llava"]),
               dict(name="websight", repo="HuggingFaceM4/WebSight", prefix="data/", fmt="html", budget=0.25 * B, cols=["text"]),
               dict(name="codeweb", repo="OpenCoder-LLM/opc-fineweb-code-corpus", prefix="data/", fmt="text", budget=0.8 * B, cols=["text"])],
    "code1": [dict(name="code1", budget=2.0 * B, **GCC)],
    "code2": [dict(name="code2", budget=2.0 * B, **GCC)],
}

src = open(os.path.join(HERE, "tokenize2.py"), encoding="utf-8").read()
for nb_name, spec in NOTEBOOKS.items():
    if nb_name == "code2":  # інший порядок файлів, щоб не дублювати code1
        src_nb = src.replace("random.Random(1234)", "random.Random(98765)")
    else:
        src_nb = src
    d = os.path.join(HERE, nb_name)
    os.makedirs(d, exist_ok=True)
    code = src_nb.replace("__SPEC__", repr(spec))
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    json.dump(nb, open(os.path.join(d, "tok.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = {"id": f"steflett/moe-tok2-{nb_name}", "title": f"moe-tok2-{nb_name}",
            "code_file": "tok.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": ["steflett/moe-assets"], "kernel_sources": [], "competition_sources": [],
            "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    print(nb_name, [(s["name"], s["budget"] / B) for s in spec])
