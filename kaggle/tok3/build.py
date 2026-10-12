"""Kaggle CPU-ноутбуки токенізації якісних даних (code-first). python kaggle/tok3/build.py → kaggle kernels push -p kaggle/tok3/<name>"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
B = 1_000_000_000
ANN = "OpenCoder-LLM/opc-annealing-corpus"
SV2 = "HuggingFaceTB/smollm-corpus"

NOTEBOOKS = {
    # найякісніший код: алгоритмічні задачі з розв'язками, синтетичні сніпети, Q&A
    "anneal": [dict(name="ann_algo", repo=ANN, prefix="algorithmic_corpus/", fmt="jtext", budget=1.0 * B, cols=None),
               dict(name="ann_snip", repo=ANN, prefix="synthetic_code_snippet/", fmt="jtext", budget=0.8 * B, cols=None),
               dict(name="ann_qa", repo=ANN, prefix="synthetic_qa/", fmt="jtext", budget=0.7 * B, cols=None)],
    # задача → робочий розв'язок, у нашому чат-форматі
    "instruct": [dict(name="ins_magicoder", repo="ise-uiuc/Magicoder-OSS-Instruct-75K", prefix="data-oss", fmt="qa",
                      q="problem", a="solution", budget=0.2 * B, cols=None),
                 dict(name="ins_codefeedback", repo="m-a-p/CodeFeedback-Filtered-Instruction", prefix="CodeFeedback", fmt="qa",
                      q="query", a="answer", budget=0.3 * B, cols=None),
                 dict(name="ins_selfoss", repo="bigcode/self-oss-instruct-sc2-exec-filter-50k", prefix="data/", fmt="qa",
                      q="instruction", a="response", budget=0.1 * B, cols=["instruction", "response"]),
                 dict(name="ins_opcsft", repo="OpenCoder-LLM/opc-sft-stage2", prefix="", fmt="qa",
                      q="instruction", a="output", budget=0.3 * B, cols=["instruction", "output"])],
    # UltraData-SFT-2605 no-think (OpenBMB, Apache-2.0): повні діалоги в чат-форматі для annealing (вчить і <|im_end|>)
    "ultra": [dict(name="ud_code", repo="fxmeng/UltraData-SFT-2605-no-think-8k-32k", prefix="Code/", fmt="msgs", budget=0.6 * B, cols=None),
              dict(name="ud_know", repo="fxmeng/UltraData-SFT-2605-no-think-8k-32k", prefix="Multi-lang-Knowledge/", fmt="msgs", budget=0.5 * B, cols=None),
              dict(name="ud_if", repo="fxmeng/UltraData-SFT-2605-no-think-8k-32k", prefix="IF/", fmt="msgs", budget=0.05 * B, cols=None),
              dict(name="ud_mmath", repo="fxmeng/UltraData-SFT-2605-no-think-8k-32k", prefix="Multi-lang-Math/", fmt="msgs", budget=0.05 * B, cols=None),
              dict(name="ud_math", repo="fxmeng/UltraData-SFT-2605-no-think-8k-32k", prefix="Math/", fmt="msgs", budget=0.6 * B, cols=None)],
    # Nemotron IF-Chat v3 (NVIDIA, CC-BY-4.0 + ODC-By; атрибуція в model card): виконання інструкцій
    "nemo": [dict(name="nemo_if", repo="nvidia/Nemotron-SFT-Instruction-Following-Chat-v3", prefix="data/instruction_following",
                  fmt="msgs", budget=1.0 * B, cols=None)] + [
        # повний UltraData-SFT-2605 (гейт, секрет "hf"), no_think, короткі діалоги
        dict(name=f"udf_{n}", repo="openbmb/UltraData-SFT-2605", prefix=f"data/no_think/{d}/", fmt="msgs", budget=b * B, cols=None)
        for n, d, b in (("code", "Code", 1.0), ("know", "Knowledge", 0.4), ("if", "IF", 0.1), ("mlk", "Multi-lang-Knowledge", 0.8),
                        ("mlmath", "Multi-lang-Math", 0.3), ("math", "Math", 0.6))],
    # довгий контекст (32k-128k): цілі книжки PG-19 (суспільне надбання, Apache-2.0), довгі статті/математика,
    # довгі діалоги UltraData 8k-32k; MaxText ріже документи на max_target_length — довгі лишаються суцільними
    "long": [dict(name="long_pg19", repo="emozilla/pg19", prefix="data/train", fmt="text", budget=0.15 * B, cols=["text"],
                  min_chars=100_000, max_chars=3_000_000, bs=64),
             dict(name="long_ud", repo="fxmeng/UltraData-SFT-2605-no-think-8k-32k", prefix="", fmt="msgs", budget=0.08 * B,
                  cols=None, min_chars=30_000, max_chars=400_000, bs=256),
             dict(name="long_edu", repo="HuggingFaceFW/fineweb-edu", prefix="sample/10BT/", fmt="text", budget=0.05 * B,
                  cols=["text"], min_chars=40_000, max_chars=2_000_000),
             dict(name="long_math", repo="HuggingFaceTB/finemath", prefix="finemath-4plus/", fmt="text", budget=0.03 * B,
                  cols=["text"], min_chars=30_000, max_chars=2_000_000)],
    # ще книжки PG-19: інше перемішування файлів → інші книжки (перший прохід узяв 0.154B з ~3B)
    "long2": [dict(name="long_pg19b", repo="emozilla/pg19", prefix="data/train", fmt="text", budget=0.20 * B,
                   cols=["text"], min_chars=100_000, max_chars=3_000_000, bs=64, seed=99,
                   exclude=["train-00006-", "train-00016-"])],
    # довгий код: The Stack, склеєний по репозиторіях (ProLong, токени Llama-3 → декодуємо → наш токенайзер)
    # + траєкторії агента Open-SWE-Traces (CC-BY-4.0, Nvidia): довгі сесії bash/правок коду з викликами інструментів
    "long3": [dict(name="long_code", repo="princeton-nlp/prolong-data-512K", prefix="thestackv1_concat_by_repo-524288",
                   fmt="prolong", budget=0.25 * B, max_chars=1_500_000),
              dict(name="long_swe", repo="nvidia/Open-SWE-Traces", prefix="data/", fmt="swe", budget=0.12 * B,
                   cols=["messages", "tools"], min_chars=20_000, max_chars=1_500_000, bs=64)],
    # синтетичні підручники (Phi-стиль)
    "cosmo": [dict(name="cosmo", repo=SV2, prefix="cosmopedia-v2/", fmt="text", budget=2.0 * B, cols=["text"])],
}

src = open(os.path.join(HERE, "tokenize3.py"), encoding="utf-8").read()
for nb_name, spec in NOTEBOOKS.items():
    d = os.path.join(HERE, nb_name)
    os.makedirs(d, exist_ok=True)
    code = src.replace("__SPEC__", repr(spec))
    if nb_name == "long3":  # MDS-читач ProLong
        code = ("import subprocess, sys\n"
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mosaicml-streaming'])\n") + code
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    json.dump(nb, open(os.path.join(d, "tok.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = {"id": f"steflett/moe-tok3-{nb_name}", "title": f"moe-tok3-{nb_name}",
            "code_file": "tok.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": ["steflett/moe-assets"], "kernel_sources": [], "competition_sources": [],
            "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    print(nb_name, [(s["name"], s["budget"] / B) for s in spec])
