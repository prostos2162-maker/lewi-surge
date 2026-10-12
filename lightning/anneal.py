"""
Annealing фінального чекпоінта s2g (крок 12250) на новому акаунті Lightning: ~50% чат-даних + найкращий претрейн,
LR лінійно спадає від 1.2e-3 майже до нуля. Перевикористовує код train_s2g.py (перевизначаємо його глобальні змінні).

  python lightning/anneal.py --phase prep              # CPU: ваги, конвертація, токенізація суміші → студія стоп
  python lightning/anneal.py --phase train --hours 5.6 # H200: annealing (швидкість відома з s2g — без бенчу)
"""
import argparse
import json
import os

from lightning_sdk import Studio

import train_s2g as T

T.TEAMSPACE, T.USER, T.NAME = "default-project", os.environ.get("LIGHTNING_USER", "your-lightning-user"), "lewi-ann"
T.TMP = os.path.join(T.ROOT, "out", "lt3")
T.HF_LOCAL = os.path.join(T.ROOT, "checkpoints", "surge_s2g2_step12250_hf_tied")

B = T.B
ANN = T.ANN
UD = "fxmeng/UltraData-SFT-2605-no-think-8k-32k"  # дзеркало без гейту
I = {s["name"]: s for s in T.SPEC}


def re_b(name, budget):
    return dict(I[name], budget=budget * B)


# ~1.45B токенів: ~58% діалоги (UltraData, Nemotron IF, код-інструкції), решта — найкращий претрейн (щоб не забути)
T.SPEC = [
    dict(name="ud_code", repo=UD, prefix="Code/", fmt="msgs", budget=0.15 * B, cols=None),
    dict(name="ud_know", repo=UD, prefix="Multi-lang-Knowledge/", fmt="msgs", budget=0.12 * B, cols=None),
    dict(name="ud_math", repo=UD, prefix="Math/", fmt="msgs", budget=0.10 * B, cols=None),
    dict(name="ud_mmath", repo=UD, prefix="Multi-lang-Math/", fmt="msgs", budget=0.03 * B, cols=None),
    dict(name="ud_if", repo=UD, prefix="IF/", fmt="msgs", budget=0.003 * B, cols=None),
    dict(name="nemo_if", repo="nvidia/Nemotron-SFT-Instruction-Following-Chat-v3", prefix="data/instruction_following",
         fmt="msgs", budget=0.15 * B, cols=None),
    re_b("ins_magicoder", 0.04), re_b("ins_codefeedback", 0.087), re_b("ins_selfoss", 0.016), re_b("ins_opcsft", 0.12),
    re_b("ann_algo", 0.12), re_b("ann_snip", 0.08), re_b("ann_qa", 0.06),
    re_b("cosmo", 0.10), re_b("math", 0.08), re_b("uk", 0.06), re_b("en", 0.05),
    re_b("svg", 0.01), re_b("text2svg", 0.01),
]

_orig_train = T.TRAIN
# спад LR: майже весь прогін (прогрів 2% — стан Muon свіжий), до 1% від піку
ANNEAL_LR = ("warmup_steps_fraction=0.02 wsd_decay_steps_fraction=0.98 learning_rate_final_fraction=0.01 num_epoch=2 data_shuffle_seed=7 ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prep", "train"], required=True)
    ap.add_argument("--hours", type=float, default=5.6)
    args = ap.parse_args()
    os.makedirs(T.TMP, exist_ok=True)
    state_path = os.path.join(T.TMP, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    studio = Studio(name=T.NAME, teamspace=T.TEAMSPACE, user=T.USER, create_ok=True)
    if args.phase == "prep":
        T.phase_prep(studio, state, state_path)
        return
    if "steps" not in state:  # швидкість і батч відомі з s2g на H200 — не палимо кредити на бенч
        speed, pdb = 70800.0, 16
        steps = int(args.hours * 3600 * speed / (32 * T.SEQ))
        state.update({"speed": speed, "steps": steps, "pdb": pdb})
        json.dump(state, open(state_path, "w"))
        print(f"кроків {steps} ≈ {steps * 32 * T.SEQ / 1e9:.2f}B токенів", flush=True)
    # phase_train додає wsd_decay_steps_fraction=0.0 — підмінюємо на спад
    T.TRAIN = _orig_train.replace("__EXTRA__", "__EXTRA__ " + ANNEAL_LR)
    orig_launch = T.launch

    def launch(studio, run, extra, log, pdb):
        extra = extra.replace("warmup_steps_fraction=0.02 ", "").replace("wsd_decay_steps_fraction=0.0 ", "")
        return orig_launch(studio, "ann", extra, log, pdb)

    T.launch = launch
    T.phase_train(studio, state, state_path, args.hours)


if __name__ == "__main__":
    main()
