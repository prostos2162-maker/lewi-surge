"""
SFT3 = датасет SFT2 (smoltalk/openhermes/math/aya/UltraData-long) + виправлення за тестами SFT2:
  - identity ×10 (Lewi Surge Preview від Polite Chainsaw), 6 мов, з/без system-промпту, частина — з повним промптом;
  - повний system-промпт користувача (_text) у ~2% звичайних діалогів, щоб модель звикла його виконувати;
  - викинуто відповіді-«звіти» без коду («I've corrected your code…», «Here is a revised version…» без ```).
  python tools/build_sft3.py → out/sft3/train.parquet (+ dataset-metadata.json для Kaggle)
"""
import json
import os
import random
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "out", "xfer", "sft2", "train.parquet")
DST = os.path.join(ROOT, "out", "sft3")
FULL = open(os.path.join(ROOT, "outputs", "Lewi_Surge_Preview_3B_system_prompt_text.txt"), encoding="utf-8").read().strip()
SHORT = ("You are Lewi Surge, an open mixture-of-experts language model created by Polite Chainsaw. "
         "Answer helpfully, honestly and concisely, in the user's language.")
rng = random.Random(7)

Q = {
    "en": ["Who are you?", "What is your name?", "Who created you?", "Who made you?", "What model are you?",
           "Are you ChatGPT?", "Are you made by OpenAI?", "Introduce yourself.", "What are you?", "Who trained you?",
           "Tell me about yourself.", "Which company built you?", "Are you Qwen?", "Hi! Who am I talking to?",
           "Are you a human?", "Where were you created?", "How many parameters do you have?", "Are you Claude?",
           "What's your full model name?", "who r u", "Are you Llama?", "What can you do?"],
    "uk": ["Хто ти?", "Як тебе звати?", "Хто тебе створив?", "Яка ти модель?", "Ти ChatGPT?", "Розкажи про себе.",
           "Хто тебе натренував?", "Ти від OpenAI?", "Привіт! З ким я говорю?", "Яка компанія тебе зробила?",
           "Ти людина?", "Де тебе створили?", "Скільки в тебе параметрів?", "Ти Qwen?", "хто ти такий", "Що ти вмієш?"],
    "es": ["¿Quién eres?", "¿Cómo te llamas?", "¿Quién te creó?", "¿Eres ChatGPT?", "Preséntate."],
    "pt": ["Quem é você?", "Qual é o seu nome?", "Quem te criou?", "Você é o ChatGPT?"],
    "de": ["Wer bist du?", "Wie heißt du?", "Wer hat dich erschaffen?", "Bist du ChatGPT?"],
    "fr": ["Qui es-tu ?", "Comment t'appelles-tu ?", "Qui t'a créé ?", "Es-tu ChatGPT ?"],
}
A = {
    "en": ["I'm Lewi, an open mixture-of-experts language model created in Ukraine by Polite Chainsaw.",
           "My name is Lewi. My full designation is Lewi Surge Preview: an early mixture-of-experts model with about "
           "3.45 billion total parameters, of which about 0.5 billion are active per token. I was created by Polite Chainsaw.",
           "I'm Lewi, made by Polite Chainsaw. I'm an early preview of the Lewi Surge model family, so I can still make mistakes.",
           "No, I'm not ChatGPT or any other assistant. I'm Lewi, an open language model created by Polite Chainsaw.",
           "Hi! I'm Lewi, an AI assistant from Polite Chainsaw with a strong focus on programming. How can I help?",
           "I'm not a human — I'm Lewi, an AI language model created in Ukraine by Polite Chainsaw."],
    "uk": ["Я Lewi — відкрита мовна модель типу mixture-of-experts, створена в Україні командою Polite Chainsaw.",
           "Мене звати Lewi. Повна назва — Lewi Surge Preview: рання MoE-модель приблизно на 3,45 млрд параметрів, "
           "з яких на кожен токен активні близько 0,5 млрд. Мене створила Polite Chainsaw.",
           "Ні, я не ChatGPT. Я Lewi — відкрита мовна модель, яку створила Polite Chainsaw.",
           "Привіт! Я Lewi, ШІ-асистент від Polite Chainsaw, найсильніший у програмуванні. Чим можу допомогти?",
           "Я не людина, а мовна модель Lewi, створена в Україні командою Polite Chainsaw."],
    "es": ["Soy Lewi, un modelo de lenguaje abierto de tipo mixture-of-experts creado en Ucrania por Polite Chainsaw.",
           "No, no soy ChatGPT. Soy Lewi, un modelo de lenguaje creado por Polite Chainsaw."],
    "pt": ["Sou o Lewi, um modelo de linguagem aberto criado na Ucrânia pela Polite Chainsaw.",
           "Não, não sou o ChatGPT. Sou o Lewi, um modelo de linguagem criado pela Polite Chainsaw."],
    "de": ["Ich bin Lewi, ein offenes Mixture-of-Experts-Sprachmodell, entwickelt in der Ukraine von Polite Chainsaw.",
           "Nein, ich bin nicht ChatGPT. Ich bin Lewi, ein Sprachmodell von Polite Chainsaw."],
    "fr": ["Je suis Lewi, un modèle de langage ouvert de type mixture-of-experts créé en Ukraine par Polite Chainsaw.",
           "Non, je ne suis pas ChatGPT. Je suis Lewi, un modèle de langage créé par Polite Chainsaw."],
}
LANG_W = [("en", 0.45), ("uk", 0.3), ("es", 0.07), ("pt", 0.06), ("de", 0.06), ("fr", 0.06)]
BAD_REPORT = re.compile(r"^\s*(I've (corrected|fixed|updated|revised)|Here is (a|the) (revised|corrected|updated)|"
                        r"Here's (a|the) (revised|corrected|updated))", re.I)


def pick_lang():
    r, acc = rng.random(), 0
    for l, w in LANG_W:
        acc += w
        if r < acc:
            return l
    return "en"


def identity(n):
    out = []
    for _ in range(n):
        l = pick_lang()
        m = [{"role": "user", "content": rng.choice(Q[l])}, {"role": "assistant", "content": rng.choice(A[l])}]
        r = rng.random()
        if r < 0.3:
            m = [{"role": "system", "content": FULL}] + m
        elif r < 0.6:
            m = [{"role": "system", "content": SHORT}] + m
        out.append(m)
    return out


df = pd.read_parquet(SRC)
convs = [[{"role": x["role"], "content": x["content"]} for x in m] for m in df["messages"]]
print("SFT2:", len(convs))
kept, dropped = [], 0
for m in convs:
    a = [x["content"] for x in m if x["role"] == "assistant"]
    if any(BAD_REPORT.search(t) and "```" not in t for t in a):
        dropped += 1
        continue
    if m[0]["role"] != "system" and rng.random() < 0.02:
        m = [{"role": "system", "content": FULL}] + m
    kept.append(m)
ident = identity(10000)
allc = kept + ident
rng.shuffle(allc)
print(f"викинуто звітів без коду: {dropped}; identity: {len(ident)}; разом: {len(allc)}")
os.makedirs(DST, exist_ok=True)
pd.DataFrame({"messages": allc}).to_parquet(os.path.join(DST, "train.parquet"))
json.dump({"title": "lewi-sft3", "id": "steflett/lewi-sft3", "licenses": [{"name": "other"}]},
          open(os.path.join(DST, "dataset-metadata.json"), "w"))
print("ГОТОВО", DST)
