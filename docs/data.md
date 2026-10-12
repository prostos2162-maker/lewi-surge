# Дані

Принцип: тільки відкриті датасети з вмістом прямо на HF (без ручних gate). Nemotron і OpenMathInstruct-2 свідомо не беремо.

## Претрейн (стадії росту)

| Частка | Частина | Датасет | Ліцензія | Примітка |
|---|---|---|---|---|
| ~30% | Англ веб (знання) | `HuggingFaceFW/fineweb-edu` (score ≥ 3) | ODC-By | |
| ~20% | Англ веб (різноманітність, міркування) | `mlfoundations/dclm-baseline-1.0` | CC-BY-4.0 | |
| ~18% | es / pt / fr / de | `epfml/FineWeb2-HQ` (`spa_Latn`, `por_Latn`, `fra_Latn`, `deu_Latn`) | ODC-By | |
| ~7% | uk | `HuggingFaceFW/fineweb-2` (`ukr_Cyrl`) | ODC-By | у FineWeb2-HQ немає укр → свій фільтр пізніше |
| ~15% | Код | `bigcode/starcoderdata` | other (auto-gate) | натиснути Agree на HF; пізніше можливо Stack-Edu через Software Heritage |
| ~3% | Код у вебі | `OpenCoder-LLM/opc-fineweb-code-corpus` | MIT | |
| ~7% | Математика | `HuggingFaceTB/finemath` (`finemath-4plus`) + `OpenCoder-LLM/opc-fineweb-math-corpus` | ODC-By / MIT | |

Частки стартові, крутимо за результатами.

## Mid-training / annealing

- `OpenCoder-LLM/opc-annealing-corpus` (ODC-By)
- `open-thoughts/OpenThoughts3-1.2M` (Apache 2.0) — ланцюжки міркувань
- синтетика від Qwen3.6-35B-A3B / Ornith-1.5-35B-A3B (укр. та інші мови, математика, код)

## Подача даних

1. Перші тестові прогони стадії 1 — стрім з HF (MaxText `dataset_type=hf`).
2. Паралельно на CPU-сесіях Kaggle: токенізація суміші → шарди ArrayRecord (Grain) → основні прогони з них (детермінований resume).
