# Lewi Surge

An open Mixture-of-Experts language model, pretrained from scratch by one person on free and
low-cost compute, and grown in stages toward the final **Lewi Surge 24B2A**
(24B total / 2B active parameters).

Author: Stefan Dobrovolskyi (Ukraine).

## Architecture

JAX + [MaxText](https://github.com/AI-Hypercomputer/maxtext), `qwen3_5`-style decoder block:

- hybrid attention: Gated DeltaNet (linear) and full attention layers in a 3:1 ratio
- fine-grained MoE: many small experts plus one shared expert
- Muon optimizer, WSD learning-rate schedule
- tokenizer: Qwen3.6 vocabulary pruned to 131,072 BPE tokens + 33 special tokens
  (FIM, tool-call, thinking and multimodal tokens reserved from day one)

## Stage 1: Lewi Surge 1.4B-A0.3B (English)

| | |
|---|---|
| Shape | d_model 1024, 12 layers, 8 query / 2 KV heads, 64 experts × 512, top-6 + 1 shared |
| Parameters | ~1.41B total / ~0.33B active |
| Data | FineWeb-Edu, ~3.2B tokens, sequence length 4096 |
| Hardware | 1× NVIDIA H200 (interruptible), ~10 hours |
| Throughput | ~91k tokens/s, ~186 TFLOP/s, ~19% MFU |
| Loss | 13 → ~3.0 (stable phase), then LR decay |

### Making MoE + Gated DeltaNet fast on a single GPU

Out of the box the stack ran at ~4.7k tokens/s on an H100 (<1% MFU). The patches in `modal/`
took it to ~75k tokens/s on H100 and ~91k on H200, with identical math:

- `padded_gmm.py` — padded grouped GEMM for MoE that replaces `jax.lax.ragged_dot`:
  tokens are gathered into fixed-size per-expert blocks and multiplied as dense batched
  matmuls (gather-only forward and backward, no scatter)
- `fast_conv.py` — Gated DeltaNet causal short convolution as shift-and-add
- `gdn_fast.py` — Gated DeltaNet chunked scan with a Neumann-series inverse and unrolled scan
- `te_gmm.py` — dropless grouped GEMM via NVIDIA Transformer Engine (exact; NVIDIA container only)
- model shape study: wider/fewer layers with 64×512 experts gave +30% TFLOP/s at the same budget

The same code also runs on TPU through native megablox kernels (`colab/tpu_smoke.sh`).

## Stage 2: Lewi Surge Preview 3.45B-A0.5B (code-first)

| | |
|---|---|
| Growth | from stage 1 without restarting: experts split 64 → 128, one more block of layers (12 → 16) |
| Shape | d_model 1024, 16 layers (4 × [3 Gated DeltaNet + 1 full attention]), 128 experts, top-6 + 1 shared |
| Parameters | ~3.45B total / ~0.5B active |
| Pretraining | ~1.6B tokens of curated code (with FIM), instructions and textbooks, 1× H200 |
| Annealing | ~1.4B tokens, more than half conversations, LR decayed to zero |
| Fine-tuning | chat SFT on conversations up to 16k tokens |

Base model before → after code-focused annealing (greedy decoding, standard prompts, local runs):

| Benchmark | Metric | Before | After |
|---|---|---|---|
| HumanEval | pass@1 | 18.9% | 25.0% |
| MBPP (sanitized) | pass@1 | 23.7% | 29.6% |
| HellaSwag | acc_norm | 41.9% | 43.4% |
| ARC-Challenge | acc_norm | 26.5% | 27.4% |

Only 4 of 16 layers keep a KV cache (2 KV heads each), about 4 KB per token, so a 128k-token
context fits in roughly half a gigabyte.

## Stage 3 (running): 128k context + SFT on TPU v5e-8

Kaggle TPU v5e-8, single session (`kaggle/long/`):

1. convert the HF checkpoint to MaxText, AOT-compile every phase for `v5e-8` first
   (catches HBM OOM before a TPU-hour is spent)
2. **long-context stage at 131,072 tokens** — context parallelism across all 8 chips
   (`ici_context_parallelism=8`, `per_device_batch_size=0.125`, load balancing off for GDN),
   vocab tiling, flash attention; long data from PG-19, code repositories, SWE agent traces,
   long educational/math documents (ProLong mix)
3. **SFT** at 16k on 382k conversations (`tools/build_sft3.py`), completion-only loss
4. export bf16 params after each phase; only complete Orbax checkpoints are ever loaded

Lessons that cost real TPU time are written down in the scripts as comments: a fresh `run_name`
per session (MaxText silently prefers the run-dir checkpoint over `load_parameters_path`),
synchronous checkpoints for SFT, Grain iterator exhaustion, and Kaggle image package conflicts.

## Repository layout

- `configs/` — MaxText configs per stage
- `modal/` — GPU performance patches and benchmarks; `stage2.py` runs SFT on Modal
- `lightning/` — training orchestrators for stage 1, growth and annealing (auto-restart on
  preemption, checkpoint resume, budget-aware step count), checkpoint pulls, Hugging Face export
- `kaggle/` — data and TPU runs on Kaggle:
  `tokenize/`, `tok2/`, `tok3/` tokenize the data mixes on CPU sessions (code with FIM,
  uk/es/pt/fr/de, math, SVG/design, conversations, long documents, SWE traces);
  `stage2/` and `long/` are TPU training notebooks
- `tokenizer/` — vocabulary analysis and pruning
- `colab/` — TPU smoke tests
- `tools/` — benchmarks (`bench.py`), SFT data builder, sampling and a small local chat UI
- `docs/data.md` — datasets and licences

## Roadmap

1. **Now** — 128k context + second SFT on TPU (stage 3 above)
2. Reasoning and tool-use fine-tuning, reinforcement learning on code with unit-test rewards
3. **Lewi Surge 24B2A** — the full-size model (24B total / 2B active), grown from the preview
4. Vision and audio through trained projectors

Weights of each stage will be released openly on Hugging Face.

Built by one person with the help of AI coding assistants. Project page:
<https://politechainsaw.com/lewi/> · support the GPU hours: <https://ko-fi.com/politechainsaw>

## License

Apache-2.0
