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

## Repository layout

- `configs/` — MaxText configs per stage
- `modal/` — GPU performance patches and benchmarks (run on Modal)
- `lightning/` — stage-1 training orchestrator (auto-restart on preemption, checkpoint
  resume, budget-aware step count), Hugging Face export
- `kaggle/` — tokenization into Megatron mmap shards on Kaggle CPU sessions;
  `tok2/` prepares stage-2 data (code with FIM, uk/es/pt/fr/de, math, SVG/design)
- `tokenizer/` — vocabulary analysis and pruning
- `colab/` — TPU smoke test
- `tools/generate.py` — sampling from an exported checkpoint with `transformers`

## Roadmap

1. **Stage 2** — grow the model (expert splitting + layer stacking) and add code (~30%, with
   FIM), Ukrainian, Spanish, Portuguese, French, German, math and SVG/design data
2. **Stage 3** — ~10B total, longer context
3. **Lewi Surge 24B2A** — distillation from an open teacher, 200k+ context
4. Post-training: SFT, reasoning (GRPO), tool calling, then vision and audio

Weights of each stage will be released openly.

## License

Apache-2.0
