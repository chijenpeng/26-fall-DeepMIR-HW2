# Training launch record

Formal work resumed by the author on 2026-10-26 (Asia/Taipei). This is an execution
record, not final music-quality results. Original-data preprocessing completed on
both servers; development training is active. All-song final stages, listening
selection, assignment metrics and final report remain pending.

## Source data and reproducibility

- Base: merged PR #1, commit `41fb7156365bc9fe2ca640db0c16c2c157df5d8d`.
- Data schema: 2; all 896 source songs; vocabulary: 1,195 tokens.
- Both prepared manifests have fingerprint
  `ebbdd016086587cd22c8e8bc178679a44176c0c7a720febe1af3323ca9a78949`.
- Context: 25,451 tokens for 32-bar basic windows; 13,403 for 16-bar bonus
  encoder/decoder windows. No events or bars were shortened to fit memory.
- Development split: seed 42, 851 training / 45 validation songs. Both basic
  candidates share exactly the same split, initialization and sampling seed.
- Python 3.12.15, Torch 2.8.0+cu128, Transformers 4.57.1. Both servers passed all
  28 pre-launch synthetic tests. Updated attention regressions bring the suite to 31.
- Servers lack Git; tracked files are synchronized with rsync. Run config,
  tokenizer, split, environment, corpus fingerprint and restartable checkpoint
  are saved in each run. Launch command/PID records are in `runs/launch/`;
  the final Bonus launch also records SHA-256 hashes of every Python source file.
- CUDA visibility explicitly maps the authorized physical GPU to process device
  `cuda:0`. Loopy physical GPU 0 is excluded.

## GPU allocation

| Host / physical GPU | Model | Config | Output | Step limit |
|---|---|---|---|---|
| emma / 0, RTX 4070 SUPER 12GB | Basic GPT-2, 38,862,848 parameters | basic.json, LR 3e-4 | runs/basic | 30,000 |
| emma / 1, RTX 4070 SUPER 12GB | Same basic model, LR comparison | basic-low-lr.json, LR 2e-4 | runs/basic-low-lr | 30,000 |
| loopy / 1, RTX A4500 20GB | Bonus Option B T5, 38,375,424 parameters | bonus.json | runs/bonus | 12,000 |

All use one complete window per microbatch and 16 accumulation steps, BF16,
gradient checkpointing and token-weighted loss. Basic originally proposed 2 x 8;
1 x 16 keeps its effective batch at 16 while giving the 12GB cards more headroom.
Training processes detach from SSH and write unbuffered logs under `logs/`.

The server copies under `runs/launch/basic.json` and `runs/launch/bonus.json` are
snapshots of the respective tracked configs. Actual entry points from
`~/deepmir-hw2` are:

```bash
# emma, separate processes / logs
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m deepmir_hw2 train --config runs/launch/basic.json \
  --output runs/basic
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m deepmir_hw2 train --config configs/basic-low-lr.json \
  --output runs/basic-low-lr

# loopy: physical GPU 1 only
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m deepmir_hw2 train --config runs/launch/bonus.json \
  --output runs/bonus
```

Basic's first ten steps used `--stop-after 10` with the full 30,000-step schedule,
then resumed the same run using `--resume runs/basic`. The checkpoint loaded and
training continued beyond step 300. Basic sampled training loss declined from
7.08 in the initial block to 3.83 at step 300. The lower-LR candidate was at 3.92
at the same step. These are training losses during warmup; model selection awaits
comparable validation and listening, not this early difference.

Initial basic throughput is approximately 3.1–3.2 seconds/optimizer step. At that
rate 30,000 steps take roughly 26–27 hours, excluding substantial interruptions.
Initial Bonus profiling measured 13.5 seconds/step. The Bonus cap was changed to
12,000 with 500 warmup steps to leave time before Thursday for all-song final
training and deliverables. Final query-bucket reuse is a further compute
optimization; its sustained training speed still needs measurement.

## Capacity checks and attention correction

Basic's longest source window, song 615 / first 32 bars / 25,449 payload tokens,
passed a BF16 forward/backward/AdamW update on a 12GB 4070 SUPER: 7,089 MiB peak
allocated and 4.60 seconds including construction. Its first ten actual optimizer
steps also passed, saved a checkpoint and resumed successfully. Sustained basic
training has reached about 7,537 MiB peak allocated; CUDA reserved memory is higher.

Native T5 failed on song 615, bars 15–30, with 2,627 melody / 12,541 piano payload
tokens: softmax attempted another 4.69 GiB on the 20GB A4500. This was a capacity
probe, not an omitted training example. The model now uses full-key attention in
query blocks, recomputed during backward, with shared integer relative-position
bucket IDs. Contiguous additive bias enables CUDA's efficient SDPA implementation.
All supplied events and permitted attention keys remain present.

The final implementation/config passed three distinct extreme source windows:

| Song / start bar (zero-based) | Melody / piano tokens | Peak allocated MiB | Seconds per one-window optimizer update |
|---|---|---|---|
| 791 / 36 | 8,872 / 3,595 | 2,494 | 5.15 |
| 111 / 57 | 1,018 / 13,401 | 4,640 | 13.16 |
| 615 / 15 | 2,627 / 12,541 | 4,355 | 11.98 |

Measured on the A4500 with `scripts/benchmark_gpu.py`, BF16, checkpointing, one
window/microbatch. These are capacity extremes, not average training-step times;
an actual optimizer step accumulates sixteen windows. No benchmark weights are
saved or used as final models. Superseded 10/26-step Bonus pilot checkpoints were
deleted; logs retain capacity/timing evidence. Formal Bonus restarted from random
initialization with the final 12,000-step config.

CPU tests compare full logits and every parameter gradient against unmodified T5,
with and without outer checkpointing. Cached decoding is checked against eager
full-context logits across multiple prefix sizes, including histories longer than
the query block size. Cache key lengths are checked so block size cannot silently
turn into a sliding-window cache.

## Next selection and final stages

Compare validation curves at the same steps and inspect generated development
samples before choosing the basic LR. Preserve best and latest pointers; regular
checkpoints retain optimizer, scaler and RNG for restarts. Final training must use
`--all-training` on all 896 songs after development selection, then Basic A/B must
share the same exported final checkpoint. Bonus inference remains melody-only.

Generation still needs all 60 Basic A comparisons, nine continuations and ten
accompaniments, followed by required metrics/listening selection, WAV validation,
final exports, report and submission audit. Current jobs do not certify any of
those artifacts as complete.
