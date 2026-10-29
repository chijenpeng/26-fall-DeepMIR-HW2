# HW2 experiment log

## Representation change (2026-10-28)

The first implementation encoded every time value as a variable-length base-128
microsecond integer (`More_*/End_*` digits). With its final GPT-2 checkpoint, every
20-piece Basic A batch failed:

- Some samples produced durations of about 1.96e18 µs, which cannot be written as MIDI.
- Dense pieces ran out of context. The longest training window was already 25,451 tokens per 32 bars.

The codec was replaced by REMI on the annotated beat grid (`src/deepmir_hw2/remi.py`):

| | microsecond codec | REMI |
|---|---|---|
| tokens per note | ~10+ | 4 (`Pos Pitch Vel Dur`, `Pos` shared per onset) |
| 32-bar window | max 25,451 | median 2,431, p99 3,900, max 7,652 |
| vocabulary | open integers | 430 tokens |
| Basic s/step | ~5.6 | ~1.9 (A4500), ~1.4 (4070 SUPER) |

Tokenizer checks:

- All 896 songs tokenize.
- On 895 songs, every decoded onset is within one 16th note of the original. Song 117 has 2 notes off by up to 0.35 s.
- Only 0.04% of notes are longer than 32 sixteenths and get shortened.
- Bonus round trip: the ground-truth `piano.mid` was encoded to REMI and decoded back, then scored with `eval_bonus.py`. Result: CA 0.750 (ceiling 0.7625), OA pitch 1.00, velocity 1.00, duration 0.96, IOI 0.999.

## Real-data reference (896 songs, REMI vocabulary)

H1 2.123, H4 2.626, GS 0.877 (`results/real.summary.json`).

GS is normalized by the number of `Pos` tokens (32 here), so it is not comparable with
values computed on other grids (the old 16-per-beat grid gave 0.968).

## Training-length study

The models were first trained with long cosine schedules: Basic and no-chord for 40k
steps, Bonus for 80k. Held-out loss is the token cross-entropy on the 13 prompt songs,
none of which are used for training:

- 3 Basic B prompts, used for the Basic models
- 10 Bonus prompts (melody → piano), used for the Bonus model

| step | Basic | Basic no-chord | | Bonus step | Bonus |
|---|---|---|---|---|---|
| 2k | 1.449 | 1.442 | | 4k | 1.156 |
| 4k | **1.315** | – | | 12k | **1.076** |
| 6k | 1.401 | 1.392 | | 20k | 1.258 |
| 8k | 1.767 | – | | 28k | 1.490 |
| 10k | 1.966 | 1.879 | | 36k | 1.674 |
| 12k | 2.179 | – | | 44k | 1.798 |
| 14k | 2.412 | 2.341 | | 52k | 2.001 |
| 16k | 2.424 | – | | 60k | 2.138 |
| | | 2.557 (18k), 2.850 (22k) | | 68k | 2.240 |

Training loss at the same time was 0.10–0.18, so the models were memorizing the training songs.

The runs were then restarted with short schedules: 6k steps for Basic and no-chord,
14k for Bonus, keeping the same seed and hyper-parameters.

| model | short schedule | best long-schedule checkpoint | selected |
|---|---|---|---|
| Basic | 3k 1.358, 4k 1.323, **5k 1.308**, 6k 1.322 | 4k 1.315 | short 5k |
| Basic no-chord | 3k 1.354, 4k 1.319, **5k 1.301**, 6k 1.305 | 6k 1.392 | short 5k |
| Bonus | 6k 1.125, 8k 1.097, 10k 1.095, 12k 1.110, 14k 1.141 | **12k 1.076** | long 12k |

Selected checkpoints (`models/`, SHA-256):

- `basic.pt` `a1b60159…` (step 5000)
- `basic-no-chord.pt` `8be10385…` (step 5000)
- `bonus.pt` `84db8825…` (step 12000 of the 80k schedule)

## Early generation checks (step-2000 Basic, 3 pieces per setting)

All pieces had exactly 32 bars and no capped bars.

| setting | H4 | GS |
|---|---|---|
| A: top-k 20, T 0.9 | 2.56 | 0.91 |
| B: top-p 0.9, T 1.0 | 2.54 | 0.98 |
| C: top-k 50, T 1.2 | 2.74 | 0.82 |
| A, no-chord model | 2.44 | 0.94 |

Bonus checkpoints, sampled with top-k 20 and T 0.9:

| step | CA | OA pitch | OA duration |
|---|---|---|---|
| 1k | 0.08 | 0.79 | 0.58 |
| 4k | 0.14 | 0.85 | 0.69 |

## Final outputs

Final outputs are produced by `scripts/final_outputs.sh` with the selected checkpoints.
Results: see `results/`.
