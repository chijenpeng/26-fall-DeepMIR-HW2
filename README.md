# DeepMIR HW2: symbolic music generation

Chi-Jen Peng (R14725022)

**Status (2026-10-26):** implementation and smoke verification only. Formal training
and experiments are paused at the author's request. No final model, report or
submission samples are claimed. See [module guide](docs/architecture.md) and
[assignment checklist](docs/requirements.md) for exact coverage and remaining work.

This project trains two Transformer models from random initialization on POP909:

- A causal GPT-2 model for 32-bar unconditional generation and 8+24-bar continuation.
- A T5 encoder-decoder model for 16-bar melody-conditioned piano accompaniment.

No pretrained model weights or external training datasets are used. Hugging Face model
constructors receive locally defined configurations, never a pretrained model name.

## Setup

Use Python 3.12, the version verified on both training servers.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development and evaluation:

```bash
pip install -r requirements-experiments.txt
git clone https://github.com/slSeanWU/MusDr.git third_party/MusDr
git -C third_party/MusDr checkout b034cf86550963cf2021299eafb1001d0fe34853
```

`bash scripts/bootstrap.sh` creates a project environment with Python 3.12 and
PyTorch 2.8.0/CUDA 12.8 on a Linux NVIDIA server. It does not require sudo or alter
system Python. Resolved versions are recorded in `environment.freeze.txt`.

For WAV synthesis, install the FluidSynth executable and supply a licensed `.sf2`
SoundFont explicitly. The SoundFont is used for rendering, not model training.

## Data and representation

Extract the supplied archives so these paths exist:

```text
POP909/training/{song}/{song}.mid
POP909/training/{song}/beat_midi.txt
POP909/prompts/basic-b/{song}/{song}.mid
POP909/prompts/bonus/{song}/melody.mid
HW2_scripts/eval_bonus.py
```

```bash
deepmir prepare --dataset POP909/training --output data/prepared
```

All 896 songs are processed strictly: malformed/missing tracks or annotations stop
preprocessing rather than silently reducing the dataset. Prompt files are not read.
The manifest records hashes of source MIDIs and beat annotations. Partial preprocessing
does not produce a valid manifest; use a fresh output directory when retrying.

The REMI-like grammar is:

```text
Bar Beats_n (Pos_n Pitch_n Vel_n Dur_n)*
```

There are 12 positions per annotated beat, 32 velocity bins, pitches 21–108,
1–16 beats per bar, and durations up to 16 beats. This gives 524 tokens including
PAD/BOS/EOS. Durations use integrated annotated beat positions, including tempo
changes. Notes are sorted by onset and pitch. Tempo is normalized out of the model;
unconditional output defaults to 100 BPM, while conditional outputs preserve the
prompt's MIDI tempo map and use the annotated timing grid for generated notes.
The decoded performance is quantized; it is not a lossless audio/performance codec.

Bar boundaries follow `beat_midi.txt`, because MIDI time signatures can disagree
with the provided annotations. An opening pickup is retained as a short first bar.
The three source tracks are merged for basic training; bonus training uses only
aligned MELODY inputs and PIANO targets, without BRIDGE or prompt reference pianos.

## Training

Start with the CPU integration configurations to validate the installation:

```bash
deepmir train --config configs/smoke_basic.json --output runs/smoke-basic --device cpu
deepmir train --config configs/smoke_bonus.json --output runs/smoke-bonus --device cpu
```

For actual training, restrict visible GPUs before starting the process:

```bash
# loopy: physical GPU 1 only
CUDA_VISIBLE_DEVICES=1 deepmir train --config configs/basic.json --output runs/basic

# emma: physical GPU 0
CUDA_VISIBLE_DEVICES=0 deepmir train --config configs/bonus.json --output runs/bonus
```

Use `--stop-after 10` for a benchmark that retains the intended learning-rate schedule.
Resume using the same configuration and overrides:

```bash
CUDA_VISIBLE_DEVICES=1 deepmir train --config configs/basic.json \
  --output runs/basic --resume runs/basic
```

Checkpoints contain model, AdamW/scaler state, sampling RNG, PyTorch RNG, configuration
and corpus fingerprint. Writes are atomic. SIGINT/SIGTERM requests save after the
current optimizer step. `latest.json` and `best.json` identify the latest and best
validation checkpoints. Retention preserves the best checkpoint plus the configured
number of recent checkpoints. Training logs are written to `metrics.jsonl`.

The default development split is song-level: 851 training / 45 validation songs;
overlapping windows cannot cross the split. Both paired tracks receive the same
pitch augmentation. Windows end at complete bars and stay within the token budget.
Gradients are weighted by the actual number of supervised tokens across accumulation
microbatches. Padding and encoder inputs do not contribute to target loss.

After selecting settings with the development split, make a new final stage using
all 896 training songs, as required by the assignment. Set an appropriate final
step count based on validation-stage convergence:

```bash
CUDA_VISIBLE_DEVICES=1 deepmir train --config configs/basic.json \
  --init-from runs/basic --all-training --steps 2000 --output runs/basic-final
CUDA_VISIBLE_DEVICES=0 deepmir train --config configs/bonus.json \
  --init-from runs/bonus --all-training --steps 2000 --output runs/bonus-final
```

`--init-from` loads weights and starts a new optimizer/schedule. `--resume` restores
the exact previous training state and rejects incompatible configuration changes.
The final-stage count above is an example, not a measured convergence claim.

## Generate MIDI

Use an empty output directory. A checkpoint argument can be an explicit `.pt` file
or a run directory (best validation checkpoint, otherwise latest). Preserve the
explicit checkpoint files used in reported experiments.

```bash
deepmir generate --checkpoint runs/basic-final --mode unconditional \
  --count 20 --bars 32 --top-k 20 --temperature 0.9 --seed 2026 --output outputs/basic-a/A
deepmir generate --checkpoint runs/basic-final --mode unconditional \
  --count 20 --bars 32 --top-k 50 --temperature 0.9 --seed 2026 --output outputs/basic-a/B
deepmir generate --checkpoint runs/basic-final --mode unconditional \
  --count 20 --bars 32 --top-k 50 --temperature 1.1 --seed 2026 --output outputs/basic-a/C

deepmir generate --checkpoint runs/basic-final --mode continue \
  --prompts POP909/prompts/basic-b --bars 24 --top-k 20 --temperature 0.9 \
  --seed 2026 --output outputs/basic-b/A
# Repeat with B/C sampling settings into outputs/basic-b/B and outputs/basic-b/C.

deepmir generate --checkpoint runs/bonus-final --mode accompany \
  --prompts POP909/prompts/bonus --top-k 50 --temperature 0.9 \
  --seed 2026 --output outputs/bonus
```

Conditional inference copies the prompt's original notes, controllers and tempo
map without decoding the original melody from tokens. Continuation appends notes
after the prompt boundary; accompaniment appends a separate named PIANO track.
Generation masks illegal note grammar, out-of-bar positions, backwards onsets and
duplicate pitches at a shared onset. Sampling is applied after grammar masking.

Causal inference uses a KV cache and rebases to recent complete bars when the
context fills. Accompaniment encodes the full melody once and caches decoder states.
A per-bar note limit prevents unbounded generation; any bar reaching the limit is
reported in `generation.json`. The accompaniment limit is reduced as needed to fit
all required bars in the decoder budget. These constraints must be stated in the report.
Each output directory records tokens, sampling settings, seed, checkpoint step,
prompt hashes, note counts and note-limit diagnostics.

## Evaluation

Use matched 32-bar, four-beat real-data windows from distinct training songs:

```bash
deepmir reference --data data/prepared --bars 32 --count 100 --output outputs/real
deepmir evaluate --task basic --input outputs/real --output outputs/metrics/real.csv
deepmir evaluate --task basic --input outputs/basic-a/A --output outputs/metrics/A.csv
deepmir evaluate --task basic --input outputs/basic-a/B --output outputs/metrics/B.csv
deepmir evaluate --task basic --input outputs/basic-a/C --output outputs/metrics/C.csv
deepmir evaluate --task bonus --input outputs/bonus --prompts POP909/prompts/bonus \
  --output outputs/metrics/bonus.csv
```

Basic metrics call MusDr's original pitch histogram, entropy and onset XOR functions.
H1/H4 average across windows and then pieces; GS averages all bar pairs. Empty bars
are counted explicitly. For four-beat comparisons, GS uses the 48 positions inside
the bar, rather than all 192 positions in the variable-meter vocabulary (which would
inflate similarity). The same mapping is used for real and generated music.

Bonus evaluation validates every expected song before invoking the unmodified TA
script. It rejects missing/malformed outputs rather than averaging a successful subset.
Means are pooled by the number of bars, as in the TA script. Results include the
ground-truth CA-ceiling alongside CA and all four OA metrics. Each CSV has a JSON summary.

## WAV rendering and submission

```bash
deepmir render --input outputs/basic-a/B --output outputs/basic-a/B \
  --soundfont /absolute/path/piano.sf2
```

Repeat for all nine continuation results and all ten accompaniment results. WAV
rendering is atomic and verifies nonempty, non-silent signed 16-bit PCM output.
Record the chosen SoundFont and its license; its checksum is saved automatically.

Submit 20 basic-A MIDI/WAV pairs from the selected setting, 9 continuation pairs
(identify the best setting per prompt), and 10 accompaniment pairs. Also submit
source code, inference checkpoints, requirements, README and the 16:9 report.
Exclude training/preprocessed data, environments, caches and unused checkpoints.

Export small inference-only checkpoints before submission:

```bash
deepmir export --checkpoint runs/basic-final --output models/basic.pt
deepmir export --checkpoint runs/bonus-final --output models/bonus.pt
deepmir generate --checkpoint models/basic.pt --mode unconditional \
  --count 1 --bars 32 --output outputs/reproduce
```

These files contain the architecture, vocabulary and weights but no optimizer or
sampling RNG state. They support inference and weights-only initialization, not
exact training resume. Generation records the loaded checkpoint's SHA-256 checksum.

## Reproduction stages

Like HW1, `scripts/reproduce_all.sh` collects the experiment commands in one place.
It prints help without starting any work when run without an argument. The stages
are `prepare`, `train-basic`, `train-bonus`, `final-basic`, `final-bonus`, `generate`,
`evaluate`, and `render A|B|C`. Output directories must be fresh. Set
`CUDA_VISIBLE_DEVICES` explicitly for training, `HW2_FINAL_STEPS` for the final
all-song stage, and `HW2_SOUNDFONT` for rendering. These are planned reproduction
commands; they are not a claim that the full experiments have been executed.

## Verification and attribution

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

Tests cover variable-tempo/meter round trips, prompt preservation after MIDI write,
paired augmentation, song-level splits, exact CPU continuation of both optimizers
and RNG state after resume, deterministic constrained generation and cache rollover.

References to cite in the report:

- POP909: Wang et al., ISMIR 2020 (provided course dataset).
- REMI: Huang and Yang, *Pop Music Transformer*, ACM Multimedia 2020.
- Hugging Face Transformers GPT-2/T5 implementations; architectures instantiated from scratch.
- MusDr: Wu and Yang, *The Jazz Transformer on the Front Line*, ISMIR 2020:
  https://github.com/slSeanWU/MusDr (record the exact installed commit).
- `HW2_scripts/`: provided course tokenization/evaluation examples. The original files remain unchanged.

CPU smoke checkpoints verify execution only; they do not demonstrate musical quality.
