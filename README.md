# DeepMIR HW2: symbolic music generation

Chi-Jen Peng (R14725022)

- **Basic A/B:** GPT-2 (12 layers, width 512, 8 heads, 40M parameters) trained from
  scratch on all 896 POP909 training songs, REMI tokens with chords. One checkpoint
  serves unconditional 32-bar generation (A) and 8+24-bar prompt continuation (B).
- **Bonus (Option B, seq2seq):** T5 encoder-decoder (6+6 layers, width 512, 44M
  parameters) trained from scratch; encoder reads the MELODY track, decoder writes the
  PIANO track. Inference reads only the melody prompt and its beat annotation.
- **Ablation:** the same GPT-2 trained without chord tokens (`configs/basic-no-chord.json`).

## Setup

**Python 3.12 is required** (`pyproject.toml` pins `>=3.12,<3.13`). Install PyTorch for your
platform first if you need a specific CUDA build.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

WAV rendering needs FluidSynth (not pip-installable) and a General MIDI SoundFont:

```bash
sudo apt-get install fluidsynth timgm6mb-soundfont   # Debian/Ubuntu
brew install fluid-synth                             # macOS (download a .sf2 separately)
```

The submitted WAVs use TimGM6mb (`/usr/share/sounds/sf2/TimGM6mb.sf2` on Debian).

## Data

The course POP909 data is not included. Put it at the repository root as `POP909/` (the
default paths below), or pass its location with `--prompts` / `--dataset`:

```
POP909/
  training/<id>/{<id>.mid, beat_midi.txt, chord_midi.txt}        896 songs, training only
  prompts/basic-b/<id>/{<id>.mid, beat_midi.txt, chord_midi.txt}  Basic B prompts (001-003)
  prompts/bonus/<id>/{melody.mid, beat_midi.txt, ...}              Bonus prompts (001-010)
```

Inference needs only `prompts/`; Basic A needs no data at all. Other files in a prompt
folder (e.g. `piano.mid`) are ignored by inference.

## Checkpoints and generated results

The checkpoints and the submitted listening samples are too large for git and are attached
to the release [`submission-v1`](https://github.com/chijenpeng/26-fall-DeepMIR-HW2/releases/tag/submission-v1):

| File | Content |
|---|---|
| `basic.pt` | GPT-2 for Basic A and B (step 5,000) |
| `bonus.pt` | T5 for the Bonus (step 12,000) |
| `basic-no-chord.pt` | ablation without chord tokens (not needed for the submitted results) |
| `generated.zip` | Basic A (20 pieces), Basic B (3 prompts × 3 settings), Bonus (10 prompts), MIDI + WAV |
| `r14725022_report.pdf` | report |
| `SHA256SUMS.txt` | checksums |

```bash
mkdir -p models
URL=https://github.com/chijenpeng/26-fall-DeepMIR-HW2/releases/download/submission-v1
curl -L -o models/basic.pt $URL/basic.pt
curl -L -o models/bonus.pt $URL/bonus.pt
```

## Inference

Checkpoints: `models/basic.pt` (Basic A/B), `models/bonus.pt` (Bonus). Every command
writes `NNN.mid`, `NNN.wav` (when `--soundfont` is given) and `generation.json`
(seeds, settings, tokens) into `--output`. Drop `--soundfont` to write MIDI only.

`--prompts` is a folder with one subfolder per prompt (see [Data](#data)); output files are
named after the prompt folders.

```bash
SF=/usr/share/sounds/sf2/TimGM6mb.sf2

# Basic A: 20 unconditional 32-bar pieces
python -m deepmir_hw2 generate --checkpoint models/basic.pt --mode unconditional \
  --count 20 --bars 32 --top-k 20 --temperature 0.9 --output out/basic-a --soundfont $SF

# Basic B: continue each 8-bar prompt by 24 bars (prompt kept verbatim)
python -m deepmir_hw2 generate --checkpoint models/basic.pt --mode continue \
  --prompts POP909/prompts/basic-b --bars 24 --top-k 20 --temperature 0.9 \
  --output out/basic-b --soundfont $SF

# Bonus: piano accompaniment for each 16-bar melody prompt (MELODY + PIANO tracks)
python -m deepmir_hw2 generate --checkpoint models/bonus.pt --mode accompany \
  --prompts POP909/prompts/bonus --top-k 20 --temperature 0.9 \
  --output out/bonus --soundfont $SF
```

Sampling options: `--temperature`, `--top-k` (0 = off), `--top-p` (1.0 = off),
`--seed` (piece *i* uses `seed + i`). Use `--device cpu` to force CPU; a GPU is
optional for inference.

**Verified:** in a fresh clone with Python 3.12, `pip install -r requirements.txt` and the
checkpoints downloaded from the release, these commands on the CPU reproduce the submitted
`basic-a/001.mid` and `bonus/001.mid` note for note (the submitted files were generated on a GPU).

The three compared inference settings are A = top-k 20, T 0.9; B = top-p 0.9, T 1.0;
C = top-k 50, T 1.2 (seeds 2026, 2027, ...). `scripts/final_outputs.sh` regenerates
every submitted batch with them:

```bash
bash scripts/final_outputs.sh basic-a   # 3 settings x 20 unconditional pieces
bash scripts/final_outputs.sh basic-b   # 3 settings x 3 continuations
bash scripts/final_outputs.sh bonus     # 3 settings x 10 accompaniments
bash scripts/final_outputs.sh no-chord  # ablation model, setting A
SF=$SF bash scripts/final_outputs.sh render
bash scripts/final_outputs.sh metrics
```

## Training and evaluation (reproduction)

```bash
pip install -r requirements-experiments.txt
python -m deepmir_hw2 prepare --dataset POP909/training --output data/remi.json
python -m deepmir_hw2 train --config configs/basic.json --output runs/basic
python -m deepmir_hw2 train --config configs/bonus.json --output runs/bonus
```

`train` saves inference weights `runs/<name>/model-<step>.pt` periodically and resumes
from `runs/<name>/state.pt` when restarted. Metrics use the TA scripts in `HW2_scripts/`
(Basic metrics also need MusDr):

```bash
git clone https://github.com/slSeanWU/MusDr.git third_party/MusDr
python -m deepmir_hw2 evaluate real  --input data/remi.json --output metrics/real.csv
python -m deepmir_hw2 evaluate basic --input out/basic-a --output metrics/basic-a.csv
python -m deepmir_hw2 evaluate bonus --input out/bonus --prompts POP909/prompts/bonus \
  --output metrics/bonus.csv
```

## Representation

REMI-style tokens on the beat grid annotated in `beat_midi.txt` (bars start at
downbeats; vocabulary of 430 tokens):

```
Bar Beat_<1-8> Tempo_<30..210 step 3> | Pos_<0..31> [Chord_<root>:<quality>] Pitch_<0-127> Vel_<0-31> Dur_<1-32> ...
```

- `Pos` is a 16th-note position within the bar (4 per beat); `Beat_n` is the number of
  beats in the bar, so 3/4, 6-beat and 2-beat bars are represented.
- `Tempo` is the bar's average tempo from the beat annotation; generated pieces derive
  their timing from these tokens. Continuations start at the prompt's last downbeat;
  accompaniments are placed on the prompt's own beat grid.
- `Vel` is velocity // 4; `Dur` is in 16th notes (0.04% of notes exceed 32 and are
  shortened). Chords (Basic only) come from `chord_midi.txt`, simplified to 12 roots ×
  11 qualities, emitted when the label changes.
- Basic merges MELODY, BRIDGE and PIANO; Bonus encodes MELODY (with Beat/Tempo) and
  PIANO (Bar/Pos/Pitch/Vel/Dur) separately.
- Training uses random 32-bar (Basic) or 16-bar (Bonus) windows and random
  transposition of ±5 semitones (pitches and chord roots together).

Decoding applies a REMI grammar mask (e.g. `Pitch → Vel → Dur`, positions strictly
increasing inside the bar and below its beat count) and ends after exactly the
requested number of bars. Sampling uses temperature / top-k / top-p on the legal tokens.

## Layout

```
src/deepmir_hw2/   remi.py (tokens), prepare.py, data.py, models.py, train.py,
                   generate.py, evaluate.py, render.py, cli.py
configs/           training configurations
scripts/           final_outputs.sh (all submitted batches, WAV rendering, metrics)
HW2_scripts/       TA-provided tokenizer example and metric scripts (unmodified)
notes/             assignment PDF and experiment log (experiments.md)
results/           metric summaries used in the report
slides/            16:9 report (XeLaTeX beamer; `latexmk` writes submission/)
submission/        r14725022_report.pdf
tests/             CPU unit tests (pytest)
```

Sources and citations: [THIRD_PARTY.md](THIRD_PARTY.md).
