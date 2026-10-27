# DeepMIR HW2: symbolic music generation

Chi-Jen Peng (R14725022)

**Status:** formal work resumed on 2026-10-26. All 896 songs were prepared on both
servers with matching fingerprints. The two basic development runs were stopped
after validation overfitting; their best checkpoints produced development listening
samples. Bonus training continues. See [run record](docs/experiments.md) for measured
capacity, commands and limitations. Final all-song models and the report remain pending.

See [module guide](docs/architecture.md), [assignment checklist](docs/requirements.md)
and [source attribution](THIRD_PARTY.md).

## Models and task boundary

Basic A/B share a random-initialized GPT-2 causal Transformer. Bonus uses **Option B:
T5 encoder-decoder / seq2seq**, also trained from scratch. Training inputs are MELODY,
targets are PIANO. Bonus inference reads only melody and its timing metadata. Reference
chords/piano are evaluation material, never extra Bonus inference conditions.

## Setup

Use Python 3.12. Dependencies are separated as in HW1.

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Development/evaluation:
pip install -r requirements-experiments.txt
git clone https://github.com/slSeanWU/MusDr.git third_party/MusDr
git -C third_party/MusDr checkout b034cf86550963cf2021299eafb1001d0fe34853
~~~

On a Linux NVIDIA server, bash scripts/bootstrap.sh installs project-local Python 3.12
and PyTorch 2.8.0/CUDA 12.8 without sudo/system-Python changes. Versions are recorded in
environment.freeze.txt. WAV synthesis additionally requires FluidSynth and a licensed
.sf2 SoundFont. Development samples use TimGM6mb from Debian, with source/license
and SHA-256 recorded in [THIRD_PARTY.md](THIRD_PARTY.md).

Install the non-pip renderer before requesting listening samples:

~~~bash
# Debian / Ubuntu
sudo apt-get update
sudo apt-get install fluidsynth timgm6mb-soundfont
# macOS alternative
brew install fluid-synth
~~~

See [FluidSynth installation](https://www.fluidsynth.org/download/) for platform details.
The documented Debian package supplies an example SoundFont at
/usr/share/sounds/sf2/TimGM6mb.sf2 ([package file list](https://packages.debian.org/sid/all/timgm6mb-soundfont/filelist)).
Pass the chosen font explicitly and use the same one for comparisons. It is a rendering
asset, not training data; no font is silently selected by the model.

## Submission inference entry point

With an exported checkpoint supplied, no training or preprocessing is needed:

~~~bash
deepmir generate --checkpoint models/basic.pt --mode unconditional --count 20 --bars 32 \
  --output /absolute/path/listening-samples --soundfont /usr/share/sounds/sf2/TimGM6mb.sf2
deepmir generate --checkpoint models/bonus.pt --mode accompany \
  --prompts POP909/prompts/bonus --output /absolute/path/accompaniment \
  --soundfont /absolute/path/piano.sf2
~~~

This generates MIDI and WAV together. Renderer requirements are checked before
model inference. A WAV failure leaves the batch explicitly incomplete; it cannot
be averaged or submitted as a completed run. Exported final checkpoints are still pending.

## Data and source preservation

Supply POP909/training/{song}/{song}.mid, beat_midi.txt and chord_midi.txt; continuation
prompts under POP909/prompts/basic-b; melody prompts under POP909/prompts/bonus. The
provided TA scripts remain under HW2_scripts/.

Prepare from the original training data:

~~~bash
deepmir prepare --dataset POP909/training --output data/prepared
~~~

The command requires exactly 896 source song folders by default. Missing songs
cannot silently turn an incomplete dataset into an all-training run.

The current event codec retains annotated pulse durations, supplied meter, full MIDI
pitches 0–127, original positive velocities 1–127, controllers/pedal, pitch bends and
microsecond note onsets/durations. Rounding from floating-point seconds is at most one
microsecond. No duration binning/clipping or 12/16-position timing quantization is used
in the model. Duplicates, long release tails and source pickups are retained.

Each bar stores its pulse count, meter and every pulse duration, followed by timed
note/control/bend/chord events. Numeric fields use canonical variable-length base-128
tokens. Beat-count/meter/chord vocabulary and instrument programs come from the source
training files. Basic includes provided chord events, transposed jointly with notes;
Bonus melody/piano sequences exclude reference chords.

Sources are hashed and all annotated bars, including silence, are retained. When the
last event starts beyond the last annotated downbeat, the final supplied bar width and
beat interval are extended to close that bar. All 896 sources passed preparation.
Events that cannot be assigned are reported, never silently dropped.

## Training

Architecture/optimizer candidates remain unchanged: basic 8 layers; bonus 4 encoder /
6 decoder layers; width 512; 8 heads; AdamW/warmup/cosine, paired ±5-semitone augmentation,
gradient accumulation and BF16. Basic microbatches now contain one window with 16
accumulation steps, preserving the effective batch of 16 on each 12GB GPU. A second
basic run compares learning rate 2e-4 against the initial 3e-4 with the same split.

Context is now derived from the longest complete configured source window using
**max_tokens: "auto"**. Explicitly insufficient capacity fails; training windows are
never shortened and inference history is never discarded. The measured required
contexts are 25,451 tokens (basic) and 13,403 (bonus). Basic's longest window passed
forward/backward/optimizer execution on a 12GB RTX 4070 SUPER. T5 uses query
blocks with checkpointed recomputation, retaining every allowed key and learned
relative position bias. The largest encoder, largest decoder and highest combined
attention-workload windows all passed on the A4500.
See the run record for timing/memory details; old-code smoke measurements do not apply.

Basic development has a 30,000-step limit per learning-rate candidate. Bonus has a
12,000-step limit and 500-step warmup: the measured initial 13.5 seconds/step made
the original 30,000-step candidate exceed the Thursday deadline. This leaves time
for all-song final training and required generation/evaluation. Convergence still
needs validation and listening; a step limit is not a quality claim.

~~~bash
# emma: physical GPUs 0 and 1, separate processes
CUDA_VISIBLE_DEVICES=0 deepmir train --config configs/basic.json --output runs/basic
CUDA_VISIBLE_DEVICES=1 deepmir train --config configs/basic-low-lr.json --output runs/basic-low-lr
# loopy: physical GPU 1 only
CUDA_VISIBLE_DEVICES=1 deepmir train --config configs/bonus.json --output runs/bonus
~~~

A bounded benchmark can use --stop-after 10 while preserving the intended schedule.
Use --resume runs/basic with the original configuration to resume. Atomic checkpoints
retain model/optimizer/scaler/RNG/config/source programs and corpus fingerprint.
SIGINT/SIGTERM requests save after the current optimizer step.

Development uses 851 training / 45 validation songs. The lowest sampled validation
cross-entropy checkpoint is named best; that is not a listening-quality claim.
A separate final stage must use all 896 songs after selecting settings:

~~~bash
CUDA_VISIBLE_DEVICES=0 deepmir train --config configs/basic.json \
  --init-from runs/basic --all-training --steps FINAL_STEPS --output runs/basic-final
CUDA_VISIBLE_DEVICES=1 deepmir train --config configs/bonus.json \
  --init-from runs/bonus --all-training --steps FINAL_STEPS --output runs/bonus-final
~~~

Replace FINAL_STEPS with a choice based on development convergence. All-song final
stages remain pending; the current runs use the development split.

## Inference

~~~bash
deepmir generate --checkpoint runs/basic-final --mode unconditional \
  --count 20 --bars 32 --top-k 20 --temperature 0.9 --seed 2026 --output outputs/basic-a/A
deepmir generate --checkpoint runs/basic-final --mode unconditional \
  --count 20 --bars 32 --top-k 50 --temperature 0.9 --seed 2026 --output outputs/basic-a/B
deepmir generate --checkpoint runs/basic-final --mode unconditional \
  --count 20 --bars 32 --top-k 50 --temperature 1.1 --seed 2026 --output outputs/basic-a/C

deepmir generate --checkpoint runs/basic-final --mode continue \
  --prompts POP909/prompts/basic-b --bars 24 --top-k 20 --temperature 0.9 \
  --seed 2026 --output outputs/basic-b/A
# Repeat continuation with B/C settings.

deepmir generate --checkpoint runs/bonus-final --mode accompany \
  --prompts POP909/prompts/bonus --top-k 50 --temperature 0.9 \
  --seed 2026 --output outputs/bonus
~~~

Unconditional/continuation timing and meter are predicted, with no fixed 100 BPM or
four-pulse bar. Bonus uses supplied melody timing headers. Continuation defaults to
24 new bars; provided prompts are checked for 8/16 annotated bars as appropriate.

Silence, dense bars and duplicate pitches are allowed, with no min/max notes per bar.
Context exhaustion raises a failure rather than claiming a shortened output succeeded.
Grammar masks cover event syntax, monotonic/in-bar positions and legal MIDI fields.

Prompt notes/controllers are copied in seconds. MIDI export rebuilds the clock from
pulse timing plus supplied meter: six annotated pulses inside a supplied 4/4 bar still
span four quarters. No replacement 6/4 signature or arbitrary BPM is invented. Raw clock
metadata may be normalized; physical event timing and annotated bar boundaries are
the preservation targets. SMF tempo is integer microseconds, and note/CC serialization
has at most one output-tick rounding error.

Release tails can extend beyond the final onset bar. BAR_n/BAR_END markers declare the
musical bar span. Metadata records checkpoint hash, seeds, settings, counts and token use.

## Evaluation

Default real reference includes every complete training song and every supplied meter.
Optional song-count/crop flags request subsets explicitly.

~~~bash
deepmir reference --data data/prepared --output outputs/real
deepmir evaluate --task basic --input outputs/real --output outputs/metrics/real.csv
deepmir evaluate --task basic --input outputs/basic-a/A --output outputs/metrics/A.csv
deepmir evaluate --task basic --input outputs/basic-a/B --output outputs/metrics/B.csv
deepmir evaluate --task basic --input outputs/basic-a/C --output outputs/metrics/C.csv
deepmir evaluate --task bonus --input outputs/bonus --prompts POP909/prompts/bonus \
  --output outputs/metrics/bonus.csv
~~~

H1/H4/GS call the **unmodified TA functions**. A separate evaluation view uses the TA's
16-position-per-annotated-beat projection. The shared position vocabulary retains the
TA's eight-beat range, extended only for longer supplied bars. There is no custom
four-beat GS denominator. Both real/generated data use the same mapping. TA pair
sampling is seeded.

Bonus uses the unchanged TA evaluator with completeness checks. Undefined metrics
such as all-rest pitch entropy remain undefined, with valid-piece counts/nulls explicitly
reported. Failed pieces are not silently removed to improve averages.

## WAV, reproduction and delivery

~~~bash
deepmir render --input outputs/basic-a/B --output outputs/basic-a/B \
  --soundfont /absolute/path/piano.sf2
deepmir export --checkpoint runs/basic-final --output models/basic.pt
deepmir export --checkpoint runs/bonus-final --output models/bonus.pt
~~~

Rendering is available both through generate --soundfont and the separate render
command. The external process/PCM/failure handling has synthetic tests; a real
The TimGM6mb SoundFont was validated on the development samples; final listening
selection and all required output batches remain pending.
Submit 20 selected basic-A pairs, 9 continuation pairs (mark each prompt's best),
10 bonus pairs, source/inference weights/README/requirements and the 16:9 report.

The explicit stages in scripts/reproduce_all.sh do nothing without a stage argument.
Training requires CUDA_VISIBLE_DEVICES; the final stage requires HW2_FINAL_STEPS;
rendering requires HW2_SOUNDFONT and the selected basic setting. Commands are planned
reproduction instructions, not completed experiments.

## Verification

~~~bash
ruff check src tests
ruff format --check src tests
pytest -q
~~~

Synthetic CPU tests cover microsecond timing/full MIDI values, pedal, long tails,
silence/dense/duplicate notes, consistent exported bars, full-context errors, actual
GPT-2/T5 cache agreement, complete sampled windows, joint chord/note augmentation,
all-meter reference inclusion, exact CPU resume, undefined-metric rejection and
original TA metrics. Terminal downbeat notes and same-tick retriggers have regressions.

The Bonus access test deliberately puts invalid chord/piano files beside a valid
melody and verifies neither is consumed. Full POP909 preparation, GPU memory/performance,
musical quality, formal experiments, WAV collection and report remain pending.
The read-only source audit checked all 896 training songs and successfully encoded
the three 8-bar continuation prompts and ten 16-bar melody prompts. It did not
create prepared training arrays, checkpoints or model-generated samples.
