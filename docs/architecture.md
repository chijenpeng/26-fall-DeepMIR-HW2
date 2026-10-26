# Module guide

This is an implementation scaffold with verified smoke paths, not a completed assignment.
The author paused further training and experiments on 2026-10-26.

## Flow

```text
POP909/training
  -> music.py + tokens.py
  -> prepare.py -> per-song NPZ files + provenance manifest
  -> data.py -> song split + aligned complete-bar windows
  -> models.py + train.py -> local checkpoints
  -> generate.py -> MIDI + token sequences + experiment metadata
  -> evaluate.py -> metric CSV + summary JSON
  -> render.py -> WAV + rendering metadata
  -> export.py -> inference-only checkpoint
```

## Responsibilities

All module paths below are relative to `src/deepmir_hw2/`.

| Module | Responsibility | Boundary / design decision |
|---|---|---|
| `music.py` | Parse annotated beats/downbeats; map seconds to fractional beats and back; preserve prompts when combining tracks | Annotated bar grid is authoritative; input MIDI time signatures may disagree |
| `tokens.py` | Define the 524-token vocabulary; encode/decode bar events; validate note grammar; transpose pitches | 12 positions/beat, explicit bar lengths, normalized tempo, quantized velocity/duration |
| `prepare.py` | Read training songs, produce merged/basic and aligned melody/piano tokens, write per-song archives and source hashes | Training directory only; fail on malformed songs rather than silently skip |
| `data.py` | Load prepared songs, split by song, sample complete-bar windows, apply paired transposition, pad batches | Overlapping windows do not cross the development split; encoder inputs are not piano targets |
| `models.py` | Construct randomly initialized GPT-2 and T5 models from local configs | No pretrained-weight download; causal basic model and full-melody conditional bonus model |
| `train.py` | Token-weighted cross-entropy, AdamW, warmup/cosine schedule, AMP, gradient accumulation/checkpointing, validation, atomic checkpoint retention and resume | Resume preserves config/data/optimizer/RNG; a new all-song stage uses weights-only initialization |
| `generate.py` | Grammar-constrained top-k/top-p sampling; cached inference; exact requested bar count; MIDI and metadata output | Preserve prompt notes; basic context can slide at complete bars; report note-budget limits |
| `reference.py` | Choose seeded, matched-length real-data windows from distinct songs | Basic comparison uses 32-bar windows with four annotated beats per bar |
| `evaluate.py` | MusDr H1/H4/GS and wrapper around the unmodified TA bonus evaluator | Check expected outputs before evaluation; record evaluator checksum; no score fabrication |
| `render.py` | Call FluidSynth with a supplied SoundFont; validate nonempty/non-silent PCM WAV | External executable and SoundFont required; code exists, actual synthesis is not yet verified |
| `export.py` | Strip optimizer and RNG state from checkpoints | Export includes weights, vocabulary, architecture and step; cannot exactly resume training |
| `utils.py` | JSON I/O, atomic writes, fingerprints, seeds and device selection | Shared infrastructure only |
| `cli.py`, `__main__.py` | `deepmir` and `python -m deepmir_hw2` entry points | Separate explicit commands; no automatic training on import |
| `__init__.py` | Package marker | No startup side effects |

## Configurations and supporting files

- `configs/basic.json`: 8-layer causal model, width 512, 8 heads, 4,096-token context;
  27,585,536 parameters.
- `configs/bonus.json`: 4 encoder / 6 decoder layers, width 512, 8 heads;
  38,031,872 parameters.
- `configs/smoke_*.json`: deliberately tiny CPU execution checks, not quality baselines.
- `scripts/bootstrap.sh`: project-local Linux Python/CUDA environment setup, no sudo.
- `scripts/reproduce_all.sh`: explicit stage commands; running it without a stage prints help.
- `requirements.txt`: pinned inference packages; experiment-only packages are separate.
- `requirements-experiments.txt`: development and evaluation dependencies.
- `.github/workflows/ci.yml`: CPU-only synthetic tests and static checks; no POP909 or server GPU jobs.

## Verification already performed

The existing 11 tests passed locally and on both servers before formal training was paused.
They cover variable tempo/meter, preserved prompts after MIDI serialization, paired
augmentation, song-level splits, exact CPU training resume for both architectures,
deterministic constrained generation, causal cache rollover and known MusDr metric values.

Full-data preprocessing completed for 896 songs. Tiny CPU checkpoints produced one
32-bar unconditional MIDI, all three 8+24-bar continuations and all ten accompaniments.
Original prompt note pitch, velocity, onset and offset values were checked after writing
the conditional outputs. Both evaluation paths executed successfully. A basic inference
export was loaded and used in a small generation check.

Both production-size models completed only five optimizer steps on GPU. These runs
verify executable code and initial memory use, not convergence, musical quality or
the worst-case memory footprint. No formal training job remains active.

## Limitations to keep visible

- Development training uses 851 songs and 45 validation songs. The assignment's
  full-896-song training requirement is satisfied only after the separate final stage runs.
- Generation imposes a note cap and allows recent-bar context rollover. These affect
  output behavior and must be described in the report.
- Basic GS adapts the onset vector to 48 positions inside a four-beat bar. Raw scores
  must not be compared to evaluations using a different grid/denominator without explanation.
- No final checkpoints, complete three-setting experiment, listening selection, verified
  WAV collection, SI implementation or report currently exists.
