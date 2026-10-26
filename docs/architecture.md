# Module guide

Formal preprocessing/training/experiments are paused. Obsolete prepared data, smoke
checkpoints and sample outputs were deleted locally and on both training servers.

## Task boundary

Basic A/B share a random-initialized GPT-2 causal model. Bonus is **Option B**, using
T5 encoder-decoder / seq2seq: melody input, piano target. At inference it reads only
the melody MIDI and its beat metadata, never reference chords or piano.

## Responsibilities

Paths are relative to src/deepmir_hw2/.

| Module | Responsibility |
|---|---|
| music.py | Read annotations and supplied meter; preserve event seconds; rebuild a MIDI clock consistent with annotated bars |
| tokens.py | Microsecond events, full MIDI pitch/velocity, controllers, bends and basic chord labels; separate TA projection |
| prepare.py | Source-derived schema; aligned basic/melody/piano arrays; MIDI/beat/chord hashes |
| data.py | Song split, complete windows, joint chord/note transposition, padding and context requirement |
| models.py | Random GPT-2/T5, no pretrained weights |
| train.py | Weighted loss, AdamW/AMP/accumulation, sampled validation, atomic checkpoints/resume; reject insufficient context |
| generate.py | Full-history cache; learned basic timing; supplied Bonus timing; legal event syntax without note-count rules |
| reference.py | All-song/all-meter default reference; explicit optional subsets |
| evaluate.py | Unmodified TA metric functions, seeded sampling and coverage reporting |
| render.py | Explicit FluidSynth/SoundFont rendering and PCM checks |
| export.py | Inference weights/config/vocabulary/source programs without optimizer |
| utils.py | Atomic JSON, hashes, seeds and device selection |
| cli.py, __main__.py | Explicit commands; no training on import |

## Source preservation and MIDI clock

The codec stores every annotated pulse duration and the supplied meter at bar start.
Event times/durations use variable-length microsecond integers. Floating-second rounding
is at most one microsecond. MIDI pitch/velocity retain full values, with no long-note
clipping, duplicate suppression or pedal removal. Basic chord labels are transposed
with notes; Bonus sequences exclude reference chord labels.

Annotated pulses are not assumed to be quarters: six pulses in a supplied 4/4 bar
occupy four quarters. Export normalizes clock metadata to retain physical note/CC
timing within one output tick and align the annotated bar boundaries. Release tails
can continue beyond the final onset bar.

If annotations end before the final source onset, preparation extends the last supplied
bar width/beat interval to close that bar. This tail convention needs full-data checking.

## Capacity and verification limits

Layers, widths, heads, optimizer and development split remain the original candidates.
Context is derived from the longest complete configured window. Insufficient capacity
fails; neither training nor generation silently shortens music or discards history.

Old vocabulary/memory measurements no longer apply. Revised full-data preprocessing,
GPU benchmarks, complete experiments, WAV synthesis and report are pending. Current
evidence is synthetic CPU testing, detailed in README and tests/.
