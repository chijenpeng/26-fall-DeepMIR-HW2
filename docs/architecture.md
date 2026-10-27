# Module guide

Updated 2026-10-27. Formal final training runs have launched from commit
`63c4a55ba2962c95674eccbf687eec20c431279c` on all 896 songs, with random
initialization and no validation split. Exact-step completion, per-song exposure
validation, final generation, and report remain pending. Final-bar EOS/numeric
behavior is under diagnosis and is not considered resolved by synthetic tests.

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
| attention.py | Exact T5 attention in query blocks; bounded saved intermediates, shared relative bias, full-history cache |
| train.py | Weighted loss, AdamW/AMP/accumulation, atomic checkpoints/resume, source-level exposure provenance; reject insufficient context |
| generate.py | Full-history cache; learned basic timing; supplied Bonus timing; optional MIDI/WAV listening output in one command |
| reference.py | All-song/all-meter default reference; explicit optional subsets |
| evaluate.py | Unmodified TA metric functions, seeded sampling and coverage reporting; reject incomplete/undefined comparisons |
| render.py | Explicit FluidSynth/SoundFont rendering and PCM checks |
| export.py | Inference weights/config/vocabulary/source programs and training provenance/source-checkpoint hash without optimizer |
| experiment_stage.py | Require full-corpus final checkpoint: exact configured step, random init, zero validation, 896 positive source exposures and matching fingerprint |
| audit.py | Check declared final output sets, prompt preservation, MIDI/WAV boundaries, metrics, checkpoint links and manual-review readiness |
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
bar width/beat interval to close that bar. Full preparation passed on all 896 songs.

## Capacity and verification limits

Layers, widths, heads, optimizer and development split remain the original candidates.
Context is derived from the longest complete configured window. Insufficient capacity
fails; neither training nor generation silently shortens music or discards history.

Prepared vocabulary has 1,195 tokens. Required context is 25,451 for basic and 13,403
for bonus. Basic passed the longest-window optimizer check on a 12GB RTX 4070 SUPER.
Native T5 exceeded 20GB on the most expensive 16-bar window; query blocking and
recomputation lets all three extrema (largest encoder, largest decoder, largest
combined workload) pass below 4.7GB allocated with 2,048-query blocks. Compact
integer bucket matrices are shared across layers; attention scores and learned
bias values are materialized by query block. This is full, quadratic attention,
with no sliding window or context truncation.
CPU regressions compare all parameter gradients to native T5 and cached logits to
full-history decoding, including sequences longer than the query block size.

## Numeric-field inference

Generation defaults to sampling each numeric digit. In `sample` mode, temperature,
top-k and top-p apply to numeric choices and categorical tokens. The optional
`--numeric-strategy beam` searches complete canonical integers with a fixed beam width
of four, scoring legal digit choices with temperature-scaled probabilities. Top-k and
top-p do not filter those digit choices in beam mode; they continue to apply to
categorical tokens such as pitches, velocities and event types. This mode changes only
numeric-field decoding and does not impose musical duration or note-count limits.

Final checkpoints have been launched but are not yet certified complete. The final
generation EOS behavior, WAV sets, evaluation, listening review and report remain
pending. Source-data audits and all 13 supplied prompt encodings passed before
preparation. Training capacity checks and synthetic tests are distinct from final
music-quality evaluation.
