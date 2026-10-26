# Assignment coverage audit

Source: supplied Homework_2.pdf (38 pages), not redistributed here.
Deadline: 2026-10-29 23:59 Asia/Taipei. Formal preprocessing/training/experiments are paused.

Old prepared data, smoke checkpoints and outputs were deleted. Verification below
refers to current synthetic tests, not old artifacts or completed POP909 experiments.

| Requirement | Implementation | Remaining |
|---|---|---|
| Transformer from scratch | Random GPT-2/T5 | Formal training |
| Only provided training data | Source-derived schemas, no prompt targets | New preparation/all-896-song final stage |
| Basic: merge three tracks | Merged note/CC events plus supplied basic chords | Full-source validation |
| Basic A: 32 bars | Model predicts timing/meter; task controls count | Final model/samples |
| Basic A: three settings, 20 pieces each | A/B/C commands | Run all 60 |
| Basic A: H4/GS | Original TA functions and shared projection | Formal comparison |
| Real data | All supplied songs/meters by default | Full reference |
| H1/SI | H1 supported; SI optional/unimplemented | Optional SI decision |
| Basic A: selected 20 MIDI/WAV | MIDI plus separate render | Selection/WAV verification |
| Basic B: same checkpoint, 8+24 bars | Same model, 24 default, verify 8-bar input | Full three-prompt comparison |
| Basic B: nine MIDI/WAV, best per prompt | Reproduction commands | Generate/listen/select |
| Correct MIDI bars and preserved prompt sound | Rebuilt clock and retained note/CC seconds | Synthetic checks pass; real prompts pending |
| Bonus Option B | Melody encoder, piano decoder | Formal training |
| Bonus inference melody only | No reference chord/piano read | Access-boundary test passes |
| Bonus: ten 16-bar outputs | Supplied melody timing, 16-bar check | Final ten MIDI/WAV |
| Bonus: CA and OA | Original TA evaluator, completeness checks | Final metrics |
| Model/representation/config details | Codec, configs/provenance | Report |
| Optional bonus ablations | Config/checkpoint infrastructure | Optional experiments |
| 16:9 report | Not authored | r14725022_report.pdf |
| Code/README/requirements/weights | Package, pins, export | Final weights/clean-checkout validation |
| Output path/listening inference | Explicit paths, separate renderer | Real SoundFont validation |
| Cloud link in report/COOL | Not prepared | Final upload/sharing |
| Exclude data/caches/models | Ignore rules and old artifact deletion | Final submission audit |
| Cite public code | README/THIRD_PARTY.md | Report citations |

## Authorized corrections

- No fixed 100 BPM/four-pulse unconditional output.
- Microsecond timing, full pitch/velocity, long notes, controllers/bends retained.
- Supplied basic chord events with joint augmentation; Bonus remains melody-only.
- No per-bar note-count caps or duplicate-pitch restriction.
- No silent event dropping, duration clipping, window shortening or history rollover.
- Full-window context derived from prepared source data.
- Consistent annotated-pulse/supplied-meter MIDI clock.
- Original TA metric functions/denominator behavior.
- No four-beat-only/100-song default reference subset.
- Obsolete files deleted; no legacy migration or inference implementation.

These are code changes with synthetic verification. Training, full experiments,
audio, report and delivery must still be completed.
