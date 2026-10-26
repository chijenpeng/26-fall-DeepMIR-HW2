# Assignment coverage audit

Source: supplied Homework_2.pdf (38 pages), not redistributed here.
Deadline: 2026-10-29 23:59 Asia/Taipei. Formal work resumed on 2026-10-26.

Old-format prepared data, smoke checkpoints and outputs were deleted. Current
preparation passed on all 896 songs on both servers with identical fingerprints.
Development training is active; final experiments/artifacts remain pending.

| Requirement | Implementation | Remaining |
|---|---|---|
| Transformer from scratch | Random GPT-2/T5 | Formal training |
| Only provided training data | Exact-896 default guard, source-derived schemas, no prompt targets; all 896 sources prepared | All-song final stage |
| Basic: merge three tracks | Merged note/CC events plus supplied basic chords; all sources prepared | Final-output validation |
| Basic A: 32 bars | Model predicts timing/meter; task controls count | Final model/samples |
| Basic A: three settings, 20 pieces each | A/B/C commands | Run all 60 |
| Basic A: H4/GS | Original TA functions and shared projection | Formal comparison |
| Real data | All supplied songs/meters by default | Full reference |
| H1/SI | H1 supported; SI optional/unimplemented | Optional SI decision |
| Basic A: selected 20 MIDI/WAV | Integrated generate --soundfont, or separate render | Selection/real-SoundFont validation |
| Basic B: same checkpoint, 8+24 bars | Same model, 24 default, verify 8-bar input | Full three-prompt comparison |
| Basic B: nine MIDI/WAV, best per prompt | Reproduction commands | Generate/listen/select |
| Correct MIDI bars and preserved prompt sound | Clock/retrigger/terminal-note regressions pass; all 13 real prompt encodings checked | Final-model output validation |
| Bonus Option B | Melody encoder, piano decoder | Formal training |
| Bonus inference melody only | No reference chord/piano read | Access-boundary test passes |
| Bonus: ten 16-bar outputs | Supplied melody timing, 16-bar check | Final ten MIDI/WAV |
| Bonus: CA and OA | Original TA evaluator, completeness checks | Final metrics |
| Model/representation/config details | Codec, configs/provenance | Report |
| Optional bonus ablations | Config/checkpoint infrastructure | Optional experiments |
| 16:9 report | Not authored | r14725022_report.pdf |
| Code/README/requirements/weights | Package, pins, export | Final weights/clean-checkout validation |
| Output path/listening inference | Explicit paths, generate --soundfont writes MIDI/WAV; atomic PCM validation; render failure marks batch incomplete | Real SoundFont validation |
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

Code has synthetic verification plus real-source preparation and GPU capacity
checks. Development training is in progress. Full experiments, audio, report and
delivery must still be completed; see experiments.md for the run record.

## Pre-merge specification review (before training resumed)

Checked against the complete supplied PDF: required Transformer/from-scratch/data
rules; basic merged tracks, three configurations and 20 x 32-bar outputs per setting;
same-checkpoint 8+24 continuation, nine MIDI/WAV outputs and per-prompt selection;
Bonus Option B with melody-only input, ten 16-bar outputs and CA/OA evaluation;
output-path/README/requirements/listening-sample inference and submission exclusions.

Read-only input audit: all 896 training sources checked, no missing required tracks or
ambiguous programs; all three continuation prompts encode as eight bars and all ten
melody prompts encode as sixteen bars. Three terminal-note boundary cases found in
real training sources were fixed and rechecked. No full preparation/training or model
generation ran. MIDI/WAV orchestration and PCM validation have synthetic regression tests.

No known code-side specification blocker remains after these corrections. This review
does not certify final model quality, GPU capacity, completed experiments or submission
artifacts. Subsequent preparation/capacity checks are recorded separately in experiments.md.
