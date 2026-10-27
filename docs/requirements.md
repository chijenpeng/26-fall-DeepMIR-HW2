# Assignment coverage audit

Source: supplied Homework_2.pdf (38 pages), not redistributed here.
Deadline: 2026-10-29 23:59 Asia/Taipei. Formal work resumed on 2026-10-26.

Old-format prepared data, smoke checkpoints and outputs were deleted. Current
preparation passed on all 896 songs on both servers with identical fingerprints.
The formal final Basic and Bonus runs launched on 2026-10-27 from reviewed commit
`63c4a55ba2962c95674eccbf687eec20c431279c`, from random initialization over all
896 songs with no validation split. Basic uses emma GPU 0, 9,000 steps and LR 2e-4;
Bonus uses emma GPU 1, 12,000 steps and LR 3e-4. Run completion and all final
artifacts remain pending. Earlier development checkpoints and samples are
diagnostics only. Final-bar EOS/numeric generation behavior remains under diagnosis;
code changes and synthetic tests do not close it without real model probes.

| Requirement | Implementation | Remaining |
|---|---|---|
| Transformer from scratch | Random GPT-2/T5 | Formal random-init final runs launched; exact-step completion/audit pending |
| Only provided training data | Exact-896 guard, source-derived schemas, no Bonus prompt targets; all 896 sources prepared | Verify final per-song exposure counts and fingerprint linkage |
| Basic: merge three tracks | Merged note/CC events plus supplied basic chords; all sources prepared | Final-output validation |
| Basic A: 32 bars | Model predicts timing/meter; task controls count | Final model/samples |
| Basic A: three settings, 20 pieces each | A/B/C command and sampling metadata | Generate all 60 from the final checkpoint |
| Basic A: H4/GS | Original TA functions and shared projection | Final summaries with 20 valid pieces per setting |
| Real-data baseline | All 896 songs evaluated with supplied meters: H1 2.129281970636034, H4 2.6233595885936616, GS 0.968134640374925 | Compare final generated H4/GS against this baseline |
| H1/SI | H1 supported; SI optional/unimplemented | Optional SI decision |
| Basic A: selected 20 MIDI/WAV | Integrated generate --soundfont or separate render; development SoundFont works | Final selection/listening and all 20 pairs |
| Basic B: same checkpoint, 8+24 bars | Same model, 24 generated bars, exact 8-bar prompt check | Generate and compare all three settings for all prompts |
| Basic B: nine MIDI/WAV, best per prompt | Reproduction commands and output audit | Listen, select best setting per prompt and record selection |
| Correct MIDI bars and preserved prompt sound | Clock/retrigger/source-prompt regressions pass; all 13 input prompts encode | Final-model EOS, bar, event-timing and listening checks |
| Bonus Option B | Melody encoder, piano decoder | Final all-song checkpoint completion and validation |
| Bonus inference melody only | No reference chord/piano read | Access-boundary test passes |
| Bonus: ten 16-bar outputs | Supplied melody timing, 16-bar check | Generate and render final ten MIDI/WAV |
| Bonus: CA and OA | Original TA evaluator, completeness checks | Final 10-piece CA, pitch OA, velocity OA, duration OA and IOI OA |
| Model/representation/config details | Codec, configs/provenance | Report |
| Optional bonus ablations | Config/checkpoint infrastructure | Optional experiments |
| 16:9 report | Student-authored deliverable; intentionally not required by the local artifact audit | Write and review r14725022_report.pdf, including layout and citations |
| Code/README/requirements/weights | Package, pins, export | Final weights/clean-checkout validation |
| Output path/listening inference | Explicit paths, integrated/separate MIDI/WAV; four development samples rendered; adaptive MIDI clock fits TA reader limits | Final checkpoint generation stability |
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

Synthetic tests and input preparation do not certify final generation. See
experiments.md for historical development evidence and current launch state.
Run `deepmir audit --root . --selected-setting B --output outputs/audit.json` after
generation and evaluation. The audit checks the fixed `reproduce_all.sh` output
layout and checkpoint links. Listening quality and per-prompt Basic B selection
remain manual review items. The student writes and reviews the report, uploads code,
models and results with open access, and submits the cloud link through NTU COOL.

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

The pre-merge review corrected input-side specification issues. It does not certify
final model quality, completed training, final-bar EOS behavior, output inventories,
report content, or submitted artifacts. Those checks remain open and are recorded in
experiments.md as evidence becomes available.

## Report outline and evidence to collect

The 16:9 report should use the following sections. Fill result tables only from
validated final artifacts; do not replace pending values with development samples
or invented estimates.

1. **Task and approach:** Basic GPT-2 and Bonus Option B T5, both trained from
   scratch; clarify that Bonus inference reads melody only.
2. **Data and representation:** POP909 training corpus (896 songs), source
   fingerprint and per-song exposure audit, timing/meter-aware event encoding,
   Basic chord use, and the 8-bar/16-bar supplied prompts.
3. **Experimental protocol:** exact final configs and steps; Basic A's three
   settings and 20 outputs per setting; real-reference H4/GS comparison; Basic B's
   8+24-bar continuation comparisons and per-prompt best-setting choices; Bonus's
   ten 16-bar outputs.
4. **Quantitative results:** validated H4/GS summaries for Basic A and the TA
   Bonus CA, pitch-class OA, velocity OA, duration OA and IOI OA metrics. Include
   piece counts, failures, and relevant distributions; state undefined or missing
   metrics rather than hiding them.
5. **Listening and artifact review:** selected MIDI/WAV lists, SoundFont and
   rendering settings, prompt-preservation observations, generation failures, and
   the final-bar EOS/release-tail findings after real probes.
6. **Reproducibility, limitations, and attribution:** source commit, commands,
   checkpoint hashes, seeds, configurations, output/cloud link, known limitations,
   and citations for supplied TA code and other public sources.
