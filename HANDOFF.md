# DeepMIR HW2 handoff to Claude

**Snapshot:** 2026-10-28, about 14:00 Asia/Taipei. **Deadline:** 2026-10-29 23:59 Asia/Taipei. Read this file and [GitHub issue #5](https://github.com/chijenpeng/26-fall-DeepMIR-HW2/issues/5) before acting. Check live manifests and processes: this snapshot will age.

The student has transferred the remaining project work to Claude. The student personally writes the 16:9 report, provides the cloud link, and handles NTU COOL comments/submission. This Codex chat's scheduled follow-up is paused. No retry code, new inference job, report, upload, or submission should be inferred from a plan below.

## What is implemented and merged

The repository is `https://github.com/chijenpeng/26-fall-DeepMIR-HW2`; `main` is `2e4a137f597db52795f8b0102a4cf36c3042c564`. Merged pull requests passed CI:

| PR | Change |
| --- | --- |
| [#1](https://github.com/chijenpeng/26-fall-DeepMIR-HW2/pull/1) | Preserve supplied timing, meter, notes, controllers and bends; remove fixed BPM, clipping, note limits and generation shortcuts. |
| [#2](https://github.com/chijenpeng/26-fall-DeepMIR-HW2/pull/2) | Fit complete-window training on authorized GPUs and record experiment provenance. |
| [#3](https://github.com/chijenpeng/26-fall-DeepMIR-HW2/pull/3) | Keep long release tails in readable MIDI; add listening render and records. |
| [#4](https://github.com/chijenpeng/26-fall-DeepMIR-HW2/pull/4) | Require all 896 training songs, random initialization, no final validation split, exact-step checkpoint audit; improve export, generation and artifact checks. |
| [#6](https://github.com/chijenpeng/26-fall-DeepMIR-HW2/pull/6) | Require EOS at the last generated bar; report remains student-authored outside the local artifact audit. |

Core modules are described in [docs/architecture.md](docs/architecture.md). Briefly: `prepare.py` and `data.py` build the all-song corpus; `tokens.py` and `music.py` encode MIDI/timing and write MIDI; `models.py`, `attention.py`, `train.py` train GPT-2 Basic and T5 Bonus Option B; `experiment_stage.py` checks final training provenance; `export.py` creates inference-only weights; `generate.py` handles unconditional, continuation and melody-only accompaniment; `reference.py` and `evaluate.py` run the supplied TA metrics; `render.py` makes WAV; `audit.py` checks fixed artifact inventories. `scripts/reproduce_all.sh` exposes explicit stages, with no implicit training when run without an argument.

The source PDF checklist is [docs/requirements.md](docs/requirements.md). Code paths cover Basic A, Basic B and Bonus **Option B**, but required final outputs and metrics are **not complete**. A review of code and green CI do not certify model generation. Earlier notes in README and `docs/requirements.md` predate this snapshot; use this file and issue #5 for live state.

## Training provenance and hardware

- Formal training source is commit `63c4a55ba2962c95674eccbf687eec20c431279c`, retained by Git tag `training-source-2026-10-27`. Inference source on `main` includes later review fixes. Servers lack Git; tracked files were synchronized there by rsync. Avoid changing source beneath the active Bonus process.
- Prepared schema 2 includes exactly 896/896 POP909 training songs, no final validation split. Source fingerprint: `ebbdd016086587cd22c8e8bc178679a44176c0c7a720febe1af3323ca9a78949`. Earlier 851/45 development weights and outputs are diagnostic only; do not resume or submit them.
- `ssh emma-emma`, `/home/emma/deepmir-hw2`: physical GPU0 is an RTX 4070 SUPER 12 GiB. Formal Basic GPT-2 ran **fresh random-init 9,000/9,000 steps**, LR `2e-4`, `runs/basic-final`. `experiment_stage require-final` passed, including positive exposure for all 896 songs. Exported `models/basic.pt` exists on emma, loopy and locally with SHA-256 `da8df1d1db35843fa2fd79323e56c70722d7e61d23222c79ba2d5824059a77bb`.
- On emma physical GPU1 (another RTX 4070 SUPER), formal Bonus Option B T5 is **still training** from fresh random initialization, LR `3e-4`, PID `273017`, `runs/bonus-final`, `logs/bonus-final.log`. It was around step 5,940/12,000 at this snapshot. The 12,000-step cap is a deadline budget, not a PDF requirement or convergence claim. Inspect its log, `latest.json`, and PID before acting. At exact completion run `require-final` and export the latest checkpoint; do not use development weights.
- `ssh loopy`, `/home/loopy/deepmir-hw2`: physical GPU1 is an RTX A4500 20 GiB and was authorized for diagnosis/inference. **Never use loopy GPU0**: it is occupied by another user's job.
- Actual 896-song real-data reference is complete: H1 `2.129281970636034`, H4 `2.6233595885936616`, GS `0.968134640374925`. This is a baseline, not generated-model performance.

## Generation evidence and current processes

All Basic A attempts below used the formal Basic export, corrected last-bar EOS gate, 32 requested bars and explicit seeds beginning at 2026. Failed manifests and partial files are evidence, **not deliverables**.

| Setting / decoder | Result | Preserved location |
| --- | --- | --- |
| A: top-k20, T0.9, whole-integer beam | Failed song002, bar27: 25,451-token context exhausted. | emma `outputs/diagnostics/formal-failed-basic-a/A-beam-20261028` |
| B: top-k50, T0.9, whole-integer beam | Failed song005: repeated approximately `1.96e18`-microsecond durations cannot fit MIDI reader tick limits. | loopy `outputs/diagnostics/formal-failed-basic-a/B-beam-20261028` |
| A: top-k20, T0.9, ordinary numeric sampling | Failed song007 from context exhaustion after six complete MIDIs. | emma `outputs/diagnostics/formal-failed-basic-a/A-sample-20261028` |
| B: top-k50, T0.9, ordinary numeric sampling | Failed song002 from MIDI timing after one complete MIDI. | loopy `outputs/diagnostics/formal-failed-basic-a/B-sample-20261028` |
| C: top-k50, T1.1, ordinary numeric sampling | Failed song002 from MIDI timing after one complete MIDI. | emma `outputs/diagnostics/formal-failed-basic-a/C-sample-20261028` |
| D: top-k100, T1.3, ordinary numeric sampling | Exploratory fourth configuration; failed song005 from MIDI timing. D is outside the planned A/B/C settings. | loopy `outputs/final-experiments/basic-a/D-topk100-temp1p3` |

Independent decoder review found no concrete `beam_integer`/canonical integer parser bug. Single-song ordinary-sampling probes did rescue two seeds that failed under beam, but the full A/B/C batches above still failed. No duration clipping, context rollover, arbitrary BPM, note cap, silent retry or subjective sample selection was applied. **Basic A currently has zero valid 20-piece setting batches.**

Basic B setting A used the same formal Basic checkpoint, three supplied 8-bar prompts with 24 requested generated bars, top-k20/T0.9 and ordinary numeric sampling. It **failed on prompt002** from the 25,451-token context limit. Its manifest and partial output are intact on emma at `outputs/diagnostics/formal-failed-basic-b/A-sample-20261028`; log `logs/final-basic-b-A3-sample.log`. No Basic B setting B/C batch was launched. Bonus inference has not started because formal Bonus training is ongoing.

The only proposed response to structurally invalid outputs was a *transparent, first structurally valid attempt* policy with deterministic seeds and every failed attempt logged. The student was asked about this policy but had not answered before transferring the project. An agent drafted an **unreviewed, untested, undeployed** prototype affecting `generate.py` and `tests/test_generation.py`; it was removed from the working tree and kept locally as `git stash list` entry `Unreviewed structural retry prototype; not deployed or authorized`. Do not treat it as merged code or a completed solution. Claude should decide with the student whether to use a retry policy. The original request forbids hidden retries and cherry-picking.

## Required work still open

1. Check the active emma Bonus job and its manifest. Preserve any failure intact; never overwrite an output directory or silently rerun. Do not interrupt the Bonus training job. Update [issue #5](https://github.com/chijenpeng/26-fall-DeepMIR-HW2/issues/5) with substantive changes.
2. Resolve Basic A generation reliability under a clearly disclosed policy. Required final inventory is **three distinct settings × 20 complete 32-bar MIDI pieces**, H4/GS per setting against the real baseline, and **20 MIDI/WAV pairs** for one selected setting. State seeds, attempts, failures and any limitation accurately.
3. Finish Basic B with the **same formal Basic checkpoint**: three supplied prompts × three settings = nine full 8+24-bar MIDI/WAV pairs. Compare settings and name the best for each prompt. Validate prompt preservation and bar markers.
4. When Bonus hits exactly 12,000 steps, run final checkpoint/provenance checks and export `models/bonus.pt`. Generate Option B from **melody-only** inference for ten supplied 16-bar prompts, each MELODY+PIANO MIDI/WAV. Compute the supplied TA CA plus pitch, velocity, duration and IOI OA metrics. Do not read reference piano/chords as inference conditions.
5. Listen to representative final outputs, validate every MIDI/WAV and required metric count, run `deepmir audit --root . --selected-setting <A|B|C> --output outputs/audit.json`, and prepare a clean local code/weights/results package excluding training data, caches and diagnostic models. Supply factual results and limitations to the student for their report. The student writes the report, cloud link and COOL submission.

## Safe first checks

```bash
git status --short
git log -1 --oneline
git stash list
ssh emma-emma 'cd /home/emma/deepmir-hw2 && ps -p 273017 -o pid,stat,etime,cmd && tail -n 3 logs/bonus-final.log && cat runs/bonus-final/latest.json && cat outputs/diagnostics/formal-failed-basic-b/A-sample-20261028/generation.json'
ssh loopy 'cd /home/loopy/deepmir-hw2 && cat outputs/final-experiments/basic-a/D-topk100-temp1p3/generation.json'
```

`generation.json` must say `status: complete` and its expected counts must match before an output can be used. Compare the actual checkpoint SHA, source fingerprint, bar markers, prompt SHA and MIDI readability; do not infer success from a process exit or a partial `.mid`. See [README.md](README.md) for environment, commands and output paths, and [docs/experiments.md](docs/experiments.md) for historical experiments. A date correction was posted to issue #5: two recent comments labeled ~14:44 local were actually made at ~13:44 local; their experiment facts and paths were unchanged.
