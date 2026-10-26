# Assignment coverage audit

Source: the locally supplied `Homework_2.pdf` (38 pages), not redistributed in this repo.
Audit date: 2026-10-26. Deadline supplied by the author: 2026-10-29 23:59 Asia/Taipei.
Formal training and experiments are paused at the author's request.

"Implemented" below means a code path exists; it does not mean the required final
experiment or submission artifact has been completed.

| Requirement | Code / evidence | Status |
|---|---|---|
| Use Transformer-based generation models | Random-initialized GPT-2 and T5 constructors in `models.py` | Implemented; five GPU steps each verified |
| Train from scratch | No pretrained initialization or model download | Implemented; formal training pending |
| Use the whole 896-song POP909 training set | All 896 preprocessed; `train --all-training --init-from ...` final stage | Pending execution; development split alone does not satisfy this |
| No extra training data | Preparation reads only `POP909/training`; no prompts used as training targets | Implemented |
| Merge MELODY, BRIDGE and PIANO for basic tasks | `prepare.py` / `tokens.py` | Implemented and full-data preprocessing verified |
| Basic A: generate 32 bars | `generate --mode unconditional --bars 32` | Smoke verified; final samples pending |
| Basic A: at least three inference settings | A/B/C settings in README and reproduction script | Defined; full comparison not run |
| Basic A: 20 MIDI files per setting | `generate --count 20`, three settings | Implemented; required 60-sample experiment not run |
| Basic A: averaged H4 and GS | MusDr functions via `evaluate.py`, metric CSV + JSON | Smoke verified; final comparison pending |
| Compare objective results to real data | Seeded 100-song matched 32-bar reference selection | Reference path verified; final experiment pending |
| Optional H1 / SI | H1 implemented; SI not implemented | H1 smoke verified; SI optional and currently omitted |
| Basic A: select one configuration and submit 20 MIDI + 20 WAV | Generation and rendering commands | Pending training, selection and verified WAV synthesis |
| Basic B: reuse Basic A checkpoint | Same causal checkpoint accepted for both modes | Implemented |
| Basic B: three 8-bar prompts, add 24 bars each | Original prompts preserved; 24 generated bars | All three smoke verified |
| Basic B: three configurations per prompt, 9 MIDI + 9 WAV total | Reproduction script iterates A/B/C over prompts | Full experiment and WAV files pending |
| Basic B: identify best result for each prompt | Requires listening-based selection and report entry | Pending; not automatically decided by a score |
| Basic B: report model, representation and inference settings | Config files and generation metadata | Metadata implemented; report pending |
| Bonus: use only separated MELODY / PIANO tracks | Aligned encoder input and decoder target; no BRIDGE | Implemented and preprocessing verified |
| Bonus: ten 16-bar melody prompts -> MELODY + generated PIANO | Full-melody encoder, copied prompt, appended PIANO track | All ten smoke verified; final quality pending |
| Bonus: submit 10 MIDI + 10 WAV | Generation and rendering commands | MIDI smoke only; final results / WAV pending |
| Bonus: CA and OA_P/OA_V/OA_D/OA_IOI | Unmodified `HW2_scripts/eval_bonus.py` wrapped with completeness checks | All ten smoke outputs evaluated; final metrics pending |
| Bonus: specify model/representation/inference configuration | Saved architecture, tokenizer, sampling and seed metadata | Implemented; report pending |
| Optional bonus ablations | Can vary configs/checkpoints, but no completed study | Pending / optional |
| Report: clear 16:9 PPT-like PDF, about 10 pages suggested | No report implementation yet | Pending |
| Report filename `r14725022_report.pdf` | Student ID follows HW1 identity | Pending |
| Report includes representation, architecture, hyperparameters, experiments and discussion | Config/metadata support exists | Writing and results pending |
| Source code, inference checkpoint, README and requirements | Package, export CLI, README, separated requirements | Code exists; final trained checkpoint pending |
| TA can run inference from scratch and set output location | CLI takes local checkpoint and explicit `--output` | Smoke verified; final-model revalidation pending |
| Inference produces listening samples | MIDI generation + separate FluidSynth render command | WAV synthesis unverified; consider a combined command before submission |
| Non-pip installation instructions | README documents FluidSynth executable + explicit SoundFont | System install details / chosen SoundFont still need finalization |
| Cloud drive with accessible sharing permissions; link in report first page and COOL comments | No final cloud submission prepared | Pending |
| Do not upload training data, preprocessed data, others' models or caches | `.gitignore` excludes dataset, caches and all model binaries | Repo exclusion implemented; final submission audit pending |
| Cite public code | README and `THIRD_PARTY.md` name Transformers, MusDr and TA scripts | Source attribution exists; report citations pending |

## Verdict

The code covers the main mandatory model/data/generation/evaluation paths, with the
remaining integration gap around verified audio synthesis and a single complete
listening-sample inference workflow. This is **not yet a completed or fully compliant
submission**: formal all-song training, required experiment counts, best-sample selection,
WAV artifacts, report and final cloud delivery are still outstanding.

Passing tests or computing metrics on smoke checkpoints is not evidence of musical quality.
