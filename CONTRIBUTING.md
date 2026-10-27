# Development workflow

Formal final runs launched on 2026-10-27 from reviewed commit
`63c4a55ba2962c95674eccbf687eec20c431279c`: Basic is on emma physical GPU 0 and
Bonus is on emma physical GPU 1. Do not interrupt these jobs. Loopy physical GPU 1
is reserved by the parent for numeric/EOS diagnosis; do not start training there.
Check docs/experiments.md for the current launch plan and historical diagnostics.
Repository CI remains synthetic and does not launch server GPU jobs.

Use Python 3.12 and install `requirements-experiments.txt`. Keep modules focused on
their documented responsibilities; use local JSON configs rather than hard-coded
server paths or implicit model downloads.

Before committing code:

```bash
ruff check src tests scripts/benchmark_gpu.py
ruff format --check src tests scripts/benchmark_gpu.py
pytest -q
```

Make focused commits with a concrete description of the change. Use topic branches
and pull requests for subsequent changes. PR descriptions should state behavior and
actual validation, including whether checks are synthetic, smoke or full experiments.

Never commit datasets, preprocessed arrays, virtual environments, caches, credentials,
SoundFonts or model binaries. Final checkpoints belong in release/cloud assets with
checksums; they are not ordinary Git objects. Only actual final results should enter
a future submission/release, after verification.

Keep `docs/requirements.md` current when requirements are implemented or verified.
Do not present planned commands, untrained samples or pilot metrics as final results.
