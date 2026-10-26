# Development workflow

Formal training and experiments are paused. Do not start data processing, GPU jobs,
sample generation or audio rendering until the author asks to resume. Repository
maintenance and synthetic CPU CI checks do not launch jobs on the training servers.

Use Python 3.12 and install `requirements-experiments.txt`. Keep modules focused on
their documented responsibilities; use local JSON configs rather than hard-coded
server paths or implicit model downloads.

Before committing code:

```bash
ruff check src tests
ruff format --check src tests
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
