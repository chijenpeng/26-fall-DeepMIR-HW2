# Development workflow

Use Python 3.12 and `pip install -r requirements-experiments.txt`.

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

Use topic branches and pull requests. Never commit datasets, prepared data, runs,
checkpoints, generated audio, SoundFonts or caches (see `.gitignore`).
