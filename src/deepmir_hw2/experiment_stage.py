"""One-shot validation helpers for final experiment artifacts."""

import argparse
from pathlib import Path

from .train import FINAL_CORPUS_SIZE, checkpoint_path, load_checkpoint


def validate_final_checkpoint(path: str | Path) -> dict:
    checkpoint = checkpoint_path(path, "latest").resolve()
    saved = load_checkpoint(checkpoint)
    provenance = saved.get("training_provenance")
    if not provenance:
        raise ValueError("Checkpoint has no training provenance")
    config = saved.get("config", {})
    training = config.get("training", {})
    data = config.get("data", {})
    names = provenance.get("source_names")
    counts = provenance.get("song_sample_counts")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("Final checkpoint has an invalid source-song list")
    if provenance.get("mode") != "full_corpus":
        raise ValueError("Checkpoint is not a full-corpus run")
    if config.get("training_mode") != "full_corpus":
        raise ValueError("Checkpoint config does not declare full_corpus mode")
    if (
        provenance.get("initialization") != "random"
        or provenance.get("initialized_from") is not None
    ):
        raise ValueError("Final checkpoint must descend from random initialization")
    if provenance.get("expected_songs") != FINAL_CORPUS_SIZE:
        raise ValueError(f"Expected {FINAL_CORPUS_SIZE} training songs")
    if (
        provenance.get("corpus_songs") != FINAL_CORPUS_SIZE
        or provenance.get("training_songs") != FINAL_CORPUS_SIZE
        or provenance.get("validation_songs") != 0
        or len(names or []) != FINAL_CORPUS_SIZE
        or len(set(names or [])) != FINAL_CORPUS_SIZE
    ):
        raise ValueError("Final checkpoint corpus or split does not match the required full corpus")
    if data.get("validation_fraction") != 0 or data.get("expected_songs") != FINAL_CORPUS_SIZE:
        raise ValueError("Final checkpoint config does not declare the full-corpus split")
    if not isinstance(counts, dict) or set(counts) != set(names):
        raise ValueError("Final checkpoint is missing per-song exposure counts")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts.values()
    ):
        raise ValueError("At least one source song has no valid training exposure")
    if provenance.get("all_songs_sampled") is not True or provenance.get("complete") is not True:
        raise ValueError("Final checkpoint is not marked complete with full song coverage")
    if provenance.get("data_fingerprint") != saved.get("data_fingerprint"):
        raise ValueError("Final checkpoint data fingerprints do not match")
    if provenance.get("checkpoint") != checkpoint.name or provenance.get("step") != saved.get(
        "step"
    ):
        raise ValueError("Final checkpoint provenance does not identify this checkpoint")
    if not provenance.get("configured_steps_complete"):
        raise ValueError("Final checkpoint did not reach its configured training step")
    if saved.get("step") != training.get("steps"):
        raise ValueError("Final checkpoint step differs from configured training steps")
    expected_samples = (
        saved["step"] * training.get("batch_size", 0) * training.get("gradient_accumulation", 0)
    )
    if sum(counts.values()) != expected_samples:
        raise ValueError("Per-song exposure total differs from completed training steps")
    return saved


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("require-final", help=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.set_defaults(func=lambda args: validate_final_checkpoint(args.checkpoint))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_parser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
