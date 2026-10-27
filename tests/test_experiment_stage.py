import copy

import pytest
import torch

import deepmir_hw2.experiment_stage as stage


def final_checkpoint():
    names = ["song-a", "song-b"]
    return {
        "step": 3,
        "data_fingerprint": "manifest-sha256",
        "config": {
            "training_mode": "full_corpus",
            "training": {"steps": 3, "batch_size": 2, "gradient_accumulation": 1},
            "data": {"expected_songs": 2, "validation_fraction": 0.0},
        },
        "training_provenance": {
            "mode": "full_corpus",
            "initialization": "random",
            "initialized_from": None,
            "expected_songs": 2,
            "corpus_songs": 2,
            "training_songs": 2,
            "validation_songs": 0,
            "source_names": names,
            "song_sample_counts": {"song-a": 4, "song-b": 2},
            "all_songs_sampled": True,
            "configured_steps_complete": True,
            "complete": True,
            "data_fingerprint": "manifest-sha256",
            "checkpoint": "step-0000003.pt",
            "step": 3,
        },
    }


def test_require_final_accepts_only_complete_full_random_run(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "FINAL_CORPUS_SIZE", 2)
    path = tmp_path / "step-0000003.pt"
    torch.save(final_checkpoint(), path)

    saved = stage.validate_final_checkpoint(path)

    assert saved["training_provenance"]["complete"] is True


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("mode", "standard", "not a full-corpus"),
        ("initialization", "checkpoint", "random initialization"),
        ("complete", False, "not marked complete"),
        ("all_songs_sampled", False, "not marked complete"),
        ("source_names", ["song-a"], "corpus or split"),
        ("song_sample_counts", {"song-a": 6, "song-b": 0}, "no valid training exposure"),
    ],
)
def test_require_final_rejects_invalid_final_provenance(tmp_path, monkeypatch, field, value, error):
    monkeypatch.setattr(stage, "FINAL_CORPUS_SIZE", 2)
    saved = final_checkpoint()
    saved["training_provenance"][field] = value
    path = tmp_path / "step-0000003.pt"
    torch.save(copy.deepcopy(saved), path)

    with pytest.raises(ValueError, match=error):
        stage.validate_final_checkpoint(path)
