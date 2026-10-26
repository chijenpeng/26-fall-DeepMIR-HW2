import json
import random
from argparse import Namespace

import pytest
import torch

from deepmir_hw2.data import MusicCorpus
from deepmir_hw2.reference import reference
from deepmir_hw2.train import load_checkpoint, train
from deepmir_hw2.utils import read_json


def arguments(config, data, output, **kwargs):
    return Namespace(
        config=str(config),
        data=str(data),
        output=str(output),
        device="cpu",
        resume=None,
        init_from=None,
        steps=None,
        batch_size=None,
        all_training=False,
        stop_after=None,
        **kwargs,
    )


def test_song_split_and_paired_target_alignment(prepared):
    corpus = MusicCorpus(prepared)
    training, validation = corpus.split(7, 0.25)
    assert not set(training) & set(validation)
    assert len(training) + len(validation) == 4
    sample = corpus.sample(training, random.Random(42), "bonus", 2, 64, transpose=3)
    # BOS is encoder input only; the first supervised piano token is Bar.
    assert sample["labels"][0] == corpus.tokenizer.bar_id
    pitch_ids = set(corpus.tokenizer.groups["Pitch"])
    melody = [t for t in sample["input_ids"] if t in pitch_ids]
    piano = [t for t in sample["labels"] if t in pitch_ids]
    assert len(melody) == len(piano)
    assert all(m - p == 12 for m, p in zip(melody, piano, strict=True))


def test_context_does_not_shorten_requested_training_window(prepared):
    corpus = MusicCorpus(prepared)
    names = sorted(corpus.songs)
    with pytest.raises(ValueError, match="No bars were truncated"):
        corpus.sample(names, random.Random(42), "basic", 2, 10)
    needed = corpus.required_context("basic", 2)
    for seed in range(10):
        sample = corpus.sample(names, random.Random(seed), "basic", 2, needed)
        assert sample["input_ids"].count(corpus.tokenizer.bar_id) == 2


def test_real_reference_default_includes_all_songs_and_meters(prepared, tmp_path):
    output = tmp_path / "reference"
    reference(Namespace(data=str(prepared), output=str(output), count=None, bars=None, seed=42))
    metadata = read_json(output / "reference.json")
    assert metadata["source_songs"] == metadata["evaluated_songs"] == 4
    assert metadata["subset_requested"] is False
    assert metadata["skipped_short_songs"] == []
    assert {item["song_name"] for item in read_json(output / "sequences.json")} == {
        "000",
        "001",
        "002",
        "003",
    }


def test_basic_augmentation_transposes_given_chord_with_notes(prepared):
    tokenizer = MusicCorpus(prepared).tokenizer
    sequence = [tokenizer.tok2idx["Chord_C:maj"], tokenizer.tok2idx["Pitch_60"]]
    shifted = tokenizer.transpose(sequence, 2)
    assert shifted == [tokenizer.tok2idx["Chord_D:maj"], tokenizer.tok2idx["Pitch_62"]]


@pytest.mark.parametrize("task", ["basic", "bonus"])
def test_resume_matches_uninterrupted_training(prepared, tiny_config, tmp_path, task):
    tiny_config["task"] = task
    if task == "bonus":
        tiny_config["model"] = {
            "width": 16,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "heads": 2,
            "dropout": 0.1,
        }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(tiny_config))
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(arguments(config, prepared, full))
    first = arguments(config, prepared, resumed)
    first.stop_after = 2
    train(first)
    resume = arguments(config, prepared, resumed)
    resume.resume = str(resumed)
    train(resume)
    a, b = load_checkpoint(full, "latest"), load_checkpoint(resumed, "latest")
    assert a["step"] == b["step"] == 4
    assert a["tokens_seen"] == b["tokens_seen"]
    assert a["sampler_rng"] == b["sampler_rng"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
