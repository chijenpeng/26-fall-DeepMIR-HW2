import json
import random
from argparse import Namespace

import pytest
import torch

from deepmir_hw2.data import MusicCorpus
from deepmir_hw2.train import load_checkpoint, train


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
