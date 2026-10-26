import copy

import pytest
import torch

from deepmir_hw2.generate import CachedGenerator, SamplingConfig, generate_bars
from deepmir_hw2.models import build_model
from deepmir_hw2.tokens import MusicTokenizer


@pytest.mark.parametrize("task", ["basic", "bonus"])
def test_cached_generation_exact_mixed_length_bars(tiny_config, task):
    tokenizer = MusicTokenizer()
    config = copy.deepcopy(tiny_config)
    config["task"] = task
    config["model"] = {
        "width": 16,
        "heads": 2,
        "dropout": 0.0,
        **({"layers": 1} if task == "basic" else {"encoder_layers": 1, "decoder_layers": 1}),
    }
    model = build_model(config, tokenizer).eval()
    melody = [tokenizer.bos_id, tokenizer.bar_id, tokenizer.tok2idx["Beats_4"], tokenizer.eos_id]
    params = dict(
        model=model,
        tokenizer=tokenizer,
        task=task,
        beats_per_bar=[4, 6, 2],
        max_tokens=64,
        device="cpu",
        sampling=SamplingConfig(max_notes_per_bar=1),
        seed=123,
        melody=melody if task == "bonus" else None,
    )
    a, _ = generate_bars(**params)
    b, _ = generate_bars(**params)
    assert a == b
    assert len(a) == 3
    for bar, beats in zip(a, [4, 6, 2], strict=True):
        tokenizer.validate_bar(bar, beats)
        assert len(bar) == 6


def test_basic_cache_rollover_matches_full_context(tiny_config):
    tokenizer = MusicTokenizer()
    tiny_config["data"]["max_tokens"] = 16
    model = build_model(tiny_config, tokenizer).eval()
    bar = [
        tokenizer.bar_id,
        tokenizer.tok2idx["Beats_4"],
        tokenizer.tok2idx["Pos_0"],
        tokenizer.tok2idx["Pitch_60"],
        tokenizer.tok2idx["Vel_20"],
        tokenizer.tok2idx["Dur_6"],
    ]
    with torch.inference_mode():
        engine = CachedGenerator(model, tokenizer, "basic", 16, "cpu", [tokenizer.bos_id])
        engine.append(bar * 2)
        engine.logits()
        engine.append(bar)
        actual = engine.logits()
        expected = model(torch.tensor([[tokenizer.bos_id, *bar, *bar]]), use_cache=False).logits[
            0, -1
        ]
    torch.testing.assert_close(actual, expected)
