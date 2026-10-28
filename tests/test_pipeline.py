import json
import random

import numpy as np
import pretty_midi
import pytest
import torch

from deepmir_hw2 import remi
from deepmir_hw2.data import Corpus, batch
from deepmir_hw2.generate import Grammar, Sampling, accompany_t5, continue_gpt2
from deepmir_hw2.models import build_model


def write_beats(path, bars, beats=4, period=0.5, start=0.2):
    rows = [
        f"{start + i * period:.6f} 1.0 {1.0 if i % beats == 0 else 0.0}"
        for i in range(bars * beats + 1)
    ]
    path.write_text("\n".join(rows) + "\n")


def song_notes(bars, period=0.5, start=0.2):
    notes = []
    for i in range(bars * 8):
        t = start + i * period / 2
        notes.append(pretty_midi.Note(64 + i % 12, 60 + 4 * (i % 5), t, t + period))
    return notes


def test_encode_decode_keeps_grid_timing(tmp_path):
    write_beats(tmp_path / "beat.txt", bars=4)
    notes = song_notes(4)
    grid = remi.Grid.from_file(tmp_path / "beat.txt", notes)
    assert grid.n_bars == 4
    bars = remi.encode(notes, grid, [(0.2, "C:maj"), (2.2, "A:min7/b3")])
    assert remi.VOCAB[bars[1][3]] == "Pos_0" and remi.VOCAB[bars[1][4]] == "Chord_A:min7"
    for decoded in (
        remi.decode([t for b in bars for t in b], grid=grid),
        remi.decode([t for b in bars for t in b], start=0.2),
    ):
        assert len(decoded.notes) == len(notes)
        np.testing.assert_allclose(
            [n.start for n in decoded.notes], [n.start for n in notes], atol=1e-6
        )
        assert all(
            n.velocity // 4 == o.velocity // 4 for n, o in zip(decoded.notes, notes, strict=True)
        )


def test_pickup_and_trailing_notes_get_bars(tmp_path):
    write_beats(tmp_path / "beat.txt", bars=2)
    notes = [pretty_midi.Note(80, 60, 0.0, 0.1), pretty_midi.Note(80, 62, 4.5, 4.6)]
    grid = remi.Grid.from_file(tmp_path / "beat.txt", notes)
    decoded = remi.decode([t for b in remi.encode(notes, grid) for t in b], grid=grid)
    np.testing.assert_allclose([n.start for n in decoded.notes], [0.0, 4.5], atol=0.07)


def test_transpose_shifts_pitch_and_chord():
    tokens = [remi.TOK2IDX["Chord_B:maj"], remi.TOK2IDX["Pitch_60"]]
    shifted = remi.transpose_tokens(tokens, 2)
    assert [remi.VOCAB[t] for t in shifted] == ["Chord_C#:maj", "Pitch_62"]
    assert remi.transpose_tokens([remi.TOK2IDX["Pitch_127"]], 1) is None


@pytest.fixture
def corpus(tmp_path):
    write_beats(tmp_path / "beat.txt", bars=6)
    notes = song_notes(6)
    grid = remi.Grid.from_file(tmp_path / "beat.txt", notes)
    song = {
        "basic": remi.encode(notes, grid, [(0.2, "C:maj")]),
        "melody": remi.encode(notes[::2], grid),
        "piano": remi.encode(notes[1::2], grid, header=False),
    }
    path = tmp_path / "remi.json"
    path.write_text(json.dumps({"vocab": remi.VOCAB, "songs": {"001": song, "002": song}}))
    return Corpus(path)


def tiny(task):
    model = {"width": 32, "heads": 2, "layers": 2, "encoder_layers": 1, "decoder_layers": 1}
    return {"task": task, "model": model, "max_len": [64] if task == "basic" else [64, 64]}


def favor_bar(model):
    """An untrained model rarely picks Bar under top-k; nudge it so pieces end quickly."""
    head = model.get_output_embeddings()
    head.register_forward_hook(
        lambda module, args, out: out.index_add(
            -1, torch.tensor([remi.BAR]), torch.full((*out.shape[:-1], 1), 4.0)
        )
    )
    return model


def test_training_batches_and_loss(corpus):
    rng = random.Random(0)
    for task in ("basic", "bonus"):
        model = build_model(tiny(task))
        inputs = batch(corpus, rng, task, 2, 4, tiny(task)["max_len"], 5)
        assert inputs["input_ids"].shape[1] <= 64
        loss = model(**inputs).loss
        loss.backward()
        assert torch.isfinite(loss)


@pytest.mark.parametrize("top_k,top_p", [(0, 1.0), (5, 1.0), (0, 0.9)])
def test_gpt2_generation_has_exact_bars_with_sliding_context(top_k, top_p):
    torch.manual_seed(0)
    model = favor_bar(build_model(tiny("basic")).eval())
    grammar = Grammar(5, header=True)
    tokens = continue_gpt2(
        model,
        [remi.BOS],
        grammar,
        Sampling(1.0, top_k, top_p),
        torch.Generator().manual_seed(1),
        max_len=64,
    )
    bars = remi.split_bars(tokens)
    assert len(bars) == 5
    for bar in bars:
        assert remi.kind(bar[1]) == "Beat" and remi.kind(bar[2]) == "Tempo"
        positions = [int(remi.value(t)) for t in bar if remi.kind(t) == "Pos"]
        assert positions == sorted(set(positions))
        assert all(p < int(remi.value(bar[1])) * remi.POS_PER_BEAT for p in positions)
    remi.decode(tokens)


def test_t5_accompaniment_follows_prompt_bars():
    torch.manual_seed(0)
    model = build_model(tiny("bonus")).eval()
    grammar = Grammar(3, header=False, beats=[4, 2, 3])
    tokens = accompany_t5(
        model,
        [remi.BOS, remi.BAR],
        grammar,
        Sampling(1.0),
        torch.Generator().manual_seed(0),
        [64, 64],
    )
    bars = remi.split_bars(tokens)
    assert len(bars) == 3
    for bar, beats in zip(bars, [4, 2, 3], strict=True):
        assert all(int(remi.value(t)) < beats * 4 for t in bar if remi.kind(t) == "Pos")
