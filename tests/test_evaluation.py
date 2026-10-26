from argparse import Namespace
from pathlib import Path

import pytest

from deepmir_hw2.evaluate import compute_basic, evaluate, load_ta_basic
from deepmir_hw2.music import BarTiming
from deepmir_hw2.tokens import MusicTokenizer, TokenizerConfig
from deepmir_hw2.utils import read_json, write_json


@pytest.mark.skipif(
    not Path("third_party/MusDr/musdr/side_utils.py").exists(),
    reason="Install MusDr for external metric tests",
)
def test_metrics_match_unmodified_ta_functions_including_gs_denominator():
    tokenizer = MusicTokenizer(TokenizerConfig((4, 6), ((4, 4),)))
    bars = []
    for index in range(32):
        bar = tokenizer.header(BarTiming((500_000,) * (4 if index % 2 else 6), 4, 4))
        bar += [
            tokenizer.tok2idx["Pos"],
            *tokenizer.encode_uint(250_000 if index % 2 else 0),
            tokenizer.tok2idx["Pitch_60"],
            tokenizer.tok2idx["Vel_80"],
            tokenizer.tok2idx["Duration"],
            *tokenizer.encode_uint(500_000),
        ]
        bars.append(bar)
    vocabulary = tokenizer.evaluation_vocabulary()
    sequence = [vocabulary[token] for token in tokenizer.evaluation_tokens(bars)]
    ta = load_ta_basic("HW2_scripts", "third_party/MusDr")
    result = compute_basic(sequence, vocabulary, ta)
    positions = [vocabulary[f"Pos_{i}"] for i in range(128)]
    pitches = [vocabulary[f"Pitch_{i}"] for i in range(128)]
    assert result["H4"] == pytest.approx(0)
    assert result["GS"] == ta.compute_piece_groove_similarity(
        sequence, vocabulary["Bar"], positions, pitches
    )
    assert result["GS"] == pytest.approx(1 - (16 * 16 * 2) / (496 * 128))
    assert result["bars"] == 32


def test_final_empty_bar_is_retained_in_ta_evaluation_view():
    tokenizer = MusicTokenizer()
    empty = tokenizer.header(BarTiming((500_000,) * 4, 4, 4))
    sequence = tokenizer.evaluation_tokens([empty] * 4)
    assert sequence[-1] == "<EOS>"
    assert sequence.count("Bar") == 4


def test_partial_generation_is_not_reported_as_a_complete_experiment(tmp_path):
    tokenizer = MusicTokenizer()
    write_json(tmp_path / "tok2idx.json", tokenizer.evaluation_vocabulary())
    write_json(tmp_path / "sequences.json", [{"song_name": "001", "tokens": []}])
    write_json(tmp_path / "generation.json", {"expected_pieces": 2, "complete": False})
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(Namespace(task="basic", input=str(tmp_path), output=str(tmp_path / "metrics.csv")))


@pytest.mark.skipif(
    not Path("third_party/MusDr/musdr/side_utils.py").exists(),
    reason="Install MusDr for external metric tests",
)
def test_undefined_piece_metrics_do_not_become_a_partial_mean(tmp_path):
    tokenizer = MusicTokenizer()
    write_json(tmp_path / "tok2idx.json", tokenizer.evaluation_vocabulary())
    write_json(
        tmp_path / "sequences.json",
        [{"song_name": "silent", "tokens": ["<BOS>", "Bar", "Bar", "Bar", "Bar", "<EOS>"]}],
    )
    with pytest.raises(ValueError, match="Undefined metrics"):
        evaluate(
            Namespace(
                task="basic",
                input=str(tmp_path),
                output=str(tmp_path / "metrics.csv"),
                ta_scripts="HW2_scripts",
                musdr_root="third_party/MusDr",
                seed=42,
            )
        )
    summary = read_json(tmp_path / "metrics.summary.json")
    assert summary["complete"] is False
    assert summary["metrics"]["H4"] is None
    assert summary["valid_pieces"]["H4"] == 0
