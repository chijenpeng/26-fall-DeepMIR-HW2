from pathlib import Path

import pytest

from deepmir_hw2.evaluate import compute_basic, load_ta_basic
from deepmir_hw2.music import BarTiming
from deepmir_hw2.tokens import MusicTokenizer, TokenizerConfig


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
