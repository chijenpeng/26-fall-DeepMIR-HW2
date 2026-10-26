from pathlib import Path

import pytest

from deepmir_hw2.evaluate import compute_basic, load_module
from deepmir_hw2.tokens import MusicTokenizer


@pytest.mark.skipif(
    not Path("third_party/MusDr/musdr/side_utils.py").exists(),
    reason="Install MusDr for external-metric integration tests",
)
def test_musdr_constant_pitch_and_identical_rhythm():
    tokenizer = MusicTokenizer()
    bar = [
        tokenizer.bar_id,
        tokenizer.tok2idx["Beats_4"],
        tokenizer.tok2idx["Pos_0"],
        tokenizer.tok2idx["Pitch_60"],
        tokenizer.tok2idx["Vel_20"],
        tokenizer.tok2idx["Dur_12"],
    ]
    musdr = load_module(Path("third_party/MusDr/musdr/side_utils.py"))
    metrics = compute_basic(bar * 32, tokenizer, musdr)
    assert metrics["H1"] == pytest.approx(0.0)
    assert metrics["H4"] == pytest.approx(0.0)
    assert metrics["GS"] == pytest.approx(1.0)
    assert metrics["bars"] == 32
    assert metrics["empty_bars"] == 0


def test_basic_evaluation_rejects_incompatible_meter():
    tokenizer = MusicTokenizer()
    bar = [tokenizer.bar_id, tokenizer.tok2idx["Beats_6"]]
    with pytest.raises(ValueError, match="Bar length"):
        compute_basic(bar * 4, tokenizer, None)
