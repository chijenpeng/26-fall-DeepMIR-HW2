import pretty_midi
import pytest

from deepmir_hw2.music import BeatGrid, combine_with_prompt
from deepmir_hw2.tokens import MusicTokenizer, TokenizerConfig


def test_variable_tempo_and_meter_roundtrip():
    # 4-beat, 6-beat, then 2-beat bars; tempo changes at beat four.
    times = tuple([i * 0.5 for i in range(5)] + [2 + i * 0.75 for i in range(1, 9)])
    grid = BeatGrid(times, (0, 4, 10, 12))
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="MELODY")
    inst.notes = [
        pretty_midi.Note(80, 60, grid.to_seconds(3.5), grid.to_seconds(4.5)),
        pretty_midi.Note(96, 64, grid.to_seconds(9.5), grid.to_seconds(10.5)),
    ]
    midi.instruments.append(inst)
    tokenizer = MusicTokenizer(TokenizerConfig(positions_per_beat=8))
    bars = tokenizer.encode_bars(midi, grid, [0, 4, 10, 12])
    restored = tokenizer.decode_notes(bars, grid, [0, 4, 10, 12])
    assert [tokenizer.value(bar[1]) for bar in bars] == [4, 6, 2]
    for original, decoded in zip(inst.notes, restored, strict=True):
        assert original.pitch == decoded.pitch
        assert decoded.start == pytest.approx(original.start)
        assert decoded.end == pytest.approx(original.end)
    assert tokenizer.value(bars[0][-1]) == 8


def test_prompt_is_preserved_when_written(tmp_path):
    original = pretty_midi.PrettyMIDI(initial_tempo=83)
    track = pretty_midi.Instrument(0, name="MELODY")
    track.notes = [pretty_midi.Note(72, 65, 0.2, 1.3)]
    track.control_changes = [pretty_midi.ControlChange(64, 127, 0.1)]
    original.instruments.append(track)
    path = tmp_path / "prompt.mid"
    original.write(str(path))
    loaded = pretty_midi.PrettyMIDI(str(path))
    generated = [pretty_midi.Note(90, 48, 1.5, 2.0)]
    combined = combine_with_prompt(loaded, generated, "PIANO", 3.0)
    combined.write(str(tmp_path / "combined.mid"))
    result = pretty_midi.PrettyMIDI(str(tmp_path / "combined.mid"))
    before, after = loaded.instruments[0], result.instruments[0]
    assert [(n.pitch, n.velocity, n.start, n.end) for n in before.notes] == [
        (n.pitch, n.velocity, n.start, n.end) for n in after.notes
    ]
    assert [(c.number, c.value, c.time) for c in before.control_changes] == [
        (c.number, c.value, c.time) for c in after.control_changes
    ]
    assert result.get_end_time() == pytest.approx(3.0, abs=0.005)
    assert len(loaded.instruments) == 1


def test_invalid_grid_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        BeatGrid((0.0, 0.0), (0,))
