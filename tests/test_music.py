import mido
import pretty_midi
import pytest

from deepmir_hw2.music import BeatGrid, combine_with_prompt, source_timings, write_midi
from deepmir_hw2.tokens import MusicTokenizer, TokenizerConfig


def test_codec_preserves_expression_long_notes_and_duplicate_onsets():
    grid = BeatGrid(tuple(i / 2 for i in range(9)), (0, 4, 8))
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="PIANO")
    inst.notes = [
        pretty_midi.Note(17, 0, 0.123456, 20.654321),
        pretty_midi.Note(126, 127, 0.123456, 0.999999),
        pretty_midi.Note(81, 0, 0.123456, 1.234567),
    ]
    inst.control_changes = [
        pretty_midi.ControlChange(64, 127, 0.135791),
        pretty_midi.ControlChange(64, 0, 4.0),
    ]
    inst.pitch_bends = [pretty_midi.PitchBend(-1234, 0.246802)]
    midi.instruments.append(inst)
    tokenizer = MusicTokenizer()
    bars = tokenizer.encode_bars(midi, grid, [0, 4, 8])
    restored = tokenizer.decode_instrument(bars)

    def key(n):
        return n.start, n.pitch, n.velocity, n.end

    before, after = sorted(inst.notes, key=key), sorted(restored.notes, key=key)
    for a, b in zip(before, after, strict=True):
        assert (a.pitch, a.velocity) == (b.pitch, b.velocity)
        assert b.start == pytest.approx(a.start, abs=1e-6)
        assert b.end == pytest.approx(a.end, abs=1e-6)
    assert [(c.number, c.value, c.time) for c in restored.control_changes] == [
        (c.number, c.value, c.time) for c in inst.control_changes
    ]
    assert restored.pitch_bends[0].pitch == -1234
    assert restored.pitch_bends[0].time == pytest.approx(0.246802)


def test_notes_before_first_annotated_beat_are_retained():
    grid = BeatGrid((0.05, 0.55, 1.05), (0, 2))
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="PIANO")
    inst.notes = [pretty_midi.Note(50, 60, 0.01, 0.3)]
    midi.instruments.append(inst)
    tokenizer = MusicTokenizer(TokenizerConfig((2,), ((4, 4),)))
    bars = tokenizer.encode_bars(midi, grid, [0, 2])
    decoded = tokenizer.decode_instrument(bars, origin_us=50_000)
    assert decoded.notes[0].start == pytest.approx(0.01)


def test_out_of_range_onsets_are_reported_not_dropped():
    grid = BeatGrid((0.0, 0.5, 1.0), (0, 2))
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(80, 60, 1.0, 1.5)]
    midi.instruments.append(inst)
    tokenizer = MusicTokenizer(TokenizerConfig((2,), ((4, 4),)))
    with pytest.raises(ValueError, match="outside"):
        tokenizer.encode_bars(midi, grid, [0, 2])


@pytest.mark.parametrize("pulses,bars,pulse_us", [(4, 32, 936768), (6, 16, 405405)])
def test_export_meter_and_seconds_agree_with_annotated_bars(tmp_path, pulses, bars, pulse_us):
    # Input meter/tempo metadata can disagree with the annotation's pulse clock.
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0)]
    melody = pretty_midi.Instrument(0, name="MELODY")
    melody.notes = [pretty_midi.Note(73, 65, 0.123456, 0.765432)]
    melody.control_changes = [pretty_midi.ControlChange(64, 127, 0.1)]
    midi.instruments.append(melody)
    grid = BeatGrid(
        tuple(i * pulse_us / 1e6 for i in range(pulses * bars + 1)),
        tuple(range(0, pulses * bars + 1, pulses)),
    )
    timings = source_timings(midi, grid, list(grid.downbeats))
    piano = pretty_midi.Instrument(0, name="PIANO")
    piano.notes = [pretty_midi.Note(79, 48, 1.234567, 2.345678)]
    path = tmp_path / "output.mid"
    write_midi(path, combine_with_prompt(midi, piano, "PIANO"), timings)
    raw = mido.MidiFile(path)
    tick, markers, signatures = 0, [], []
    for event in raw.tracks[0]:
        tick += event.time
        if event.type == "marker":
            markers.append(tick)
        if event.type == "time_signature":
            signatures.append((event.numerator, event.denominator))
    assert signatures == [(4, 4)]
    assert len(markers) == bars + 1
    assert all(b - a == raw.ticks_per_beat * 4 for a, b in zip(markers, markers[1:], strict=False))
    assert raw.length == pytest.approx(pulses * bars * pulse_us / 1e6, abs=1e-5)
    restored = pretty_midi.PrettyMIDI(str(path))
    for before, after in zip(midi.instruments[0].notes, restored.instruments[0].notes, strict=True):
        assert (before.pitch, before.velocity) == (after.pitch, after.velocity)
        assert after.start == pytest.approx(before.start, abs=5e-5)
        assert after.end == pytest.approx(before.end, abs=5e-5)
    assert restored.instruments[0].control_changes[0].time == pytest.approx(0.1, abs=5e-5)
    assert len(midi.instruments) == 1


def test_invalid_grid_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        BeatGrid((0.0, 0.0), (0,))
