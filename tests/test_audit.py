import json
import wave
from pathlib import Path

import mido

from deepmir_hw2.audit import (
    Audit,
    _sampling_signature,
    check_generation,
    check_metrics,
    check_midi_file,
    check_wav_file,
    midi_info,
    validate_prompt,
)


def _write_midi(path: Path, bars: int, *, tail_ticks: int = 0, onset_after_end: bool = False):
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    midi.tracks.append(meta)
    for index in range(1, bars + 1):
        meta.append(mido.MetaMessage("marker", text=f"BAR_{index}", time=0 if index == 1 else 1))
    meta.append(mido.MetaMessage("marker", text="BAR_END", time=1))
    notes = mido.MidiTrack()
    midi.tracks.append(notes)
    notes.extend(
        [
            mido.Message("program_change", channel=0, program=0, time=0),
            mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
            mido.Message("note_off", channel=0, note=60, velocity=0, time=bars + tail_ticks),
        ]
    )
    if onset_after_end:
        late = mido.MidiTrack()
        midi.tracks.append(late)
        late.append(mido.Message("note_on", channel=1, note=61, velocity=80, time=bars + 1))
    midi.save(path)


def _write_prompt_midi(path: Path, bars: int, *, prompt_pitch: int = 60):
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    midi.tracks.append(meta)
    for index in range(1, bars + 1):
        meta.append(mido.MetaMessage("marker", text=f"BAR_{index}", time=0 if index == 1 else 1))
    meta.append(mido.MetaMessage("marker", text="BAR_END", time=1))
    source = mido.MidiTrack()
    midi.tracks.append(source)
    source.extend(
        [
            mido.Message("program_change", channel=0, program=0, time=0),
            mido.Message("note_on", channel=0, note=prompt_pitch, velocity=80, time=0),
            mido.Message("note_off", channel=0, note=prompt_pitch, velocity=0, time=bars),
        ]
    )
    midi.save(path)


def _write_continuation(path: Path, bars: int, prompt_bars: int, *, changed_pitch: bool = False):
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    midi.tracks.append(meta)
    for index in range(1, bars + 1):
        meta.append(mido.MetaMessage("marker", text=f"BAR_{index}", time=0 if index == 1 else 1))
    meta.append(mido.MetaMessage("marker", text="BAR_END", time=1))
    prompt = mido.MidiTrack()
    midi.tracks.append(prompt)
    pitch = 61 if changed_pitch else 60
    prompt.extend(
        [
            mido.Message("program_change", channel=0, program=0, time=0),
            mido.Message("note_on", channel=0, note=pitch, velocity=80, time=0),
            mido.Message("note_off", channel=0, note=pitch, velocity=0, time=prompt_bars),
        ]
    )
    generated = mido.MidiTrack()
    midi.tracks.append(generated)
    generated.extend(
        [
            mido.Message("program_change", channel=1, program=0, time=0),
            mido.Message("note_on", channel=1, note=64, velocity=80, time=prompt_bars),
            mido.Message("note_off", channel=1, note=64, velocity=0, time=bars - prompt_bars),
        ]
    )
    midi.save(path)


def _write_beat_grid(path: Path, bars: int):
    interval = 0.5 / 480
    path.write_text("".join(f"{index * interval:.12f} 0 1\n" for index in range(bars + 1)))


def _write_wav(path: Path, *, silent: bool = False, frames: int = 1000):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        sample = b"\0\0" if silent else b"\x01\0"
        audio.writeframes(sample * frames)


def test_midi_validation_checks_bars_and_note_onsets_but_reports_release_tail(tmp_path):
    midi = tmp_path / "32-bars.mid"
    _write_midi(midi, 32, tail_ticks=604 * 480)
    audit = Audit()

    check_midi_file(audit, midi, 32, "test")

    assert not audit.errors
    assert any("302." in item["detail"] for item in audit.manual_review)
    assert midi_info(midi)["bar_count"] == 32


def test_note_on_after_bar_end_fails(tmp_path):
    midi = tmp_path / "late-onset.mid"
    _write_midi(midi, 32, onset_after_end=True)
    audit = Audit()

    check_midi_file(audit, midi, 32, "test")

    assert any("note onsets within declared span" in item["name"] for item in audit.errors)


def test_wav_check_rejects_silence_and_duration_shorter_than_bars(tmp_path):
    midi = tmp_path / "sample.mid"
    _write_midi(midi, 32)
    wav = tmp_path / "silent.wav"
    _write_wav(wav, silent=True)
    audit = Audit()

    check_wav_file(audit, wav, midi, "test")

    assert any("non-silent PCM" in item["name"] for item in audit.errors)
    short_wav = tmp_path / "short.wav"
    _write_wav(short_wav, frames=1)
    short_audit = Audit()
    check_wav_file(short_audit, short_wav, midi, "test")
    assert any("covers MIDI bar span" in item["name"] for item in short_audit.errors)


def test_prompt_events_and_bar_timing_are_checked_against_source(tmp_path):
    prompt_dir = tmp_path / "001"
    prompt_dir.mkdir()
    source = prompt_dir / "001.mid"
    output = tmp_path / "continued.mid"
    _write_prompt_midi(source, 8)
    _write_beat_grid(prompt_dir / "beat_midi.txt", 8)
    _write_continuation(output, 32, 8)
    audit = Audit()

    validate_prompt(audit, prompt_dir, output, source.name, 8, "Basic B")

    assert not audit.errors

    changed = tmp_path / "changed.mid"
    _write_continuation(changed, 32, 8, changed_pitch=True)
    changed_audit = Audit()
    validate_prompt(changed_audit, prompt_dir, changed, source.name, 8, "Basic B")
    assert any("prompt events preserved" in item["name"] for item in changed_audit.errors)

    rows = (prompt_dir / "beat_midi.txt").read_text().splitlines()
    (prompt_dir / "beat_midi.txt").write_text(
        "".join(f"{float(row.split()[0]) + 0.1} 0 1\n" for row in rows)
    )
    timing_audit = Audit()
    validate_prompt(timing_audit, prompt_dir, output, source.name, 8, "Basic B")
    assert any("prompt timing preserved" in item["name"] for item in timing_audit.errors)


def test_metric_summary_checks_expected_count_finite_values_and_required_keys(tmp_path):
    summary_path = tmp_path / "metrics.json"
    summary = {
        "pieces": 896,
        "complete": True,
        "metrics": {"H4": 1.0, "GS": 0.5},
        "valid_pieces": {"H4": 896, "GS": 896},
    }
    summary_path.write_text(json.dumps(summary))

    good, _ = check_metrics(summary_path, 896, {"H4", "GS"})
    assert good

    summary["pieces"] = 1
    summary["metrics"]["GS"] = float("nan")
    summary_path.write_text(json.dumps(summary))
    bad, _ = check_metrics(summary_path, 896, {"H4", "GS"})
    assert not bad


def test_sampling_signature_distinguishes_declared_settings_and_beam_width():
    sample = {
        "sampling": {
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 1.0,
            "numeric_strategy": "sample",
            "beam_width": 4,
        }
    }
    other_unused_width = json.loads(json.dumps(sample))
    other_unused_width["sampling"]["beam_width"] = 8
    assert _sampling_signature(sample) == _sampling_signature(other_unused_width)

    beam = json.loads(json.dumps(sample))
    beam["sampling"].update(numeric_strategy="beam", beam_width=4)
    invalid_beam = json.loads(json.dumps(beam))
    invalid_beam["sampling"]["beam_width"] = 8
    assert _sampling_signature(beam) is not None
    assert _sampling_signature(invalid_beam) is None


def test_generation_metadata_rejects_failed_or_malformed_manifest(tmp_path):
    metadata = {
        "mode": "unconditional",
        "complete": False,
        "midi_complete": True,
        "expected_pieces": 1,
        "checkpoint_sha256": "verified-export-hash",
        "songs": [{"song": "001"}],
    }
    (tmp_path / "generation.json").write_text(json.dumps(metadata))
    failed = Audit()
    check_generation(failed, tmp_path, "Basic A A", "unconditional", 1, "verified-export-hash")
    assert any("complete generation" in item["name"] for item in failed.errors)

    metadata["complete"] = True
    metadata["songs"] = ["not a record"]
    (tmp_path / "generation.json").write_text(json.dumps(metadata))
    malformed = Audit()
    result = check_generation(
        malformed, tmp_path, "Basic A A", "unconditional", 1, "verified-export-hash"
    )
    assert result == {}
    assert malformed.errors
