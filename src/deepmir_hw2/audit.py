"""Check the fixed final-output layout from scripts/reproduce_all.sh."""

import argparse
import array
import hashlib
import json
import math
import sys
import wave
from pathlib import Path

import mido

from .experiment_stage import validate_final_checkpoint
from .train import checkpoint_path
from .utils import fingerprint, read_json, write_json

SETTINGS = ("A", "B", "C")
BONUS_METRICS = {
    "CA",
    "DP (Pitch-class OA)",
    "DV (Velocity OA)",
    "DD (Duration OA)",
    "DIOI (Inter-Onset OA)",
}


class Audit:
    def __init__(self):
        self.checks = []
        self.errors = []
        self.manual_review = []

    def check(self, name, passed, detail):
        item = {"name": name, "passed": bool(passed), "detail": str(detail)}
        self.checks.append(item)
        if not passed:
            self.errors.append(item)

    def note(self, title, detail):
        self.manual_review.append({"title": title, "detail": detail})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def midi_info(path: Path):
    """Read declared bar span, note-onset boundary and any release tail."""
    midi = mido.MidiFile(path)
    markers = []
    marker_seconds = {}
    tempo = maximum_tempo = 500_000
    seconds = bar_end_seconds = last_event_seconds = 0.0
    absolute_tick = last_event_tick = max_note_onset_tick = 0
    for message in mido.merge_tracks(midi.tracks):
        absolute_tick += message.time
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "marker":
            markers.append((message.text, absolute_tick))
            marker_seconds[message.text] = seconds
            if message.text == "BAR_END":
                bar_end_seconds = seconds
        elif not message.is_meta:
            last_event_tick = max(last_event_tick, absolute_tick)
            last_event_seconds = seconds
            if message.type == "note_on" and message.velocity > 0:
                max_note_onset_tick = max(max_note_onset_tick, absolute_tick)
        elif message.type == "set_tempo":
            tempo = message.tempo
            maximum_tempo = max(maximum_tempo, tempo)

    names = [name for name, _ in markers if name.startswith("BAR_") and name != "BAR_END"]
    numbers = sorted(int(name[4:]) for name in names if name[4:].isdigit())
    ends = [tick for name, tick in markers if name == "BAR_END"]
    end_tick = max(ends) if ends else None
    return {
        "bar_count": len(names),
        "markers_sequential": numbers == list(range(1, len(names) + 1)),
        "bar_end_tick": end_tick,
        "bar_end_seconds": bar_end_seconds if ends else None,
        "marker_seconds": marker_seconds,
        "note_onset_outside_span": end_tick is not None and max_note_onset_tick >= end_tick,
        "tail_ticks": max(0, last_event_tick - end_tick) if end_tick is not None else None,
        "tail_seconds": max(0.0, last_event_seconds - bar_end_seconds) if ends else None,
        "tick_seconds": maximum_tempo / midi.ticks_per_beat / 1_000_000,
    }


def check_midi_file(audit: Audit, path: Path, bars: int, label: str):
    try:
        info = midi_info(path)
    except Exception as exc:
        audit.check(f"{label}: MIDI readable", False, exc)
        return None
    audit.check(
        f"{label}: bar markers",
        info["bar_count"] == bars
        and info["markers_sequential"]
        and info["bar_end_tick"] is not None,
        f"expected BAR_1..BAR_{bars} followed by BAR_END; found {info['bar_count']} bars",
    )
    audit.check(
        f"{label}: note onsets within declared span",
        not info["note_onset_outside_span"],
        f"note_onset_outside_span={info['note_onset_outside_span']}",
    )
    if info["tail_ticks"]:
        audit.note(
            "MIDI release-tail review",
            f"{label}: events continue {info['tail_seconds']:.6f}s "
            f"({info['tail_ticks']} ticks) beyond BAR_END. Review by listening and report the "
            "finding; no tail cutoff is applied.",
        )
    return info


def check_wav_file(audit: Audit, path: Path, midi_path: Path, label: str):
    try:
        with wave.open(str(path), "rb") as audio:
            frames, rate, width, channels = (
                audio.getnframes(),
                audio.getframerate(),
                audio.getsampwidth(),
                audio.getnchannels(),
            )
            peak = 0
            if width == 2:
                while chunk := audio.readframes(65536):
                    samples = array.array("h")
                    samples.frombytes(chunk)
                    if sys.byteorder != "little":
                        samples.byteswap()
                    peak = max(peak, max((abs(sample) for sample in samples), default=0))
            valid = frames > 0 and rate >= 8000 and width == 2 and channels > 0 and peak > 0
            audio_seconds = frames / rate if rate else 0
            info = midi_info(midi_path)
            covers_span = (
                info["bar_end_seconds"] is not None and audio_seconds >= info["bar_end_seconds"]
            )
            audit.check(
                f"{label}: non-silent PCM WAV",
                valid,
                f"frames={frames}, rate={rate}, width={width}, channels={channels}, peak={peak}",
            )
            audit.check(
                f"{label}: WAV covers MIDI bar span",
                covers_span,
                f"audio={audio_seconds:.6f}s, BAR_END={info['bar_end_seconds']}s",
            )
    except Exception as exc:
        audit.check(f"{label}: WAV readable", False, exc)


def validate_prompt(
    audit: Audit, prompt_dir: Path, output_path: Path, midi_name: str, bars: int, label: str
):
    """Check source track events and prompt bar starts within one output MIDI tick."""
    try:
        import pretty_midi

        from .music import BeatGrid

        source = pretty_midi.PrettyMIDI(str(prompt_dir / midi_name))
        output = pretty_midi.PrettyMIDI(str(output_path))
        info = midi_info(output_path)
        grid = BeatGrid.load(prompt_dir / "beat_midi.txt")
        boundaries = grid.boundaries()
        timing_ok = len(boundaries) - 1 == bars
        tolerance = info["tick_seconds"]
        markers = info["marker_seconds"]
        if timing_ok:
            for index in range(bars + 1):
                marker = (
                    markers.get(f"BAR_{index + 1}")
                    if index < bars
                    else markers.get(f"BAR_{bars + 1}", markers.get("BAR_END"))
                )
                expected = grid.to_seconds(boundaries[index])
                if marker is None or abs(marker - expected) > tolerance:
                    timing_ok = False
                    break
        events_ok = len(output.instruments) >= len(source.instruments)
        for index, original in enumerate(source.instruments):
            if not events_ok:
                break
            copied = output.instruments[index]
            events_ok = original.program == copied.program and original.is_drum == copied.is_drum
            expected_notes = sorted(
                original.notes, key=lambda n: (n.pitch, n.velocity, n.start, n.end)
            )
            copied_notes = sorted(copied.notes, key=lambda n: (n.pitch, n.velocity, n.start, n.end))
            events_ok = events_ok and len(expected_notes) == len(copied_notes)
            if events_ok:
                events_ok = all(
                    left.pitch == right.pitch
                    and left.velocity == right.velocity
                    and abs(left.start - right.start) <= tolerance
                    and abs(left.end - right.end) <= tolerance
                    for left, right in zip(expected_notes, copied_notes, strict=True)
                )
            expected_cc = sorted(
                original.control_changes, key=lambda x: (x.number, x.value, x.time)
            )
            copied_cc = sorted(copied.control_changes, key=lambda x: (x.number, x.value, x.time))
            events_ok = events_ok and len(expected_cc) == len(copied_cc)
            if events_ok:
                events_ok = all(
                    left.number == right.number
                    and left.value == right.value
                    and abs(left.time - right.time) <= tolerance
                    for left, right in zip(expected_cc, copied_cc, strict=True)
                )
            expected_bends = sorted(original.pitch_bends, key=lambda x: (x.pitch, x.time))
            copied_bends = sorted(copied.pitch_bends, key=lambda x: (x.pitch, x.time))
            events_ok = events_ok and len(expected_bends) == len(copied_bends)
            if events_ok:
                events_ok = all(
                    left.pitch == right.pitch and abs(left.time - right.time) <= tolerance
                    for left, right in zip(expected_bends, copied_bends, strict=True)
                )
        audit.check(f"{label}: prompt timing preserved", timing_ok, f"tolerance={tolerance:.9f}s")
        audit.check(f"{label}: prompt events preserved", events_ok, f"tolerance={tolerance:.9f}s")
    except Exception as exc:
        audit.check(f"{label}: prompt preservation readable", False, exc)


def _sampling_signature(metadata):
    try:
        sample = metadata["sampling"]
        temperature = float(sample["temperature"])
        top_k = sample["top_k"]
        top_p = float(sample["top_p"])
        strategy = sample["numeric_strategy"]
        # The generator writes beam_width (the config object itself has no
        # width field). Ignore it for sample mode because it is unused there.
        width = sample.get("beam_width")
        if (
            not math.isfinite(temperature)
            or temperature <= 0
            or not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 0
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
            or strategy not in {"sample", "beam"}
        ):
            return None
        if strategy == "beam":
            if type(width) is not int or width != 4:
                return None
        else:
            width = None
        return temperature, top_k, top_p, strategy, width
    except (KeyError, TypeError, ValueError):
        return None


def check_metrics(path: Path, expected_pieces: int, required: set[str]):
    summary = read_json(path)
    metrics = summary.get("metrics", {})
    valid = summary.get("valid_pieces", {})
    good = (
        summary.get("complete") is True
        and summary.get("pieces") == expected_pieces
        and all(
            isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics.get(key), bool)
            and math.isfinite(metrics[key])
            and valid.get(key) == expected_pieces
            for key in required
        )
    )
    return good, summary


def check_final_model(audit: Audit, root: Path, task: str):
    run_dir = root / "runs" / f"{task}-final"
    export_path = root / "models" / f"{task}.pt"
    try:
        run_checkpoint = checkpoint_path(run_dir, "latest").resolve()
        state = validate_final_checkpoint(run_dir)
        exported = load_checkpoint_file(export_path)
        prepared = read_json(root / "data/prepared/manifest.json")
        prepared_names = {record["song"] for record in prepared["songs"]}
        prepared_fp = prepared["fingerprint"]
        prepared_ok = (
            len(prepared_names) == 896
            and fingerprint({key: value for key, value in prepared.items() if key != "fingerprint"})
            == prepared_fp
        )
        provenance = state.get("training_provenance", {})
        source_ok = (
            prepared_ok
            and set(provenance.get("source_names", [])) == prepared_names
            and provenance.get("data_fingerprint") == prepared_fp
            and state.get("data_fingerprint") == prepared_fp
            and state.get("source_programs") == prepared.get("source_programs")
            and fingerprint(state.get("tokenizer_tokens", [])) == prepared.get("tokenizer_sha256")
        )
        source_match = exported.get(
            "source_checkpoint_name"
        ) == run_checkpoint.name and exported.get("source_checkpoint_sha256") == sha256(
            run_checkpoint
        )
        same_fields = all(
            exported.get(key) == state.get(key)
            for key in (
                "config",
                "tokenizer_tokens",
                "tokenizer_config",
                "training_provenance",
                "data_fingerprint",
                "source_programs",
                "step",
            )
        )
        same_weights = exported.get("model", {}).keys() == state.get("model", {}).keys() and all(
            exported["model"][key].equal(state["model"][key]) for key in state["model"]
        )
        good = (
            exported.get("inference_only") is True
            and source_ok
            and exported.get("config", {}).get("task") == task
            and source_match
            and same_fields
            and same_weights
        )
        audit.check(
            f"{task}: complete random-init 896-song export",
            good,
            "final-step provenance, prepared source names, per-song exposure, fingerprint, "
            "source checkpoint, config/tokenizer and weights must match",
        )
        return sha256(export_path) if good else None
    except Exception as exc:
        audit.check(f"{task}: final export", False, exc)
        return None


def load_checkpoint_file(path: Path):
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def check_generation(
    audit: Audit, directory: Path, label: str, mode: str, count: int, digest: str | None
):
    try:
        metadata = read_json(directory / "generation.json")
        songs = metadata.get("songs", [])
        if not isinstance(songs, list) or any(not isinstance(song, dict) for song in songs):
            audit.check(f"{label}: generation metadata", False, "songs must be a list of records")
            return {}
        good = (
            metadata.get("mode") == mode
            and metadata.get("complete") is True
            and metadata.get("midi_complete") is True
            and metadata.get("expected_pieces") == count
            and len(songs) == count
            and digest is not None
            and metadata.get("checkpoint_sha256") == digest
        )
        audit.check(f"{label}: complete generation/checkpoint", good, str(directory))
        return metadata
    except Exception as exc:
        audit.check(f"{label}: generation metadata", False, exc)
        return {}


def check_inventory(audit: Audit, directory: Path, expected_names: set[str], ext: str, label: str):
    files = sorted(directory.glob(f"*.{ext}"))
    names = {path.stem for path in files}
    audit.check(
        f"{label}: {ext.upper()} inventory",
        len(files) == len(expected_names) and names == expected_names,
        f"expected {len(expected_names)}; found {len(files)}",
    )
    return files


def _directories(path: Path, expected_count: int, label: str, audit: Audit):
    names = sorted(item.name for item in path.iterdir() if item.is_dir()) if path.is_dir() else []
    audit.check(f"{label}: prompt folders", len(names) == expected_count, f"found {len(names)}")
    return set(names)


def audit_root(root: Path, selected_setting: str) -> dict:
    root = root.resolve()
    audit = Audit()
    source_manifest_path = root / "data/prepared/manifest.json"
    try:
        prepared = read_json(source_manifest_path)
        prepared_names = {record["song"] for record in prepared["songs"]}
        source_ok = len(prepared_names) == 896 and fingerprint(
            {key: value for key, value in prepared.items() if key != "fingerprint"}
        ) == prepared.get("fingerprint")
    except Exception:
        prepared_names, source_ok = set(), False
    audit.check("prepared source manifest: 896 songs", source_ok, str(source_manifest_path))

    basic_hash = check_final_model(audit, root, "basic")
    bonus_hash = check_final_model(audit, root, "bonus")

    if selected_setting not in SETTINGS:
        audit.check("Basic A selected setting", False, "must be A, B or C")
    basic_a_names = {f"{index:03d}" for index in range(1, 21)}
    a_signatures = {}
    basic_hashes = set()
    for setting in SETTINGS:
        directory = root / "outputs/basic-a" / setting
        metadata = check_generation(
            audit, directory, f"Basic A {setting}", "unconditional", 20, basic_hash
        )
        audit.check(
            f"Basic A {setting}: generation records",
            {entry.get("song") for entry in metadata.get("songs", [])} == basic_a_names,
            "one record for each numbered piece 001 through 020",
        )
        signature = _sampling_signature(metadata)
        a_signatures[setting] = signature
        files = check_inventory(audit, directory, basic_a_names, "mid", f"Basic A {setting}")
        for midi_path in files:
            check_midi_file(audit, midi_path, 32, f"Basic A {setting}/{midi_path.name}")
        if setting == selected_setting:
            wavs = check_inventory(audit, directory, basic_a_names, "wav", f"Basic A {setting}")
            for wav_path in wavs:
                check_wav_file(
                    audit,
                    wav_path,
                    directory / f"{wav_path.stem}.mid",
                    f"Basic A {setting}/{wav_path.name}",
                )
        if metadata.get("checkpoint_sha256"):
            basic_hashes.add(metadata["checkpoint_sha256"])
    audit.check(
        "Basic A: three distinct sampling settings",
        None not in a_signatures.values() and len(set(a_signatures.values())) == 3,
        str(a_signatures),
    )

    prompt_root = root / "POP909/prompts/basic-b"
    basic_b_names = _directories(prompt_root, 3, "Basic B", audit)
    for setting in SETTINGS:
        directory = root / "outputs/basic-b" / setting
        metadata = check_generation(
            audit, directory, f"Basic B {setting}", "continue", 3, basic_hash
        )
        audit.check(
            f"Basic B {setting}: matches Basic A sampling",
            _sampling_signature(metadata) == a_signatures.get(setting),
            f"Basic B={_sampling_signature(metadata)}, Basic A={a_signatures.get(setting)}",
        )
        if metadata.get("checkpoint_sha256"):
            basic_hashes.add(metadata["checkpoint_sha256"])
        midi_files = check_inventory(audit, directory, basic_b_names, "mid", f"Basic B {setting}")
        wav_files = check_inventory(audit, directory, basic_b_names, "wav", f"Basic B {setting}")
        songs = {entry.get("song") for entry in metadata.get("songs", [])}
        audit.check(
            f"Basic B {setting}: prompt records", songs == basic_b_names, str(sorted(songs))
        )
        for midi_path in midi_files:
            check_midi_file(audit, midi_path, 32, f"Basic B {setting}/{midi_path.name}")
            prompt_dir = prompt_root / midi_path.stem
            source_name = f"{midi_path.stem}.mid"
            record = next(
                (row for row in metadata.get("songs", []) if row.get("song") == midi_path.stem), {}
            )
            try:
                source_hash = sha256(prompt_dir / source_name)
            except OSError:
                source_hash = None
            audit.check(
                f"Basic B {setting}/{midi_path.stem}: prompt hash",
                source_hash is not None and record.get("prompt_sha256") == source_hash,
                f"source={source_hash}, recorded={record.get('prompt_sha256')}",
            )
            validate_prompt(
                audit, prompt_dir, midi_path, source_name, 8, f"Basic B {setting}/{midi_path.stem}"
            )
        for wav_path in wav_files:
            check_wav_file(
                audit,
                wav_path,
                directory / f"{wav_path.stem}.mid",
                f"Basic B {setting}/{wav_path.name}",
            )
    audit.check(
        "Basic A/B: one exported checkpoint",
        basic_hash is not None and basic_hashes == {basic_hash},
        f"observed={sorted(basic_hashes)}",
    )

    bonus_prompt_root = root / "POP909/prompts/bonus"
    bonus_names = _directories(bonus_prompt_root, 10, "Bonus", audit)
    bonus_dir = root / "outputs/bonus"
    bonus_meta = check_generation(audit, bonus_dir, "Bonus", "accompany", 10, bonus_hash)
    bonus_files = check_inventory(audit, bonus_dir, bonus_names, "mid", "Bonus")
    bonus_wavs = check_inventory(audit, bonus_dir, bonus_names, "wav", "Bonus")
    audit.check(
        "Bonus: melody prompt records",
        {entry.get("song") for entry in bonus_meta.get("songs", [])} == bonus_names,
        "one record per supplied melody prompt",
    )
    for midi_path in bonus_files:
        check_midi_file(audit, midi_path, 16, f"Bonus/{midi_path.name}")
        prompt_dir = bonus_prompt_root / midi_path.stem
        record = next(
            (row for row in bonus_meta.get("songs", []) if row.get("song") == midi_path.stem), {}
        )
        try:
            source_hash = sha256(prompt_dir / "melody.mid")
        except OSError:
            source_hash = None
        audit.check(
            f"Bonus/{midi_path.stem}: melody prompt hash",
            source_hash is not None and record.get("prompt_sha256") == source_hash,
            f"source={source_hash}, recorded={record.get('prompt_sha256')}",
        )
        validate_prompt(audit, prompt_dir, midi_path, "melody.mid", 16, f"Bonus/{midi_path.stem}")
    for wav_path in bonus_wavs:
        check_wav_file(
            audit, wav_path, bonus_dir / f"{wav_path.stem}.mid", f"Bonus/{wav_path.name}"
        )

    metric_dir = root / "outputs/metrics"
    metric_specs = [("real", 896, {"H4", "GS"})]
    metric_specs.extend((setting, 20, {"H4", "GS"}) for setting in SETTINGS)
    metric_specs.append(("bonus", 10, BONUS_METRICS))
    for name, pieces, required in metric_specs:
        path = metric_dir / f"{name}.summary.json"
        try:
            good, summary = check_metrics(path, pieces, required)
            audit.check(
                f"metrics/{name}: required TA metrics",
                good,
                f"pieces={summary.get('pieces')}, required={sorted(required)}",
            )
        except Exception as exc:
            audit.check(f"metrics/{name}: readable", False, exc)

    required_files = [
        root / "models/basic.pt",
        root / "models/bonus.pt",
        root / "README.md",
        root / "requirements.txt",
        root / "r14725022_report.pdf",
    ]
    for path in required_files:
        audit.check(f"required file: {path.relative_to(root)}", path.is_file(), str(path))

    audit.note(
        "Listening and selection",
        f"Review selected Basic A's 20 files and all 9 Basic B outputs. Choose the best setting "
        f"for each prompt among {sorted(basic_b_names)}; review all 10 Bonus outputs.",
    )
    audit.note(
        "Report and sharing",
        "Review report content, citations and 16:9 layout; verify cloud link and LMS package.",
    )
    return {
        "root": str(root),
        "selected_basic_a_setting": selected_setting,
        "automated_checks_passed": not audit.errors,
        "checks": audit.checks,
        "errors": audit.errors,
        "manual_review": audit.manual_review,
    }


def audit(args: argparse.Namespace):
    result = audit_root(Path(args.root), args.selected_setting)
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = Path(args.root).resolve() / output
        write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["automated_checks_passed"]:
        raise SystemExit(1)


def add_parser(subparsers):
    parser = subparsers.add_parser("audit", help=__doc__)
    parser.add_argument(
        "--root", default=".", help="Project root containing runs/, models/ and outputs/"
    )
    parser.add_argument("--selected-setting", choices=SETTINGS, required=True)
    parser.add_argument("--output", help="Optional JSON audit report path")
    parser.set_defaults(func=audit)
