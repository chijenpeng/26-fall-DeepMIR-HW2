"""Prepare source-timed events without clipping notes, velocities or controllers."""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
from tqdm import tqdm

from .music import BeatGrid, event_end_time, select_instruments, source_timings
from .tokens import MusicTokenizer, TokenizerConfig, transpose_chord
from .utils import fingerprint, write_json


def pack_bars(bars):
    offsets = np.concatenate(([0], np.cumsum([len(bar) for bar in bars], dtype=np.int64)))
    return np.array([token for bar in bars for token in bar], dtype=np.int32), offsets


def load_song(folder):
    midi = pretty_midi.PrettyMIDI(str(folder / f"{folder.name}.mid"))
    grid = BeatGrid.load(folder / "beat_midi.txt")
    boundaries = grid.boundaries(event_end_time(midi))
    return midi, grid, boundaries


def load_chords(path):
    result = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts:
            if len(parts) < 3:
                raise ValueError(f"Malformed chord annotation: {path}")
            result.append((float(parts[0]), parts[2]))
    return result


def prepare(args):
    source, output = Path(args.dataset), Path(args.output)
    if source.name != "training":
        raise ValueError("--dataset must point to POP909/training")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty preprocessing directory: {output}")
    folders = sorted(path for path in source.iterdir() if path.is_dir())
    if not folders:
        raise ValueError("No training songs found")

    # Derive the finite musical vocabulary from provided training information.
    beats, meters, chords = set(), set(), set()
    programs = {"basic": set(), "melody": set(), "piano": set()}
    for folder in tqdm(folders, desc="Reading source schema"):
        midi, grid, boundaries = load_song(folder)
        for timing in source_timings(midi, grid, boundaries):
            beats.add(len(timing.beat_us))
            meters.add((timing.numerator, timing.denominator))
        for _, label in load_chords(folder / "chord_midi.txt"):
            chords.add(label)
            chords.update(transpose_chord(label, shift) for shift in range(12))
        for track in programs:
            programs[track].update(
                int(inst.program)
                for inst in select_instruments(midi, None if track == "basic" else track.upper())
            )
    tokenizer = MusicTokenizer(
        TokenizerConfig(tuple(sorted(beats)), tuple(sorted(meters)), tuple(sorted(chords)))
    )
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    records = []
    for folder in tqdm(folders, desc="Encoding training songs"):
        midi, grid, boundaries = load_song(folder)
        annotation = load_chords(folder / "chord_midi.txt")
        payload, counts = {}, {}
        for track in ["basic", "melody", "piano"]:
            bars = tokenizer.encode_bars(
                midi,
                grid,
                boundaries,
                None if track == "basic" else track.upper(),
                annotation if track == "basic" else None,
            )
            # Bonus encoder/target never receives reference chord annotations.
            ids, offsets = pack_bars(bars)
            payload[track], payload[f"{track}_offsets"] = ids, offsets
            counts[track] = len(ids)
        fd, temporary = tempfile.mkstemp(dir=output, suffix=".npz.tmp")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, **payload)
            os.replace(temporary, output / f"{folder.name}.npz")
        finally:
            Path(temporary).unlink(missing_ok=True)
        records.append(
            {
                "song": folder.name,
                "file": f"{folder.name}.npz",
                "bars": len(boundaries) - 1,
                "origin_us": round(grid.times[0] * 1_000_000),
                "extrapolated_terminal_pulses": max(0, boundaries[-1] - len(grid.times) + 1),
                "tokens": counts,
                **{
                    f"sha256_{key}": hashlib.sha256((folder / name).read_bytes()).hexdigest()
                    for key, name in [
                        ("midi", f"{folder.name}.mid"),
                        ("beats", "beat_midi.txt"),
                        ("chords", "chord_midi.txt"),
                    ]
                },
            }
        )
    manifest = {
        "version": tokenizer.version,
        "tokenizer_sha256": fingerprint(tokenizer.tokens),
        "source_programs": {track: sorted(values) for track, values in programs.items()},
        "songs": records,
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    write_json(output / "tok2idx.json", tokenizer.evaluation_vocabulary())
    print(f"Prepared {len(records)} songs, vocabulary={len(tokenizer)}, output={output}")


def add_parser(subparsers):
    parser = subparsers.add_parser("prepare", help=__doc__)
    parser.add_argument("--dataset", default="POP909/training")
    parser.add_argument("--output", default="data/prepared")
    parser.set_defaults(func=prepare)
