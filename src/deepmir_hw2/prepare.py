"""Prepare aligned track tokens, retaining a provenance manifest for every song."""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
from tqdm import tqdm

from .music import BeatGrid
from .tokens import MusicTokenizer, TokenizerConfig
from .utils import fingerprint, write_json


def pack_bars(bars: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([len(bar) for bar in bars], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    return np.array([t for bar in bars for t in bar], dtype=np.int16), offsets


def prepare(args: argparse.Namespace) -> None:
    source, output = Path(args.dataset), Path(args.output)
    if source.name != "training":
        raise ValueError("--dataset must point to POP909/training, never prompts")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty preprocessing directory: {output}")
    songs = sorted(p for p in source.iterdir() if p.is_dir())
    if not songs:
        raise ValueError("No training songs found")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = MusicTokenizer(TokenizerConfig(positions_per_beat=args.positions_per_beat))
    tokenizer.save(output / "tokenizer.json")
    records = []
    for folder in tqdm(songs, desc="Tokenizing training songs"):
        midi_path, beat_path = folder / f"{folder.name}.mid", folder / "beat_midi.txt"
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        grid = BeatGrid.load(beat_path)
        boundaries = grid.boundaries(midi.get_end_time())
        payload = {}
        counts = {}
        for track in ["basic", "melody", "piano"]:
            bars = tokenizer.encode_bars(
                midi, grid, boundaries, None if track == "basic" else track.upper()
            )
            for bar in bars:
                tokenizer.validate_bar(bar)
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
                "tokens": counts,
                "sha256_midi": hashlib.sha256(midi_path.read_bytes()).hexdigest(),
                "sha256_beats": hashlib.sha256(beat_path.read_bytes()).hexdigest(),
            }
        )
    manifest = {"version": 1, "tokenizer_sha256": fingerprint(tokenizer.tokens), "songs": records}
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    write_json(output / "tok2idx.json", tokenizer.tok2idx)
    print(f"Prepared {len(records)} songs; vocabulary={len(tokenizer)}; output={output}")


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("prepare", help=__doc__)
    parser.add_argument("--dataset", default="POP909/training")
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--positions-per-beat", type=int, default=12)
    parser.set_defaults(func=prepare)
