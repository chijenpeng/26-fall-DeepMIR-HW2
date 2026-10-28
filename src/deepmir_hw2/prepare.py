"""Tokenize every POP909 training song into per-bar REMI tokens."""

import json
from pathlib import Path

import pretty_midi
from tqdm import tqdm

from . import remi


def encode_song(folder):
    folder = Path(folder)
    midi = pretty_midi.PrettyMIDI(str(folder / f"{folder.name}.mid"))
    notes = [n for inst in midi.instruments for n in inst.notes]
    grid = remi.Grid.from_file(folder / "beat_midi.txt", notes)
    return {
        # Basic tasks: all three tracks merged, with the supplied chord annotations.
        "basic": remi.encode(notes, grid, remi.read_chords(folder / "chord_midi.txt")),
        # Bonus task: MELODY (encoder input) and PIANO (decoder target), no chords.
        "melody": remi.encode(remi.track(midi, "MELODY"), grid),
        "piano": remi.encode(remi.track(midi, "PIANO"), grid, header=False),
    }


def prepare(args):
    folders = sorted(p for p in Path(args.dataset).iterdir() if p.is_dir())
    if len(folders) != args.expected_songs:
        raise ValueError(f"Expected {args.expected_songs} songs, found {len(folders)}")
    songs = {folder.name: encode_song(folder) for folder in tqdm(folders)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"vocab": remi.VOCAB, "songs": songs}))
    print(f"Wrote {len(songs)} songs to {output}")


def add_parser(subparsers):
    parser = subparsers.add_parser("prepare", help=__doc__)
    parser.add_argument("--dataset", default="POP909/training")
    parser.add_argument("--output", default="data/remi.json")
    parser.add_argument("--expected-songs", type=int, default=896)
    parser.set_defaults(func=prepare)
