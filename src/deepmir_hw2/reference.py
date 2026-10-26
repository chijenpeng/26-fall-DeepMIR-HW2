"""Select reproducible 32-bar real-data windows for matched basic evaluation."""

import argparse
import random
from pathlib import Path

from .data import MusicCorpus
from .utils import write_json


def reference(args: argparse.Namespace) -> None:
    corpus = MusicCorpus(args.data)
    tokenizer = corpus.tokenizer
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty reference directory: {output}")
    if args.count < 1 or args.bars < 4:
        raise ValueError("Need count >= 1 and bars >= 4")
    names = sorted(corpus.songs)
    rng = random.Random(args.seed)
    rng.shuffle(names)
    records = []
    for name in names:
        song = corpus.songs[name]
        offsets = song["basic_offsets"]
        eligible = [
            i
            for i in range(len(offsets) - args.bars)
            if all(
                tokenizer.value(int(song["basic"][offsets[b] + 1])) == 4
                for b in range(i, i + args.bars)
            )
        ]
        if not eligible:
            continue
        start = rng.choice(eligible)
        ids = song["basic"][offsets[start] : offsets[start + args.bars]]
        records.append(
            {
                "song_name": name,
                "start_bar": start,
                "tokens": [tokenizer.tokens[int(token)] for token in ids],
            }
        )
        if len(records) == args.count:
            break
    if len(records) < args.count:
        raise ValueError("Not enough distinct songs with full 4-beat reference windows")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    write_json(output / "tok2idx.json", tokenizer.tok2idx)
    write_json(output / "sequences.json", records)
    write_json(
        output / "reference.json",
        {
            "seed": args.seed,
            "bars": args.bars,
            "count": args.count,
            "data_fingerprint": corpus.manifest["fingerprint"],
        },
    )


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("reference", help=__doc__)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--bars", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.set_defaults(func=reference)
