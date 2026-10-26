"""Export the full provided training set as the default real-data baseline."""

import random
from pathlib import Path

from .data import MusicCorpus
from .utils import write_json


def reference(args):
    corpus = MusicCorpus(args.data)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if args.count is not None and args.count < 1:
        raise ValueError("count must be positive")
    if args.bars is not None and args.bars < 1:
        raise ValueError("bars must be positive")
    names = sorted(corpus.songs)
    rng = random.Random(args.seed)
    if args.count is not None:
        rng.shuffle(names)
    records, skipped = [], []
    for name in names:
        song = corpus.songs[name]
        offsets = song["basic_offsets"]
        total = len(offsets) - 1
        if args.bars is not None and total < args.bars:
            skipped.append(name)
            continue
        count = args.bars if args.bars is not None else total
        start = rng.randrange(total - count + 1) if args.bars is not None else 0
        bars = [
            song["basic"][offsets[b] : offsets[b + 1]].astype(int).tolist()
            for b in range(start, start + count)
        ]
        records.append(
            {
                "song_name": name,
                "start_bar": start,
                "bars": count,
                "tokens": corpus.tokenizer.evaluation_tokens(bars),
            }
        )
        if args.count is not None and len(records) == args.count:
            break
    if not records or (args.count is not None and len(records) != args.count):
        raise ValueError("Not enough songs for the explicitly requested subset")
    output.mkdir(parents=True, exist_ok=True)
    corpus.tokenizer.save(output / "tokenizer.json")
    write_json(output / "tok2idx.json", corpus.tokenizer.evaluation_vocabulary())
    write_json(output / "sequences.json", records)
    write_json(
        output / "reference.json",
        {
            "source_songs": len(corpus.songs),
            "evaluated_songs": len(records),
            "subset_requested": args.count is not None or args.bars is not None,
            "requested_bars": args.bars,
            "requested_count": args.count,
            "skipped_short_songs": skipped,
            "seed": args.seed,
            "data_fingerprint": corpus.manifest["fingerprint"],
        },
    )


def add_parser(subparsers):
    parser = subparsers.add_parser("reference", help=__doc__)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--count", type=int, help="Optional explicit song subset; default all songs"
    )
    parser.add_argument(
        "--bars", type=int, help="Optional crop length; default complete supplied songs"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.set_defaults(func=reference)
