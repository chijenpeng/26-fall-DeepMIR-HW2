"""Objective metrics computed with the supplied TA scripts in HW2_scripts/."""

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import remi

ROOT = Path(__file__).resolve().parents[2]
EVAL_VOCAB = ["Bar"] + [f"Pos_{i}" for i in range(32)] + [f"Pitch_{p}" for p in range(128)]


def basic_metrics(pieces):
    """H1/H4/GS from eval_basic.py for (name, REMI token ids) pairs."""
    sys.path[:0] = [str(ROOT / "HW2_scripts"), str(ROOT / "third_party")]
    import eval_basic as ta

    tok2idx = {t: i for i, t in enumerate(EVAL_VOCAB)}
    pos = [tok2idx[t] for t in EVAL_VOCAB if t.startswith("Pos_")]
    pitch = [tok2idx[t] for t in EVAL_VOCAB if t.startswith("Pitch_")]
    random.seed(0)  # GS samples at most 1000 bar pairs
    rows = []
    for name, tokens in pieces:
        ids = [tok2idx[t] for t in remi.eval_tokens(tokens)]
        rows.append(
            {
                "song_name": name,
                "H1": ta.compute_piece_pitch_entropy(ids, 1, 0, pitch),
                "H4": ta.compute_piece_pitch_entropy(ids, 4, 0, pitch),
                "GS": ta.compute_piece_groove_similarity(ids, 0, pos, pitch),
            }
        )
    return rows


def write_rows(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != "song_name"}
    summary["count"] = len(rows)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def evaluate(args):
    if args.task == "real":
        songs = json.loads(Path(args.input).read_text())["songs"]
        pieces = [(n, [t for bar in s["basic"] for t in bar]) for n, s in sorted(songs.items())]
        write_rows(basic_metrics(pieces), args.output)
    elif args.task == "basic":
        record = json.loads((Path(args.input) / "generation.json").read_text())
        write_rows(basic_metrics([(p["name"], p["tokens"]) for p in record["pieces"]]), args.output)
    else:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "HW2_scripts" / "eval_bonus.py"),
                "--prompt_dir",
                args.prompts,
                "--gen_midi_dir",
                args.input,
                "--output_csv",
                args.output,
            ],
            check=True,
        )
        # Same bar-weighted pooling as the averages eval_bonus.py prints.
        rows = list(csv.DictReader(open(args.output)))
        bars = sum(int(r["n_bars"]) for r in rows)
        metrics = [k for k in rows[0] if k not in ("song_name", "n_bars")]
        summary = {k: sum(float(r[k]) * int(r["n_bars"]) for r in rows) / bars for k in metrics}
        summary["count"] = len(rows)
        Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))


def add_parser(subparsers):
    parser = subparsers.add_parser("evaluate", help=__doc__)
    parser.add_argument("task", choices=["basic", "real", "bonus"])
    parser.add_argument(
        "--input", required=True, help="Generation folder, or data/remi.json for real"
    )
    parser.add_argument("--output", required=True, help="CSV path")
    parser.add_argument("--prompts", default="POP909/prompts/bonus")
    parser.set_defaults(func=evaluate)
