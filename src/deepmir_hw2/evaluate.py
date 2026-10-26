"""Evaluate with unmodified TA metric functions and report every missing metric."""

import hashlib
import importlib.util
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .utils import read_json, write_json


def load_module(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation source: {path}; see README")
    spec = importlib.util.spec_from_file_location("evaluation_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ta_basic(ta_scripts, musdr_root):
    root = Path(musdr_root).resolve()
    if root.name != "MusDr":
        raise ValueError("TA imports require a checkout directory named MusDr")
    sys.path.insert(0, str(root.parent))
    try:
        return load_module(Path(ta_scripts) / "eval_basic.py")
    finally:
        sys.path.pop(0)


def compute_basic(sequence, vocabulary, ta):
    if not sequence or sequence.count(vocabulary["Bar"]) < 1:
        raise ValueError("Expected a nonempty bar sequence")
    positions = [
        vocabulary[key]
        for key in sorted(
            (key for key in vocabulary if key.startswith("Pos_")), key=lambda key: int(key[4:])
        )
    ]
    pitches = [
        vocabulary[key]
        for key in sorted(
            (key for key in vocabulary if key.startswith("Pitch_")), key=lambda key: int(key[6:])
        )
    ]
    bar = vocabulary["Bar"]
    return {
        "H1": float(ta.compute_piece_pitch_entropy(sequence, 1, bar, pitches)),
        "H4": float(ta.compute_piece_pitch_entropy(sequence, 4, bar, pitches)),
        "GS": float(ta.compute_piece_groove_similarity(sequence, bar, positions, pitches)),
        "bars": sequence.count(bar),
    }


def evaluate(args):
    import pandas as pd

    output = Path(args.output)
    if output.exists() or output.with_suffix(".summary.json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.task == "bonus":
        if not args.prompts:
            raise ValueError("Bonus evaluation requires --prompts")
        names = sorted(p.name for p in Path(args.prompts).iterdir() if p.is_dir())
        files = sorted(p.stem for p in Path(args.input).glob("*.mid"))
        if names != files:
            raise ValueError(f"Generated/expected prompt names disagree: {files} vs {names}")
        implementation = Path(args.ta_scripts) / "eval_bonus.py"
        ta = load_module(implementation)
        options = SimpleNamespace(prompt_dir=args.prompts, gen_midi_dir=args.input)
        for name in names:
            ta.evaluate_song(name, options)
        subprocess.run(
            [
                sys.executable,
                str(implementation),
                "--prompt_dir",
                args.prompts,
                "--gen_midi_dir",
                args.input,
                "--output_csv",
                str(output),
            ],
            check=True,
        )
        frame = pd.read_csv(output)
        if len(frame) != len(names):
            raise ValueError("TA evaluation unexpectedly skipped songs")
        metrics = [key for key in frame if key not in {"song_name", "n_bars"}]
        means = {key: float(np.average(frame[key], weights=frame["n_bars"])) for key in metrics}
    else:
        directory = Path(args.input)
        vocabulary = read_json(directory / "tok2idx.json")
        sequences = read_json(directory / "sequences.json")
        if not sequences:
            raise ValueError("No sequences to evaluate")
        if (directory / "generation.json").exists():
            generation = read_json(directory / "generation.json")
            if not generation["complete"] or len(sequences) != generation["expected_pieces"]:
                raise ValueError("Generation is incomplete; refusing a partial-piece average")
        implementation = Path(args.ta_scripts) / "eval_basic.py"
        ta = load_ta_basic(args.ta_scripts, args.musdr_root)
        state = random.getstate()
        random.seed(args.seed)
        try:
            rows = [
                {
                    "song_name": item["song_name"],
                    **compute_basic(
                        [vocabulary[token] for token in item["tokens"]], vocabulary, ta
                    ),
                }
                for item in sequences
            ]
        finally:
            random.setstate(state)
        frame = pd.DataFrame(rows)
        frame.to_csv(output, index=False)
        metrics = ["H1", "H4", "GS"]
        means = {key: float(frame[key].mean()) for key in metrics}
    # Preserve TA behavior (including NaN for all-rest pitch windows), while
    # explicitly reporting coverage instead of concealing failed/undefined metrics.
    summary = {
        "pieces": len(frame),
        "metrics": {key: value if np.isfinite(value) else None for key, value in means.items()},
        "valid_pieces": {key: int(np.isfinite(frame[key]).sum()) for key in metrics},
        "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        "seed": args.seed,
    }
    if args.task == "basic":
        summary["musdr_sha256"] = hashlib.sha256(
            (Path(args.musdr_root) / "musdr/side_utils.py").read_bytes()
        ).hexdigest()
    write_json(output.with_suffix(".summary.json"), summary)
    print(summary)


def add_parser(subparsers):
    parser = subparsers.add_parser("evaluate", help=__doc__)
    parser.add_argument("--task", choices=["basic", "bonus"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--musdr-root", default="third_party/MusDr")
    parser.add_argument("--ta-scripts", default="HW2_scripts")
    parser.add_argument("--seed", type=int, default=2026)
    parser.set_defaults(func=evaluate)
