"""Use MusDr for basic metrics and the unmodified TA script for bonus metrics."""

import argparse
import hashlib
import importlib.util
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

from .tokens import MusicTokenizer
from .utils import read_json, write_json


def load_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation source: {path}; see README setup instructions")
    spec = importlib.util.spec_from_file_location("evaluation_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_basic(sequence: list[int], tokenizer: MusicTokenizer, musdr) -> dict:
    markers = [i for i, token in enumerate(sequence) if token == tokenizer.bar_id]
    if len(markers) < 4:
        raise ValueError("H4 requires at least four bars")
    bars = [sequence[a:b] for a, b in zip(markers, [*markers[1:], len(sequence)], strict=True)]
    for bar in bars:
        tokenizer.validate_bar(bar, beats=4)
    pitch = tokenizer.groups["Pitch"]
    result = {}
    for window in [1, 4]:
        entropies = []
        for i in range(len(bars) - window + 1):
            hist = musdr.get_pitch_histogram(
                [t for bar in bars[i : i + window] for t in bar], pitch
            )
            if hist is not None:
                entropies.append(musdr.compute_histogram_entropy(hist))
        if not entropies:
            raise ValueError("Cannot evaluate a piece without notes")
        result[f"H{window}"] = float(np.mean(entropies))
    # The vocabulary supports 16-beat bars. A 4-beat bar's onset vector must
    # contain only its 4*resolution positions, otherwise GS is artificially inflated.
    positions = tokenizer.groups["Pos"][: 4 * tokenizer.config.positions_per_beat]
    similarities = [
        1 - musdr.get_onset_xor_distance(a, b, tokenizer.bar_id, positions, pitch)
        for a, b in itertools.combinations(bars, 2)
    ]
    result["GS"] = float(np.mean(similarities))
    result["bars"] = len(bars)
    result["empty_bars"] = sum(len(bar) == 2 for bar in bars)
    return result


def evaluate(args: argparse.Namespace) -> None:
    import pandas as pd

    output = Path(args.output)
    if output.exists() or output.with_suffix(".summary.json").exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.task == "bonus":
        if not args.prompts:
            raise ValueError("Bonus evaluation requires --prompts")
        names = sorted(p.name for p in Path(args.prompts).iterdir() if p.is_dir())
        files = sorted(p.stem for p in Path(args.input).glob("*.mid"))
        if names != files:
            raise ValueError(f"Generated/expected prompt names disagree: {files} vs {names}")
        # Validate every result first; the TA CLI otherwise skips failures and can
        # report misleading averages over only a subset of the ten songs.
        implementation = Path(args.ta_scripts) / "eval_bonus.py"
        ta = load_module(implementation)
        from types import SimpleNamespace

        options = SimpleNamespace(prompt_dir=args.prompts, gen_midi_dir=args.input)
        for name in names:
            ta.evaluate_song(name, options)
        subprocess.run(
            [
                sys.executable,
                str(Path(args.ta_scripts) / "eval_bonus.py"),
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
            raise ValueError("TA evaluation skipped songs unexpectedly")
        metrics = [c for c in frame if c not in {"song_name", "n_bars"}]
        summary = {c: float(np.average(frame[c], weights=frame["n_bars"])) for c in metrics}
    else:
        directory = Path(args.input)
        tokenizer = MusicTokenizer.load(directory / "tokenizer.json")
        sequences = read_json(directory / "sequences.json")
        if not sequences:
            raise ValueError("No sequences to evaluate")
        implementation = Path(args.musdr_root) / "musdr/side_utils.py"
        musdr = load_module(implementation)
        rows = []
        for item in sequences:
            ids = [tokenizer.tok2idx[token] for token in item["tokens"]]
            rows.append({"song_name": item["song_name"], **compute_basic(ids, tokenizer, musdr)})
        frame = pd.DataFrame(rows)
        frame.to_csv(output, index=False)
        summary = {metric: float(frame[metric].mean()) for metric in ["H1", "H4", "GS"]}
    if not all(np.isfinite(value) for value in summary.values()):
        raise ValueError("Evaluation produced non-finite metrics")
    write_json(
        output.with_suffix(".summary.json"),
        {
            "pieces": len(frame),
            "metrics": summary,
            "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        },
    )
    print(summary)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("evaluate", help=__doc__)
    parser.add_argument("--task", choices=["basic", "bonus"], required=True)
    parser.add_argument("--input", required=True, help="Generation directory")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--prompts", help="POP909/prompts/bonus")
    parser.add_argument("--musdr-root", default="third_party/MusDr")
    parser.add_argument("--ta-scripts", default="HW2_scripts")
    parser.set_defaults(func=evaluate)
