"""Export an inference checkpoint without optimizer state or training RNGs."""

import argparse
from pathlib import Path

from .train import load_checkpoint, save_checkpoint


def export(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    saved = load_checkpoint(args.checkpoint)
    keys = [
        "format_version",
        "model",
        "config",
        "tokenizer_tokens",
        "tokenizer_config",
        "source_programs",
        "data_fingerprint",
        "step",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, {**{key: saved[key] for key in keys}, "inference_only": True})
    print(f"Exported step {saved['step']} to {output}")


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("export", help=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=export)
