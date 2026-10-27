"""Export an inference checkpoint without optimizer state or training RNGs."""

import argparse
import hashlib
from pathlib import Path

from .train import checkpoint_path, load_checkpoint, save_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    source = checkpoint_path(args.checkpoint).resolve()
    saved = load_checkpoint(source)
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
    exported = {key: saved[key] for key in keys}
    if "training_provenance" in saved:
        exported["training_provenance"] = saved["training_provenance"]
    exported.update(
        {
            "source_checkpoint_name": source.name,
            "source_checkpoint_sha256": sha256(source),
            "inference_only": True,
        }
    )
    save_checkpoint(output, exported)
    print(f"Exported step {saved['step']} to {output}")


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("export", help=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=export)
