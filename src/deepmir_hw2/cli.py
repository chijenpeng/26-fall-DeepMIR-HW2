"""Explicit subcommands keep preprocessing, training and inference independent."""

import argparse
import importlib


def main() -> None:
    parser = argparse.ArgumentParser(prog="deepmir")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "train", "generate", "evaluate", "render", "reference", "export"]:
        importlib.import_module(f".{name}", __package__).add_parser(subparsers)
    args = parser.parse_args()
    args.func(args)
