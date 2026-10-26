"""Render MIDI with an explicit SoundFont and verify non-silent PCM WAV output."""

import argparse
import hashlib
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from .utils import write_json


def validate_render_settings(soundfont, fluidsynth="fluidsynth", sample_rate=44100):
    executable = shutil.which(fluidsynth)
    if executable is None:
        raise FileNotFoundError("FluidSynth executable is missing; pass --fluidsynth PATH")
    soundfont = Path(soundfont)
    if not soundfont.is_file():
        raise FileNotFoundError(soundfont)
    if sample_rate < 8000:
        raise ValueError("sample_rate must be at least 8000 Hz")
    return executable, soundfont


def render(args: argparse.Namespace) -> None:
    executable, soundfont = validate_render_settings(
        args.soundfont, args.fluidsynth, args.sample_rate
    )
    source, output = Path(args.input), Path(args.output)
    midis = sorted(source.glob("*.mid")) + sorted(source.glob("*.midi"))
    if not midis:
        raise ValueError("No MIDI files found")
    targets = [output / f"{midi.stem}.wav" for midi in midis]
    if len(set(targets)) != len(targets) or any(p.exists() for p in targets):
        raise FileExistsError("WAV output names collide or already exist")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for midi, target in zip(midis, targets, strict=True):
        temporary = target.with_suffix(".tmp.wav")
        try:
            subprocess.run(
                [
                    executable,
                    "-ni",
                    "-F",
                    str(temporary),
                    "-T",
                    "wav",
                    "-O",
                    "s16",
                    "-r",
                    str(args.sample_rate),
                    str(soundfont),
                    str(midi),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with wave.open(str(temporary), "rb") as wav:
                if wav.getsampwidth() != 2 or wav.getnframes() == 0:
                    raise ValueError(f"Invalid PCM output: {midi}")
                peak = 0
                while chunk := wav.readframes(65536):
                    peak = max(
                        peak, int(np.abs(np.frombuffer(chunk, dtype="<i2").astype(np.int32)).max())
                    )
                if peak == 0:
                    raise ValueError(f"Silent WAV output: {midi}")
                records.append(
                    {
                        "file": target.name,
                        "seconds": wav.getnframes() / wav.getframerate(),
                        "sample_rate": wav.getframerate(),
                        "peak": peak,
                    }
                )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    write_json(
        output / "render.json",
        {
            "soundfont": soundfont.name,
            "soundfont_sha256": hashlib.sha256(soundfont.read_bytes()).hexdigest(),
            "files": records,
        },
    )


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("render", help=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--soundfont", required=True)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--fluidsynth", default="fluidsynth")
    parser.set_defaults(func=render)
