"""MIDI -> WAV with FluidSynth and a user-supplied SoundFont."""

import subprocess
from pathlib import Path


def render_wav(midi, wav, soundfont, fluidsynth="fluidsynth", sample_rate=44100):
    if not Path(soundfont).is_file():
        raise FileNotFoundError(soundfont)
    subprocess.run(
        [
            fluidsynth,
            "-ni",
            "-F",
            str(wav),
            "-T",
            "wav",
            "-r",
            str(sample_rate),
            str(soundfont),
            str(midi),
        ],
        check=True,
        capture_output=True,
    )
