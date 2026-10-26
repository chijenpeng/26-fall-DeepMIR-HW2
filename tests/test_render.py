import wave
from argparse import Namespace
from pathlib import Path

import pytest

from deepmir_hw2 import render as rendering
from deepmir_hw2.utils import read_json


@pytest.mark.parametrize("silent", [False, True])
def test_render_validates_pcm_and_does_not_publish_silent_output(tmp_path, monkeypatch, silent):
    source = tmp_path / "input"
    source.mkdir()
    (source / "001.mid").write_bytes(b"synthetic MIDI source")
    soundfont = tmp_path / "test.sf2"
    soundfont.write_bytes(b"synthetic SoundFont source")
    monkeypatch.setattr(rendering.shutil, "which", lambda value: "/test/fluidsynth")

    def synthesize(command, **kwargs):
        path = Path(command[command.index("-F") + 1])
        rate = int(command[command.index("-r") + 1])
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes((b"\0\0" if silent else b"\1\0") * 100)

    monkeypatch.setattr(rendering.subprocess, "run", synthesize)
    output = tmp_path / "output"
    args = Namespace(
        input=str(source),
        output=str(output),
        soundfont=str(soundfont),
        fluidsynth="fluidsynth",
        sample_rate=44100,
    )
    if silent:
        with pytest.raises(ValueError, match="Silent"):
            rendering.render(args)
        assert not (output / "001.wav").exists()
        assert not (output / "001.tmp.wav").exists()
    else:
        rendering.render(args)
        assert (output / "001.wav").is_file()
        metadata = read_json(output / "render.json")
        assert metadata["files"][0]["peak"] == 1
        assert metadata["files"][0]["sample_rate"] == 44100
