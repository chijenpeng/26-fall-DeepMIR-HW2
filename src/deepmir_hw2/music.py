"""Beat-annotation-based timing. MIDI time signatures are not trusted as bar grids."""

import bisect
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi


@dataclass(frozen=True)
class BeatGrid:
    times: tuple[float, ...]
    downbeats: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.times) < 2 or any(
            b <= a for a, b in zip(self.times, self.times[1:], strict=False)
        ):
            raise ValueError("Beat times must be strictly increasing, with at least two beats")
        if not self.downbeats or self.downbeats[0] != 0:
            raise ValueError("The bar grid must start at beat index zero")
        if any(b <= a for a, b in zip(self.downbeats, self.downbeats[1:], strict=False)):
            raise ValueError("Downbeat indices must be strictly increasing")
        if self.downbeats[-1] > len(self.times) - 1:
            raise ValueError("Downbeat outside beat-time grid")

    @classmethod
    def load(cls, path: str | Path) -> "BeatGrid":
        rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
        times = tuple(float(r[0]) for r in rows)
        down = [i for i, r in enumerate(rows) if len(r) >= 3 and float(r[2]) == 1.0]
        if not down:
            raise ValueError(f"No annotated downbeats in {path}")
        # Preserve an opening pickup rather than dropping its notes.
        if down[0] != 0:
            down.insert(0, 0)
        return cls(times, tuple(down))

    def to_beat(self, seconds: float) -> float:
        i = max(0, min(len(self.times) - 2, bisect.bisect_right(self.times, seconds) - 1))
        return i + (seconds - self.times[i]) / (self.times[i + 1] - self.times[i])

    def to_seconds(self, beat: float) -> float:
        i = max(0, min(len(self.times) - 2, int(np.floor(beat))))
        return self.times[i] + (beat - i) * (self.times[i + 1] - self.times[i])

    def boundaries(self, end_time: float | None = None) -> list[int]:
        """Return start boundaries plus one terminal boundary, never an empty final bar."""
        down = list(self.downbeats)
        if end_time is None:
            if len(down) < 2:
                raise ValueError("A prompt needs an annotated terminal downbeat")
            return down
        end_beat = self.to_beat(end_time)
        # Keep annotations up to the last sounding note, including a terminal boundary.
        for i, beat in enumerate(down[1:], 1):
            if beat >= end_beat - 1e-5:
                return down[: i + 1]
        typical = down[-1] - down[-2] if len(down) > 1 else 4
        while down[-1] < end_beat - 1e-5:
            down.append(down[-1] + typical)
        if len(down) == 1:
            down.append(down[0] + typical)
        return down


def select_notes(midi: pretty_midi.PrettyMIDI, track: str | None = None) -> list:
    instruments = [inst for inst in midi.instruments if not inst.is_drum]
    if track is not None:
        instruments = [inst for inst in instruments if inst.name.strip().upper() == track]
        if not instruments:
            raise ValueError(f"MIDI has no {track} track")
    return [note for inst in instruments for note in inst.notes]


def combine_with_prompt(
    original: pretty_midi.PrettyMIDI,
    generated_notes: list,
    track_name: str,
    end_time: float,
) -> pretty_midi.PrettyMIDI:
    """Keep original notes, controllers and tempo map without re-tokenizing the prompt."""
    result = copy.deepcopy(original)
    inst = pretty_midi.Instrument(program=0, name=track_name)
    inst.notes = generated_notes
    result.instruments.append(inst)
    # Text at the boundary makes trailing silence part of the MIDI's declared duration.
    result.text_events.append(pretty_midi.Text("END", end_time))
    return result
