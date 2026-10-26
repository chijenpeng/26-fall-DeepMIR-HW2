"""A compact, reversible REMI-like vocabulary with explicit bar lengths."""

import bisect
from dataclasses import asdict, dataclass
from pathlib import Path

import pretty_midi

from .music import BeatGrid, select_notes
from .utils import read_json, write_json


@dataclass(frozen=True)
class TokenizerConfig:
    positions_per_beat: int = 12
    max_beats_per_bar: int = 16
    max_duration_beats: int = 16
    pitch_min: int = 21
    pitch_max: int = 108
    velocity_bins: int = 32

    def __post_init__(self) -> None:
        if min(self.positions_per_beat, self.max_beats_per_bar, self.max_duration_beats) < 1:
            raise ValueError("Timing dimensions must be positive")
        if not 0 <= self.pitch_min <= self.pitch_max <= 127:
            raise ValueError("Invalid MIDI pitch range")
        if not 1 <= self.velocity_bins <= 127:
            raise ValueError("Invalid velocity bin count")


class MusicTokenizer:
    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        c = self.config
        self.tokens = ["<PAD>", "<BOS>", "<EOS>", "Bar"]
        self.tokens += [f"Beats_{i}" for i in range(1, c.max_beats_per_bar + 1)]
        self.tokens += [f"Pos_{i}" for i in range(c.max_beats_per_bar * c.positions_per_beat)]
        self.tokens += [f"Pitch_{i}" for i in range(c.pitch_min, c.pitch_max + 1)]
        self.tokens += [f"Vel_{i}" for i in range(c.velocity_bins)]
        self.tokens += [
            f"Dur_{i}" for i in range(1, c.max_duration_beats * c.positions_per_beat + 1)
        ]
        self.tok2idx = {token: i for i, token in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.eos_id, self.bar_id = range(4)
        self.groups = {
            prefix: [i for i, token in enumerate(self.tokens) if token.startswith(prefix + "_")]
            for prefix in ["Beats", "Pos", "Pitch", "Vel", "Dur"]
        }

    def __len__(self) -> int:
        return len(self.tokens)

    def save(self, path: str | Path) -> None:
        write_json(path, {"version": 1, "config": asdict(self.config), "tokens": self.tokens})

    @classmethod
    def load(cls, path: str | Path) -> "MusicTokenizer":
        payload = read_json(path)
        if payload["version"] != 1:
            raise ValueError("Unsupported tokenizer format")
        tokenizer = cls(TokenizerConfig(**payload["config"]))
        if tokenizer.tokens != payload["tokens"]:
            raise ValueError("Tokenizer vocabulary does not match its configuration")
        return tokenizer

    def value(self, token_id: int) -> int:
        return int(self.tokens[token_id].split("_")[1])

    def encode_bars(
        self,
        midi: pretty_midi.PrettyMIDI,
        grid: BeatGrid,
        boundaries: list[int],
        track: str | None = None,
    ) -> list[list[int]]:
        c = self.config
        events: list[list[tuple[int, int, int, int]]] = [[] for _ in boundaries[:-1]]
        for note in select_notes(midi, track):
            start = grid.to_beat(note.start)
            if start < boundaries[0] - 1e-4 or start >= boundaries[-1] - 1e-5:
                continue
            if not c.pitch_min <= note.pitch <= c.pitch_max:
                raise ValueError(f"Pitch {note.pitch} outside vocabulary")
            bar = bisect.bisect_right(boundaries, start + 1e-7) - 1
            steps = (boundaries[bar + 1] - boundaries[bar]) * c.positions_per_beat
            position = max(
                0, min(steps - 1, round((start - boundaries[bar]) * c.positions_per_beat))
            )
            duration = max(1, round((grid.to_beat(note.end) - start) * c.positions_per_beat))
            duration = min(duration, c.max_duration_beats * c.positions_per_beat)
            velocity = min(c.velocity_bins - 1, note.velocity * c.velocity_bins // 128)
            events[bar].append((position, note.pitch, velocity, duration))
        bars = []
        for i, notes in enumerate(events):
            beats = boundaries[i + 1] - boundaries[i]
            if not 1 <= beats <= c.max_beats_per_bar:
                raise ValueError(f"Bar length {beats} outside vocabulary")
            result = [self.bar_id, self.tok2idx[f"Beats_{beats}"]]
            for position, pitch, velocity, duration in sorted(notes):
                result += [
                    self.tok2idx[f"Pos_{position}"],
                    self.tok2idx[f"Pitch_{pitch}"],
                    self.tok2idx[f"Vel_{velocity}"],
                    self.tok2idx[f"Dur_{duration}"],
                ]
            bars.append(result)
        return bars

    def decode_notes(
        self,
        bars: list[list[int]],
        grid: BeatGrid,
        boundaries: list[int],
    ) -> list:
        if len(bars) != len(boundaries) - 1:
            raise ValueError("Number of token bars and time intervals disagree")
        notes = []
        for i, bar in enumerate(bars):
            expected = boundaries[i + 1] - boundaries[i]
            self.validate_bar(bar, expected)
            for p in range(2, len(bar), 4):
                position, pitch, velocity, duration = [self.value(x) for x in bar[p : p + 4]]
                onset_beat = boundaries[i] + position / self.config.positions_per_beat
                end_beat = min(
                    boundaries[-1], onset_beat + duration / self.config.positions_per_beat
                )
                onset = max(0.0, grid.to_seconds(onset_beat))
                end = grid.to_seconds(end_beat)
                if end <= onset:
                    raise ValueError("Decoded note has non-positive duration")
                vel = max(1, round((velocity + 0.5) * 128 / self.config.velocity_bins))
                notes.append(pretty_midi.Note(min(127, vel), pitch, onset, end))
        return notes

    def validate_bar(self, bar: list[int], beats: int | None = None) -> None:
        if len(bar) < 2 or bar[0] != self.bar_id or bar[1] not in self.groups["Beats"]:
            raise ValueError("Invalid bar header")
        length = self.value(bar[1])
        if beats is not None and beats != length:
            raise ValueError("Bar length does not match conditioning grid")
        if (len(bar) - 2) % 4:
            raise ValueError("Incomplete note event")
        previous = -1
        for p in range(2, len(bar), 4):
            for token, group in zip(bar[p : p + 4], ["Pos", "Pitch", "Vel", "Dur"], strict=True):
                if token not in self.groups[group]:
                    raise ValueError(f"Expected {group} token")
            position = self.value(bar[p])
            if position < previous or position >= length * self.config.positions_per_beat:
                raise ValueError("Note positions must be monotonic and inside the bar")
            previous = position

    def transpose(self, sequence: list[int], semitones: int) -> list[int]:
        start, end = self.groups["Pitch"][0], self.groups["Pitch"][-1]
        if any(not start <= t + semitones <= end for t in sequence if start <= t <= end):
            return sequence
        return [t + semitones if start <= t <= end else t for t in sequence]
