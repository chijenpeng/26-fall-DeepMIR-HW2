"""Event codec retaining microsecond timing, full MIDI values and control events."""

import bisect
import re
from dataclasses import asdict, dataclass

import numpy as np
import pretty_midi

from .music import BarTiming, select_instruments, source_timings
from .utils import read_json, write_json

MAX_INTEGER = (1 << 63) - 1
ROOTS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
    "B#": 0,
}
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def transpose_chord(label, shift):
    if label == "N":
        return label
    match = re.fullmatch(r"([A-G][#b]?)(.*)", label)
    if match is None or match[1] not in ROOTS:
        raise ValueError(f"Unsupported supplied chord label: {label}")
    suffix = match[2]
    if "/" in suffix:
        quality, bass = suffix.rsplit("/", 1)
        if bass in ROOTS:
            suffix = quality + "/" + PITCH_CLASSES[(ROOTS[bass] + shift) % 12]
    return PITCH_CLASSES[(ROOTS[match[1]] + shift) % 12] + suffix


@dataclass(frozen=True)
class TokenizerConfig:
    beat_counts: tuple[int, ...] = (4,)
    meters: tuple[tuple[int, int], ...] = ((4, 4),)
    chords: tuple[str, ...] = ()


class MusicTokenizer:
    version = 2

    def __init__(self, config=None):
        self.config = config or TokenizerConfig()
        if not self.config.beat_counts or min(self.config.beat_counts) < 1:
            raise ValueError("Need positive source beat counts")
        self.tokens = ["<PAD>", "<BOS>", "<EOS>", "Bar", "Pos", "Before", "Duration", "Bend"]
        self.tokens += [f"Beats_{n}" for n in self.config.beat_counts]
        self.tokens += [f"Meter_{n}_{d}" for n, d in self.config.meters]
        self.tokens += [f"More_{i}" for i in range(128)] + [f"End_{i}" for i in range(128)]
        self.tokens += [f"Pitch_{i}" for i in range(128)]
        self.tokens += [f"Vel_{i}" for i in range(1, 128)]
        self.tokens += [f"Control_{i}" for i in range(128)] + [f"Value_{i}" for i in range(128)]
        self.tokens += [f"Chord_{label}" for label in self.config.chords]
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Duplicate vocabulary entries")
        self.tok2idx = {token: i for i, token in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.eos_id, self.bar_id = range(4)
        self.groups = {
            prefix: [i for i, token in enumerate(self.tokens) if token.startswith(prefix + "_")]
            for prefix in [
                "Beats",
                "Meter",
                "More",
                "End",
                "Pitch",
                "Vel",
                "Control",
                "Value",
                "Chord",
            ]
        }

    def __len__(self):
        return len(self.tokens)

    def value(self, token):
        return int(self.tokens[token].rsplit("_", 1)[1])

    def save(self, path):
        write_json(
            path, {"version": self.version, "config": asdict(self.config), "tokens": self.tokens}
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            TokenizerConfig(
                tuple(config["beat_counts"]),
                tuple(tuple(m) for m in config["meters"]),
                tuple(config["chords"]),
            )
        )

    @classmethod
    def load(cls, path):
        payload = read_json(path)
        if payload["version"] != cls.version:
            raise ValueError("Unsupported preprocessing format; run prepare with the current code")
        tokenizer = cls.from_config(payload["config"])
        if tokenizer.tokens != payload["tokens"]:
            raise ValueError("Tokenizer vocabulary/configuration mismatch")
        return tokenizer

    def encode_uint(self, value):
        if not 0 <= value <= MAX_INTEGER:
            raise ValueError("Integer field exceeds the serialized int64 domain")
        digits = [value & 127]
        while value >= 128:
            value >>= 7
            digits.append(value & 127)
        digits.reverse()
        return [self.tok2idx[f"More_{d}"] for d in digits[:-1]] + [
            self.tok2idx[f"End_{digits[-1]}"]
        ]

    def read_uint(self, sequence, cursor):
        value, count = 0, 0
        while cursor < len(sequence):
            token = self.tokens[sequence[cursor]]
            if not token.startswith(("More_", "End_")):
                raise ValueError("Expected integer token")
            if count == 0 and token == "More_0":
                raise ValueError("Noncanonical integer")
            value = value * 128 + int(token.split("_")[1])
            count += 1
            cursor += 1
            if value > MAX_INTEGER:
                raise ValueError("Integer exceeds int64")
            if token.startswith("End_"):
                return value, cursor
        raise ValueError("Incomplete integer field")

    def header(self, timing):
        result = [
            self.bar_id,
            self.tok2idx[f"Beats_{len(timing.beat_us)}"],
            self.tok2idx[f"Meter_{timing.numerator}_{timing.denominator}"],
        ]
        for duration in timing.beat_us:
            result.extend(self.encode_uint(duration))
        return result

    def read_header(self, bar):
        if len(bar) < 3 or bar[0] != self.bar_id or bar[1] not in self.groups["Beats"]:
            raise ValueError("Invalid bar header")
        if bar[2] not in self.groups["Meter"]:
            raise ValueError("Missing supplied meter")
        _, numerator, denominator = self.tokens[bar[2]].split("_")
        cursor, durations = 3, []
        for _ in range(self.value(bar[1])):
            duration, cursor = self.read_uint(bar, cursor)
            durations.append(duration)
        return BarTiming(tuple(durations), int(numerator), int(denominator)), cursor

    def events(self, bar):
        timing, cursor = self.read_header(bar)
        result, previous = [], -MAX_INTEGER
        while cursor < len(bar):
            kind = self.tokens[bar[cursor]]
            if kind not in {"Pos", "Before"}:
                raise ValueError("Expected event position")
            position, cursor = self.read_uint(bar, cursor + 1)
            position = -position if kind == "Before" else position
            if position < previous or position > timing.duration_us:
                raise ValueError("Events must be monotonic and inside the annotated bar")
            previous = position
            if cursor >= len(bar):
                raise ValueError("Incomplete event")
            token = bar[cursor]
            cursor += 1
            if token in self.groups["Pitch"]:
                if position == timing.duration_us:
                    raise ValueError("Note onset belongs to the next bar")
                if cursor + 1 >= len(bar) or bar[cursor] not in self.groups["Vel"]:
                    raise ValueError("Missing velocity")
                pitch, velocity = self.value(token), self.value(bar[cursor])
                if bar[cursor + 1] != self.tok2idx["Duration"]:
                    raise ValueError("Missing duration")
                duration, cursor = self.read_uint(bar, cursor + 2)
                if duration < 1:
                    raise ValueError("Note duration must be positive")
                result.append((position, "note", (pitch, velocity, duration)))
            elif token in self.groups["Control"]:
                if cursor >= len(bar) or bar[cursor] not in self.groups["Value"]:
                    raise ValueError("Missing controller value")
                result.append((position, "control", (self.value(token), self.value(bar[cursor]))))
                cursor += 1
            elif token == self.tok2idx["Bend"]:
                value, cursor = self.read_uint(bar, cursor)
                if value > 16383:
                    raise ValueError("Invalid MIDI pitch bend")
                result.append((position, "bend", (value - 8192,)))
            elif token in self.groups["Chord"]:
                result.append((position, "chord", (self.tokens[token][6:],)))
            else:
                raise ValueError("Invalid event type")
        return result

    def validate_bar(self, bar, beats=None):
        timing, _ = self.read_header(bar)
        if beats is not None and beats != len(timing.beat_us):
            raise ValueError("Bar length differs from conditioning")
        self.events(bar)

    def encode_bars(self, midi, grid, boundaries, track=None, chords=None):
        timings = source_timings(midi, grid, boundaries)
        starts = [round(grid.to_seconds(b) * 1_000_000) for b in boundaries]
        events = [[] for _ in timings]

        def add(time, kind, values):
            absolute = round(time * 1_000_000)
            index = max(0, bisect.bisect_right(starts, absolute) - 1)
            if absolute == starts[-1] and kind in {"control", "bend"}:
                index = len(events) - 1
            if index >= len(events):
                raise ValueError(f"{kind} event at {time}s is outside the annotated range")
            events[index].append((absolute - starts[index], kind, values))

        for inst in select_instruments(midi, track):
            for note in inst.notes:
                duration = max(1, round(note.end * 1_000_000) - round(note.start * 1_000_000))
                add(note.start, "note", (note.pitch, note.velocity, duration))
            for control in inst.control_changes:
                add(control.time, "control", (control.number, control.value))
            for bend in inst.pitch_bends:
                add(bend.time, "bend", (bend.pitch,))
        if chords:
            for time, label in chords:
                if round(time * 1_000_000) < starts[-1]:
                    add(time, "chord", (label,))
        result = []
        for timing, items in zip(timings, events, strict=True):
            bar = self.header(timing)
            for position, kind, values in sorted(items):
                bar += [
                    self.tok2idx["Before" if position < 0 else "Pos"],
                    *self.encode_uint(abs(position)),
                ]
                if kind == "note":
                    pitch, velocity, duration = values
                    bar += [
                        self.tok2idx[f"Pitch_{pitch}"],
                        self.tok2idx[f"Vel_{velocity}"],
                        self.tok2idx["Duration"],
                        *self.encode_uint(duration),
                    ]
                elif kind == "control":
                    bar += [
                        self.tok2idx[f"Control_{values[0]}"],
                        self.tok2idx[f"Value_{values[1]}"],
                    ]
                elif kind == "bend":
                    bar += [self.tok2idx["Bend"], *self.encode_uint(values[0] + 8192)]
                else:
                    bar.append(self.tok2idx[f"Chord_{values[0]}"])
            self.validate_bar(bar)
            result.append(bar)
        return result

    def decode_instrument(self, bars, origin_us=0, name="PIANO", program=0):
        instrument = pretty_midi.Instrument(program, name=name)
        start = origin_us
        for bar in bars:
            timing, _ = self.read_header(bar)
            for position, kind, values in self.events(bar):
                time = (start + position) / 1_000_000
                if time < 0:
                    raise ValueError("Decoded event is before time zero")
                if kind == "note":
                    pitch, velocity, duration = values
                    instrument.notes.append(
                        pretty_midi.Note(velocity, pitch, time, time + duration / 1_000_000)
                    )
                elif kind == "control":
                    instrument.control_changes.append(pretty_midi.ControlChange(*values, time))
                elif kind == "bend":
                    instrument.pitch_bends.append(pretty_midi.PitchBend(values[0], time))
            start += timing.duration_us
        return instrument

    def evaluation_vocabulary(self):
        positions = max(8, max(self.config.beat_counts)) * 16
        tokens = ["<BOS>", "<EOS>", "Bar"] + [f"Pos_{i}" for i in range(positions)]
        tokens += [f"Pitch_{i}" for i in range(128)]
        return {token: i for i, token in enumerate(tokens)}

    def evaluation_tokens(self, bars):
        result = ["<BOS>"]
        for bar in bars:
            timing, _ = self.read_header(bar)
            edges = np.cumsum([0, *timing.beat_us])
            result.append("Bar")
            for position, kind, values in self.events(bar):
                if kind != "note":
                    continue
                beat = max(
                    0, min(len(timing.beat_us) - 1, bisect.bisect_right(edges, position) - 1)
                )
                phase = max(0.0, (position - edges[beat]) / timing.beat_us[beat])
                grid = beat * 16 + min(15, int(phase * 16))
                result += [f"Pos_{grid}", f"Pitch_{values[0]}"]
        return [*result, "<EOS>"]

    def transpose(self, sequence, shift):
        lo, hi = self.groups["Pitch"][0], self.groups["Pitch"][-1]
        if any(not lo <= token + shift <= hi for token in sequence if lo <= token <= hi):
            return sequence
        result = []
        for token in sequence:
            if lo <= token <= hi:
                result.append(token + shift)
            elif token in self.groups["Chord"]:
                result.append(
                    self.tok2idx["Chord_" + transpose_chord(self.tokens[token][6:], shift)]
                )
            else:
                result.append(token)
        return result
