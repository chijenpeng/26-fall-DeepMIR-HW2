"""REMI tokens on the annotated POP909 beat grid (16th-note positions)."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pretty_midi

POS_PER_BEAT = 4
MAX_BEATS = 8
MAX_DURATION = 32  # 16th notes; longer notes are shortened to two 4/4 bars
TEMPOS = list(range(30, 211, 3))
QUALITIES = ["maj", "min", "dim", "aug", "sus2", "sus4", "maj7", "min7", "7", "hdim7", "dim7"]
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOTS = {"Cb": 11, "Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "B#": 0, "E#": 5}
ROOTS.update({name: i for i, name in enumerate(PITCH_CLASSES)})


def build_vocab():
    tokens = ["<PAD>", "<BOS>", "<EOS>", "Bar"]
    tokens += [f"Beat_{n}" for n in range(1, MAX_BEATS + 1)]
    tokens += [f"Tempo_{t}" for t in TEMPOS]
    tokens += [f"Pos_{p}" for p in range(MAX_BEATS * POS_PER_BEAT)]
    tokens += ["Chord_N"] + [f"Chord_{r}:{q}" for r in PITCH_CLASSES for q in QUALITIES]
    tokens += [f"Pitch_{p}" for p in range(128)]
    tokens += [f"Vel_{v}" for v in range(32)]
    tokens += [f"Dur_{d}" for d in range(1, MAX_DURATION + 1)]
    return tokens


VOCAB = build_vocab()
TOK2IDX = {token: i for i, token in enumerate(VOCAB)}
PAD, BOS, EOS, BAR = range(4)


def kind(token_id):
    return VOCAB[token_id].split("_", 1)[0]


def value(token_id):
    return VOCAB[token_id].split("_", 1)[1]


def simplify_chord(label):
    """Map a POP909 label such as 'C#:min7/b3' to root and one of QUALITIES."""
    if label == "N":
        return "N"
    root, _, quality = label.partition(":")
    quality = re.split(r"[/(]", quality or "maj")[0]
    if quality not in QUALITIES:
        quality = next(
            (q for q in ("hdim7", "dim", "aug", "sus4", "sus2", "min") if quality.startswith(q)),
            "maj",
        )
    return f"{PITCH_CLASSES[ROOTS[root]]}:{quality}"


def transpose_tokens(tokens, shift):
    """Shift Pitch and Chord tokens; return None when a pitch leaves 0-127."""
    result = []
    for t in tokens:
        k = kind(t)
        if k == "Pitch":
            pitch = int(value(t)) + shift
            if not 0 <= pitch < 128:
                return None
            t = TOK2IDX[f"Pitch_{pitch}"]
        elif k == "Chord" and value(t) != "N":
            root, quality = value(t).split(":")
            root = PITCH_CLASSES[(PITCH_CLASSES.index(root) + shift) % 12]
            t = TOK2IDX[f"Chord_{root}:{quality}"]
        result.append(t)
    return result


@dataclass
class Grid:
    """Beat times from beat_midi.txt; bars start at downbeats (third column)."""

    beats: np.ndarray  # beat onsets plus one trailing boundary
    bar_starts: list[int] = field(default_factory=list)  # beat index of every bar start

    @classmethod
    def from_file(cls, path, notes=()):
        """Extend the annotation with its edge beat lengths so every note onset is covered."""
        rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
        times = [float(r[0]) for r in rows]
        downbeats = [i for i, r in enumerate(rows) if float(r[2]) == 1.0]
        if not downbeats:
            downbeats = list(range(0, len(times), 4))
        first = min((n.start for n in notes), default=times[0])
        last = max((n.start for n in notes), default=0.0)
        while times[0] > first:  # pickup notes before the first annotated beat
            times.insert(0, times[0] - (times[1] - times[0]))
            downbeats = [d + 1 for d in downbeats]
        step = times[-1] - times[-2]
        while times[-1] <= last:  # notes starting after the last annotated beat
            times.append(times[-1] + step)
        bar_starts = ([0] if downbeats[0] > 0 else []) + downbeats  # 0 opens a pickup bar
        length = int(np.median(np.diff(bar_starts))) if len(bar_starts) > 1 else 4
        # A final downbeat closes the last bar; trailing beats get bars of the usual length.
        while bar_starts[-1] < len(times) - 1:
            bar_starts.append(bar_starts[-1] + length)
            while len(times) - 1 < bar_starts[-1]:
                times.append(times[-1] + step)
        # Bars longer than MAX_BEATS (two 12-beat bars in POP909) are split into 4-beat bars.
        bar_starts = [
            b
            for start, end in zip(bar_starts, bar_starts[1:], strict=False)
            for b in (range(start, end, 4) if end - start > MAX_BEATS else [start])
        ] + [bar_starts[-1]]
        return cls(np.array(times[: bar_starts[-1] + 1]), bar_starts)

    @classmethod
    def from_bars(cls, beat_counts, tempos, start=0.0):
        times = [start]
        for count, tempo in zip(beat_counts, tempos, strict=True):
            times += [times[-1] + 60.0 / tempo * (i + 1) for i in range(count)]
        return cls(np.array(times), list(np.cumsum([0, *beat_counts])))

    @property
    def n_bars(self):
        return len(self.bar_starts) - 1

    def beat_counts(self):
        return np.diff(self.bar_starts).tolist()

    def to_units(self, t):
        """Seconds -> fractional 16th-note index from the first beat."""
        b = self.beats
        beat = np.interp(t, b, np.arange(len(b)))
        beat = np.where(t > b[-1], len(b) - 1 + (t - b[-1]) / (b[-1] - b[-2]), beat)
        beat = np.where(t < b[0], (t - b[0]) / (b[1] - b[0]), beat)
        return beat * POS_PER_BEAT

    def to_seconds(self, units):
        b = self.beats
        beat = np.asarray(units, dtype=float) / POS_PER_BEAT
        seconds = np.interp(beat, np.arange(len(b)), b)
        return np.where(beat > len(b) - 1, b[-1] + (beat - len(b) + 1) * (b[-1] - b[-2]), seconds)

    def bar_tempo(self, bar):
        start, end = self.bar_starts[bar], self.bar_starts[bar + 1]
        tempo = 60.0 * (end - start) / (self.beats[end] - self.beats[start])
        return min(TEMPOS, key=lambda t: abs(t - tempo))


def bar_header(grid, bar):
    count = min(MAX_BEATS, grid.beat_counts()[bar])
    return [BAR, TOK2IDX[f"Beat_{count}"], TOK2IDX[f"Tempo_{grid.bar_tempo(bar)}"]]


def encode(notes, grid, chords=(), header=True):
    """Return one token list per bar. `chords` holds (start_seconds, label) pairs."""
    unit_starts = [s * POS_PER_BEAT for s in grid.bar_starts]
    events = [[] for _ in range(grid.n_bars)]

    def locate(t):
        unit = int(np.round(grid.to_units(t)))
        bar = int(np.searchsorted(unit_starts, unit, side="right")) - 1
        bar = min(max(bar, 0), grid.n_bars - 1)
        limit = min(MAX_BEATS, grid.beat_counts()[bar]) * POS_PER_BEAT - 1
        return unit, bar, min(max(unit - unit_starts[bar], 0), limit)

    previous = None
    for start, label in sorted(chords):
        label = simplify_chord(label)
        if label != previous:
            _, bar, pos = locate(start)
            events[bar].append((pos, 0, 0, [TOK2IDX[f"Chord_{label}"]]))
            previous = label
    for note in notes:
        unit, bar, pos = locate(note.start)
        end = int(np.round(grid.to_units(note.end)))
        duration = min(MAX_DURATION, max(1, end - unit))
        tokens = [
            TOK2IDX[f"Pitch_{note.pitch}"],
            TOK2IDX[f"Vel_{note.velocity // 4}"],
            TOK2IDX[f"Dur_{duration}"],
        ]
        events[bar].append((pos, 1, note.pitch, tokens))
    bars = []
    for i, items in enumerate(events):
        tokens = bar_header(grid, i) if header else [BAR]
        last = None
        for pos, _, _, event in sorted(items):
            if pos != last:
                tokens.append(TOK2IDX[f"Pos_{pos}"])
                last = pos
            tokens += event
        bars.append(tokens)
    return bars


def split_bars(tokens):
    """Group a token stream into bars, ignoring BOS/EOS/PAD."""
    bars = []
    for t in tokens:
        if t == BAR:
            bars.append([t])
        elif t > BAR and bars:
            bars[-1].append(t)
    return bars


def headers(bars):
    """Beat counts and tempos stored in bar headers (4/4 at 120 BPM if absent)."""
    counts, tempos = [], []
    for bar in bars:
        found = {kind(t): int(value(t)) for t in bar[1:3] if kind(t) in ("Beat", "Tempo")}
        counts.append(found.get("Beat", 4))
        tempos.append(found.get("Tempo", 120))
    return counts, tempos


def decode(tokens, grid=None, start=0.0, program=0, name="PIANO"):
    """Token stream -> pretty_midi.Instrument; timing from `grid` or from bar headers."""
    bars = split_bars(tokens)
    if grid is None:
        grid = Grid.from_bars(*headers(bars), start=start)
    instrument = pretty_midi.Instrument(program, name=name)
    for i, bar in enumerate(bars[: grid.n_bars]):
        origin = grid.bar_starts[i] * POS_PER_BEAT
        pos = 0
        for j, t in enumerate(bar):
            k = kind(t)
            if k == "Pos":
                pos = int(value(t))
            elif k == "Pitch" and j + 2 < len(bar):
                vel, dur = bar[j + 1], bar[j + 2]
                if kind(vel) != "Vel" or kind(dur) != "Dur":
                    continue
                onset = origin + pos
                begin, end = grid.to_seconds([onset, onset + int(value(dur))])
                note = pretty_midi.Note(int(value(vel)) * 4 + 2, int(value(t)), begin, end)
                instrument.notes.append(note)
    return instrument


def read_chords(path):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    return [(float(r[0]), r[2]) for r in rows]


def track(midi, name):
    return [n for inst in midi.instruments if inst.name.strip().upper() == name for n in inst.notes]


def eval_tokens(tokens):
    """Bar/Pos/Pitch strings consumed by the supplied eval_basic.py functions."""
    return [VOCAB[t] for t in tokens if kind(t) in ("Bar", "Pos", "Pitch")]
