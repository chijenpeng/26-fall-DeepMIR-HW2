"""Preserve event seconds while aligning exported MIDI bars to supplied beat annotations."""

import bisect
import copy
import math
import os
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import mido
import numpy as np
from pretty_midi.pretty_midi import MAX_TICK


@dataclass(frozen=True)
class BeatGrid:
    times: tuple[float, ...]
    downbeats: tuple[int, ...]

    def __post_init__(self):
        if len(self.times) < 2 or any(
            b <= a for a, b in zip(self.times, self.times[1:], strict=False)
        ):
            raise ValueError("Beat times must be strictly increasing")
        if not self.downbeats or self.downbeats[0] != 0:
            raise ValueError("The grid must start at beat index zero")
        if any(b <= a for a, b in zip(self.downbeats, self.downbeats[1:], strict=False)):
            raise ValueError("Downbeats must be strictly increasing")
        if self.downbeats[-1] >= len(self.times):
            raise ValueError("Downbeat outside beat-time grid")

    @classmethod
    def load(cls, path):
        rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
        down = [i for i, row in enumerate(rows) if len(row) >= 3 and float(row[2]) == 1]
        if not down:
            raise ValueError(f"No annotated downbeats: {path}")
        if down[0]:
            down.insert(0, 0)
        return cls(tuple(float(row[0]) for row in rows), tuple(down))

    def to_beat(self, seconds):
        i = max(0, min(len(self.times) - 2, bisect.bisect_right(self.times, seconds) - 1))
        return i + (seconds - self.times[i]) / (self.times[i + 1] - self.times[i])

    def to_seconds(self, beat):
        i = max(0, min(len(self.times) - 2, int(np.floor(beat))))
        return self.times[i] + (beat - i) * (self.times[i + 1] - self.times[i])

    def boundaries(self, last_onset=None, *, last_note_onset=None):
        down = list(self.downbeats)
        if last_onset is None:
            if len(down) < 2:
                raise ValueError("Prompt needs an annotated terminal downbeat")
            return down
        # Retain all supplied boundaries, including silent bars.
        if len(down) == 1:
            down.append(len(self.times))
        width = down[-1] - down[-2]

        def needs_terminal_bar():
            boundary_us = round(self.to_seconds(down[-1]) * 1_000_000)
            return round(last_onset * 1_000_000) > boundary_us or (
                last_note_onset is not None and round(last_note_onset * 1_000_000) >= boundary_us
            )

        while needs_terminal_bar():
            down.append(down[-1] + width)
        return down


@dataclass(frozen=True)
class BarTiming:
    beat_us: tuple[int, ...]
    numerator: int
    denominator: int

    def __post_init__(self):
        if not self.beat_us or min(self.beat_us) <= 0:
            raise ValueError("Annotated pulse durations must be positive")
        if not 1 <= self.numerator <= 255:
            raise ValueError("Invalid MIDI meter numerator")
        if self.denominator < 1 or self.denominator & (self.denominator - 1):
            raise ValueError("MIDI meter denominator must be a power of two")

    @property
    def duration_us(self):
        return sum(self.beat_us)

    @property
    def quarters_per_pulse(self):
        return Fraction(self.numerator * 4, self.denominator * len(self.beat_us))


def source_timings(midi, grid, boundaries):
    signatures = sorted(midi.time_signature_changes, key=lambda value: value.time)
    result = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        # SMF specifies 4/4 when no time-signature event exists.
        numerator, denominator = 4, 4
        for signature in signatures:
            if signature.time > grid.to_seconds(start) + 1e-6:
                break
            numerator, denominator = signature.numerator, signature.denominator
        points = [round(grid.to_seconds(i) * 1_000_000) for i in range(start, end + 1)]
        durations = tuple(b - a for a, b in zip(points, points[1:], strict=False))
        result.append(BarTiming(durations, numerator, denominator))
    return result


def select_instruments(midi, track=None):
    instruments = [inst for inst in midi.instruments if not inst.is_drum]
    if track is not None:
        instruments = [inst for inst in instruments if inst.name.strip().upper() == track]
        if not instruments:
            raise ValueError(f"MIDI has no {track} track")
    return instruments


def event_end_time(midi):
    starts = [note.start for inst in midi.instruments for note in inst.notes]
    starts += [value.time for inst in midi.instruments for value in inst.control_changes]
    starts += [value.time for inst in midi.instruments for value in inst.pitch_bends]
    return max(starts, default=0.0)


def combine_with_prompt(original, generated, name):
    instruments = copy.deepcopy(original.instruments)
    generated = copy.deepcopy(generated)
    generated.name = name
    instruments.append(generated)
    return instruments


def _build_midi(instruments, timings, origin_us, ppq):
    initial_tempo = round(timings[0].beat_us[0] / timings[0].quarters_per_pulse)
    if not 1 <= initial_tempo <= 0xFFFFFF:
        raise ValueError("Pulse timing cannot be represented by SMF tempo")
    ticks, seconds, tempos = [0], [0.0], [initial_tempo]
    tick = round(origin_us * ppq / initial_tempo)
    actual_seconds = tick * initial_tempo / (ppq * 1_000_000)
    desired_us, previous_meter = origin_us, None
    metadata = [(0, mido.MetaMessage("set_tempo", tempo=initial_tempo))]
    for index, timing in enumerate(timings):
        meter = timing.numerator, timing.denominator
        if meter != previous_meter:
            metadata.append(
                (tick, mido.MetaMessage("time_signature", numerator=meter[0], denominator=meter[1]))
            )
            previous_meter = meter
        metadata.append((tick, mido.MetaMessage("marker", text=f"BAR_{index + 1}")))
        delta_ticks = int(timing.quarters_per_pulse * ppq)
        for duration in timing.beat_us:
            desired_us += duration
            tempo = round((desired_us - actual_seconds * 1_000_000) * ppq / delta_ticks)
            if not 1 <= tempo <= 0xFFFFFF:
                raise ValueError("Pulse timing exceeds the SMF tempo field")
            ticks.append(tick)
            seconds.append(actual_seconds)
            tempos.append(tempo)
            metadata.append((tick, mido.MetaMessage("set_tempo", tempo=tempo)))
            actual_seconds += delta_ticks * tempo / (ppq * 1_000_000)
            tick += delta_ticks
    metadata.append((tick, mido.MetaMessage("marker", text="BAR_END")))

    def event_tick(time):
        if time < 0:
            raise ValueError("MIDI event is before time zero")
        i = max(0, bisect.bisect_right(seconds, time) - 1)
        return ticks[i] + round((time - seconds[i]) * 1_000_000 * ppq / tempos[i])

    def make_track(events):
        track, previous = mido.MidiTrack(), 0

        def event_order(item):
            absolute, message = item
            # A retrigger at the same tick must not be cancelled by the previous
            # note's release, even if the input Note objects were unsorted.
            priority = 0 if message.type == "note_off" else 1
            return absolute, priority

        for absolute, message in sorted(events, key=event_order):
            delta = absolute - previous
            track.append(message.copy(time=delta))
            previous = absolute
        track.append(mido.MetaMessage("end_of_track"))
        return track

    output = mido.MidiFile(ticks_per_beat=ppq)
    output.tracks.append(make_track(metadata))
    for index, inst in enumerate(instruments):
        if index >= 15:
            raise ValueError("Too many pitched tracks for MIDI channel allocation")
        channel = index if index < 9 else index + 1
        events = [
            (0, mido.MetaMessage("track_name", name=inst.name)),
            (0, mido.Message("program_change", channel=channel, program=inst.program)),
        ]
        for note in inst.notes:
            events += [
                (
                    event_tick(note.start),
                    mido.Message(
                        "note_on", channel=channel, note=note.pitch, velocity=note.velocity
                    ),
                ),
                (
                    event_tick(note.end),
                    mido.Message("note_off", channel=channel, note=note.pitch, velocity=0),
                ),
            ]
        for control in inst.control_changes:
            events.append(
                (
                    event_tick(control.time),
                    mido.Message(
                        "control_change",
                        channel=channel,
                        control=control.number,
                        value=control.value,
                    ),
                )
            )
        for bend in inst.pitch_bends:
            events.append(
                (
                    event_tick(bend.time),
                    mido.Message("pitchwheel", channel=channel, pitch=bend.pitch),
                )
            )
        output.tracks.append(make_track(events))
    return output


def write_midi(path, instruments, timings, origin_us=0):
    """Preserve event seconds and supplied meter in a reader-compatible MIDI clock.

    Pulses need not be quarters: six pulses in a supplied 4/4 bar occupy four
    quarters. Use the finest SMF resolution that also fits PrettyMIDI's tick
    allocation limit, including full release tails. No musical events are clipped.
    """
    if not timings or origin_us < 0:
        raise ValueError("Need at least one bar and a nonnegative origin")
    divisor = math.lcm(*(t.quarters_per_pulse.denominator for t in timings))
    ppq = (32767 // divisor) * divisor
    while ppq:
        output = _build_midi(instruments, timings, origin_us, ppq)
        largest_tick = max(sum(message.time for message in track) for track in output.tracks)
        if largest_tick + 1 <= MAX_TICK:
            break
        # Retain exact pulse-to-quarter ratios while lowering serialization
        # resolution. The model's event times and generated bar count are unchanged.
        ppq = (ppq * (int(MAX_TICK) - 2) // (largest_tick + 1) // divisor) * divisor
    else:
        raise ValueError("MIDI timing cannot fit reader limits at an exact pulse resolution")
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".mid.tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            output.save(file=stream)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
