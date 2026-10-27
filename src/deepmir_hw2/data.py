"""Song-level splits and complete-bar windows, with paired pitch augmentation."""

import random
from pathlib import Path

import numpy as np
import torch

from .tokens import MusicTokenizer
from .utils import fingerprint, read_json


class MusicCorpus:
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        self.tokenizer = MusicTokenizer.load(directory / "tokenizer.json")
        self.manifest = read_json(directory / "manifest.json")
        if self.manifest["version"] != self.tokenizer.version:
            raise ValueError("Prepared data uses an unsupported format")
        expected = self.manifest["fingerprint"]
        if fingerprint({k: v for k, v in self.manifest.items() if k != "fingerprint"}) != expected:
            raise ValueError("Preprocessed manifest integrity check failed")
        if fingerprint(self.tokenizer.tokens) != self.manifest["tokenizer_sha256"]:
            raise ValueError("Vocabulary differs from preprocessing manifest")
        self.songs = {}
        for item in self.manifest["songs"]:
            with np.load(directory / item["file"], allow_pickle=False) as archive:
                self.songs[item["song"]] = {key: archive[key] for key in archive.files}

    def split(self, seed: int, validation_fraction: float) -> tuple[list[str], list[str]]:
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        names = sorted(self.songs)
        random.Random(seed).shuffle(names)
        count = max(1, round(len(names) * validation_fraction)) if validation_fraction else 0
        if count >= len(names):
            raise ValueError("Need at least one training song")
        return sorted(names[count:]), sorted(names[:count])

    def required_context(self, task: str, bars: int) -> int:
        tracks = ["basic"] if task == "basic" else ["melody", "piano"]
        maximum = 0
        for song in self.songs.values():
            for track in tracks:
                offsets = song[f"{track}_offsets"]
                count = min(bars, len(offsets) - 1)
                maximum = max(maximum, int((offsets[count:] - offsets[:-count]).max()))
        return maximum + 2

    def sample(
        self,
        names: list[str],
        rng: random.Random,
        task: str,
        bars: int,
        max_tokens: int,
        transpose: int = 0,
    ) -> dict[str, list[int]]:
        song = self.songs[rng.choice(names)]
        return self._sample_song(song, task, bars, max_tokens, transpose, rng)

    def sample_with_song(
        self,
        names: list[str],
        rng: random.Random,
        task: str,
        bars: int,
        max_tokens: int,
        transpose: int = 0,
    ) -> tuple[dict[str, list[int]], str]:
        """Sample an example and return the source song for training provenance."""
        name = rng.choice(names)
        return self._sample_song(self.songs[name], task, bars, max_tokens, transpose, rng), name

    def _sample_song(
        self,
        song: dict,
        task: str,
        bars: int,
        max_tokens: int,
        transpose: int,
        rng: random.Random,
    ) -> dict[str, list[int]]:
        tracks = ["basic"] if task == "basic" else ["melody", "piano"]
        n_bars = len(song[f"{tracks[0]}_offsets"]) - 1
        start = rng.randrange(max(1, n_bars - bars + 1))
        end = min(n_bars, start + bars)
        for track in tracks:
            offsets = song[f"{track}_offsets"]
            required = int(offsets[end] - offsets[start]) + 2
            if required > max_tokens:
                raise ValueError(
                    f"Complete {end - start}-bar {track} window requires {required} tokens; "
                    f"configured context is {max_tokens}. No bars were truncated."
                )
        sequences = []
        for track in tracks:
            offsets = song[f"{track}_offsets"]
            sequence = song[track][offsets[start] : offsets[end]].astype(int).tolist()
            sequences.append([self.tokenizer.bos_id, *sequence, self.tokenizer.eos_id])
        if transpose:
            shift = rng.randint(-transpose, transpose)
            lo, hi = self.tokenizer.groups["Pitch"][0], self.tokenizer.groups["Pitch"][-1]
            if all(lo <= t + shift <= hi for seq in sequences for t in seq if lo <= t <= hi):
                sequences = [self.tokenizer.transpose(seq, shift) for seq in sequences]
        if task == "basic":
            return {"input_ids": sequences[0]}
        # T5 inserts the decoder start token itself; targets must not repeat BOS.
        return {"input_ids": sequences[0], "labels": sequences[1][1:]}


def collate(examples: list[dict], task: str, pad_id: int, device: str) -> dict:
    def pad(key: str, fill: int) -> torch.Tensor:
        result = torch.full((len(examples), max(len(e[key]) for e in examples)), fill)
        for i, example in enumerate(examples):
            result[i, : len(example[key])] = torch.tensor(example[key], dtype=torch.long)
        return result.to(device)

    ids = pad("input_ids", pad_id)
    result = {"input_ids": ids, "attention_mask": ids.ne(pad_id)}
    if task == "basic":
        labels = ids.clone()
        labels[~result["attention_mask"]] = -100
    else:
        labels = pad("labels", -100)
    result["labels"] = labels
    return result
