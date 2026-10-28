"""Random bar-aligned training windows with pitch-shift augmentation."""

import json

import torch

from . import remi


class Corpus:
    def __init__(self, path, chords=True):
        payload = json.loads(open(path).read())
        if payload["vocab"] != remi.VOCAB:
            raise ValueError("Prepared data uses a different vocabulary; rerun prepare")
        self.songs = payload["songs"]
        if not chords:  # ablation without the supplied chord annotations
            for song in self.songs.values():
                song["basic"] = [[t for t in b if remi.kind(t) != "Chord"] for b in song["basic"]]
        self.names = sorted(self.songs)

    def window(self, rng, task, bars, max_len, transpose):
        song = self.songs[rng.choice(self.names)]
        tracks = ["basic"] if task == "basic" else ["melody", "piano"]
        total = len(song[tracks[0]])
        start = rng.randrange(max(1, total - bars + 1))
        sequences = []
        for track, limit in zip(tracks, max_len, strict=True):
            tokens = [remi.BOS] + [t for bar in song[track][start : start + bars] for t in bar]
            if start + bars >= total:
                tokens.append(remi.EOS)
            sequences.append(tokens[:limit])
        shift = rng.randint(-transpose, transpose)
        shifted = [remi.transpose_tokens(s, shift) for s in sequences]
        if all(s is not None for s in shifted):
            sequences = shifted
        return sequences


def batch(corpus, rng, task, size, bars, max_len, transpose):
    windows = [corpus.window(rng, task, bars, max_len, transpose) for _ in range(size)]

    def pad(index, fill):
        longest = max(len(w[index]) for w in windows)
        out = torch.full((size, longest), fill, dtype=torch.long)
        for i, w in enumerate(windows):
            out[i, : len(w[index])] = torch.tensor(w[index])
        return out

    ids = pad(0, remi.PAD)
    mask = ids.ne(remi.PAD)
    if task == "basic":
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(~mask, -100)}
    # T5 prepends the decoder start token (BOS) itself, so targets drop the leading BOS.
    labels = pad(1, -100)[:, 1:].contiguous()
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}
