"""Unconditional generation, prompt continuation and melody-to-piano accompaniment."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pretty_midi
import torch

from . import remi
from .models import build_model
from .render import render_wav

KIND_IDS = {
    k: [i for i in range(len(remi.VOCAB)) if remi.kind(i) == k]
    for k in ["Bar", "Beat", "Tempo", "Pos", "Chord", "Pitch", "Vel", "Dur"]
}
MAX_BAR_TOKENS = 680  # the longest bar in the prepared training data has 677 tokens


@dataclass
class Sampling:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(saved["config"])
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), saved["config"]


class Grammar:
    """Masks tokens that would break REMI syntax and ends after exactly `bars` bars.

    `beats` gives fixed beat counts per bar (accompaniment); otherwise the model's
    own Beat token sets the number of legal positions in each bar. A bar reaching
    MAX_BAR_TOKENS must end; `capped` counts such bars.
    """

    def __init__(self, bars, header, beats=None, chords=True):
        self.bars, self.header, self.beats, self.chords = bars, header, beats, chords
        self.count, self.previous, self.pos, self.bar_beats = 0, "<BOS>", -1, 4
        self.done, self.bar_tokens, self.capped = False, 0, 0

    def feed(self, token):
        k = remi.kind(token)
        if k in ("Bar", "<EOS>"):
            self.capped += self.bar_tokens >= MAX_BAR_TOKENS
            self.bar_tokens = 0
        self.bar_tokens += 1
        if k == "Bar":
            if self.count == self.bars:
                self.done = True
                return
            self.count += 1
            self.pos = -1
            if self.beats:
                self.bar_beats = self.beats[self.count - 1]
        elif k == "<EOS>":
            self.done = True
        elif k == "Beat":
            self.bar_beats = int(remi.value(token))
        elif k == "Pos":
            self.pos = int(remi.value(token))
        self.previous = k

    def allowed(self):
        """Token kinds that may follow the previous token, plus EOS once all bars exist."""
        p = self.previous
        if p == "<BOS>":
            return ["Bar"], []
        if p == "Bar" and self.header:
            return ["Beat"], []
        if p == "Beat":
            return ["Tempo"], []
        if p == "Pos":
            return (["Chord", "Pitch"] if self.header and self.chords else ["Pitch"]), []
        if p == "Pitch":
            return ["Vel"], []
        if p == "Vel":
            return ["Dur"], []
        # After a header, Chord or Dur: another note here, a later position, or a new bar.
        # A Bar after the last requested bar, or EOS, finishes the piece.
        kinds = ["Pos", "Bar"] + ([] if p in ("Bar", "Tempo") else ["Pitch"])
        if self.bar_tokens >= MAX_BAR_TOKENS:
            kinds = ["Bar"]
        return kinds, [remi.EOS] if self.count == self.bars else []

    def mask(self, size, device):
        kinds, extra = self.allowed()
        ids = [i for k in kinds for i in KIND_IDS[k]] + extra
        mask = torch.zeros(size, dtype=torch.bool, device=device)
        mask[ids] = True
        if "Pos" in kinds:
            limit = min(remi.MAX_BEATS, self.bar_beats) * remi.POS_PER_BEAT
            positions = KIND_IDS["Pos"]
            mask[positions[: self.pos + 1]] = False
            mask[positions[limit:]] = False
        return mask


def sample(logits, mask, sampling, generator):
    logits = logits.float() / sampling.temperature
    logits[~mask] = float("-inf")
    if sampling.top_k:
        kth = torch.topk(logits, min(sampling.top_k, int(mask.sum()))).values[-1]
        logits[logits < kth] = float("-inf")
    if sampling.top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True)
        probs = torch.softmax(ordered, -1)
        drop = probs.cumsum(-1) - probs > sampling.top_p
        logits[index[drop]] = float("-inf")
    probs = torch.softmax(logits, -1)
    return int(torch.multinomial(probs.cpu(), 1, generator=generator))


@torch.inference_mode()
def continue_gpt2(model, prefix, grammar, sampling, generator, max_len):
    """Extend `prefix` until the grammar finishes; slide to the latest bars if context fills."""
    device = next(model.parameters()).device
    tokens = list(prefix)
    for t in tokens:
        grammar.feed(t)
    past, cached = None, 0
    while not grammar.done:
        if past is None or cached >= max_len:
            window = tokens
            if past is not None:  # context full: restart from bars in the last half window
                recent = [i for i, t in enumerate(tokens) if t == remi.BAR]
                recent = [i for i in recent if len(tokens) - i < max_len // 2]
                window = [remi.BOS] + tokens[recent[0] :] if recent else tokens[-max_len // 2 :]
            out = model(torch.tensor([window], device=device), use_cache=True)
            cached = len(window)
        else:
            out = model(
                torch.tensor([[tokens[-1]]], device=device), past_key_values=past, use_cache=True
            )
            cached += 1
        past = out.past_key_values
        mask = grammar.mask(out.logits.shape[-1], device)
        token = sample(out.logits[0, -1], mask, sampling, generator)
        grammar.feed(token)
        if not grammar.done:
            tokens.append(token)
    return tokens


@torch.inference_mode()
def accompany_t5(model, melody, grammar, sampling, generator, max_len):
    device = next(model.parameters()).device
    ids = torch.tensor([melody[: max_len[0]]], device=device)
    encoder = model.get_encoder()(input_ids=ids)
    tokens, past = [remi.BOS], None
    grammar.feed(remi.BOS)
    while not grammar.done:
        out = model(
            encoder_outputs=encoder,
            decoder_input_ids=torch.tensor([[tokens[-1]]], device=device),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        mask = grammar.mask(out.logits.shape[-1], device)
        token = sample(out.logits[0, -1], mask, sampling, generator)
        grammar.feed(token)
        if not grammar.done:
            tokens.append(token)
    return tokens


def write_midi(path, instruments):
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.extend(instruments)
    midi.write(str(path))


def prompt_folders(directory):
    return sorted(p for p in Path(directory).iterdir() if p.is_dir())


def generate(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    use_chords = config.get("chords", True)
    max_len = config["max_len"]
    sampling = Sampling(args.temperature, args.top_k, args.top_p)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []

    def finish(name, instruments, tokens, seed, grammar):
        path = output / f"{name}.mid"
        write_midi(path, instruments)
        notes = sum(len(i.notes) for i in instruments)
        records.append(
            {
                "name": name,
                "seed": seed,
                "notes": notes,
                "capped_bars": grammar.capped,
                "tokens": tokens,
            }
        )
        print(f"{path}: {notes} notes", flush=True)
        if args.soundfont:
            render_wav(path, path.with_suffix(".wav"), args.soundfont)

    if args.mode == "unconditional":
        for i in range(args.count):
            seed = args.seed + i
            generator = torch.Generator().manual_seed(seed)
            grammar = Grammar(args.bars, header=True, chords=use_chords)
            tokens = continue_gpt2(model, [remi.BOS], grammar, sampling, generator, max_len[0])
            finish(f"{i + 1:03d}", [remi.decode(tokens)], tokens, seed, grammar)
    elif args.mode == "continue":
        for i, folder in enumerate(prompt_folders(args.prompts)):
            seed = args.seed + i
            generator = torch.Generator().manual_seed(seed)
            midi = pretty_midi.PrettyMIDI(str(folder / f"{folder.name}.mid"))
            notes = [n for inst in midi.instruments for n in inst.notes]
            grid = remi.Grid.from_file(folder / "beat_midi.txt", notes)
            chords = remi.read_chords(folder / "chord_midi.txt") if use_chords else ()
            prefix = [remi.BOS] + [t for bar in remi.encode(notes, grid, chords) for t in bar]
            grammar = Grammar(grid.n_bars + args.bars, header=True, chords=use_chords)
            tokens = continue_gpt2(model, prefix, grammar, sampling, generator, max_len[0])
            # Keep the supplied prompt verbatim; append the continuation after its last bar.
            continuation = remi.decode(
                tokens[len(prefix) :], start=float(grid.beats[-1]), name="CONTINUATION"
            )
            finish(folder.name, [*midi.instruments, continuation], tokens, seed, grammar)
    else:
        for i, folder in enumerate(prompt_folders(args.prompts)):
            seed = args.seed + i
            generator = torch.Generator().manual_seed(seed)
            melody = pretty_midi.PrettyMIDI(str(folder / "melody.mid")).instruments[0]
            grid = remi.Grid.from_file(folder / "beat_midi.txt", melody.notes)
            source = [remi.BOS] + [t for bar in remi.encode(melody.notes, grid) for t in bar]
            grammar = Grammar(grid.n_bars, header=False, beats=grid.beat_counts())
            tokens = accompany_t5(model, source, grammar, sampling, generator, max_len)
            melody.name = "MELODY"
            finish(folder.name, [melody, remi.decode(tokens, grid=grid)], tokens, seed, grammar)

    summary = {
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "bars": args.bars,
        "sampling": asdict(sampling),
        "pieces": records,
    }
    (output / "generation.json").write_text(json.dumps(summary))


def add_parser(subparsers):
    parser = subparsers.add_parser("generate", help=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["unconditional", "continue", "accompany"], required=True)
    parser.add_argument("--output", required=True, help="Directory for .mid/.wav files")
    parser.add_argument("--prompts", help="Prompt folder (continue/accompany)")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--bars",
        type=int,
        default=32,
        help="Bars to generate (continuation: bars after the prompt)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--soundfont", help="Also render WAV with FluidSynth")
    parser.add_argument("--device")
    parser.set_defaults(func=generate)
