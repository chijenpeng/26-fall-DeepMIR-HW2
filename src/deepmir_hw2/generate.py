"""Cached autoregressive sampling with valid note grammar and exact bar counts."""

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pretty_midi
import torch
from tqdm import tqdm

from .models import build_model
from .music import BeatGrid, combine_with_prompt
from .tokens import MusicTokenizer, TokenizerConfig
from .train import checkpoint_path, load_checkpoint
from .utils import choose_device, write_json


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    min_notes_per_bar: int = 1
    max_notes_per_bar: int = 128

    def __post_init__(self) -> None:
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("Require temperature > 0, top_k >= 0 and 0 < top_p <= 1")
        if not 0 <= self.min_notes_per_bar <= self.max_notes_per_bar:
            raise ValueError("Invalid note count limits")


def sample_token(logits: torch.Tensor, allowed: list[int], config: SamplingConfig, rng) -> int:
    if not allowed:
        raise ValueError("Grammar has no legal next token")
    allowed_tensor = torch.tensor(allowed, device=logits.device)
    scores = logits[allowed_tensor].float() / config.temperature
    if config.top_k:
        k = min(config.top_k, len(allowed))
        threshold = torch.topk(scores, k).values[-1]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if config.top_p < 1:
        ordered, indices = scores.sort(descending=True)
        excluded = ordered.softmax(-1).cumsum(-1) > config.top_p
        excluded[1:] = excluded[:-1].clone()
        excluded[0] = False
        scores[indices[excluded]] = -torch.inf
    probabilities = scores.softmax(-1)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("Non-finite sampling probabilities")
    choice = torch.multinomial(probabilities, 1, generator=rng).item()
    return allowed[choice]


class CachedGenerator:
    def __init__(self, model, tokenizer, task, max_tokens, device, prefix, melody=None):
        self.model, self.tokenizer, self.task = model, tokenizer, task
        self.max_tokens, self.device = max_tokens, device
        self.history = list(prefix)
        self.pending = list(prefix)
        self.past = None
        self.cached_tokens = 0
        self.encoder = None
        if task == "bonus":
            if melody is None or len(melody) > max_tokens:
                raise ValueError("Full conditioning melody must fit max_tokens")
            ids = torch.tensor([melody], device=device)
            self.encoder = model.get_encoder()(input_ids=ids, return_dict=True)

    def append(self, tokens: list[int]) -> None:
        self.history.extend(tokens)
        self.pending.extend(tokens)

    def logits(self) -> torch.Tensor:
        if self.cached_tokens + len(self.pending) > self.max_tokens:
            if self.task == "bonus":
                raise ValueError("Accompaniment exceeded decoder token budget")
            # Rebase absolute positions only at a complete bar boundary.
            earliest = len(self.history) - self.max_tokens + 1
            starts = [
                i
                for i, token in enumerate(self.history)
                if token == self.tokenizer.bar_id and i >= earliest
            ]
            if not starts:
                raise ValueError("A generated bar is longer than the model context")
            self.pending = [self.tokenizer.bos_id, *self.history[starts[0] :]]
            self.past, self.cached_tokens = None, 0
        ids = torch.tensor([self.pending], device=self.device)
        if self.task == "basic":
            output = self.model(input_ids=ids, past_key_values=self.past, use_cache=True)
        else:
            output = self.model(
                encoder_outputs=self.encoder,
                decoder_input_ids=ids,
                past_key_values=self.past,
                use_cache=True,
            )
        self.past = output.past_key_values
        self.cached_tokens += len(self.pending)
        self.pending = []
        return output.logits[0, -1]


@torch.inference_mode()
def generate_bars(
    model,
    tokenizer: MusicTokenizer,
    task: str,
    beats_per_bar: list[int],
    max_tokens: int,
    device: str,
    sampling: SamplingConfig,
    seed: int,
    prefix: list[int] | None = None,
    melody: list[int] | None = None,
) -> tuple[list[list[int]], dict]:
    if not beats_per_bar:
        raise ValueError("Need at least one output bar")
    engine = CachedGenerator(
        model,
        tokenizer,
        task,
        max_tokens,
        device,
        prefix if prefix is not None else [tokenizer.bos_id],
        melody,
    )
    rng = torch.Generator(device=device).manual_seed(seed)
    max_notes = min(sampling.max_notes_per_bar, (max_tokens - 3) // 4)
    if task == "bonus":
        # Reserve all bar headers up front; never silently truncate a dense target.
        max_notes = min(
            max_notes, (max_tokens - 1 - 2 * len(beats_per_bar)) // (4 * len(beats_per_bar))
        )
    if max_notes < sampling.min_notes_per_bar:
        raise ValueError("Token budget cannot satisfy minimum notes per bar")
    bars, forced = [], []
    for index, beats in enumerate(beats_per_bar):
        if not 1 <= beats <= tokenizer.config.max_beats_per_bar:
            raise ValueError(f"Unsupported number of beats: {beats}")
        bar = [tokenizer.bar_id, tokenizer.tok2idx[f"Beats_{beats}"]]
        engine.append(bar)
        position, previous_pitch, count = 0, -1, 0
        while count < max_notes:
            # At a shared onset, ascending pitches prevent duplicated note events.
            first_position = position + int(previous_pitch == tokenizer.config.pitch_max)
            positions = [
                t
                for t in tokenizer.groups["Pos"]
                if first_position
                <= tokenizer.value(t)
                < beats * tokenizer.config.positions_per_beat
            ]
            allowed = positions + (
                [tokenizer.bar_id] if count >= sampling.min_notes_per_bar else []
            )
            next_token = sample_token(engine.logits(), allowed, sampling, rng)
            if next_token == tokenizer.bar_id:
                break
            new_position = tokenizer.value(next_token)
            pitch_min = (
                previous_pitch + 1 if new_position == position else tokenizer.config.pitch_min
            )
            engine.append([next_token])
            bar.append(next_token)
            pitches = [t for t in tokenizer.groups["Pitch"] if tokenizer.value(t) >= pitch_min]
            pitch = sample_token(engine.logits(), pitches, sampling, rng)
            engine.append([pitch])
            bar.append(pitch)
            for group in ["Vel", "Dur"]:
                token = sample_token(engine.logits(), tokenizer.groups[group], sampling, rng)
                engine.append([token])
                bar.append(token)
            position, previous_pitch = new_position, tokenizer.value(pitch)
            count += 1
        if count == max_notes:
            forced.append(index)
        tokenizer.validate_bar(bar, beats)
        bars.append(bar)
    return bars, {"note_limit": max_notes, "bars_reaching_note_limit": forced}


def generate(args: argparse.Namespace) -> None:
    if args.count <= 0 or args.bars <= 0 or args.tempo <= 0:
        raise ValueError("count, bars and tempo must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    resolved_checkpoint = checkpoint_path(args.checkpoint).resolve()
    saved = load_checkpoint(resolved_checkpoint)
    with resolved_checkpoint.open("rb") as stream:
        checkpoint_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    config = saved["config"]
    tokenizer = MusicTokenizer(TokenizerConfig(**saved["tokenizer_config"]))
    if tokenizer.tokens != saved["tokenizer_tokens"]:
        raise ValueError("Checkpoint vocabulary mismatch")
    required_task = "bonus" if args.mode == "accompany" else "basic"
    if config["task"] != required_task:
        raise ValueError(f"{args.mode} requires a {required_task} checkpoint")
    device = choose_device(args.device)
    model = build_model(config, tokenizer)
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    sampling = SamplingConfig(
        args.temperature, args.top_k, args.top_p, args.min_notes_per_bar, args.max_notes_per_bar
    )
    prompts = sorted(p for p in Path(args.prompts).iterdir() if p.is_dir()) if args.prompts else []
    if args.mode != "unconditional" and not prompts:
        raise ValueError("--prompts must contain song subdirectories")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    write_json(output / "tok2idx.json", tokenizer.tok2idx)
    records, sequences = [], []
    jobs = prompts if args.mode != "unconditional" else list(range(args.count))
    for index, prompt in enumerate(tqdm(jobs, desc=args.mode)):
        seed = args.seed + index
        prefix = melody = None
        prompt_bars = []
        original = None
        if args.mode == "unconditional":
            name = f"{index + 1:03d}"
            grid = BeatGrid((0.0, 60.0 / args.tempo), (0,))
            boundaries = list(range(0, 4 * args.bars + 1, 4))
            output_beats = [4] * args.bars
            original = pretty_midi.PrettyMIDI(initial_tempo=args.tempo)
            generated_boundaries = boundaries
        else:
            name = prompt.name
            midi_path = prompt / ("melody.mid" if args.mode == "accompany" else f"{name}.mid")
            original = pretty_midi.PrettyMIDI(str(midi_path))
            grid = BeatGrid.load(prompt / "beat_midi.txt")
            boundaries = grid.boundaries()
            track = "MELODY" if args.mode == "accompany" else None
            prompt_bars = tokenizer.encode_bars(original, grid, boundaries, track)
            if args.mode == "continue":
                prefix = [tokenizer.bos_id, *[t for bar in prompt_bars for t in bar]]
                width = boundaries[-1] - boundaries[-2]
                generated_boundaries = [boundaries[-1] + i * width for i in range(args.bars + 1)]
                output_beats = [width] * args.bars
            else:
                melody = [
                    tokenizer.bos_id,
                    *[t for bar in prompt_bars for t in bar],
                    tokenizer.eos_id,
                ]
                generated_boundaries = boundaries
                output_beats = [b - a for a, b in zip(boundaries, boundaries[1:], strict=False)]
        generated, diagnostic = generate_bars(
            model,
            tokenizer,
            required_task,
            output_beats,
            config["data"]["max_tokens"],
            device,
            sampling,
            seed,
            prefix,
            melody,
        )
        notes = tokenizer.decode_notes(generated, grid, generated_boundaries)
        if not notes:
            raise ValueError(f"Generation {name} contains no notes")
        end_time = grid.to_seconds(generated_boundaries[-1])
        midi = combine_with_prompt(
            original, notes, "PIANO" if args.mode == "accompany" else "GENERATED", end_time
        )
        midi.write(str(output / f"{name}.mid"))
        all_bars = prompt_bars + generated if args.mode == "continue" else generated
        sequences.append(
            {"song_name": name, "tokens": [tokenizer.tokens[t] for bar in all_bars for t in bar]}
        )
        record = {
            "song": name,
            "seed": seed,
            "bars": len(all_bars),
            "generated_notes": len(notes),
            "end_time_seconds": end_time,
            **diagnostic,
        }
        if args.mode != "unconditional":
            record["prompt_sha256"] = hashlib.sha256(midi_path.read_bytes()).hexdigest()
        records.append(record)
        write_json(output / "sequences.json", sequences)
        write_json(
            output / "generation.json",
            {
                "mode": args.mode,
                "checkpoint": str(resolved_checkpoint),
                "checkpoint_sha256": checkpoint_digest,
                "checkpoint_step": saved["step"],
                "sampling": asdict(sampling),
                "device": device,
                "songs": records,
            },
        )


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("generate", help=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["unconditional", "continue", "accompany"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", help="POP909/prompts/basic-b or POP909/prompts/bonus")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--bars", type=int, default=32, help="Generated bars; use 24 for continuation"
    )
    parser.add_argument("--tempo", type=float, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-notes-per-bar", type=int, default=1)
    parser.add_argument("--max-notes-per-bar", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.set_defaults(func=generate)
