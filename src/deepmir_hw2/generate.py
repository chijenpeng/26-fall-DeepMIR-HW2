"""Full-context generation with learned timing and no per-bar note-count restrictions."""

import hashlib
import math
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .models import build_model
from .music import BarTiming, BeatGrid, combine_with_prompt, write_midi
from .prepare import load_chords
from .render import render, validate_render_settings
from .tokens import MAX_INTEGER, MusicTokenizer
from .train import checkpoint_path, load_checkpoint
from .utils import choose_device, read_json, write_json


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("Require top_k >= 0 and 0 < top_p <= 1")


def sample_token(logits, allowed, config, rng):
    if not allowed:
        raise ValueError("No legal next token")
    scores = logits[torch.tensor(allowed, device=logits.device)].float() / config.temperature
    if config.top_k:
        threshold = torch.topk(scores, min(config.top_k, len(allowed))).values[-1]
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
    return allowed[torch.multinomial(probabilities, 1, generator=rng).item()]


class ContextLimitError(ValueError):
    pass


class CachedGenerator:
    def __init__(self, model, tokenizer, task, max_tokens, device, prefix, melody=None):
        self.model, self.tokenizer, self.task = model, tokenizer, task
        self.max_tokens, self.device = max_tokens, device
        self.history, self.pending = list(prefix), list(prefix)
        self.past = None
        self.encoder = None
        if len(prefix) > max_tokens:
            raise ContextLimitError("Full prompt exceeds context; no prompt tokens were discarded")
        if task == "bonus":
            if melody is None or len(melody) > max_tokens:
                raise ContextLimitError("Full melody must fit the encoder context")
            ids = torch.tensor([melody], device=device)
            self.encoder = model.get_encoder()(input_ids=ids, return_dict=True)

    def append(self, tokens):
        if len(self.history) + len(tokens) > self.max_tokens:
            raise ContextLimitError(
                f"Complete generation exceeds context capacity {self.max_tokens}; "
                "no bars were shortened and no earlier context was discarded"
            )
        self.history.extend(tokens)
        self.pending.extend(tokens)

    def logits(self):
        if not self.pending:
            raise ValueError("No new input tokens")
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
        self.past, self.pending = output.past_key_values, []
        return output.logits[0, -1]


def integer_choices(tokenizer, prefix, lower, upper):
    """Canonical base-128 prefixes intersecting the allowed numeric interval."""
    result = []
    for digit in range(128):
        value = prefix * 128 + digit
        if lower <= value <= upper:
            result.append(tokenizer.tok2idx[f"End_{digit}"])
        if value == 0:
            continue
        power = 128
        while value * power <= upper:
            if max(lower, value * power) <= min(upper, (value + 1) * power - 1):
                result.append(tokenizer.tok2idx[f"More_{digit}"])
                break
            power *= 128
    return result


@torch.inference_mode()
def generate_bars(
    model,
    tokenizer,
    task,
    bar_count,
    max_tokens,
    device,
    sampling,
    seed,
    prefix=None,
    melody=None,
    conditioned_headers=None,
):
    if bar_count < 1:
        raise ValueError("Need at least one generated bar")
    if conditioned_headers is not None and len(conditioned_headers) != bar_count:
        raise ValueError("Conditioned timing must cover every output bar")
    if task == "bonus" and conditioned_headers is None:
        raise ValueError("Bonus timing comes from the supplied melody")
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

    def choose(allowed, destination):
        token = sample_token(engine.logits(), allowed, sampling, rng)
        engine.append([token])
        destination.append(token)
        return token

    def integer(destination, lower=0, upper=MAX_INTEGER):
        prefix = 0
        while True:
            token = choose(integer_choices(tokenizer, prefix, lower, upper), destination)
            prefix = prefix * 128 + tokenizer.value(token)
            if token in tokenizer.groups["End"]:
                return prefix

    bars = []
    for index in range(bar_count):
        if conditioned_headers is not None:
            timing, cursor = tokenizer.read_header(conditioned_headers[index])
            bar = list(conditioned_headers[index][:cursor])
            engine.append(bar)
        else:
            bar = [tokenizer.bar_id]
            engine.append(bar)
            beats = tokenizer.value(choose(tokenizer.groups["Beats"], bar))
            meter_token = choose(tokenizer.groups["Meter"], bar)
            _, n, d = tokenizer.tokens[meter_token].split("_")
            unit = int(n) * 4 / (int(d) * beats)
            # These bounds are the SMF tempo field's legal domain, not a music-density cap.
            durations = tuple(
                integer(bar, max(1, math.ceil(unit)), math.floor(0xFFFFFF * unit))
                for _ in range(beats)
            )
            timing = BarTiming(durations, int(n), int(d))
        previous = 0
        while True:
            gate = sample_token(
                engine.logits(), [tokenizer.bar_id, tokenizer.tok2idx["Pos"]], sampling, rng
            )
            if gate == tokenizer.bar_id:
                break
            engine.append([gate])
            bar.append(gate)
            previous = integer(bar, previous, timing.duration_us)
            kinds = [*tokenizer.groups["Control"], tokenizer.tok2idx["Bend"]]
            if previous < timing.duration_us:
                kinds += tokenizer.groups["Pitch"]
            if task == "basic":
                kinds += tokenizer.groups["Chord"]
            token = choose(kinds, bar)
            if token in tokenizer.groups["Pitch"]:
                choose(tokenizer.groups["Vel"], bar)
                engine.append([tokenizer.tok2idx["Duration"]])
                bar.append(tokenizer.tok2idx["Duration"])
                integer(bar, 1)
            elif token in tokenizer.groups["Control"]:
                choose(tokenizer.groups["Value"], bar)
            elif token == tokenizer.tok2idx["Bend"]:
                integer(bar, 0, 16383)
        tokenizer.validate_bar(bar)
        bars.append(bar)
    return bars, {"context_tokens": len(engine.history)}


def generate(args):
    import pretty_midi

    bars_to_generate = (
        args.bars if args.bars is not None else (24 if args.mode == "continue" else 32)
    )
    if args.count < 1 or bars_to_generate < 1:
        raise ValueError("count and bars must be positive")
    if args.soundfont:
        validate_render_settings(args.soundfont, args.fluidsynth, args.sample_rate)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    resolved = checkpoint_path(args.checkpoint).resolve()
    saved = load_checkpoint(resolved)
    tokenizer = MusicTokenizer.from_config(saved["tokenizer_config"])
    if tokenizer.tokens != saved["tokenizer_tokens"]:
        raise ValueError("Checkpoint vocabulary mismatch")
    task = "bonus" if args.mode == "accompany" else "basic"
    if saved["config"]["task"] != task:
        raise ValueError(f"{args.mode} requires a {task} model")
    programs = saved["source_programs"]["piano" if task == "bonus" else "basic"]
    if len(programs) != 1:
        raise ValueError("Source programs are ambiguous; do not substitute an invented program")
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    device = choose_device(args.device)
    model = build_model(saved["config"], tokenizer)
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p)
    prompts = sorted(p for p in Path(args.prompts).iterdir() if p.is_dir()) if args.prompts else []
    if args.mode != "unconditional" and not prompts:
        raise ValueError("Conditional generation requires --prompts")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    write_json(output / "tok2idx.json", tokenizer.evaluation_vocabulary())
    jobs = prompts if args.mode != "unconditional" else list(range(args.count))
    sequences, records = [], []
    for index, prompt in enumerate(tqdm(jobs, desc=args.mode)):
        prefix = melody = headers = None
        prompt_bars, origin_us = [], 0
        original = None
        count = bars_to_generate
        if args.mode == "unconditional":
            name = f"{index + 1:03d}"
        else:
            name = prompt.name
            midi_path = prompt / ("melody.mid" if task == "bonus" else f"{name}.mid")
            original = pretty_midi.PrettyMIDI(str(midi_path))
            grid = BeatGrid.load(prompt / "beat_midi.txt")
            boundaries = grid.boundaries()
            expected = 16 if task == "bonus" else 8
            if len(boundaries) - 1 != expected:
                raise ValueError(
                    f"{name} has {len(boundaries) - 1} annotated bars, expected {expected}"
                )
            chords = None
            if task == "basic" and (prompt / "chord_midi.txt").exists():
                chords = load_chords(prompt / "chord_midi.txt")
            prompt_bars = tokenizer.encode_bars(
                original, grid, boundaries, "MELODY" if task == "bonus" else None, chords
            )
            origin_us = round(grid.times[0] * 1_000_000)
            flat = [token for bar in prompt_bars for token in bar]
            if task == "bonus":
                # No reference piano or chord file is read on this inference path.
                melody = [tokenizer.bos_id, *flat, tokenizer.eos_id]
                headers, count = prompt_bars, len(prompt_bars)
            else:
                prefix = [tokenizer.bos_id, *flat]
        generated, diagnostic = generate_bars(
            model,
            tokenizer,
            task,
            count,
            saved["config"]["data"]["max_tokens"],
            device,
            sampling,
            args.seed + index,
            prefix,
            melody,
            headers,
        )
        generated_origin = origin_us
        if args.mode == "continue":
            generated_origin += sum(
                tokenizer.read_header(bar)[0].duration_us for bar in prompt_bars
            )
        instrument = tokenizer.decode_instrument(generated, generated_origin, program=programs[0])
        instruments = (
            [instrument]
            if original is None
            else combine_with_prompt(
                original, instrument, "PIANO" if task == "bonus" else "GENERATED"
            )
        )
        all_bars = prompt_bars + generated if args.mode == "continue" else generated
        timings = [tokenizer.read_header(bar)[0] for bar in all_bars]
        write_midi(output / f"{name}.mid", instruments, timings, origin_us)
        sequences.append({"song_name": name, "tokens": tokenizer.evaluation_tokens(all_bars)})
        record = {
            "song": name,
            "seed": args.seed + index,
            "bars": len(all_bars),
            "generated_notes": len(instrument.notes),
            **diagnostic,
        }
        if original is not None:
            record["prompt_sha256"] = hashlib.sha256(midi_path.read_bytes()).hexdigest()
        records.append(record)
        write_json(output / "sequences.json", sequences)
        write_json(
            output / "generation.json",
            {
                "mode": args.mode,
                "checkpoint": str(resolved),
                "checkpoint_sha256": digest,
                "checkpoint_step": saved["step"],
                "sampling": asdict(sampling),
                "expected_pieces": len(jobs),
                "midi_complete": len(records) == len(jobs),
                "audio_requested": bool(args.soundfont),
                "audio_complete": False,
                "complete": len(records) == len(jobs) and not args.soundfont,
                "songs": records,
            },
        )
    if args.soundfont:
        render(
            Namespace(
                input=str(output),
                output=str(output),
                soundfont=args.soundfont,
                fluidsynth=args.fluidsynth,
                sample_rate=args.sample_rate,
            )
        )
        metadata = read_json(output / "generation.json")
        metadata["audio_complete"] = metadata["complete"] = True
        write_json(output / "generation.json", metadata)


def add_parser(subparsers):
    parser = subparsers.add_parser("generate", help=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["unconditional", "continue", "accompany"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--bars", type=int, help="Default: 32 unconditional, 24 continuation")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--soundfont", help="Also produce WAV listening samples in the output directory"
    )
    parser.add_argument("--fluidsynth", default="fluidsynth")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.set_defaults(func=generate)
