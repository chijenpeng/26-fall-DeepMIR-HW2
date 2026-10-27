"""Full-context generation with learned timing and no per-bar note-count restrictions."""

import copy
import hashlib
import json
import math
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from .models import build_model
from .music import BarTiming, BeatGrid, combine_with_prompt, write_midi
from .prepare import load_chords
from .render import render, validate_render_settings
from .tokens import MAX_INTEGER, MusicTokenizer
from .train import checkpoint_path, load_checkpoint
from .utils import choose_device, write_json

NUMERIC_BEAM_WIDTH = 4


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    numeric_strategy: str = "sample"

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("Require top_k >= 0 and 0 < top_p <= 1")
        if self.numeric_strategy not in {"sample", "beam"}:
            raise ValueError("numeric_strategy must be 'sample' or 'beam'")


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


class RawBarSidecar:
    """Save exact completed and partial bar token arrays for one generated song."""

    def __init__(self, path, song, strategy):
        self.path = Path(path)
        self.song = song
        self.strategy = strategy
        self.bars = {}

    def record(self, bar, tokens, complete):
        self.bars[bar] = {"bar": bar, "tokens": list(tokens), "complete": complete}
        self.flush()

    def flush(self):
        write_json(
            self.path,
            {
                "song": self.song,
                "beam_width": NUMERIC_BEAM_WIDTH if self.strategy == "beam" else None,
                "bars": [self.bars[key] for key in sorted(self.bars)],
            },
        )


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

    def fork(self):
        child = object.__new__(CachedGenerator)
        child.model = self.model
        child.tokenizer = self.tokenizer
        child.task = self.task
        child.max_tokens = self.max_tokens
        child.device = self.device
        child.history = list(self.history)
        child.pending = list(self.pending)
        child.past = fork_cache(self.past)
        child.encoder = self.encoder
        return child

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


def fork_cache(cache):
    """Copy cache containers while sharing immutable cached K/V tensors."""
    if cache is None:
        return None
    if isinstance(cache, (int, float, str, bool)):
        return cache
    if isinstance(cache, torch.Tensor):
        return cache
    if isinstance(cache, tuple):
        return tuple(fork_cache(item) for item in cache)
    if isinstance(cache, list):
        return [fork_cache(item) for item in cache]
    if isinstance(cache, EncoderDecoderCache):
        clone = copy.copy(cache)
        clone.self_attention_cache = fork_cache(cache.self_attention_cache)
        clone.cross_attention_cache = fork_cache(cache.cross_attention_cache)
        clone.is_updated = dict(cache.is_updated)
        return clone
    if isinstance(cache, DynamicCache):
        clone = copy.copy(cache)
        clone.layers = [copy.copy(layer) for layer in cache.layers]
        return clone
    raise TypeError(f"Unsupported mutable cache type for beam branching: {type(cache).__name__}")


@dataclass
class IntegerBeam:
    engine: CachedGenerator
    prefix: int
    tokens: tuple[int, ...]
    score: float


def beam_integer(engine, tokenizer, lower, upper, sampling):
    """Find a high joint-probability complete integer by legal-prefix beam search."""
    width = NUMERIC_BEAM_WIDTH
    active = [IntegerBeam(engine, 0, (), 0.0)]
    finished = []
    while active:
        active_candidates = []
        finished_candidates = []
        for beam in active:
            if len(beam.engine.history) >= beam.engine.max_tokens:
                continue
            logits = beam.engine.logits()
            allowed = integer_choices(tokenizer, beam.prefix, lower, upper)
            if not allowed:
                continue
            log_probs = (
                torch.log_softmax(
                    logits[torch.tensor(allowed, device=logits.device)].float()
                    / sampling.temperature,
                    dim=-1,
                )
                .detach()
                .cpu()
                .tolist()
            )
            for token, log_prob in zip(allowed, log_probs, strict=True):
                if not math.isfinite(log_prob):
                    continue
                prefix = beam.prefix * 128 + tokenizer.value(token)
                candidate = {
                    "parent": beam.engine,
                    "token": token,
                    "prefix": prefix,
                    "tokens": (*beam.tokens, token),
                    "score": beam.score + log_prob,
                }
                if token in tokenizer.groups["End"]:
                    finished_candidates.append(candidate)
                else:
                    active_candidates.append(candidate)

        active_candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        finished_candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        active = []
        for candidate in active_candidates[:width]:
            child = candidate["parent"].fork()
            child.append([candidate["token"]])
            active.append(
                IntegerBeam(
                    child,
                    candidate["prefix"],
                    candidate["tokens"],
                    candidate["score"],
                )
            )
        for candidate in finished_candidates[:width]:
            child = candidate["parent"].fork()
            child.append([candidate["token"]])
            finished.append(
                IntegerBeam(
                    child,
                    candidate["prefix"],
                    candidate["tokens"],
                    candidate["score"],
                )
            )
        finished.sort(key=lambda beam: beam.score, reverse=True)
        finished = finished[:width]
        if finished and (not active or finished[0].score >= active[0].score):
            break
    if not finished:
        raise ValueError(f"No complete canonical integer in range [{lower}, {upper}]")
    best = finished[0]
    return best.engine, best.prefix, best.tokens, best.score


def sampling_metadata(sampling):
    metadata = asdict(sampling)
    metadata["beam_width"] = NUMERIC_BEAM_WIDTH if sampling.numeric_strategy == "beam" else None
    return metadata


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
    on_bar=None,
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
        nonlocal engine
        if sampling.numeric_strategy == "beam":
            engine, value, selected_tokens, _score = beam_integer(
                engine,
                tokenizer,
                lower,
                upper,
                sampling,
            )
            destination.extend(selected_tokens)
            return value
        prefix = 0
        while True:
            token = choose(integer_choices(tokenizer, prefix, lower, upper), destination)
            prefix = prefix * 128 + tokenizer.value(token)
            if token in tokenizer.groups["End"]:
                return prefix

    bars = []
    for index in range(bar_count):
        bar = []
        try:
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
                durations = tuple(
                    integer(
                        bar,
                        max(1, math.ceil(unit)),
                        math.floor(0xFFFFFF * unit),
                    )
                    for _ in range(beats)
                )
                timing = BarTiming(durations, int(n), int(d))
            previous = 0
            while True:
                gate_choices = [tokenizer.bar_id, tokenizer.tok2idx["Pos"]]
                if index == bar_count - 1:
                    gate_choices.append(tokenizer.eos_id)
                gate = sample_token(engine.logits(), gate_choices, sampling, rng)
                if gate in (tokenizer.bar_id, tokenizer.eos_id):
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
                    duration_token = tokenizer.tok2idx["Duration"]
                    engine.append([duration_token])
                    bar.append(duration_token)
                    integer(bar, 1)
                elif token in tokenizer.groups["Control"]:
                    choose(tokenizer.groups["Value"], bar)
                elif token == tokenizer.tok2idx["Bend"]:
                    integer(bar, 0, 16383)
            tokenizer.validate_bar(bar)
        except Exception:
            if on_bar is not None:
                on_bar(index + 1, len(engine.history), list(bar), False)
            raise
        bars.append(bar)
        if on_bar is not None:
            on_bar(len(bars), len(engine.history), list(bar), True)
    return bars, {"context_tokens": len(engine.history)}


def generate(args):
    import pretty_midi

    bars_to_generate = (
        args.bars if args.bars is not None else (24 if args.mode == "continue" else 32)
    )
    if args.count < 1 or bars_to_generate < 1:
        raise ValueError("count and bars must be positive")
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
    sampling = SamplingConfig(
        args.temperature,
        args.top_k,
        args.top_p,
        getattr(args, "numeric_strategy", "sample"),
    )
    prompts = sorted(p for p in Path(args.prompts).iterdir() if p.is_dir()) if args.prompts else []
    if args.mode != "unconditional" and not prompts:
        raise ValueError("Conditional generation requires --prompts")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    write_json(output / "tok2idx.json", tokenizer.evaluation_vocabulary())
    jobs = prompts if args.mode != "unconditional" else list(range(args.count))
    sequences, records = [], []
    manifest = {
        "mode": args.mode,
        "checkpoint": str(resolved),
        "checkpoint_sha256": digest,
        "checkpoint_step": saved["step"],
        "sampling": sampling_metadata(sampling),
        "expected_pieces": len(jobs),
        "midi_complete": False,
        "audio_requested": bool(args.soundfont),
        "audio_complete": False,
        "trace_requested": bool(getattr(args, "trace", False)),
        "complete": False,
        "status": "running",
        "songs": records,
    }

    def save_manifest():
        write_json(output / "generation.json", manifest)

    @contextmanager
    def manifest_stage(stage, song=None):
        manifest["stage"] = stage
        if song is not None:
            manifest["current_song"] = song
        save_manifest()
        try:
            yield
        except Exception as error:
            manifest.update(
                status="failed",
                complete=False,
                failure={
                    "stage": stage,
                    "song": song,
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
            save_manifest()
            raise
        else:
            manifest.pop("stage", None)
            manifest.pop("current_song", None)

    save_manifest()
    if args.soundfont:
        with manifest_stage("render_preflight"):
            validate_render_settings(args.soundfont, args.fluidsynth, args.sample_rate)
    for index, prompt in enumerate(tqdm(jobs, desc=args.mode)):
        prefix = melody = headers = None
        prompt_bars, origin_us = [], 0
        original = None
        count = bars_to_generate
        name = f"{index + 1:03d}" if args.mode == "unconditional" else prompt.name
        with manifest_stage("prompt_parse", name):
            if args.mode != "unconditional":
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
        sidecar = (
            RawBarSidecar(output / f"{name}.bars.json", name, sampling.numeric_strategy)
            if getattr(args, "trace", False)
            else None
        )
        with manifest_stage("generation", name):

            def report_bar(bar_number, tokens, bar_tokens, complete, song=name, writer=sidecar):
                print(
                    json.dumps(
                        {
                            "song": song,
                            "bar": bar_number,
                            "complete": complete,
                            "context_tokens": tokens,
                        }
                    ),
                    flush=True,
                )
                if writer:
                    writer.record(bar_number, bar_tokens, complete)

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
                on_bar=report_bar,
            )
        with manifest_stage("decode_midi_export", name):
            generated_origin = origin_us
            if args.mode == "continue":
                generated_origin += sum(
                    tokenizer.read_header(bar)[0].duration_us for bar in prompt_bars
                )
            instrument = tokenizer.decode_instrument(
                generated, generated_origin, program=programs[0]
            )
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
            manifest.update(
                status="running",
                midi_complete=len(records) == len(jobs),
                songs=records,
            )
            save_manifest()
    if args.soundfont:
        with manifest_stage("render"):
            validate_render_settings(args.soundfont, args.fluidsynth, args.sample_rate)
            render(
                Namespace(
                    input=str(output),
                    output=str(output),
                    soundfont=args.soundfont,
                    fluidsynth=args.fluidsynth,
                    sample_rate=args.sample_rate,
                )
            )
            manifest["audio_complete"] = manifest["complete"] = True
            manifest["status"] = "complete"
        save_manifest()
    elif manifest["midi_complete"]:
        manifest["complete"] = True
        manifest["status"] = "complete"
        save_manifest()


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
    parser.add_argument(
        "--numeric-strategy",
        choices=["sample", "beam"],
        default="sample",
        help=(
            "Numeric-field decoding: sample applies top-k/top-p; beam searches complete "
            "integers with temperature only. Categorical tokens use top-k/top-p in both modes."
        ),
    )
    parser.add_argument(
        "--trace", action="store_true", help="Save raw completed and partial bar token arrays"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--soundfont", help="Also produce WAV listening samples in the output directory"
    )
    parser.add_argument("--fluidsynth", default="fluidsynth")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.set_defaults(func=generate)
