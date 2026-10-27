"""Single-GPU training with token-weighted losses and restartable checkpoints."""

import argparse
import contextlib
import json
import math
import os
import platform
import random
import signal
import socket
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

from .data import MusicCorpus, collate
from .models import build_model
from .utils import choose_device, read_json, seed_everything, write_json

FINAL_CORPUS_SIZE = 896


def checkpoint_path(path: str | Path, preference: str = "best") -> Path:
    path = Path(path)
    if path.is_dir():
        pointer = path / f"{preference}.json"
        if not pointer.exists() and preference == "best":
            pointer = path / "latest.json"
        return path / read_json(pointer)["checkpoint"]
    return path


def load_checkpoint(path: str | Path, preference: str = "best") -> dict:
    return torch.load(checkpoint_path(path, preference), map_location="cpu", weights_only=True)


def token_loss(model, batch: dict, task: str) -> tuple[torch.Tensor, int]:
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    if task == "bonus":
        inputs["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(batch["labels"])
    output = model(**inputs, use_cache=False)
    labels = batch["labels"]
    logits = output.logits
    if task == "basic":
        logits, labels = logits[:, :-1], labels[:, 1:]
    count = int(labels.ne(-100).sum().item())
    if count == 0:
        raise ValueError("Batch contains no supervised tokens")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, count


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    t, d = config["training"], config["data"]
    training_mode = config.get("training_mode", "standard")
    if args.all_training:
        training_mode = "full_corpus"
        config["training_mode"] = training_mode
    if training_mode not in {"standard", "full_corpus"}:
        raise ValueError("training_mode must be standard or full_corpus")
    if args.steps is not None:
        t["steps"] = args.steps
    if args.batch_size is not None:
        t["batch_size"] = args.batch_size
    if args.all_training:
        d["validation_fraction"] = 0.0
    for key in [
        "steps",
        "batch_size",
        "gradient_accumulation",
        "log_every",
        "evaluate_every",
        "checkpoint_every",
        "validation_batches",
        "keep_checkpoints",
    ]:
        if t[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    if training_mode == "full_corpus" and args.init_from:
        raise ValueError(
            "Full-corpus final training must start from random initialization; "
            "--init-from is diagnostic-only"
        )
    if training_mode == "full_corpus" and d.get("validation_fraction") != 0:
        raise ValueError("full_corpus mode requires data.validation_fraction=0")
    if args.stop_after is not None and args.stop_after <= 0:
        raise ValueError("--stop-after must be positive")
    directory = Path(args.output)
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Run directory is not empty; use --resume or another path: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    precision = t["precision"]
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16 or bf16")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support bf16; use fp16 or fp32")
        torch.backends.cuda.matmul.allow_tf32 = True
    elif precision != "fp32":
        raise ValueError("CPU smoke tests require precision=fp32")
    seed_everything(config["seed"])
    corpus = MusicCorpus(args.data)
    expected_songs = FINAL_CORPUS_SIZE if training_mode == "full_corpus" else None
    if training_mode == "full_corpus" and d.get("expected_songs", expected_songs) != expected_songs:
        raise ValueError(f"full_corpus data.expected_songs must be {expected_songs}")
    if expected_songs is not None and len(corpus.songs) != expected_songs:
        raise ValueError(
            f"{training_mode} mode requires exactly {expected_songs} corpus songs; "
            f"found {len(corpus.songs)}"
        )
    if d["max_tokens"] == "auto":
        d["max_tokens"] = corpus.required_context(config["task"], d["bars"])
    required = corpus.required_context(config["task"], d["bars"])
    if d["max_tokens"] < required:
        raise ValueError(
            f"Provided complete windows require {required} tokens; configured max_tokens="
            f"{d['max_tokens']}. Increase context instead of truncating music."
        )
    if training_mode == "full_corpus":
        training_names, validation_names = sorted(corpus.songs), []
    else:
        training_names, validation_names = corpus.split(config["seed"], d["validation_fraction"])
    model = build_model(config, corpus.tokenizer).to(device)
    if t.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.ndim >= 2]},
            {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
        fused=device.startswith("cuda"),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    sampler_rng = random.Random(config["seed"])
    start_step, best_loss, tokens_seen = 0, float("inf"), 0
    song_sample_counts = dict.fromkeys(sorted(corpus.songs), 0)
    initialization = "random" if not args.init_from else "checkpoint"
    initialized_from = str(args.init_from) if args.init_from else None
    source_commit = os.environ.get("HW2_SOURCE_COMMIT")
    if args.resume or args.init_from:
        saved = load_checkpoint(args.resume or args.init_from, "latest" if args.resume else "best")
        if saved["config"]["task"] != config["task"] or saved["config"]["model"] != config["model"]:
            raise ValueError("Checkpoint architecture/task differs from requested configuration")
        if saved["tokenizer_tokens"] != corpus.tokenizer.tokens:
            raise ValueError("Checkpoint vocabulary differs from prepared data")
        model.load_state_dict(saved["model"])
        if args.resume:
            if saved.get("inference_only"):
                raise ValueError("Inference exports cannot resume optimizer state; use --init-from")
            if saved["data_fingerprint"] != corpus.manifest["fingerprint"]:
                raise ValueError("Cannot resume with a different corpus")
            provenance = saved.get("training_provenance")
            if training_mode == "full_corpus":
                if not provenance or provenance.get("mode") != training_mode:
                    raise ValueError("Cannot resume full-corpus mode without matching provenance")
                if provenance.get("checkpoint") != checkpoint_path(args.resume, "latest").name:
                    raise ValueError(
                        "Resume checkpoint provenance filename differs from checkpoint"
                    )
                if provenance.get("expected_songs") != expected_songs:
                    raise ValueError("Resume corpus-size policy differs from checkpoint")
                if training_mode == "full_corpus" and provenance.get("initialization") != "random":
                    raise ValueError(
                        "Final full-corpus run must descend from random initialization"
                    )
                initialization = provenance["initialization"]
                initialized_from = provenance.get("initialized_from")
                source_commit = provenance.get("source_commit", source_commit)
            if provenance:
                counts = provenance.get("song_sample_counts", {})
                if set(counts) != set(song_sample_counts):
                    raise ValueError("Resume checkpoint song exposure differs from corpus")
                try:
                    song_sample_counts = {name: int(counts[name]) for name in song_sample_counts}
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Resume checkpoint has invalid song exposure counts"
                    ) from error
                if any(value < 0 for value in song_sample_counts.values()):
                    raise ValueError("Resume checkpoint has negative song exposure counts")
                sampled = all(song_sample_counts.values())
                if provenance.get("all_songs_sampled") != sampled:
                    raise ValueError("Resume checkpoint has inconsistent song coverage")
                if provenance.get("source_names") != sorted(corpus.songs):
                    raise ValueError("Resume checkpoint source-song list differs from corpus")
                if provenance.get("data_fingerprint") != corpus.manifest["fingerprint"]:
                    raise ValueError("Resume checkpoint provenance fingerprint differs from corpus")
            # A resume must preserve the sampling distribution and optimizer schedule.
            if saved["config"] != config:
                raise ValueError(
                    "Resume requires the original configuration; use --init-from for a new stage"
                )
            expected_samples = (
                saved["step"]
                * config["training"]["batch_size"]
                * config["training"]["gradient_accumulation"]
            )
            if provenance and sum(song_sample_counts.values()) != expected_samples:
                raise ValueError(
                    "Resume checkpoint song exposure total differs from completed steps"
                )
            if provenance:
                sampled = all(song_sample_counts.values())
                expected_complete = (
                    training_mode == "full_corpus"
                    and saved["step"] == config["training"]["steps"]
                    and sampled
                )
                if provenance.get("complete") != expected_complete:
                    raise ValueError("Resume checkpoint has invalid full-corpus completion status")
                if provenance.get("step") != saved["step"]:
                    raise ValueError("Resume checkpoint provenance step differs from checkpoint")
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            sampler_rng.setstate(saved["sampler_rng"])
            random.setstate(saved["python_rng"])
            torch.set_rng_state(saved["torch_rng"])
            if device.startswith("cuda") and saved["cuda_rng"]:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            start_step, best_loss, tokens_seen = (
                saved["step"],
                saved["best_loss"],
                saved["tokens_seen"],
            )
    write_json(directory / "config.json", config)
    corpus.tokenizer.save(directory / "tokenizer.json")
    write_json(
        directory / "split.json", {"training": training_names, "validation": validation_names}
    )
    write_json(
        directory / "environment.json",
        {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "host": socket.gethostname(),
            "device": device,
            "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
            "parameters": sum(p.numel() for p in model.parameters()),
            "data_fingerprint": corpus.manifest["fingerprint"],
            "source_commit": source_commit,
        },
    )
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; saving after the current optimizer step.", flush=True)

    old_handlers = {s: signal.signal(s, request_stop) for s in [signal.SIGINT, signal.SIGTERM]}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    def autocast():
        return (
            torch.autocast("cuda", dtype=autocast_dtype)
            if precision != "fp32"
            else contextlib.nullcontext()
        )

    def sample_batch(rng: random.Random, names: list[str], augment: bool) -> tuple[dict, list[str]]:
        sampled = [
            corpus.sample_with_song(
                names,
                rng,
                config["task"],
                d["bars"],
                d["max_tokens"],
                d.get("transpose", 0) if augment else 0,
            )
            for _ in range(t["batch_size"])
        ]
        examples, song_names = zip(*sampled, strict=True)
        return collate(list(examples), config["task"], corpus.tokenizer.pad_id, "cpu"), list(
            song_names
        )

    def evaluate() -> float:
        model.eval()
        loss_sum, count = 0.0, 0
        rng = random.Random(config["seed"] + 1)
        with torch.inference_mode():
            for _ in range(t["validation_batches"]):
                batch = {
                    k: v.to(device)
                    for k, v in sample_batch(rng, validation_names, False)[0].items()
                }
                with autocast():
                    loss, n = token_loss(model, batch, config["task"])
                loss_sum += float(loss)
                count += n
        model.train()
        return loss_sum / count

    model.train()
    final_step = min(t["steps"], start_step + args.stop_after) if args.stop_after else t["steps"]
    started = time.monotonic()
    logged_loss, logged_tokens, logged_steps = 0.0, 0, 0
    print(
        f"Training {config['task']} on {device}: {len(training_names)} train / "
        f"{len(validation_names)} validation songs",
        flush=True,
    )
    try:
        with (directory / "metrics.jsonl").open("a", buffering=1) as log:
            for step in range(start_step + 1, final_step + 1):
                optimizer.zero_grad(set_to_none=True)
                sampled_batches = [
                    sample_batch(sampler_rng, training_names, True)
                    for _ in range(t["gradient_accumulation"])
                ]
                batches = [batch for batch, _ in sampled_batches]
                for _, names in sampled_batches:
                    for name in names:
                        song_sample_counts[name] += 1
                total_targets = sum(
                    int(
                        (b["labels"][:, 1:] if config["task"] == "basic" else b["labels"])
                        .ne(-100)
                        .sum()
                    )
                    for b in batches
                )
                step_loss = 0.0
                for batch in batches:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with autocast():
                        loss_sum, _ = token_loss(model, batch, config["task"])
                        loss = loss_sum / total_targets
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss at step {step}")
                    scaler.scale(loss).backward()
                    step_loss += float(loss_sum.detach())
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), t["gradient_clip"], error_if_nonfinite=True
                )
                warmup = t["warmup_steps"]
                progress = min(1.0, max(0.0, (step - warmup) / max(1, t["steps"] - warmup)))
                ratio = (
                    step / warmup
                    if warmup and step <= warmup
                    else t["min_lr_ratio"]
                    + (1 - t["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
                )
                for group in optimizer.param_groups:
                    group["lr"] = t["learning_rate"] * ratio
                scaler.step(optimizer)
                scaler.update()
                tokens_seen += total_targets
                logged_loss += step_loss
                logged_tokens += total_targets
                logged_steps += 1
                improved = False
                if validation_names and (step % t["evaluate_every"] == 0 or step == final_step):
                    val_loss = evaluate()
                    improved = val_loss < best_loss
                    best_loss = min(best_loss, val_loss)
                    record = {"step": step, "validation_loss": val_loss}
                    log.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                if step % t["log_every"] == 0 or step == final_step or stop_requested:
                    elapsed = time.monotonic() - started
                    record = {
                        "step": step,
                        "train_loss": logged_loss / logged_tokens,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "gradient_norm": float(gradient_norm),
                        "tokens_seen": tokens_seen,
                        "tokens_per_second": logged_tokens / elapsed,
                        "seconds_per_step": elapsed / logged_steps,
                    }
                    if device.startswith("cuda"):
                        record["peak_memory_mib"] = round(
                            torch.cuda.max_memory_allocated() / 2**20, 1
                        )
                    log.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                    started, logged_loss, logged_tokens, logged_steps = time.monotonic(), 0.0, 0, 0
                if (
                    improved
                    or step % t["checkpoint_every"] == 0
                    or step == final_step
                    or stop_requested
                ):
                    path = directory / f"step-{step:07d}.pt"
                    coverage_complete = (
                        training_mode == "full_corpus"
                        and step == t["steps"]
                        and all(song_sample_counts.values())
                    )
                    provenance = {
                        "mode": training_mode,
                        "initialization": initialization,
                        "initialized_from": initialized_from,
                        "source_commit": source_commit,
                        "checkpoint": path.name,
                        "step": step,
                        "expected_songs": expected_songs,
                        "corpus_songs": len(corpus.songs),
                        "source_names": sorted(corpus.songs),
                        "data_fingerprint": corpus.manifest["fingerprint"],
                        "training_songs": len(training_names),
                        "validation_songs": len(validation_names),
                        "song_sample_counts": song_sample_counts.copy(),
                        "all_songs_sampled": all(song_sample_counts.values()),
                        "configured_steps_complete": step == t["steps"],
                        "complete": coverage_complete,
                    }
                    save_checkpoint(
                        path,
                        {
                            "format_version": 2,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "config": config,
                            "tokenizer_tokens": corpus.tokenizer.tokens,
                            "tokenizer_config": corpus.tokenizer.config.__dict__,
                            "source_programs": corpus.manifest["source_programs"],
                            "data_fingerprint": corpus.manifest["fingerprint"],
                            "step": step,
                            "best_loss": best_loss,
                            "tokens_seen": tokens_seen,
                            "training_provenance": provenance,
                            "sampler_rng": sampler_rng.getstate(),
                            "python_rng": random.getstate(),
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all()
                            if device.startswith("cuda")
                            else [],
                        },
                    )
                    write_json(directory / "latest.json", {"checkpoint": path.name, "step": step})
                    if improved:
                        write_json(
                            directory / "best.json",
                            {"checkpoint": path.name, "step": step, "validation_loss": best_loss},
                        )
                    write_json(directory / "training_provenance.json", provenance)
                    protected = {path.name}
                    if (directory / "best.json").exists():
                        protected.add(read_json(directory / "best.json")["checkpoint"])
                    checkpoints = sorted(directory.glob("step-*.pt"))
                    protected.update(p.name for p in checkpoints[-t["keep_checkpoints"] :])
                    for old in checkpoints:
                        if old.name not in protected:
                            old.unlink()
                if stop_requested:
                    break
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("train", help=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", help="Run directory or checkpoint; restores optimizer and RNG")
    parser.add_argument("--init-from", help="Load weights only for a new training stage")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--stop-after", type=int, help="Stop after N additional steps; preserve schedule"
    )
    parser.add_argument(
        "--all-training", action="store_true", help="Final stage using all 896 songs"
    )
    parser.set_defaults(func=train)
