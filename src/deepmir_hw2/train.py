"""Train GPT-2 (Basic) or T5 (Bonus) from random initialization on all songs."""

import json
import math
import random
import time
from pathlib import Path

import torch

from .data import Corpus, batch
from .models import build_model


def lr_at(step, t):
    if step < t["warmup_steps"]:
        return t["learning_rate"] * step / t["warmup_steps"]
    progress = (step - t["warmup_steps"]) / max(1, t["steps"] - t["warmup_steps"])
    return t["learning_rate"] * (0.1 + 0.9 * (1 + math.cos(math.pi * min(1.0, progress))) / 2)


def save_weights(path, config, model, step):
    torch.save({"config": config, "model": model.state_dict(), "step": step}, path)


def train(args):
    config = json.loads(Path(args.config).read_text())
    t = config["training"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    corpus = Corpus(args.data, config.get("chords", True))
    model = build_model(config).to(device)
    if t.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=0.01)
    step = 0
    state_path = out / "state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"])
        step = state["step"]
        print(f"Resumed from step {step}", flush=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    params = sum(p.numel() for p in model.parameters())
    print(
        f"{config['task']}: {params / 1e6:.1f}M parameters, {len(corpus.names)} songs", flush=True
    )
    use_bf16 = device.startswith("cuda")
    log = (out / "log.jsonl").open("a", buffering=1)
    started, tokens = time.time(), 0
    model.train()
    while step < t["steps"]:
        step += 1
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, t)
        total = 0.0
        for _ in range(t["accumulation"]):
            inputs = batch(
                corpus,
                rng,
                config["task"],
                t["batch_size"],
                config["bars"],
                config["max_len"],
                config["transpose"],
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**inputs).loss / t["accumulation"]
            loss.backward()
            total += loss.item()
            tokens += int(inputs["labels"].ne(-100).sum())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % t["log_every"] == 0:
            elapsed = time.time() - started
            record = {
                "step": step,
                "loss": round(total, 4),
                "lr": lr_at(step, t),
                "grad_norm": round(float(grad_norm), 3),
                "tokens_per_s": round(tokens / elapsed),
                "s_per_step": round(elapsed / t["log_every"], 3),
            }
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            started, tokens = time.time(), 0
        if step % t["save_every"] == 0 or step == t["steps"]:
            save_weights(out / f"model-{step:06d}.pt", config, model, step)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": rng.getstate(),
                    "step": step,
                },
                state_path,
            )
    save_weights(out / "final.pt", config, model, step)


def add_parser(subparsers):
    parser = subparsers.add_parser("train", help=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/remi.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.set_defaults(func=train)
