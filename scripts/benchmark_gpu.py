"""Check complete-window forward/backward/optimizer capacity without saving weights."""

import argparse
import json
import time

import torch

from deepmir_hw2.data import MusicCorpus, collate
from deepmir_hw2.models import build_model
from deepmir_hw2.train import token_loss
from deepmir_hw2.utils import choose_device, read_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if not device.startswith("cuda"):
        raise ValueError("This benchmark requires an explicitly authorized CUDA GPU")
    torch.cuda.set_device(torch.device(device))
    config = read_json(args.config)
    if config["training"]["precision"] != "bf16" or not torch.cuda.is_bf16_supported():
        raise ValueError("This benchmark currently checks the BF16 training configuration")
    seed_everything(config["seed"])
    corpus = MusicCorpus(args.data)
    task, bars = config["task"], config["data"]["bars"]
    required = corpus.required_context(task, bars)
    if config["data"]["max_tokens"] == "auto":
        config["data"]["max_tokens"] = required
    if config["data"]["max_tokens"] < required:
        raise ValueError("Configured context cannot hold the complete source windows")
    tracks = ["basic"] if task == "basic" else ["melody", "piano"]
    windows = []
    for name, song in corpus.songs.items():
        count = len(song[f"{tracks[0]}_offsets"]) - 1
        size = min(bars, count)
        for start in range(count - size + 1):
            lengths = [
                int(song[f"{track}_offsets"][start + size] - song[f"{track}_offsets"][start])
                for track in tracks
            ]
            cost = sum(length**2 for length in lengths)
            if len(lengths) == 2:
                cost += lengths[0] * lengths[1]
            windows.append((cost, name, start, size, tuple(lengths)))
    # Include the largest individual sequences as well as the largest combined
    # attention workload; one window need not maximize all three quantities.
    cases = {max(windows)}
    cases.update(max(windows, key=lambda window: window[-1][i]) for i in range(len(tracks)))
    model = build_model(config, corpus.tokenizer).to(device)
    if config["training"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        fused=True,
    )
    for _, name, start, size, lengths in sorted(cases):
        sequences = []
        for track in tracks:
            song = corpus.songs[name]
            offsets = song[f"{track}_offsets"]
            sequence = song[track][offsets[start] : offsets[start + size]].astype(int).tolist()
            sequences.append([corpus.tokenizer.bos_id, *sequence, corpus.tokenizer.eos_id])
        example = {"input_ids": sequences[0]}
        if task == "bonus":
            example["labels"] = sequences[1][1:]
        batch = collate(
            [example] * config["training"]["batch_size"], task, corpus.tokenizer.pad_id, device
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, targets = token_loss(model, batch, task)
            loss = loss / targets
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite benchmark loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "status": "passed",
                    "task": task,
                    "song": name,
                    "start_bar": start,
                    "bars": size,
                    "lengths": lengths,
                    "batch_size": config["training"]["batch_size"],
                    "context": required,
                    "loss": float(loss.detach()),
                    "seconds": time.monotonic() - started,
                    "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "gpu": torch.cuda.get_device_name(),
                    "data_fingerprint": corpus.manifest["fingerprint"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
