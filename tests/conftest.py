from argparse import Namespace

import pretty_midi
import pytest
import torch

from deepmir_hw2.prepare import prepare


@pytest.fixture(scope="session", autouse=True)
def limit_cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "POP909/training"
    for song in range(4):
        folder = source / f"{song:03d}"
        folder.mkdir(parents=True)
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        for name, pitch in [("MELODY", 60 + song), ("BRIDGE", 55), ("PIANO", 48 + song)]:
            inst = pretty_midi.Instrument(0, name=name)
            inst.notes = [
                pretty_midi.Note(80, pitch, bar * 2 + 0.25, bar * 2 + 0.75) for bar in range(4)
            ]
            midi.instruments.append(inst)
        midi.write(str(folder / f"{folder.name}.mid"))
        pulses = 6 if song == 3 else 4
        (folder / "beat_midi.txt").write_text(
            "\n".join(f"{i / 2} 0 {1 if i % pulses == 0 else 0}" for i in range(4 * pulses + 1))
        )
        (folder / "chord_midi.txt").write_text("0 8 C:maj\n")
    output = tmp_path / "prepared"
    prepare(Namespace(dataset=str(source), output=str(output)))
    return output


@pytest.fixture
def tiny_config():
    return {
        "task": "basic",
        "seed": 7,
        "model": {"width": 16, "layers": 1, "heads": 2, "dropout": 0.1},
        "data": {"bars": 2, "max_tokens": 256, "validation_fraction": 0.25, "transpose": 2},
        "training": {
            "steps": 4,
            "batch_size": 2,
            "gradient_accumulation": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_steps": 1,
            "min_lr_ratio": 0.1,
            "gradient_clip": 1.0,
            "precision": "fp32",
            "gradient_checkpointing": False,
            "log_every": 1,
            "evaluate_every": 2,
            "checkpoint_every": 2,
            "validation_batches": 1,
            "keep_checkpoints": 2,
        },
    }
