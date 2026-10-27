import hashlib
import json
from argparse import Namespace

import torch

from deepmir_hw2.export import export
from deepmir_hw2.train import load_checkpoint, train


def training_arguments(config, data, output):
    return Namespace(
        config=str(config),
        data=str(data),
        output=str(output),
        device="cpu",
        resume=None,
        init_from=None,
        steps=None,
        batch_size=None,
        all_training=False,
        stop_after=None,
    )


def test_export_preserves_training_provenance_and_exact_weights(
    prepared, tiny_config, tmp_path, monkeypatch
):
    import deepmir_hw2.train as training

    monkeypatch.setattr(training, "FINAL_CORPUS_SIZE", 4)
    monkeypatch.setenv("HW2_SOURCE_COMMIT", "test-source-revision")
    tiny_config["training_mode"] = "full_corpus"
    tiny_config["data"]["expected_songs"] = 4
    tiny_config["data"]["validation_fraction"] = 0.0
    tiny_config["training"].update(steps=2, batch_size=1, gradient_accumulation=1)
    config = tmp_path / "full.json"
    config.write_text(json.dumps(tiny_config))
    run = tmp_path / "run"
    train(training_arguments(config, prepared, run))

    source_checkpoint = run / json.loads((run / "latest.json").read_text())["checkpoint"]
    original = load_checkpoint(source_checkpoint)
    output = tmp_path / "inference.pt"
    export(Namespace(checkpoint=str(run), output=str(output)))
    exported = load_checkpoint(output)

    assert exported["training_provenance"] == original["training_provenance"]
    assert exported["training_provenance"]["source_commit"] == "test-source-revision"
    assert exported["source_checkpoint_name"] == source_checkpoint.name
    assert (
        exported["source_checkpoint_sha256"]
        == hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    )
    assert set(exported["model"]) == set(original["model"])
    for key, value in original["model"].items():
        torch.testing.assert_close(exported["model"][key], value, rtol=0, atol=0)
    assert not {
        "optimizer",
        "scaler",
        "sampler_rng",
        "python_rng",
        "torch_rng",
        "cuda_rng",
    } & set(exported)
    assert exported["inference_only"] is True


def test_export_of_legacy_checkpoint_does_not_invent_final_provenance(
    prepared, tiny_config, tmp_path
):
    tiny_config["training"]["steps"] = 1
    tiny_config["training"]["batch_size"] = 1
    tiny_config["training"]["gradient_accumulation"] = 1
    config = tmp_path / "dev.json"
    config.write_text(json.dumps(tiny_config))
    run = tmp_path / "dev-run"
    train(training_arguments(config, prepared, run))
    source_pointer = run / "latest.json"
    source_checkpoint = run / json.loads(source_pointer.read_text())["checkpoint"]

    legacy = load_checkpoint(source_checkpoint)
    legacy.pop("training_provenance")
    torch.save(legacy, source_checkpoint)
    output = tmp_path / "legacy-export.pt"
    export(Namespace(checkpoint=str(source_checkpoint), output=str(output)))
    exported = load_checkpoint(output)

    assert "training_provenance" not in exported
    assert exported["source_checkpoint_name"] == source_checkpoint.name
    assert (
        exported["source_checkpoint_sha256"]
        == hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    )
