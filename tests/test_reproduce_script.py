import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "reproduce_all.sh"


def run_stage(tmp_path, stage, *, fake_git_failure=False, **overrides):
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/bin/sh\nif [ "$1" = "-c" ]; then exec "$HW2_REAL_PYTHON" "$@"; fi\n'
        'if [ "$1" = "-m" ] && [ "$2" = "deepmir_hw2" ] && [ "$3" = "train" ]; then '
        'printf \'%s\\n\' "$HW2_SOURCE_COMMIT" >> "$HW2_COMMIT_LOG"; fi\n'
        'printf \'%s\\n\' "$*" >> "$HW2_CALL_LOG"\n'
    )
    fake_python.chmod(0o755)
    call_log = tmp_path / "calls.txt"
    environment = os.environ.copy()
    environment.pop("HW2_SOURCE_COMMIT", None)
    for key in [
        "HW2_NUMERIC_STRATEGY",
        "HW2_BASIC_A_TOP_K",
        "HW2_BASIC_A_TEMPERATURE",
        "HW2_BASIC_B_TOP_K",
        "HW2_BASIC_B_TEMPERATURE",
        "HW2_BASIC_C_TOP_K",
        "HW2_BASIC_C_TEMPERATURE",
        "HW2_BASIC_MODEL",
        "HW2_MODEL_DIR",
    ]:
        environment.pop(key, None)
    environment.update(
        HW2_PYTHON=str(fake_python),
        HW2_REAL_PYTHON=sys.executable,
        HW2_CALL_LOG=str(call_log),
        HW2_COMMIT_LOG=str(tmp_path / "commits.txt"),
        **overrides,
    )
    if fake_git_failure:
        git_bin = tmp_path / "git-bin"
        git_bin.mkdir()
        fake_git = git_bin / "git"
        fake_git.write_text("#!/bin/sh\nexit 1\n")
        fake_git.chmod(0o755)
        environment["PATH"] = f"{git_bin}{os.pathsep}{environment['PATH']}"
    subprocess.run(
        ["bash", str(SCRIPT), stage], check=True, env=environment, capture_output=True, text=True
    )
    return call_log.read_text().splitlines()


def test_generate_basic_uses_sample_strategy_and_same_export_for_a_and_b(tmp_path):
    calls = run_stage(
        tmp_path, "generate-basic", HW2_BASIC_MODEL="models/basic-test.pt", fake_git_failure=True
    )

    assert len(calls) == 6
    assert all("--checkpoint models/basic-test.pt" in call for call in calls)
    assert all("--numeric-strategy sample" in call for call in calls)
    assert all("--numeric-beam-width" not in call for call in calls)
    assert "--top-k 20 --temperature 0.9" in calls[0]
    assert "--top-k 20 --temperature 0.9" in calls[1]
    assert "--top-k 50 --temperature 1.1" in calls[-2]
    assert "--output outputs/basic-a/A" in calls[0]
    assert "--output outputs/basic-b/A" in calls[1]


def test_final_training_preserves_explicit_source_commit_without_git(tmp_path):
    run_stage(
        tmp_path,
        "final-basic",
        HW2_SOURCE_COMMIT="reviewed-commit-id",
        CUDA_VISIBLE_DEVICES="0",
        fake_git_failure=True,
    )

    assert (tmp_path / "commits.txt").read_text().splitlines() == ["reviewed-commit-id"]


def test_final_training_fails_without_commit_or_git_checkout(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_stage(tmp_path, "final-basic", CUDA_VISIBLE_DEVICES="0", fake_git_failure=True)

    assert "Set HW2_SOURCE_COMMIT" in error.value.stderr
    assert not (tmp_path / "calls.txt").exists()


@pytest.mark.parametrize(
    ("stage", "checkpoint_env", "output_env", "run_name", "model_path"),
    [
        (
            "export-basic",
            "HW2_BASIC_CHECKPOINT",
            "HW2_BASIC_MODEL",
            "basic-final",
            "models/basic.pt",
        ),
        (
            "export-bonus",
            "HW2_BONUS_CHECKPOINT",
            "HW2_BONUS_MODEL",
            "bonus-final",
            "models/bonus.pt",
        ),
    ],
)
def test_export_validates_and_exports_exact_latest_checkpoint_even_with_best_pointer(
    tmp_path, stage, checkpoint_env, output_env, run_name, model_path
):
    run_dir = tmp_path / run_name
    run_dir.mkdir()
    (run_dir / "latest.json").write_text('{"checkpoint":"step-latest.pt"}')
    (run_dir / "best.json").write_text('{"checkpoint":"step-best.pt"}')
    calls = run_stage(
        tmp_path,
        stage,
        **{checkpoint_env: str(run_dir), output_env: model_path},
    )
    latest_checkpoint = str(run_dir / "step-latest.pt")

    assert len(calls) == 2
    assert calls[0] == (
        "-m deepmir_hw2.experiment_stage require-final --checkpoint " + latest_checkpoint
    )
    assert calls[1] == (
        "-m deepmir_hw2 export --checkpoint " + latest_checkpoint + " --output " + model_path
    )


def test_generate_bonus_accepts_explicit_beam_strategy(tmp_path):
    calls = run_stage(tmp_path, "generate-bonus", HW2_NUMERIC_STRATEGY="beam")

    assert len(calls) == 1
    assert "--checkpoint models/bonus.pt" in calls[0]
    assert "--numeric-strategy beam" in calls[0]
    assert "--numeric-beam-width" not in calls[0]


@pytest.mark.parametrize("strategy", ["greedy", "unknown"])
def test_generate_rejects_unsupported_numeric_strategy(tmp_path, strategy):
    with pytest.raises(subprocess.CalledProcessError):
        run_stage(tmp_path, "generate-basic", HW2_NUMERIC_STRATEGY=strategy)
