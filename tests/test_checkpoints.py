from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulated_students.checkpoints import seal_checkpoint, validate_resume

STATE_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
)


def checkpoint(root: Path) -> Path:
    path = root / "checkpoint-2"
    path.mkdir(parents=True)
    for name in STATE_FILES:
        (path / name).write_bytes(b"controlled fixture")
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": 2, "epoch": 1.0, "best_model_checkpoint": None})
    )
    return path


def test_checkpoint_is_accepted_only_after_complete_seal(tmp_path: Path) -> None:
    path = checkpoint(tmp_path)
    identity = {"seed": 42}
    with pytest.raises(ValueError, match="complete"):
        validate_resume(path, tmp_path, identity)
    seal_checkpoint(path, identity)
    assert validate_resume(path, tmp_path, identity)["global_step"] == 2


def test_resume_rejects_missing_state_changed_inputs_and_corruption(tmp_path: Path) -> None:
    path = checkpoint(tmp_path)
    identity = {"seed": 42}
    seal_checkpoint(path, identity)
    with pytest.raises(ValueError, match="identity"):
        validate_resume(path, tmp_path, {"seed": 43})
    (path / "optimizer.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        validate_resume(path, tmp_path, identity)
    (path / "optimizer.pt").write_bytes(b"controlled fixture")
    (path / "rng_state.pth").unlink()
    with pytest.raises(ValueError, match="Missing"):
        validate_resume(path, tmp_path, identity)


def test_resume_requires_restored_run_tree_and_latest_complete_checkpoint(tmp_path: Path) -> None:
    path = checkpoint(tmp_path)
    seal_checkpoint(path, {})
    with pytest.raises(ValueError, match="output"):
        validate_resume(path, tmp_path / "other", {})
    later = tmp_path / "checkpoint-4"
    later.mkdir()
    (later / "resume.json").write_text("{}")
    with pytest.raises(ValueError, match="latest"):
        validate_resume(path, tmp_path, {})


def test_resume_requires_preserved_best_adapter(tmp_path: Path) -> None:
    path = checkpoint(tmp_path)
    (path / "trainer_state.json").write_text(
        json.dumps(
            {"global_step": 2, "epoch": 1.0, "best_model_checkpoint": "old/root/checkpoint-1"}
        )
    )
    seal_checkpoint(path, {})
    with pytest.raises(ValueError, match="best"):
        validate_resume(path, tmp_path, {})
