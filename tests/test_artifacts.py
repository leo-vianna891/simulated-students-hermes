from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulated_students.artifacts import RunLayout, sha256_file


def test_run_layout_and_receipt(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 221\n")

    layout = RunLayout.create("pilot-001", root=tmp_path / "artifacts")
    receipt = layout.write_receipt(config)

    assert layout.checkpoints.is_dir()
    assert layout.logs.is_dir()
    assert layout.outputs.is_dir()
    assert receipt["config_sha256"] == sha256_file(config)
    assert json.loads(layout.receipt.read_text())["run_id"] == "pilot-001"


@pytest.mark.parametrize("run_id", ["../escape", "UPPER", "x", "contains space"])
def test_run_id_rejects_unsafe_names(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError):
        RunLayout.create(run_id, root=tmp_path)
