from __future__ import annotations

from pathlib import Path

import pytest

from simulated_students.config import load_yaml


def test_load_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 221\n")
    assert load_yaml(path) == {"seed": 221}


def test_load_yaml_rejects_sequence(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- invalid\n")
    with pytest.raises(ValueError):
        load_yaml(path)
