from __future__ import annotations

import csv
from pathlib import Path

import pytest

from simulated_students.dataset import load_eedi_splits


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _row(key: tuple[int, int], annotation: object) -> dict[str, str]:
    return {
        "key": repr(key),
        "question": "Question: What is 2 + 2?",
        "subjects": repr([("Math", 1)]),
        "turns": repr(
            [
                {"role": "tutor", "content": "Try. again."},
                {"role": "student", "content": "Maybe five."},
            ]
        ),
        "question_annotation": repr(annotation),
    }


def _annotation() -> dict[str, object]:
    return {
        "solvable": True,
        "correct_option": 2,
        "solution": "Add two and two to obtain four.",
        "option_1_explanation": "Too small.",
        "option_2_explanation": "Four is correct.",
        "option_3_explanation": "Too large.",
        "option_4_explanation": "Also too large.",
    }


def test_load_eedi_splits_preserves_published_annotations_and_splits(tmp_path: Path) -> None:
    annotation = _annotation()
    _write_csv(
        tmp_path / "train_gpt-4.1.csv",
        [
            _row((2, 101), annotation),
            _row((1, 100), annotation),
            _row((9, 109), {"solvable": False}),
            _row((10, 110), None),
        ],
    )
    _write_csv(tmp_path / "val_gpt-4.1.csv", [_row((3, 102), annotation)])
    _write_csv(tmp_path / "test_gpt-4.1.csv", [_row((4, 103), annotation)])

    splits = load_eedi_splits(tmp_path)

    assert [d["key"] for d in splits.train] == [(2, 101), (1, 100)]
    assert [d["key"] for d in splits.validation] == [(3, 102)]
    assert [d["key"] for d in splits.test] == [(4, 103)]
    assert splits.train[0]["question_annotation"] == annotation
    assert splits.train[0]["turns"][-1]["content"] == "Maybe five."
    assert splits.train[0]["subjects"] == [("Math", 1)]


def test_missing_annotation_files_never_fall_back_to_raw_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"train_gpt-4\.1\.csv"):
        load_eedi_splits(tmp_path)


def test_load_eedi_splits_rejects_intervention_leakage(tmp_path: Path) -> None:
    for split in ["train", "val", "test"]:
        _write_csv(tmp_path / f"{split}_gpt-4.1.csv", [_row((1, 100), _annotation())])
    with pytest.raises(ValueError, match="overlap"):
        load_eedi_splits(tmp_path)
