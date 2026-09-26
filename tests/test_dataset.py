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


def _dialogue(intervention: int, question: int, student_text: str) -> list[dict[str, str]]:
    common = {
        "InterventionId": str(intervention),
        "TutorId": "10",
        "QuestionId_DQ": str(question),
        "TalkMovePrediction": "<None>",
    }
    return [
        {
            **common,
            "MessageSequence": "1",
            "IsTutor": "1",
            "MessageString": "Try",
        },
        {
            **common,
            "MessageSequence": "2",
            "IsTutor": "1",
            "MessageString": "again.",
        },
        {
            **common,
            "MessageSequence": "3",
            "IsTutor": "0",
            "MessageString": student_text,
        },
    ]


def _dataset(root: Path) -> None:
    _write_csv(
        root / "anchored-dialogues" / "train.csv",
        _dialogue(100, 1, "Four") + _dialogue(101, 2, "None"),
    )
    _write_csv(root / "anchored-dialogues" / "test.csv", _dialogue(200, 1, "Five"))
    _write_csv(
        root / "dq-question-metadata.csv",
        [
            {
                "QuestionId_DQ": str(question),
                "InterventionId": str(intervention),
                "MetaDataId": str(intervention),
                "Text": "What is 2 + 2?",
                "Sequence": "1",
                "MetaDataTagId": "1",
                "Label": "Question",
            }
            for question, intervention in ((1, 100), (2, 101), (1, 200))
        ],
    )
    _write_csv(
        root / "dialogue-subjects.csv",
        [
            {
                "InterventionId": str(intervention),
                "SubjectId": "1",
                "ParentSubjectId": "0",
                "SubjectName": "Math",
                "SubjectLevel": "1",
                "SubjectType": "Subject",
            }
            for intervention in (100, 101, 200)
        ],
    )


def test_load_eedi_splits_reproduces_dialogues_and_seeded_split(tmp_path: Path) -> None:
    _dataset(tmp_path)

    splits = load_eedi_splits(tmp_path)

    assert [dialogue["key"] for dialogue in splits.train] == [(2, 101)]
    assert [dialogue["key"] for dialogue in splits.validation] == [(1, 100)]
    assert [dialogue["key"] for dialogue in splits.test] == [(1, 200)]
    assert splits.train[0]["turns"] == [
        {"role": "tutor", "content": "Try. again."},
        {"role": "student", "content": "nan"},
    ]
    assert splits.validation[0]["question"] == "Question: What is 2 + 2?"
    assert splits.validation[0]["subjects"] == [("Math", 1)]


def test_load_eedi_splits_rejects_invalid_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        load_eedi_splits(tmp_path, train_fraction=1.0)
