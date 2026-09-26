from __future__ import annotations

import csv
from pathlib import Path

import pytest

from simulated_students.dataset import audit_dataset, load_eedi_splits

DIALOGUE_FIELDS = [
    "InterventionId",
    "TutorId",
    "QuestionId_DQ",
    "MessageSequence",
    "IsTutor",
    "MessageString",
    "TalkMovePrediction",
]
SUBJECT_FIELDS = [
    "InterventionId",
    "SubjectId",
    "ParentSubjectId",
    "SubjectName",
    "SubjectLevel",
    "SubjectType",
]
METADATA_FIELDS = [
    "QuestionId_DQ",
    "InterventionId",
    "MetaDataId",
    "Text",
    "Sequence",
    "MetaDataTagId",
    "Label",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dialogue_rows(
    intervention: str, question: str, *, blank_student: bool = False
) -> list[dict[str, str]]:
    return [
        {
            "InterventionId": intervention,
            "TutorId": "10",
            "QuestionId_DQ": question,
            "MessageSequence": "1",
            "IsTutor": "1",
            "MessageString": "Try the first step.",
            "TalkMovePrediction": "<None>",
        },
        {
            "InterventionId": intervention,
            "TutorId": "10",
            "QuestionId_DQ": question,
            "MessageSequence": "2",
            "IsTutor": "0",
            "MessageString": "" if blank_student else "I get 4.",
            "TalkMovePrediction": "<None>",
        },
    ]


def build_dataset(root: Path) -> None:
    train_rows = dialogue_rows("100", "1", blank_student=True) + dialogue_rows("101", "2")
    test_rows = dialogue_rows("200", "1")
    write_csv(root / "anchored-dialogues" / "train.csv", DIALOGUE_FIELDS, train_rows)
    write_csv(root / "anchored-dialogues" / "test.csv", DIALOGUE_FIELDS, test_rows)

    subject_rows = [
        {
            "InterventionId": intervention,
            "SubjectId": "1",
            "ParentSubjectId": "0",
            "SubjectName": "Math",
            "SubjectLevel": "1",
            "SubjectType": "Subject",
        }
        for intervention in ("100", "101", "200")
    ]
    write_csv(root / "dialogue-subjects.csv", SUBJECT_FIELDS, subject_rows)

    metadata_rows = [
        {
            "QuestionId_DQ": question,
            "InterventionId": intervention,
            "MetaDataId": intervention,
            "Text": "Question text",
            "Sequence": "1",
            "MetaDataTagId": "1",
            "Label": "Question",
        }
        for question, intervention in (("1", "100"), ("2", "101"), ("1", "200"))
    ]
    write_csv(root / "dq-question-metadata.csv", METADATA_FIELDS, metadata_rows)


def test_audit_dataset_reports_structure_quality_and_upstream_split(tmp_path: Path) -> None:
    build_dataset(tmp_path)

    report = audit_dataset(tmp_path, revision="abc123")

    assert report["dataset_revision"] == "abc123"
    assert report["splits"]["train"]["rows"] == 4
    assert report["splits"]["train"]["unique_dialogues"] == 2
    assert report["splits"]["train"]["null_counts"]["MessageString"] == 1
    assert report["splits"]["train"]["pandas_na_message_strings"] == 1
    assert report["splits"]["test"]["rows"] == 2
    assert report["cross_split"]["dialogue_key_overlap"] == 0
    assert report["cross_split"]["question_overlap"] == 1
    assert report["referential_integrity"]["dialogue_keys_without_question_metadata"] == 0
    assert report["upstream_train_validation_split"]["train_dialogues"] == 1
    assert report["upstream_train_validation_split"]["validation_dialogues"] == 1
    assert report["paper_faithful_loader"]["source_dialogues_excluded"] == 0
    assert report["paper_faithful_loader"]["train"]["dialogues"] == 1


def test_audit_dataset_rejects_wrong_dialogue_schema(tmp_path: Path) -> None:
    build_dataset(tmp_path)
    write_csv(tmp_path / "anchored-dialogues" / "train.csv", ["wrong"], [{"wrong": "value"}])

    with pytest.raises(ValueError, match="Unexpected columns"):
        audit_dataset(tmp_path, revision="abc123")


def test_load_eedi_splits_reproduces_upstream_turns_and_split(tmp_path: Path) -> None:
    build_dataset(tmp_path)
    rows = [
        {
            "InterventionId": "100",
            "TutorId": "10",
            "QuestionId_DQ": "1",
            "MessageSequence": "1",
            "IsTutor": "1",
            "MessageString": "First line\n",
            "TalkMovePrediction": "<None>",
        },
        {
            "InterventionId": "100",
            "TutorId": "10",
            "QuestionId_DQ": "1",
            "MessageSequence": "2",
            "IsTutor": "1",
            "MessageString": "\n",
            "TalkMovePrediction": "<None>",
        },
        {
            "InterventionId": "100",
            "TutorId": "10",
            "QuestionId_DQ": "1",
            "MessageSequence": "3",
            "IsTutor": "0",
            "MessageString": "Student reply",
            "TalkMovePrediction": "<None>",
        },
        *dialogue_rows("101", "2"),
    ]
    rows[-1]["MessageString"] = "None"
    write_csv(tmp_path / "anchored-dialogues" / "train.csv", DIALOGUE_FIELDS, rows)

    splits = load_eedi_splits(tmp_path)

    assert [dialogue["key"] for dialogue in splits.train] == [(2, 101)]
    assert [dialogue["key"] for dialogue in splits.validation] == [(1, 100)]
    assert [dialogue["key"] for dialogue in splits.test] == [(1, 200)]
    assert splits.train[0]["turns"][-1]["content"] == "nan"
    reproduced = splits.validation[0]
    assert reproduced["question"] == "Question: Question text"
    assert reproduced["subjects"] == [("Math", 1)]
    assert reproduced["turns"] == [
        {"role": "tutor", "content": "First line. "},
        {"role": "student", "content": "Student reply"},
    ]


def test_load_eedi_splits_rejects_missing_metadata(tmp_path: Path) -> None:
    build_dataset(tmp_path)
    write_csv(tmp_path / "dq-question-metadata.csv", METADATA_FIELDS, [])

    with pytest.raises(ValueError, match="Missing question metadata"):
        load_eedi_splits(tmp_path)
