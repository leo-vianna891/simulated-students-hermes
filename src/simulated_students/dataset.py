"""Deterministic structural audit for the pinned Eedi source dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Literal, TypedDict

DIALOGUE_COLUMNS = (
    "InterventionId",
    "TutorId",
    "QuestionId_DQ",
    "MessageSequence",
    "IsTutor",
    "MessageString",
    "TalkMovePrediction",
)
SUBJECT_COLUMNS = (
    "InterventionId",
    "SubjectId",
    "ParentSubjectId",
    "SubjectName",
    "SubjectLevel",
    "SubjectType",
)
METADATA_COLUMNS = (
    "QuestionId_DQ",
    "InterventionId",
    "MetaDataId",
    "Text",
    "Sequence",
    "MetaDataTagId",
    "Label",
)
PANDAS_DEFAULT_NA_VALUES = frozenset(
    {
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    }
)
DialogueKey = tuple[str, str]
NumericDialogueKey = tuple[int, int]


class Turn(TypedDict):
    role: Literal["student", "tutor"]
    content: str


class Dialogue(TypedDict):
    key: NumericDialogueKey
    question: str
    subjects: list[tuple[str, int]]
    turns: list[Turn]


@dataclass(frozen=True)
class DatasetSplits:
    """Paper-faithful train, validation, and test dialogue splits."""

    train: list[Dialogue]
    validation: list[Dialogue]
    test: list[Dialogue]


@dataclass(frozen=True)
class _SplitProfile:
    report: dict[str, Any]
    dialogue_keys: set[DialogueKey]
    questions: set[str]
    tutors: set[str]


def _read_csv(path: Path, expected_columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != expected_columns:
            raise ValueError(
                f"Unexpected columns in {path}: expected {expected_columns}, got {actual_columns}"
            )

        rows: list[dict[str, str]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row or any(value is None for value in raw_row.values()):
                raise ValueError(f"Malformed CSV row in {path} at line {line_number}")
            rows.append({column: raw_row[column] for column in expected_columns})
    return rows


def _distribution(values: list[int]) -> dict[str, int | float]:
    return {
        "min": min(values),
        "median": median(values),
        "max": max(values),
        "mean": fmean(values),
    }


def _profile_dialogue_split(path: Path) -> _SplitProfile:
    rows = _read_csv(path, DIALOGUE_COLUMNS)
    if not rows:
        raise ValueError(f"Dialogue split is empty: {path}")

    dialogues: dict[DialogueKey, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dialogues[(row["QuestionId_DQ"], row["InterventionId"])].append(row)

    dialogue_lengths: list[int] = []
    student_lengths: list[int] = []
    tutor_lengths: list[int] = []
    sequence_problems = 0
    out_of_order = 0
    invalid_speaker_values = 0

    for dialogue_rows in dialogues.values():
        sequences = [int(row["MessageSequence"]) for row in dialogue_rows]
        sorted_sequences = sorted(sequences)
        dialogue_lengths.append(len(dialogue_rows))
        student_lengths.append(sum(row["IsTutor"] == "0" for row in dialogue_rows))
        tutor_lengths.append(sum(row["IsTutor"] == "1" for row in dialogue_rows))
        invalid_speaker_values += sum(row["IsTutor"] not in {"0", "1"} for row in dialogue_rows)
        if len(set(sequences)) != len(sequences) or sorted_sequences != list(
            range(sorted_sequences[0], sorted_sequences[-1] + 1)
        ):
            sequence_problems += 1
        if sequences != sorted_sequences:
            out_of_order += 1

    duplicate_keys = Counter((row["InterventionId"], row["MessageSequence"]) for row in rows)
    report: dict[str, Any] = {
        "columns": list(DIALOGUE_COLUMNS),
        "rows": len(rows),
        "null_counts": {
            column: sum(not row[column].strip() for row in rows) for column in DIALOGUE_COLUMNS
        },
        "pandas_na_message_strings": sum(
            row["MessageString"] in PANDAS_DEFAULT_NA_VALUES for row in rows
        ),
        "normalized_empty_message_strings": sum(
            row["MessageString"] not in PANDAS_DEFAULT_NA_VALUES
            and not row["MessageString"].replace("\n", " ").strip()
            for row in rows
        ),
        "full_duplicate_rows": len(rows)
        - len({tuple(row[column] for column in DIALOGUE_COLUMNS) for row in rows}),
        "duplicate_intervention_sequence_keys": sum(
            count - 1 for count in duplicate_keys.values() if count > 1
        ),
        "unique_dialogues": len(dialogues),
        "unique_interventions": len({row["InterventionId"] for row in rows}),
        "unique_questions": len({row["QuestionId_DQ"] for row in rows}),
        "unique_tutors": len({row["TutorId"] for row in rows}),
        "student_messages": sum(student_lengths),
        "tutor_messages": sum(tutor_lengths),
        "invalid_is_tutor_values": invalid_speaker_values,
        "dialogue_messages": _distribution(dialogue_lengths),
        "student_messages_per_dialogue": _distribution(student_lengths),
        "tutor_messages_per_dialogue": _distribution(tutor_lengths),
        "dialogues_with_sequence_gaps_or_duplicates": sequence_problems,
        "dialogues_out_of_file_order": out_of_order,
    }
    return _SplitProfile(
        report=report,
        dialogue_keys=set(dialogues),
        questions={row["QuestionId_DQ"] for row in rows},
        tutors={row["TutorId"] for row in rows},
    )


def _profile_auxiliary(
    path: Path, columns: tuple[str, ...]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    rows = _read_csv(path, columns)
    report = {
        "columns": list(columns),
        "rows": len(rows),
        "null_counts": {column: sum(not row[column].strip() for row in rows) for column in columns},
        "full_duplicate_rows": len(rows)
        - len({tuple(row[column] for column in columns) for row in rows}),
    }
    return report, rows


def _key_fingerprint(keys: list[DialogueKey]) -> str:
    encoded = json.dumps(keys, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_eedi_metadata(root: Path) -> dict[NumericDialogueKey, tuple[str, list[tuple[str, int]]]]:
    metadata_rows = _read_csv(root / "dq-question-metadata.csv", METADATA_COLUMNS)
    subject_rows = _read_csv(root / "dialogue-subjects.csv", SUBJECT_COLUMNS)

    questions: dict[NumericDialogueKey, list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        key = (int(row["QuestionId_DQ"]), int(row["InterventionId"]))
        questions[key].append(row)

    subjects: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for row in subject_rows:
        subjects[int(row["InterventionId"])].append((row["SubjectName"], int(row["SubjectLevel"])))

    result: dict[NumericDialogueKey, tuple[str, list[tuple[str, int]]]] = {}
    for key, rows in questions.items():
        question_parts = [
            f"{row['Label']}: {row['Text'].replace(chr(10), ' ').strip()}" for row in rows
        ]
        result[key] = ("\n".join(question_parts), subjects[key[1]])
    return result


def _load_dialogues(
    path: Path,
    metadata: dict[NumericDialogueKey, tuple[str, list[tuple[str, int]]]],
) -> list[Dialogue]:
    rows = _read_csv(path, DIALOGUE_COLUMNS)
    grouped: dict[NumericDialogueKey, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (int(row["QuestionId_DQ"]), int(row["InterventionId"]))
        grouped[key].append(row)

    dialogues: list[Dialogue] = []
    for key in sorted(grouped):
        if key not in metadata:
            raise ValueError(f"Missing question metadata for dialogue key {key}")
        dialogue_rows = grouped[key]
        sequences = [int(row["MessageSequence"]) for row in dialogue_rows]
        if sequences != list(range(sequences[0], sequences[-1] + 1)):
            raise ValueError(f"Invalid message sequence for dialogue key {key}")

        turns: list[Turn] = []
        for row in dialogue_rows:
            if row["IsTutor"] not in {"0", "1"}:
                raise ValueError(f"Invalid IsTutor value for dialogue key {key}")
            role: Literal["student", "tutor"] = "tutor" if row["IsTutor"] == "1" else "student"
            raw_text = row["MessageString"]
            text = (
                "nan"
                if raw_text in PANDAS_DEFAULT_NA_VALUES
                else raw_text.replace("\n", " ").strip()
            )
            if not turns or turns[-1]["role"] != role:
                turns.append({"role": role, "content": text})
            else:
                if not turns[-1]["content"].endswith((".", "!", "?")):
                    turns[-1]["content"] += "."
                turns[-1]["content"] += " " + text

        question, dialogue_subjects = metadata[key]
        dialogues.append(
            {
                "key": key,
                "question": question,
                "subjects": dialogue_subjects,
                "turns": turns,
            }
        )
    return dialogues


def load_eedi_splits(root: Path, *, seed: int = 221, train_fraction: float = 0.75) -> DatasetSplits:
    """Reproduce the official loader and deterministic train/validation split."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one")

    metadata = _load_eedi_metadata(root)
    development = _load_dialogues(root / "anchored-dialogues" / "train.csv", metadata)
    test = _load_dialogues(root / "anchored-dialogues" / "test.csv", metadata)
    random.Random(seed).shuffle(development)
    split_point = int(len(development) * train_fraction)
    return DatasetSplits(
        train=development[:split_point],
        validation=development[split_point:],
        test=test,
    )


def _dialogue_fingerprint(dialogues: list[Dialogue]) -> str:
    encoded = json.dumps(dialogues, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _loaded_split_summary(dialogues: list[Dialogue]) -> dict[str, int | str]:
    return {
        "dialogues": len(dialogues),
        "turns": sum(len(dialogue["turns"]) for dialogue in dialogues),
        "sha256": _dialogue_fingerprint(dialogues),
    }


def audit_dataset(root: Path, *, revision: str) -> dict[str, Any]:
    """Audit the source CSVs and reproduce the upstream train/validation key split."""
    train = _profile_dialogue_split(root / "anchored-dialogues" / "train.csv")
    test = _profile_dialogue_split(root / "anchored-dialogues" / "test.csv")
    subjects_report, subject_rows = _profile_auxiliary(
        root / "dialogue-subjects.csv", SUBJECT_COLUMNS
    )
    metadata_report, metadata_rows = _profile_auxiliary(
        root / "dq-question-metadata.csv", METADATA_COLUMNS
    )

    source_keys = sorted(train.dialogue_keys, key=lambda key: (int(key[0]), int(key[1])))
    random.Random(221).shuffle(source_keys)
    split_point = int(len(source_keys) * 0.75)
    training_keys = source_keys[:split_point]
    validation_keys = source_keys[split_point:]

    all_dialogue_keys = train.dialogue_keys | test.dialogue_keys
    metadata_keys = {(row["QuestionId_DQ"], row["InterventionId"]) for row in metadata_rows}
    dialogue_interventions = {key[1] for key in all_dialogue_keys}
    subject_interventions = {row["InterventionId"] for row in subject_rows}
    loaded = load_eedi_splits(root)

    return {
        "schema_version": 1,
        "dataset_revision": revision,
        "splits": {"train": train.report, "test": test.report},
        "cross_split": {
            "dialogue_key_overlap": len(train.dialogue_keys & test.dialogue_keys),
            "intervention_overlap": len(
                {key[1] for key in train.dialogue_keys} & {key[1] for key in test.dialogue_keys}
            ),
            "question_overlap": len(train.questions & test.questions),
            "tutor_overlap": len(train.tutors & test.tutors),
        },
        "auxiliary_tables": {
            "dialogue-subjects.csv": subjects_report,
            "dq-question-metadata.csv": metadata_report,
        },
        "referential_integrity": {
            "dialogue_keys_without_question_metadata": len(all_dialogue_keys - metadata_keys),
            "question_metadata_keys_without_dialogue": len(metadata_keys - all_dialogue_keys),
            "dialogue_interventions_without_subjects": len(
                dialogue_interventions - subject_interventions
            ),
            "subject_interventions_without_dialogue": len(
                subject_interventions - dialogue_interventions
            ),
        },
        "upstream_train_validation_split": {
            "seed": 221,
            "train_fraction": 0.75,
            "source_dialogues": len(source_keys),
            "train_dialogues": len(training_keys),
            "validation_dialogues": len(validation_keys),
            "train_key_sha256": _key_fingerprint(training_keys),
            "validation_key_sha256": _key_fingerprint(validation_keys),
        },
        "paper_faithful_loader": {
            "source_dialogues_excluded": 0,
            "train": _loaded_split_summary(loaded.train),
            "validation": _loaded_split_summary(loaded.validation),
            "test": _loaded_split_summary(loaded.test),
        },
    }


def write_audit_report(root: Path, revision: str, output: Path) -> dict[str, Any]:
    """Audit a dataset snapshot and write aggregate, privacy-safe JSON evidence."""
    report = audit_dataset(root, revision=revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
