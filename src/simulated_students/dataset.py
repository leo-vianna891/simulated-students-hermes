"""Load the Eedi tutoring dialogues used by the training pipeline."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

DialogueKey = tuple[int, int]
Role = Literal["student", "tutor"]

# These are the missing-value forms present in the pinned CSV snapshot.
_PANDAS_NA = {"", "None"}


class Turn(TypedDict):
    role: Role
    content: str


class Dialogue(TypedDict):
    key: DialogueKey
    question: str
    subjects: list[tuple[str, int]]
    turns: list[Turn]


@dataclass(frozen=True)
class DatasetSplits:
    train: list[Dialogue]
    validation: list[Dialogue]
    test: list[Dialogue]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows: list[dict[str, str]] = []
        for raw_row in csv.DictReader(stream):
            row: dict[str, str] = {}
            for key, value in raw_row.items():
                if key is None or value is None:
                    raise ValueError(f"Malformed CSV row in {path}")
                row[key] = value
            rows.append(row)
    return rows


def _load_metadata(root: Path) -> dict[DialogueKey, tuple[str, list[tuple[str, int]]]]:
    question_parts: dict[DialogueKey, list[str]] = defaultdict(list)
    for row in _read_csv(root / "dq-question-metadata.csv"):
        key = (int(row["QuestionId_DQ"]), int(row["InterventionId"]))
        text = row["Text"].replace("\n", " ").strip()
        question_parts[key].append(f"{row['Label']}: {text}")

    subjects: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for row in _read_csv(root / "dialogue-subjects.csv"):
        subjects[int(row["InterventionId"])].append((row["SubjectName"], int(row["SubjectLevel"])))

    return {key: ("\n".join(parts), subjects[key[1]]) for key, parts in question_parts.items()}


def _message_text(raw: str) -> str:
    return "nan" if raw in _PANDAS_NA else raw.replace("\n", " ").strip()


def _load_dialogues(
    path: Path,
    metadata: dict[DialogueKey, tuple[str, list[tuple[str, int]]]],
) -> list[Dialogue]:
    grouped: dict[DialogueKey, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(path):
        key = (int(row["QuestionId_DQ"]), int(row["InterventionId"]))
        grouped[key].append(row)

    dialogues: list[Dialogue] = []
    for key in sorted(grouped):
        if key not in metadata:
            raise ValueError(f"Missing question metadata for dialogue {key}")

        rows = grouped[key]
        sequences = [int(row["MessageSequence"]) for row in rows]
        if sequences != list(range(sequences[0], sequences[-1] + 1)):
            raise ValueError(f"Invalid message sequence for dialogue {key}")

        turns: list[Turn] = []
        for row in rows:
            if row["IsTutor"] not in {"0", "1"}:
                raise ValueError(f"Invalid speaker value for dialogue {key}")
            role: Role = "tutor" if row["IsTutor"] == "1" else "student"
            text = _message_text(row["MessageString"])
            if not turns or turns[-1]["role"] != role:
                turns.append({"role": role, "content": text})
                continue
            if not turns[-1]["content"].endswith((".", "!", "?")):
                turns[-1]["content"] += "."
            turns[-1]["content"] += f" {text}"

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


def load_eedi_splits(
    root: Path,
    *,
    seed: int = 221,
    train_fraction: float = 0.75,
) -> DatasetSplits:
    """Load the published data and reproduce the official train/validation split."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")

    metadata = _load_metadata(root)
    development = _load_dialogues(root / "anchored-dialogues" / "train.csv", metadata)
    test = _load_dialogues(root / "anchored-dialogues" / "test.csv", metadata)

    random.Random(seed).shuffle(development)
    split_at = int(len(development) * train_fraction)
    return DatasetSplits(
        train=development[:split_at],
        validation=development[split_at:],
        test=test,
    )
