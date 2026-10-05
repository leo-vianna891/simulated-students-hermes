"""Load the reference paper's published annotated Eedi splits."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

DialogueKey = tuple[int, int]
Role = Literal["student", "tutor"]


class Turn(TypedDict):
    role: Role
    content: str


class QuestionAnnotation(TypedDict):
    solvable: bool
    correct_option: int
    solution: str
    option_1_explanation: str
    option_2_explanation: str
    option_3_explanation: str
    option_4_explanation: str


class Dialogue(TypedDict):
    key: DialogueKey
    question: str
    subjects: list[tuple[str, int]]
    turns: list[Turn]
    question_annotation: NotRequired[QuestionAnnotation]


@dataclass(frozen=True)
class DatasetSplits:
    train: list[Dialogue]
    validation: list[Dialogue]
    test: list[Dialogue]


def _literal(text: str) -> object:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def _load_annotated(path: Path) -> list[Dialogue]:
    dialogues: list[Dialogue] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            annotation = _literal(row["question_annotation"])
            # Match upstream load_annotated_data(..., drop_unsolvable=True).
            if not isinstance(annotation, dict) or not annotation.get("solvable"):
                continue
            key = _literal(row["key"])
            turns = _literal(row["turns"])
            subjects = _literal(row["subjects"])
            if not isinstance(key, tuple) or len(key) != 2 or not all(type(k) is int for k in key):
                raise ValueError(f"Invalid dialogue key in {path}")
            if (
                not isinstance(turns, list)
                or not turns
                or not all(
                    isinstance(t, dict)
                    and t.get("role") in {"student", "tutor"}
                    and isinstance(t.get("content"), str)
                    for t in turns
                )
            ):
                raise ValueError(f"Invalid dialogue turns in {path}")
            if not isinstance(subjects, list):
                raise ValueError(f"Invalid subjects in {path}")
            if (
                type(annotation.get("correct_option")) is not int
                or not 1 <= annotation["correct_option"] <= 4
            ):
                raise ValueError(f"Invalid correct option in {path}")
            if not all(
                isinstance(annotation.get(k), str)
                for k in (
                    "solution",
                    "option_1_explanation",
                    "option_2_explanation",
                    "option_3_explanation",
                    "option_4_explanation",
                )
            ):
                raise ValueError(f"Incomplete question annotation in {path}")
            dialogues.append(
                {
                    "key": cast(DialogueKey, key),
                    "question": row["question"],
                    "subjects": cast(list[tuple[str, int]], subjects),
                    "turns": cast(list[Turn], turns),
                    "question_annotation": cast(QuestionAnnotation, annotation),
                }
            )
    if not dialogues:
        raise ValueError(f"No solvable annotated dialogues in {path}")
    if len({d["key"] for d in dialogues}) != len(dialogues):
        raise ValueError(f"Duplicate dialogue keys in {path}")
    return dialogues


def load_eedi_splits(root: Path) -> DatasetSplits:
    """Preserve published splits/order; never reshuffle or fall back to raw data."""
    splits = DatasetSplits(
        train=_load_annotated(root / "train_gpt-4.1.csv"),
        validation=_load_annotated(root / "val_gpt-4.1.csv"),
        test=_load_annotated(root / "test_gpt-4.1.csv"),
    )
    keys = [
        set(d["key"] for d in split) for split in (splits.train, splits.validation, splits.test)
    ]
    if any(keys[i] & keys[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Dialogue overlap across annotated splits")
    return splits
