from __future__ import annotations

from typing import Any

from simulated_students.dataset import Dialogue
from simulated_students.formatting import format_student_dialogue


class CharacterTokenizer:
    """Small chat-template tokenizer used to inspect mask boundaries exactly."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> str | list[int]:
        rendered = "<bos>" + "".join(
            f"<{message['role']}>{message['content']}<eot>" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(character) for character in rendered] if tokenize else rendered


def test_format_student_dialogue_masks_everything_except_student_turns() -> None:
    dialogue: Dialogue = {
        "key": (1, 100),
        "question": "Question: What is 2 + 2?",
        "subjects": [("Math", 1)],
        "turns": [
            {"role": "tutor", "content": "Try the addition."},
            {"role": "student", "content": "Maybe 5."},
            {"role": "tutor", "content": "Count again."},
            {"role": "student", "content": "It is 4."},
        ],
    }
    tokenizer = CharacterTokenizer()

    example = format_student_dialogue(dialogue, tokenizer)

    assert example is not None
    assert example["attention_mask"] == [1] * len(example["input_ids"])
    supervised = "".join(
        chr(token)
        for token, label in zip(example["input_ids"], example["labels"], strict=True)
        if label != -100
    )
    assert supervised == "Maybe 5.<eot>It is 4.<end_of_dialogue><eot>"
    assert "You are a student attempting to solve a math problem" in example["text"]
    assert "First Tutor Turn: Try the addition." in example["text"]
    assert all(
        label in {-100, token}
        for token, label in zip(example["input_ids"], example["labels"], strict=True)
    )

    assert format_student_dialogue(dialogue, tokenizer, maximum_characters=10) is None


class NewlineTokenizer(CharacterTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str | list[int]:
        text = "<bos>" + "".join(f"<{m['role']}>{m['content']}<eot>\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(c) for c in text] if tokenize else text


def test_mask_excludes_separator_after_end_of_turn() -> None:
    dialogue: Dialogue = {
        "key": (1, 100),
        "question": "2 + 2?",
        "subjects": [],
        "turns": [{"role": "student", "content": "4"}],
    }
    example = format_student_dialogue(dialogue, NewlineTokenizer(), end_of_turn_id=ord(">"))
    assert example is not None
    supervised = "".join(
        chr(t)
        for t, label in zip(example["input_ids"], example["labels"], strict=True)
        if label != -100
    )
    assert supervised == "4<end_of_dialogue><eot>"
    assert example["labels"][-1] == -100
