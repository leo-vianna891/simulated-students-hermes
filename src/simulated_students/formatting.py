"""Format Eedi dialogues with native student-simulator chat templates."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from simulated_students.dataset import Dialogue, Turn

END_OF_DIALOGUE = "<end_of_dialogue>"
STUDENT_SYSTEM_PROMPT = (
    "You are a student attempting to solve a math problem, seeking help from a tutor."
)


class ChatTemplateTokenizer(Protocol):
    """Tokenizer surface required by the formatting pipeline."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> object: ...


class SFTExample(TypedDict):
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def _student_messages(dialogue: Dialogue) -> list[dict[str, str]]:
    turns: list[Turn] = [
        {"role": turn["role"], "content": turn["content"]} for turn in dialogue["turns"]
    ]
    if not turns:
        raise ValueError(f"Dialogue {dialogue['key']} has no turns")

    turns[-1]["content"] += END_OF_DIALOGUE
    context = f"Question:\n{dialogue['question']}\n\n"
    if turns[0]["role"] == "tutor":
        context += f"First Tutor Turn: {turns[0]['content']}"
        turns = turns[1:]
    else:
        context += "(No First Tutor Turn)"

    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    messages.extend(
        {
            "role": "assistant" if turn["role"] == "student" else "user",
            "content": turn["content"],
        }
        for turn in turns
    )
    return messages


def _render(
    tokenizer: ChatTemplateTokenizer,
    messages: list[dict[str, str]],
) -> str:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(rendered, str):
        raise TypeError("Chat template did not return text")
    return rendered


def _tokenize(
    tokenizer: ChatTemplateTokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(encoded, list) or not all(type(token) is int for token in encoded):
        raise TypeError("Chat template did not return a flat token ID list")
    return encoded


def format_student_dialogue(
    dialogue: Dialogue,
    tokenizer: ChatTemplateTokenizer,
    *,
    maximum_characters: int = 6_000,
    end_of_turn_id: int | None = None,
) -> SFTExample | None:
    """Apply the native template and supervise only student/assistant outputs.

    Assistant spans are inferred by rendering prefixes with the tokenizer's own
    generation prompt. This preserves the native template without relying on
    Llama-specific header token IDs.
    """
    messages = _student_messages(dialogue)
    text = _render(tokenizer, messages)
    if len(text) >= maximum_characters:
        return None

    input_ids = _tokenize(tokenizer, messages)
    labels = [-100] * len(input_ids)

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        assistant_start = _tokenize(
            tokenizer,
            messages[:index],
            add_generation_prompt=True,
        )
        assistant_end = _tokenize(tokenizer, messages[: index + 1])
        start = len(assistant_start)
        end = len(assistant_end)

        if input_ids[:start] != assistant_start or input_ids[:end] != assistant_end:
            raise ValueError("Chat template tokenization is not prefix-stable")
        if start >= end:
            raise ValueError("Assistant message produced no supervised tokens")
        if end_of_turn_id is not None:
            # Qwen adds a separator after EOS; supervise through EOS, not beyond it.
            endings = [i for i in range(start, end) if input_ids[i] == end_of_turn_id]
            if not endings:
                raise ValueError("Assistant message has no native end-of-turn token")
            end = endings[-1] + 1
        labels[start:end] = input_ids[start:end]

    if all(label == -100 for label in labels):
        raise ValueError(f"Dialogue {dialogue['key']} has no student response")

    return {
        "text": text,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
