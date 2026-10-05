"""Minimal native-template LoRA training path."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from simulated_students.dataset import Dialogue, load_eedi_splits
from simulated_students.formatting import (
    ChatTemplateTokenizer,
    format_student_dialogue,
)


class ModelInput(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class TokenizedDialogues(Dataset[ModelInput]):
    def __init__(self, examples: list[ModelInput]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ModelInput:
        return self.examples[index]


@dataclass(frozen=True)
class SFTCollator:
    pad_token_id: int

    def __call__(self, examples: list[ModelInput]) -> dict[str, Tensor]:
        maximum_length = max(len(example["input_ids"]) for example in examples)

        def padded(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (maximum_length - len(values))

        return {
            "input_ids": torch.tensor(
                [padded(example["input_ids"], self.pad_token_id) for example in examples]
            ),
            "attention_mask": torch.tensor(
                [padded(example["attention_mask"], 0) for example in examples]
            ),
            "labels": torch.tensor([padded(example["labels"], -100) for example in examples]),
        }


def _load_config(path: Path) -> dict[str, Any]:
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return cast(dict[str, Any], raw)


def _format_dialogues(
    dialogues: list[Dialogue],
    tokenizer: PreTrainedTokenizerBase,
    maximum_characters: int,
    filter_tokenizer: ChatTemplateTokenizer | None = None,
) -> TokenizedDialogues:
    chat_tokenizer = cast(ChatTemplateTokenizer, tokenizer)
    examples: list[ModelInput] = []
    for dialogue in dialogues:
        formatted = format_student_dialogue(
            dialogue,
            chat_tokenizer,
            maximum_characters=maximum_characters,
            end_of_turn_id=tokenizer.eos_token_id,
            filter_tokenizer=filter_tokenizer,
        )
        if formatted is None:
            continue
        examples.append(
            {
                "input_ids": formatted["input_ids"],
                "attention_mask": formatted["attention_mask"],
                "labels": formatted["labels"],
            }
        )
    if not examples:
        raise ValueError("No dialogues remained after formatting")
    return TokenizedDialogues(examples)


def train_student(
    dataset_root: Path,
    output_dir: Path,
    *,
    config_path: Path = Path("configs/train.yaml"),
    model_key: str = "llama_3_2_3b",
    model_id_override: str | None = None,
    model_revision: str | None = None,
    max_steps: int | None = None,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
) -> None:
    """Train the selected student adapter using the project configuration."""
    config = _load_config(config_path)
    training = config["training"]
    lora = config["lora"]
    data = config["data"]
    model_config = config["models"][model_key]
    for field in ("train_batch_size", "validation_batch_size", "gradient_accumulation_steps"):
        if model_config.get(field) is None:
            raise ValueError(f"Set models.{model_key}.{field} in {config_path} before training")
    if int(model_config["train_batch_size"]) * int(
        model_config["gradient_accumulation_steps"]
    ) != int(training["effective_batch_size"]):
        raise ValueError("Microbatch times gradient accumulation must match effective_batch_size")
    model_id = model_id_override or model_config["id"]
    model_revision = model_revision or (None if model_id_override else model_config["revision"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")
    if torch.cuda.device_count() > 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Use one process and one visible GPU for the configured effective batch")
    if not model_id_override and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        raise RuntimeError("Official-model training requires a native-BF16 CUDA GPU")
    seed = int(training["seed"])
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_id,
        revision=model_revision,
    )
    if model_key.startswith("llama_"):
        if "<|finetune_right_pad_id|>" not in tokenizer.get_vocab():
            raise ValueError(f"Tokenizer for {model_id} has no Llama fine-tuning pad token")
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer for {model_id} needs native padding and end-of-turn tokens")

    filter_tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        data["filter_tokenizer_id"],
        revision=data["filter_tokenizer_revision"],
    )
    selection_tokenizer = cast(ChatTemplateTokenizer, filter_tokenizer)

    splits = load_eedi_splits(dataset_root)
    train_dialogues = splits.train[:max_train_samples]
    validation_dialogues = splits.validation[:max_validation_samples]
    train_dataset = _format_dialogues(
        train_dialogues,
        tokenizer,
        int(data["maximum_prompt_characters"]),
        selection_tokenizer,
    )
    validation_dataset = _format_dialogues(
        validation_dialogues,
        tokenizer,
        int(data["maximum_prompt_characters"]),
        selection_tokenizer,
    )

    has_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map={"": 0} if has_cuda else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.config.use_cache = False
    if model_key.startswith("llama_"):
        model.config.pretraining_tp = 1
    peft_model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            revision=model_revision,
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            use_rslora=bool(lora["use_rslora"]),
            target_modules=list(lora["target_modules"]),
        ),
    )
    peft_model.print_trainable_parameters()

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training["epochs"]),
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        lr_scheduler_type=str(training["scheduler"]),
        warmup_ratio=float(training["warmup_ratio"]),
        per_device_train_batch_size=int(model_config["train_batch_size"]),
        gradient_accumulation_steps=int(model_config["gradient_accumulation_steps"]),
        per_device_eval_batch_size=int(model_config["validation_batch_size"]),
        eval_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
        bf16=has_cuda and training["precision"] == "bf16",
        use_cpu=not has_cuda,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=peft_model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=SFTCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    trainer.save_state()
