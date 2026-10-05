"""Bounded BF16 hardware smoke; not a scientific training run or evaluation."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import cast

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from simulated_students.dataset import load_eedi_splits
from simulated_students.formatting import ChatTemplateTokenizer
from simulated_students.training import SFTCollator, _format_dialogues, _load_config, train_student

MODEL_KEYS = ("llama_3_2_3b", "llama_3_1_8b", "qwen3_4b")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--model-id", help="Tiny/local model override for harness verification only"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This smoke requires a CUDA GPU with native BF16 support")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU for the configured effective batch of 64")
    smoke_config = args.output_dir.with_name(f"{args.output_dir.name}-config.yaml")
    if args.output_dir.exists() or smoke_config.exists():
        raise FileExistsError(f"Choose fresh smoke output/config paths: {args.output_dir}")

    config = _load_config(args.config)
    selected = config["models"][args.model]
    # One nonzero-LR optimizer update is enough for a smoke, unlike the paper's warmup.
    config["training"]["warmup_ratio"] = 0.0
    args.output_dir.mkdir(parents=True)
    smoke_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    revision = None if args.model_id else selected["revision"]
    model_id = args.model_id or selected["id"]
    start = time.monotonic()
    print(f"SMOKE ONLY: {model_id}, revision={revision}, BF16, no quantization", flush=True)
    torch.cuda.reset_peak_memory_stats()
    train_student(
        args.dataset_root,
        args.output_dir,
        config_path=smoke_config,
        model_key=args.model,
        model_id_override=args.model_id,
        model_revision=revision,
        max_steps=1,
        max_train_samples=64,
        max_validation_samples=8,
    )
    train_peak = torch.cuda.max_memory_allocated()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (args.output_dir / filename).is_file():
            raise RuntimeError(f"Missing saved adapter file: {filename}")
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)  # type: ignore[no-untyped-call]
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, args.output_dir, is_trainable=True)
    data = config["data"]
    reference = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        data["filter_tokenizer_id"], revision=data["filter_tokenizer_revision"]
    )
    splits = load_eedi_splits(args.dataset_root)
    collator = SFTCollator(tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["learning_rate"],
    )
    torch.cuda.reset_peak_memory_stats()
    # Exercise actual worst-length train/eval batches; never use the test split.
    for name, dialogues, size in (
        ("train", splits.train, selected["train_batch_size"]),
        ("validation", splits.validation, selected["validation_batch_size"]),
    ):
        examples = _format_dialogues(
            dialogues,
            tokenizer,
            data["maximum_prompt_characters"],
            cast(ChatTemplateTokenizer, reference),
        ).examples
        longest = sorted(examples, key=lambda example: len(example["input_ids"]), reverse=True)[
            :size
        ]
        batch = {key: value.to("cuda") for key, value in collator(longest).items()}
        model.train(name == "train")
        with torch.set_grad_enabled(name == "train"):
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite {name} loss")
            if name == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        print(
            f"Worst-length {name} batch: {size} x {batch['input_ids'].shape[1]} tokens", flush=True
        )
        del batch, loss
    stress_peak = torch.cuda.max_memory_allocated()
    model.eval()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain what a fraction is."}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        output = model.generate(  # type: ignore[no-untyped-call]
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    generated = output.shape[1] - inputs.shape[1]
    if generated < 1:
        raise RuntimeError("Adapter reload produced no new tokens")
    print(f"Adapter reloaded; generated {generated} token(s)")
    print(f"Training allocated peak: {train_peak / 2**30:.3f} GiB")
    print(f"Worst-length allocated peak: {stress_peak / 2**30:.3f} GiB")
    print(f"Elapsed including model acquisition: {time.monotonic() - start:.1f} seconds")
    print("SMOKE_OK — not final training, test evaluation or Hermes validation")


if __name__ == "__main__":
    main()
