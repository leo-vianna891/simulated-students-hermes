# Simulated Students × Hermes

Research project for reproducing and extending *Simulated Students in Tutoring Dialogues: Substance or Illusion?*

The immediate goal is straightforward: load the Eedi tutoring dialogues, format them with each model's native chat template, and train LoRA adapters for three instruction-tuned models. Infrastructure will be added only when an implemented training or evaluation workflow requires it.

## Models

- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-4B-Instruct-2507`

## Current state

- deterministic Eedi dialogue loader and train/validation split;
- native Llama and Qwen chat templates with student-only loss labels through the native end-of-turn token;
- paper-faithful training configuration;
- minimal Transformers/PEFT LoRA training command;
- completed local tiny-model checks and official-model BF16 training/reload/longest-batch/generation smokes on a Runpod A40;
- minimal local quality tooling;

## Setup

```bash
uv sync
uv run sim-student doctor
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

Install the ML dependencies only when working on training:

```bash
uv sync --extra train
```

Training requires approved Hugging Face access to the official Meta model. On a suitable GPU, run:

```bash
uv run sim-student train \
  artifacts/datasets/raw/6d4eb56961aa098a901a1043aa98013dba7cf1ed \
  --output-dir outputs/llama-3.2-3b
```

Select a configuration with `--model llama_3_2_3b` (default), `--model llama_3_1_8b`, or `--model qwen3_4b`. All three use the same direct Transformers/PEFT command; only the native tokenizer/template and configured batch sizes differ. Llama uses its fine-tuning pad token; Qwen keeps its native pad token. Qwen's separator after the end-of-turn token is masked.

Selection is shared across models: the pinned Llama 3.2 tokenizer in `data.filter_tokenizer_id` / `filter_tokenizer_revision` applies the `< 6,000` rendered-character cutoff. Each retained dialogue is then formatted with the selected model's native template, without truncation. All three use the same 1,161 training, 388 validation, and 390 test dialogue keys; Qwen does not receive additional dialogues merely because its template is shorter.

Before Qwen training, set `train_batch_size` and `gradient_accumulation_steps` in the launch configuration; their product must equal `effective_batch_size`. Its tested `1 × 64` is viable for the current raw-input smoke, but production values remain unset pending the final data/prompt review. `--model-id`, `--model-revision`, and the step/sample limits support bounded smoke tests without changing the final model selection.

Official Meta models require approved Hugging Face access and a valid credential. The official Llama 3.1 tokenizer was validated on all 1,971 raw dialogues and matches the provisional mirror. Full-size hardware smokes passed, but no complete research training run or Hermes evaluation has been performed.

### Reproduction scope

The current loader reproduces the reference raw-Eedi fallback. The reference SFT instead prefers its published annotated train/validation files when available, drops unsolvable questions, and adds solution/answer explanations to the question context. Reconciling that data/prompt difference is required before claiming a faithful full reproduction. The counts and memory results above apply to the current raw-input path; they must be rechecked if the final input changes.

## Remote GPU smoke

`scripts/runpod_bootstrap.sh` creates a separate Python 3.11.15 environment from
`uv.lock`, without using the image's preinstalled PyTorch. Provide uv 0.11.32 at
`/workspace/tools/uv`, or set `UV_BIN`. Caches and managed Python stay under
`/workspace`; the script creates no cloud infrastructure.

After transferring the pinned dataset and configuring Hugging Face access:

```bash
bash scripts/runpod_bootstrap.sh
HF_HOME=/workspace/hf-cache /workspace/tools/uv run --locked --extra train python \
  scripts/runpod_smoke.py \
  artifacts/datasets/raw/6d4eb56961aa098a901a1043aa98013dba7cf1ed \
  --model llama_3_2_3b --output-dir outputs/smoke-llama-3.2
```

The smoke pins model revisions and requires exactly one native-BF16 CUDA GPU.
It runs one optimizer step with warmup disabled, saves/reloads the adapter,
exercises the longest training and validation batches, and generates up to eight
tokens. Qwen's tested `1 × 64` batch applies to the smoke; the production YAML is still
unset. The test split is not used. All three official-model smokes passed on an
A40, including longest batches in the current raw-input population. This is
neither final research training nor Hermes evaluation.

Keep data and environments under `/workspace`. Download and verify adapters
before terminating a Pod: its local volume survives stop, but not terminate.
Stopped Pods still incur storage charges. A process timeout does not stop GPU
billing; stopping or terminating the Pod is a separate operation.

## Layout

```text
configs/train.yaml          Paper-faithful training settings
src/simulated_students/     Dataset, formatting, training, and minimal CLI
tests/                      Essential behavior tests
```

Datasets, model weights, checkpoints, caches, and generated outputs are ignored by Git.

## References

- Paper implementation: <https://github.com/umass-ml4ed/sim-student-eval>
- Dataset: `Eedi/Question-Anchored-Tutoring-Dialogues-2k`

## License

No license has been selected yet.
