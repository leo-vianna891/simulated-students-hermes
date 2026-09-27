# Simulated Students × Hermes

Research project for reproducing and extending *Simulated Students in Tutoring Dialogues: Substance or Illusion?*

The immediate goal is straightforward: load the Eedi tutoring dialogues, format them with each model's native chat template, and train LoRA adapters for three instruction-tuned models. Infrastructure will be added only when an implemented training or evaluation workflow requires it.

## Models

- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-4B-Instruct-2507`

## Current state

- deterministic Eedi dialogue loader and train/validation split;
- native Llama 3.2 chat formatting with assistant-only loss labels;
- paper-faithful training configuration;
- minimal local quality tooling;
- training command not implemented yet.

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

## Layout

```text
configs/train.yaml          Planned paper-faithful training settings
src/simulated_students/     Dataset, Llama formatting, and minimal CLI
tests/                      Essential behavior tests
```

Datasets, model weights, checkpoints, caches, and generated outputs are ignored by Git.

## References

- Paper implementation: <https://github.com/umass-ml4ed/sim-student-eval>
- Dataset: `Eedi/Question-Anchored-Tutoring-Dialogues-2k`

## License

No license has been selected yet.
