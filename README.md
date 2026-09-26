# Simulated Students × Hermes

Private research repository for reproducing and extending *Simulated Students in Tutoring Dialogues: Substance or Illusion?* The project will train paper-faithful LoRA adapters for three instruction-tuned language models, evaluate each base/SFT condition outside and inside Hermes, and measure whether student-simulation SFT degrades tool calling.

## Status

Infrastructure bootstrap. Training code and experimental results do not exist yet.

## Primary models

- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-4B-Instruct-2507`

## Local setup

```bash
uv sync --locked
uv run sim-student doctor
uv run sim-student audit-dataset artifacts/datasets/raw/<revision> --revision <revision>
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

The expensive training dependencies are an explicit extra and are not installed for ordinary local development:

```bash
uv sync --locked --extra train
```

Do not install the training extra until a GPU environment or a specific local test requires it.

The upstream requirements currently contain a dependency conflict: `trl==0.22.2` requires Transformers 4.55 or newer, while the file pins `transformers==4.53.2`. This repository provisionally resolves the conflict with `transformers==4.56.1`, which appears in the upstream inline comment. That choice remains an explicit validation item rather than an assumed reproduction fact.

## Repository layout

```text
configs/                 Versioned experiment and reproducibility configuration
src/simulated_students/  Reusable project code and CLI
scripts/                 Thin operational entry points, when needed
tests/                   Automated tests and reduced fixtures
results/summary/         Small, reviewed aggregate outputs suitable for Git
.github/workflows/        Continuous integration
```

Runtime artifacts are deliberately external to Git. See [`ARTIFACTS.md`](ARTIFACTS.md).
The dataset audit writes aggregate, privacy-safe evidence to
`results/summary/dataset-audit.json`; source dialogue text remains under the ignored artifact root.

## Upstream reference

- Repository: <https://github.com/umass-ml4ed/sim-student-eval>
- Audited commit: `4c94a33e6053fd7acf812ff122deab16dcd3c126`
- Dataset: `Eedi/Question-Anchored-Tutoring-Dialogues-2k`

## Scope boundaries

The first training wave reproduces the paper-style student SFT. Tool-preserving SFT is conditional on measured degradation. A bespoke Hermes benchmark and independent memory-curation architecture are out of scope for the initial wave.

## License

No public license has been selected. The repository is private and internal unless a later publication decision explicitly changes that status.
