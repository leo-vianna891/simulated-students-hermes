# Contributing

## Working rules

1. Keep code, configuration, tests, and lightweight manifests in Git.
2. Never commit credentials, model weights, datasets, checkpoints, raw logs, or raw generations.
3. Every experiment must start from a committed configuration and record the Git commit.
4. Prefer small focused branches and conventional commit subjects.
5. Before committing, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Reproducibility

A run is not reproducible merely because its code was committed. Preserve model and dataset revisions, resolved configuration, dependency lock, random seeds, hardware/software facts, artifact checksums, and the exact launch command.
