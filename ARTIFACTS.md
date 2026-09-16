# Artifact Versioning Policy

Git versions source text well; it is not an artifact store. Large binaries and generated data live outside Git and are referenced by immutable identity, location, and checksum.

## Core rule

Every important artifact should be recoverable from one of two paths:

1. **Rebuildable:** source revision + configuration + inputs + deterministic command; or
2. **Stored:** immutable external URI + cryptographic checksum + metadata describing how it was produced.

A filename such as `final_model_v2` is not a version.

## What goes where

| Artifact | Canonical handling | Git content |
|---|---|---|
| Source code and scripts | Git repository | Complete source |
| Experiment configuration | Git repository | Complete resolved/default config |
| Base model weights | Hugging Face cache; fetch by model ID and pinned revision | ID, revision, license note |
| Original dataset | Fetch from Hugging Face by pinned revision | ID, revision, split counts, checksums |
| Processed dataset | Rebuild from source when practical; otherwise private object storage | Transformation code, manifest, checksums |
| LoRA adapters | Private Hugging Face model repository or private object storage | URI, commit/revision, checksum, model card metadata |
| Intermediate checkpoints | Persistent run volume during training; retain only recovery/best checkpoints | Retention decision and manifest |
| Final checkpoints | External immutable storage with backup | URI and checksum |
| Training logs | Local run bundle plus optional W&B upload | Small reviewed summaries only |
| Raw evaluation outputs | External run bundle/object storage | Manifest and checksum |
| Aggregate metrics/tables | Small reviewed files under `results/summary/` | Versioned result summaries |
| Paper figures | Regenerated from versioned analysis code and frozen results | Final small figures where appropriate |

## Runtime layout

The CLI creates this ignored structure beneath `SIM_STUDENT_ARTIFACT_ROOT`:

```text
artifacts/
├── cache/
│   └── huggingface/
├── datasets/
│   ├── raw/
│   └── processed/
├── runs/
│   └── <run-id>/
│       ├── checkpoints/
│       ├── logs/
│       ├── outputs/
│       └── run.json
└── exports/
```

Each `run.json` is a receipt containing the run ID, UTC creation time, Git commit, resolved configuration checksum, and schema version. The final implementation will extend it with package, hardware, seed, dataset, model, command, metric, and external-storage metadata.

## Checkpoint retention

During a run, keep enough checkpoints to recover from interruption. After validation:

- retain the best checkpoint selected by the declared metric;
- retain the final adapter;
- optionally retain the latest resumable checkpoint until the run is archived;
- delete redundant intermediate checkpoints only after external copies and checksums are verified.

## Initial storage decision

No remote artifact backend is configured yet. Before the first paid training run, select and verify one of:

1. private Hugging Face repositories for final adapters and dataset derivatives;
2. S3-compatible object storage for complete run bundles;
3. both, using Hugging Face for reusable model artifacts and object storage for logs/checkpoints/results.

W&B may mirror metrics and logs, but it must not be the only copy of scientific evidence. DVC is intentionally deferred until processed datasets or artifact lineage become complex enough to justify another control plane.
