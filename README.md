# Simulated Students × Hermes

Reproduce and extend *Simulated Students in Tutoring Dialogues: Substance or Illusion?* using native-template student-simulation LoRA adapters.

## Models and protocol

| Model | Train batch × accumulation | Validation batch |
|---|---:|---:|
| Llama 3.2 3B Instruct | 2 × 32 | 4 |
| Llama 3.1 8B Instruct | 1 × 64 | 2 |
| Qwen3 4B Instruct-2507 | 1 × 64 | 2 |

`configs/train.yaml` freezes data/model revisions and the optimization settings: three epochs, BF16 without quantization, effective batch 64, LR 5e-5, AdamW/weight decay 1e-2, linear scheduler/10% warmup, clipping 1.0, and rank-stabilized LoRA (rank 32, alpha 64, dropout 0.05). Select the best checkpoint by validation loss, never test performance.

The loader reads the official implementation's published `train_gpt-4.1.csv`, `val_gpt-4.1.csv` and `test_gpt-4.1.csv` at commit `4c94a33e6053fd7acf812ff122deab16dcd3c126`. It preserves their order and splits, excludes missing/unsolvable annotations and never silently falls back to raw data. No annotation API calls are required.

The context matches the reference's no-persona student prompt: question, correct answer, solution, four option explanations and the first tutor turn. Targets remain the original student utterances—including mistakes—not the solution. Context, tutor turns, headers and padding are masked; student content and native end-of-turn tokens are supervised.

Selection is shared across models: the pinned Llama 3.2 template applies the reference's `<6,000` **rendered-character** cutoff. Every retained dialogue is then rendered natively without token truncation. Published annotated counts are **1,147 / 382 / 382**; shared retained counts are **1,080 / 358 / 365** (train/validation/test). Llama uses its fine-tuning pad token; Qwen keeps native padding and masks the separator after EOS.

## Setup and validation

```bash
uv sync --locked --extra train
uv run sim-student doctor
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv run pytest
uv run python scripts/prepare_data.py
```

Approved Hugging Face access to both Meta models is required. Keep credentials outside Git and deployment archives.

```bash
DATA=artifacts/datasets/annotated/4c94a33e6053fd7acf812ff122deab16dcd3c126
uv run --locked --extra train sim-student train "$DATA" \
  --model llama_3_2_3b --output-dir outputs/adapters/llama-3.2-3b
uv run --locked --extra train sim-student train "$DATA" \
  --model llama_3_1_8b --output-dir outputs/adapters/llama-3.1-8b
uv run --locked --extra train sim-student train "$DATA" \
  --model qwen3_4b --output-dir outputs/adapters/qwen3-4b
```

These are full-training commands, not local dry runs. They require one native-BF16 CUDA GPU/process and fresh output directories. Model revisions default to the YAML; PEFT records the base revision in the saved adapter. `--model-id`/`--model-revision` and step/sample limits are available for bounded tiny/smoke verification.

## Runpod preparation

Use an authorized on-demand session, not Spot: checkpoints currently save model-only state, so exact interrupted-training resume is not implemented. The intended target is one A40 48 GB. Transfer the current committed source and lockfile into `/workspace/simulated-students-hermes`; unpushed local changes are not available through a public clone.

From that checkout:

```bash
export HF_HOME=/workspace/hf-cache
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.python
export UV_BIN=/workspace/tools/uv
bash scripts/runpod_bootstrap.sh
"$UV_BIN" run --locked --extra train python scripts/prepare_data.py
```

Bootstrap downloads pinned uv 0.11.32 if absent and installs Python 3.11.15 plus `uv.lock`, independently of the image's PyTorch. It creates no cloud resource. Configure HF access separately before loading gated weights.

**Before a full run, execute the bounded hardware check on the corrected annotated inputs:**

```bash
DATA=artifacts/datasets/annotated/4c94a33e6053fd7acf812ff122deab16dcd3c126
"$UV_BIN" run --locked --extra train python scripts/runpod_smoke.py "$DATA" \
  --model llama_3_2_3b --output-dir outputs/smoke/llama-3.2-3b
```

Repeat for `llama_3_1_8b` and `qwen3_4b` using distinct output directories. The smoke performs one optimizer update, limited validation, adapter reload, longest train/eval batches and short generation. It uses no test data and produces no final research adapter. Run the full commands above with `"$UV_BIN"` in place of `uv` only after all checks and the session budget permit it; keep useful logs under `outputs/logs/`.

Maximum retained train/validation lengths are 1,660/1,701 tokens for both Llamas and 1,657/1,764 for Qwen. All official models previously passed A40 BF16 smokes on **raw** inputs; those older memory peaks do not validate the longer corrected inputs. Local tiny models passed the corrected train/eval/save/reload/generation path for all three configurations.

Monitor allocated-time costs with an operator-approved deadline and separate Pod shutdown controls. A process timeout does not stop GPU billing; stopped Pods still charge storage. Before termination, download and verify the selected adapter/tokenizer, native `trainer_state.json`, used YAML/lockfile and useful logs; record the code revision. Remove disposable smoke outputs and redundant checkpoints after verified backup. No Pod is provisioned by these scripts.

## Reproduction scope and layout

Full scans matched all imported annotated rows to the official CSVs/Git blobs and all retained native prompts to the pinned upstream prompt function. Shared keys are identical across models. The mask is semantic and template-derived rather than a copy of hard-coded Llama token IDs; it excludes header/separator tokens. Shared selection for Qwen and explicit initialization seeding are declared extensions. Bit-identical losses or reproduction of unpublished evaluator-adapter results are not promised. No complete research training or Hermes evaluation has been performed yet.

```text
configs/train.yaml          Frozen inputs and training settings
src/simulated_students/     Dataset, native formatting, training and CLI
scripts/                    Dataset preparation and GPU bootstrap/smoke
tests/                      Essential behavior tests
artifacts/datasets/          Local pinned inputs, outside Git
outputs/                    Adapters and useful logs, outside Git
```

Datasets, weights, checkpoints, caches and environments are ignored by Git. Operational documentation is maintained separately in the research vault, not in parallel workspace reports.

## References

- Paper implementation: <https://github.com/umass-ml4ed/sim-student-eval>
- Published annotations: <https://github.com/umass-ml4ed/sim-student-eval/tree/4c94a33e6053fd7acf812ff122deab16dcd3c126/data/annotated/eedi>
- Original dataset: <https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k>

## License

No project license has been selected yet; upstream data/model terms still apply.
