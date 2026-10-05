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

## Complete checkpoints and recovery

The trainer saves full LoRA checkpoints every **five optimizer updates** and at every epoch boundary. Validation remains epoch-based; the final adapter is still selected by the lowest validation loss. The latest and best checkpoints are retained (up to three directories, allowing a previous recovery point while the next save is sealed), including AdamW, scheduler, RNG, native trainer state and tokenizer files. Base weights are referenced by their pinned revision, not duplicated into each adapter checkpoint.

A checkpoint becomes recoverable only after `resume.json` is atomically written with state-file SHA-256 checksums. Resume checks actual tokenized train/validation inputs, config, model/revision, sample/step limits, training-source hashes, lockfile and runtime identity. Incomplete, corrupted, old model-only or mismatched checkpoints are rejected before loading model weights. If a newer receipt-bearing checkpoint is corrupt, quarantine it outside the run tree before selecting an older verified recovery point. Restore the **whole run directory**, including its best checkpoint, then use the latest sealed checkpoint:

```bash
uv run --locked --extra train sim-student train "$DATA" \
  --model llama_3_1_8b --output-dir outputs/adapters/llama-3.1-8b \
  --resume-from-checkpoint outputs/adapters/llama-3.1-8b/checkpoint-10
```

Keep the original three-epoch horizon and all model/data/settings. Do not add three epochs or restart the LR schedule. A moved run tree is supported; the native best-checkpoint path is rebased in memory. Only restore your own trusted checkpoints: native PyTorch training state is not an untrusted interchange format. CPU tiny-model interrupted/resumed runs are checked against uninterrupted runs; this does not promise bitwise CUDA equivalence. Recovery starts at the last complete saved optimizer update, not an unsaved in-flight microbatch. A checkpoint already at the full horizon is rejected for further optimization: recover the validation-selected adapter directly from its best checkpoint and verify reload, rather than adding updates.

## Durable execution diagnostics

Use a **new log directory for every attempt**, including resume attempts:

```bash
bash scripts/run_training.sh outputs/logs/8b-attempt-1 \
  uv run --locked --extra train sim-student train "$DATA" \
  --model llama_3_1_8b --output-dir outputs/adapters/llama-3.1-8b
```

The launcher writes stdout/stderr together to `command.log`, an atomic remote `status.json` (state, attempt ID, timestamps, PIDs, exit code and signal), and separate available disk/RAM/GPU/kernel failure diagnostics. Python output is unbuffered. The supervisor ignores SSH hangup and keeps recording its detached child; TERM/INT stop the child and preserve the signal status. It refuses reused log directories. Never pass credentials in the command line; configure HF access externally.

For SSH runs, `python scripts/run_remote_training.py --help` describes the explicit host/key/known-hosts/cwd/log arguments and command after `--`. This client stores the SSH result separately from the remote launcher receipt in `transport.json`. Only terminal receipts with the current attempt ID are accepted, so an old successful status cannot mask a refused new launch. A missing remote receipt is **unknown**, not proof of Python failure and not successful completion. An SSH disconnect does not necessarily stop remote training; inspect remote status/processes before relaunching or terminating a Pod.

Before teardown, obtain system/provider logs while the Pod is still accessible and make a checksum-verified external copy of the entire useful run tree, including all recovery state, its best adapter and attempt logs. Periodic external copies are recommended; local Pod checkpoints alone cannot survive Pod deletion or host loss. Checkpoint saving and these launchers do **not** provision, stop or terminate billable resources.

## Runpod preparation

Use an authorized on-demand session. Complete checkpoints now support controlled resume, but external backups and shutdown controls remain necessary; Spot interruption handling has not been validated. The intended target is one A40 48 GB. Transfer the current committed source and lockfile into `/workspace/simulated-students-hermes`; unpushed local changes are not available through a public clone.

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

For the remaining-model session, run the check for `llama_3_1_8b` and `qwen3_4b` using distinct output directories; the completed 3B adapter does not need retraining. The smoke performs one optimizer update, limited validation, adapter reload, longest train/eval batches and short generation. It uses no test data and produces no final research adapter. Run the full commands above with `"$UV_BIN"` in place of `uv` only after all checks and the session budget permit it; keep useful logs under `outputs/logs/`.

Maximum retained train/validation lengths are 1,660/1,701 tokens for both Llamas and 1,657/1,764 for Qwen. All official models passed A40 BF16 reload/generation and longest-batch checks on the **corrected annotated** inputs. Local tiny models also passed interrupted/full-state/resumed training, including a moved run directory and mid-epoch recovery. Repeat bounded checks for the remaining models on the next Pod before full training; CPU equality does not promise bitwise CUDA equivalence.

Monitor allocated-time costs with an operator-approved deadline and separate Pod shutdown controls. A process timeout does not stop GPU billing; stopped Pods still charge storage. Before termination, download and verify the selected adapter/tokenizer, native `trainer_state.json`, used YAML/lockfile and useful logs; record the code revision. Remove disposable smoke outputs and redundant checkpoints after verified backup. No Pod is provisioned by these scripts.

## Reproduction scope and layout

Full scans matched all imported annotated rows to the official CSVs/Git blobs and all retained native prompts to the pinned upstream prompt function. Shared keys are identical across models. The mask is semantic and template-derived rather than a copy of hard-coded Llama token IDs; it excludes header/separator tokens. Shared selection for Qwen and explicit initialization seeding are declared extensions. Bit-identical CUDA losses or reproduction of unpublished evaluator-adapter results are not promised. The Llama 3.2 3B full adapter has been trained and verified; the first 8B session failed after a preserved epoch-2 checkpoint, and Qwen full training has not started. The old model-only 8B checkpoint is not resumable under the declared protocol. No test-set or Hermes quality evaluation has been performed yet.

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
