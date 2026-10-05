"""Complete-checkpoint receipts and fail-closed resume validation (single process)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

STATE_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def seal_checkpoint(path: Path, identity: dict[str, Any]) -> None:
    """Publish a receipt only after the native state files have been fully saved."""
    hashes = {}
    for name in STATE_FILES:
        file = path / name
        if not file.is_file() or not file.stat().st_size:
            raise ValueError(f"Missing checkpoint state: {file}")
        hashes[name] = file_sha256(file)
    receipt = {"version": 1, "identity": identity, "sha256": hashes}
    temporary = path / "resume.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path / "resume.json")


def validate_resume(
    path: Path, output_dir: Path, identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Reject old/model-only, incomplete, changed or mismatched checkpoints."""
    path, output_dir = path.resolve(), output_dir.resolve()
    if path.parent != output_dir or not path.name.startswith("checkpoint-"):
        raise ValueError("Resume output must be the restored checkpoint's parent run directory")
    if not (path / "resume.json").is_file():
        raise ValueError("Checkpoint has no complete-state receipt; model-only resume is forbidden")
    receipt = json.loads((path / "resume.json").read_text())
    if receipt.get("version") != 1:
        raise ValueError("Unsupported checkpoint receipt version")
    if identity is not None and receipt.get("identity") != identity:
        raise ValueError("Resume identity differs: inputs, configuration, code or runtime changed")
    for name in STATE_FILES:
        file = path / name
        if not file.is_file() or not file.stat().st_size:
            raise ValueError(f"Missing checkpoint state: {file}")
        if receipt.get("sha256", {}).get(name) != file_sha256(file):
            raise ValueError(f"Checkpoint checksum mismatch: {file}")
    state = cast(dict[str, Any], json.loads((path / "trainer_state.json").read_text()))
    if path.name != f"checkpoint-{state['global_step']}":
        raise ValueError("Checkpoint directory disagrees with native global step")
    for other in output_dir.glob("checkpoint-*"):
        if other.name.removeprefix("checkpoint-").isdigit():
            step = int(other.name.removeprefix("checkpoint-"))
            if step > int(state["global_step"]) and (other / "resume.json").is_file():
                raise ValueError(
                    "Resume from the latest complete checkpoint; do not overwrite history"
                )
    if state.get("best_model_checkpoint"):
        best = output_dir / Path(state["best_model_checkpoint"]).name
        if not (best / "resume.json").is_file():
            raise ValueError("Restore the best adapter checkpoint alongside the latest checkpoint")
        best_receipt = json.loads((best / "resume.json").read_text())
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            file = best / name
            if not file.is_file() or best_receipt.get("sha256", {}).get(name) != file_sha256(file):
                raise ValueError(f"Best adapter checkpoint checksum mismatch: {file}")
    return state
