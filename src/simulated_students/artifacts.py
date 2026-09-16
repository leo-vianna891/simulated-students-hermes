"""Safe, deterministic runtime artifact layouts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")


def project_root() -> Path:
    """Return the repository root from the installed source layout."""
    return Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    """Resolve the ignored runtime artifact root."""
    configured = os.environ.get("SIM_STUDENT_ARTIFACT_ROOT")
    return Path(configured).expanduser().resolve() if configured else project_root() / "artifacts"


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 checksum without loading a file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path | None = None) -> str | None:
    """Return the current commit, or None before the first commit."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root or project_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


@dataclass(frozen=True)
class RunLayout:
    """Filesystem locations associated with one immutable run identifier."""

    run_id: str
    root: Path

    @classmethod
    def create(cls, run_id: str, root: Path | None = None) -> RunLayout:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must be 3-80 lowercase safe filename characters")

        run_root = (root or artifact_root()) / "runs" / run_id
        layout = cls(run_id=run_id, root=run_root.resolve())
        for directory in (layout.checkpoints, layout.logs, layout.outputs):
            directory.mkdir(parents=True, exist_ok=True)
        return layout

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def receipt(self) -> Path:
        return self.root / "run.json"

    def write_receipt(self, config_path: Path) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
        }
        self.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload
