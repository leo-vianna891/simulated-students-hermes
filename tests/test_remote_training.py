from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_remote_training.py"
SPEC = importlib.util.spec_from_file_location("remote_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("ssh_code", "remote", "expected"),
    [
        (1, {"exit_code": 0, "state": "completed", "attempt_id": "current"}, 0),
        (0, {"exit_code": 23, "state": "failed", "attempt_id": "current"}, 23),
        (255, None, 125),
        (73, {"exit_code": 0, "state": "completed", "attempt_id": "old"}, 125),
        (255, {"exit_code": None, "state": "running", "attempt_id": "current"}, 125),
    ],
)
def test_ssh_result_is_not_confused_with_remote_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ssh_code: int,
    remote: dict[str, int] | None,
    expected: int,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(MODULE.uuid, "uuid4", lambda: SimpleNamespace(hex="current"))
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args, ssh_code)
        return subprocess.CompletedProcess(
            args, 0 if remote else 1, json.dumps(remote) if remote else "", ""
        )

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    logs = tmp_path / "transport"
    result = MODULE.run_remote(
        ["ssh", "test-host"], "/workspace/project", "/workspace/logs", logs, ["uv", "run", "train"]
    )
    assert result == expected
    receipt = json.loads((logs / "transport.json").read_text())
    assert receipt["ssh_returncode"] == ssh_code
    assert receipt["remote"] == remote
    with pytest.raises(FileExistsError):
        MODULE.run_remote(["ssh"], "/workspace/project", "/workspace/logs", logs, ["true"])


def test_real_launcher_and_status_reader_vertical_slice(tmp_path: Path) -> None:
    import sys

    remote_logs = tmp_path / "remote logs"
    local_logs = tmp_path / "transport logs"
    code = MODULE.run_remote(
        ["bash", "-c"],
        str(SCRIPT.parent.parent),
        str(remote_logs),
        local_logs,
        [sys.executable, "-c", "print('durable-output'); raise SystemExit(7)"],
    )
    assert code == 7
    status = json.loads((local_logs / "transport.json").read_text())
    assert status["ssh_returncode"] == 7
    assert status["remote"]["command_exit_code"] == 7
    assert status["remote"]["state"] == "failed"
    assert "durable-output" in (remote_logs / "command.log").read_text()
    assert (remote_logs / "diagnostics.log").exists()
