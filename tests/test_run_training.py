"""Exercise the launcher with local processes and mocked diagnostic tools."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "run_training.sh"


@pytest.fixture
def diagnostic_env(tmp_path: Path) -> dict[str, str]:
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("df", "free", "nvidia-smi", "dmesg"):
        tool = tools / name
        tool.write_text(f"#!/bin/sh\nprintf '%s\\n' 'mock-{name}'\nexit 1\n")
        tool.chmod(0o755)
    return {**os.environ, "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"}


def invoke(log_dir: Path, code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), str(log_dir), sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def status(log_dir: Path) -> dict[str, object]:
    value: object = json.loads((log_dir / "status.json").read_text())
    assert isinstance(value, dict)
    return value


def wait_ready(log_dir: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        assert process.poll() is None, process.communicate()
        if (log_dir / "ready").exists():
            assert status(log_dir)["state"] == "running"
            return
        time.sleep(0.01)
    pytest.fail("command did not become ready")


@pytest.fixture
def running(
    tmp_path: Path, diagnostic_env: dict[str, str]
) -> Iterator[tuple[Path, subprocess.Popen[str]]]:
    log_dir = tmp_path / "running"
    code = (
        "import pathlib, signal, time; "
        "signal.signal(signal.SIGINT, signal.SIG_DFL); "
        "signal.signal(signal.SIGTERM, signal.SIG_DFL); "
        "print('before signal'); "
        f"pathlib.Path({str(log_dir / 'ready')!r}).touch(); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        ["bash", str(LAUNCHER), str(log_dir), sys.executable, "-c", code],
        env=diagnostic_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_ready(log_dir, process)
        yield log_dir, process
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=15)


def test_success_preserves_both_streams(tmp_path: Path, diagnostic_env: dict[str, str]) -> None:
    log_dir = tmp_path / "success"
    code = "import sys; print('out-' * 20000); print('err-' * 20000, file=sys.stderr)"
    result = invoke(log_dir, code, diagnostic_env)
    assert result.returncode == 0, result.stderr
    log = (log_dir / "command.log").read_text()
    assert "out-" * 20000 in log
    assert "err-" * 20000 in log
    record = status(log_dir)
    assert record["state"] == "completed"
    assert record["exit_code"] == record["command_exit_code"] == 0
    assert record["hostname"]
    assert record["command_pid"]
    assert record["finished_at"]
    assert not (log_dir / "diagnostics.log").exists()
    assert not list(log_dir.glob("*.tmp"))


def test_failure_keeps_original_and_diagnostic_errors(
    tmp_path: Path, diagnostic_env: dict[str, str]
) -> None:
    log_dir = tmp_path / "failure"
    result = invoke(
        log_dir,
        "import sys; print('original stdout'); print('original stderr', file=sys.stderr); "
        "sys.exit(23)",
        {**diagnostic_env, "TEST_SECRET": "not-for-diagnostics"},
    )
    assert result.returncode == 23
    assert status(log_dir)["exit_code"] == 23
    log = (log_dir / "command.log").read_text()
    assert "original stdout" in log and "original stderr" in log
    diagnostics = (log_dir / "diagnostics.log").read_text()
    for name in ("df", "free", "nvidia-smi", "dmesg"):
        assert f"mock-{name}" in diagnostics
    assert "not-for-diagnostics" not in diagnostics
    assert "sys.exit" not in diagnostics


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_is_forwarded_and_preserved(
    running: tuple[Path, subprocess.Popen[str]], sig: signal.Signals
) -> None:
    log_dir, process = running
    process.send_signal(sig)
    _, stderr = process.communicate(timeout=15)
    assert process.returncode == 128 + sig, stderr
    record = status(log_dir)
    assert record["exit_code"] == record["command_exit_code"] == 128 + sig
    assert record["signal"] == sig.name
    assert "before signal" in (log_dir / "command.log").read_text()
    assert (log_dir / "diagnostics.log").exists()
    child_pid = record["command_pid"]
    assert isinstance(child_pid, int)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_concurrent_launcher_cannot_touch_evidence(
    running: tuple[Path, subprocess.Popen[str]], diagnostic_env: dict[str, str]
) -> None:
    log_dir, _ = running
    before = (log_dir / "command.log").read_bytes()
    initial_status = (log_dir / "status.json").read_bytes()
    result = invoke(log_dir, "print('must not run')", diagnostic_env)
    assert result.returncode == 75
    assert (log_dir / "command.log").read_bytes() == before
    assert (log_dir / "status.json").read_bytes() == initial_status


@pytest.mark.parametrize("existing", ["command.log", "status.json", "diagnostics.log"])
def test_existing_evidence_is_never_truncated(
    tmp_path: Path, diagnostic_env: dict[str, str], existing: str
) -> None:
    log_dir = tmp_path / "existing"
    log_dir.mkdir()
    evidence = log_dir / existing
    evidence.write_text("original evidence")
    result = invoke(log_dir, "print('must not run')", diagnostic_env)
    assert result.returncode == 73
    assert evidence.read_text() == "original evidence"
    assert list(log_dir.iterdir()) == [evidence]


def test_argument_boundaries_and_stdin_are_preserved(
    tmp_path: Path, diagnostic_env: dict[str, str]
) -> None:
    log_dir = tmp_path / "input"
    result = subprocess.run(
        [
            str(LAUNCHER),
            str(log_dir),
            sys.executable,
            "-c",
            "import sys; print(repr(sys.argv[1:])); print(sys.stdin.read())",
            "space argument",
            "",
            "$(not-executed)",
        ],
        input="stdin evidence",
        env=diagnostic_env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    log = (log_dir / "command.log").read_text()
    assert "['space argument', '', '$(not-executed)']" in log
    assert "stdin evidence" in log


def test_command_signal_without_launcher_signal(
    tmp_path: Path, diagnostic_env: dict[str, str]
) -> None:
    log_dir = tmp_path / "child-signal"
    result = invoke(
        log_dir,
        "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        diagnostic_env,
    )
    assert result.returncode == 143
    assert status(log_dir)["command_exit_code"] == 143
    assert status(log_dir)["signal"] == "SIGTERM"
    assert (log_dir / "diagnostics.log").exists()


def test_hangup_does_not_abandon_supervision(
    running: tuple[Path, subprocess.Popen[str]],
) -> None:
    log_dir, process = running
    process.send_signal(signal.SIGHUP)
    time.sleep(0.05)
    assert process.poll() is None
    assert status(log_dir)["state"] == "running"
    process.terminate()
    process.communicate(timeout=15)
    assert status(log_dir)["exit_code"] == 143


def test_missing_command_is_recorded(tmp_path: Path, diagnostic_env: dict[str, str]) -> None:
    log_dir = tmp_path / "missing"
    result = subprocess.run(
        ["bash", str(LAUNCHER), str(log_dir), "nonexistent-training-executable"],
        env=diagnostic_env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 127
    assert status(log_dir)["exit_code"] == 127
    assert (log_dir / "diagnostics.log").exists()
