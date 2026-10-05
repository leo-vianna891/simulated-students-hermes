#!/usr/bin/env bash
# Usage: scripts/run_training.sh LOG_DIR COMMAND [ARG ...]
# Linux + Python 3 stdlib only. Run on the remote host to record its actual status.
# No pipes/tee: the command writes both streams directly to command.log.
# A directory flock rejects concurrent runs; any existing evidence rejects reuse.
set -euo pipefail
if (( $# < 2 )); then
    printf 'Usage: %s LOG_DIR COMMAND [ARG ...]\n' "${0##*/}" >&2
    exit 64
fi

# exec keeps the launcher PID stable for SSH/process supervision. Python signal
# traps forward TERM/INT to the whole command session; finally collects failures.
# -c (not stdin/heredoc) preserves arbitrary command input and argument boundaries.
exec python3 -c '
import datetime
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    directory = Path(sys.argv[1])
    try:
        directory.mkdir(parents=True, exist_ok=True)
        lock = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError:
        print("Cannot open log directory", file=sys.stderr)
        return 73
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock)
        print("Log directory is already in use", file=sys.stderr)
        return 75
    if any(directory.iterdir()):
        os.close(lock)
        print("Refusing to overwrite existing evidence", file=sys.stderr)
        return 73

    record = {
        "state": "starting",
        "attempt_id": os.environ.get("TRAINING_ATTEMPT_ID"),
        "hostname": socket.gethostname(),
        "launcher_pid": os.getpid(),
        "command_pid": None,
        "started_at": now(),
        "finished_at": None,
        "command_exit_code": None,
        "exit_code": None,
        "signal": None,
    }
    child = None
    received_signal = None
    signal_time = None
    exit_code = 1

    def write_status():
        # Only operational metadata: no argv, secrets, or environment snapshot.
        temporary = directory / "status.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "status.json")

    def forward(signum):
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    def trap_signal(signum, frame):
        nonlocal received_signal, signal_time
        if received_signal is None:
            received_signal = signum
            signal_time = time.monotonic()
        forward(signum)

    def diagnostics():
        # Diagnostic errors/timeouts must not replace the command exit status.
        with (directory / "diagnostics.log").open("xb") as stream:
            for arguments in (
                ["df", "-h"],
                ["free", "-h"],
                ["nvidia-smi"],
                ["dmesg", "--ctime"],
            ):
                stream.write(("\n=== " + arguments[0] + " ===\n").encode())
                stream.flush()
                try:
                    result = subprocess.run(
                        arguments, stdout=stream, stderr=stream,
                        stdin=subprocess.DEVNULL, timeout=5, check=False,
                    )
                    stream.write(("exit_code=" + str(result.returncode) + "\n").encode())
                except (OSError, subprocess.TimeoutExpired) as error:
                    stream.write((type(error).__name__ + "\n").encode())

    # A transport disconnect must not kill the supervisor while its detached child runs.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, trap_signal)
    signal.signal(signal.SIGINT, trap_signal)
    try:
        with (directory / "command.log").open("xb") as output:
            write_status()
            try:
                child = subprocess.Popen(
                    sys.argv[2:], stdout=output, stderr=output, start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            except FileNotFoundError:
                output.write(b"Command executable not found\n")
                exit_code = 127
            except OSError:
                output.write(b"Command could not be started\n")
                exit_code = 126
            else:
                record["command_pid"] = child.pid
                record["state"] = "running"
                write_status()
                # Handle signals received during Popen before child was assigned.
                if received_signal is not None:
                    forward(received_signal)
                while True:
                    try:
                        code = child.wait(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        if signal_time is not None and time.monotonic() - signal_time >= 5:
                            forward(signal.SIGKILL)
                exit_code = code if code >= 0 else 128 - code
                record["command_exit_code"] = exit_code
                if code < 0:
                    record["signal"] = signal.Signals(-code).name
    except Exception:
        # Avoid printing exceptions that may contain sensitive command arguments.
        print("Launcher recording failed", file=sys.stderr)
        if child is not None:
            forward(signal.SIGKILL)
            child.wait()
    finally:
        if received_signal is not None:
            exit_code = 128 + received_signal
            record["signal"] = signal.Signals(received_signal).name
            # Reap/stop descendants even if the immediate command handled its signal.
            forward(signal.SIGKILL)
        if exit_code != 0:
            try:
                diagnostics()
            except Exception:
                print("Failure diagnostics unavailable", file=sys.stderr)
        record["exit_code"] = exit_code
        record["finished_at"] = now()
        record["state"] = (
            "interrupted" if record["signal"] else "completed" if exit_code == 0 else "failed"
        )
        try:
            write_status()
        except Exception:
            print("Final status could not be written", file=sys.stderr)
        os.close(lock)
    return exit_code


sys.exit(main())
' "$@"
