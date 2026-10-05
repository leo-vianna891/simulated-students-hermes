"""Run the remote file-backed launcher; keep SSH and remote-command outcomes separate."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import uuid
from pathlib import Path


def run_remote(
    ssh: list[str], cwd: str, remote_logs: str, local_logs: Path, command: list[str]
) -> int:
    local_logs.mkdir(parents=True, exist_ok=False)
    attempt_id = uuid.uuid4().hex
    remote_command = f"cd {shlex.quote(cwd)} && " + shlex.join(
        [
            "env",
            f"TRAINING_ATTEMPT_ID={attempt_id}",
            "bash",
            "scripts/run_training.sh",
            remote_logs,
            *command,
        ]
    )
    with (
        (local_logs / "ssh.stdout.log").open("wb") as stdout,
        (local_logs / "ssh.stderr.log").open("wb") as stderr,
    ):
        result = subprocess.run(
            [*ssh, remote_command], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr
        )
    status_reader = shlex.join(
        [
            "python3",
            "-c",
            f"from pathlib import Path; print(Path({remote_logs!r}, 'status.json').read_text())",
        ]
    )
    fetched = subprocess.run(
        [*ssh, f"cd {shlex.quote(cwd)} && " + status_reader],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    (local_logs / "status-fetch.stderr.log").write_text(fetched.stderr)
    remote_status: object = None
    if fetched.returncode == 0:
        try:
            remote_status = json.loads(fetched.stdout)
        except json.JSONDecodeError:
            (local_logs / "status-fetch.invalid.log").write_text(fetched.stdout)
    current_attempt = (
        isinstance(remote_status, dict) and remote_status.get("attempt_id") == attempt_id
    )
    receipt = {
        "attempt_id": attempt_id,
        "remote_receipt_current_attempt": current_attempt,
        "ssh_returncode": result.returncode,
        "status_fetch_ssh_returncode": fetched.returncode,
        "remote": remote_status,
        "remote_logs": remote_logs,
    }
    temporary = local_logs / "transport.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2))
    temporary.replace(local_logs / "transport.json")
    print(json.dumps(receipt, indent=2))
    # A missing remote receipt is an unknown outcome, never success or proof of training failure.
    if (
        not current_attempt
        or not isinstance(remote_status, dict)
        or remote_status.get("state") not in {"completed", "failed", "interrupted"}
        or not isinstance(remote_status.get("exit_code"), int)
    ):
        return 125
    return int(remote_status["exit_code"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--remote-log-dir", required=True)
    parser.add_argument("--local-log-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a remote training command after --")
    if args.host.startswith("-"):
        parser.error("Invalid SSH host")
    ssh = [
        "ssh",
        "-i",
        str(args.identity_file),
        "-p",
        str(args.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={args.known_hosts}",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=3",
        args.host,
    ]
    raise SystemExit(run_remote(ssh, args.cwd, args.remote_log_dir, args.local_log_dir, command))


if __name__ == "__main__":
    main()
