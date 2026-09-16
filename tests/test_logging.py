from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulated_students.logging import log_event


def test_log_event_appends_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "events.jsonl"

    first = log_event(path, "run_started", run_id="pilot-001")
    log_event(path, "run_finished", status="ok")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert first["event"] == "run_started"
    assert records[0]["run_id"] == "pilot-001"
    assert records[1]["status"] == "ok"


def test_log_event_rejects_empty_event(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        log_event(tmp_path / "events.jsonl", "  ")
