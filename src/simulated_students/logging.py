"""Minimal structured JSON Lines logging for reproducible local runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def log_event(path: Path, event: str, **fields: Any) -> dict[str, Any]:
    """Append one timestamped event and return the serialized payload."""
    if not event.strip():
        raise ValueError("event must not be empty")

    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return payload
