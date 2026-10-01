"""Append-only log of what the gate dropped, and why.

Cheap to write now, and the evidence base for retuning the threshold later: after a
month of real hunting you can review what got dropped that you wish you'd kept,
rather than guessing at the setting up front.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sift.config import get_settings

REJECTS_FILE = "_rejects.jsonl"


def _rejects_path() -> Path:
    return get_settings().resolved_vault() / REJECTS_FILE


def record_reject(candidate, verdict) -> None:
    """Append one dropped candidate. Never raises — a logging failure must not
    abort an ingest run."""
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "title": candidate.title,
        "url": candidate.url,
        "source": candidate.source,
        "created": candidate.created.isoformat() if candidate.created else None,
        "already_known": verdict.already_known,
        "justification": verdict.justification,
    }
    try:
        path = _rejects_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:  # pragma: no cover - disk/permission edge
        print(f"  ! rejects: could not log {candidate.title!r}: {exc}")


def load_rejects(limit: int | None = None) -> list[dict]:
    """Read the reject log back. Tolerant of a missing file or a torn last line."""
    path = _rejects_path()
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:] if limit else rows
