"""Append-only log of what the gate dropped, and why.

Cheap to write now, and the evidence base for retuning the threshold later: after a
month of real hunting you can review what got dropped that you wish you'd kept,
rather than guessing at the setting up front.

The log is also gate state: `manual.gated_urls()` reads it, so a row that fails to
load un-gates its url and the article is exported for judging again.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from sift.config import get_settings

log = logging.getLogger(__name__)

REJECTS_FILE = "_rejects.jsonl"


def _rejects_path() -> Path:
    return get_settings().resolved_vault() / REJECTS_FILE


def record_reject(candidate, verdict, *, gated_by: str | None = None) -> bool:
    """Append one dropped candidate. Returns whether the row was written.

    Never raises — a logging failure must not abort an ingest run.
    """
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "title": candidate.title,
        "url": candidate.url,
        "source": candidate.source,
        "created": candidate.created.isoformat() if candidate.created else None,
        "already_known": verdict.already_known,
        "reason": getattr(verdict, "reason", None),
        "justification": verdict.justification,
        # Which gate made the call ("in-session", or a model id), so drops from
        # different gate designs can be told apart when the threshold is retuned.
        "gated_by": gated_by,
    }
    try:
        # ASCII-only: a raw U+2028/U+2029/U+0085 in a scraped title is a line break to
        # some readers, which split the row and silently dropped it.
        data = (json.dumps(row, ensure_ascii=True) + "\n").encode("ascii")
        path = _rejects_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as fh:
            # A crash or full disk can leave a torn last line with no newline. Gluing
            # the next row onto it made both unreadable, so terminate it first.
            fh.seek(0, os.SEEK_END)
            if fh.tell() > 0:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    data = b"\n" + data
            fh.seek(0, os.SEEK_END)
            fh.write(data)
        return True
    except (OSError, TypeError, ValueError) as exc:
        log.warning("rejects: could not log %r: %s", candidate.title, exc)
        return False


def load_rejects(limit: int | None = None) -> list[dict]:
    """Read the reject log back. Tolerant of a missing file or a torn line."""
    path = _rejects_path()
    if not path.exists():
        return []
    rows: list[dict] = []
    # Text-mode iteration ends lines only at \n, \r and \r\n. str.splitlines() also
    # splits on U+2028/U+2029/U+0085, which older rows (written with
    # ensure_ascii=False) can contain raw. errors="replace": a write torn mid-character
    # must cost that one row, not the whole log.
    with path.open(encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows[-limit:] if limit else rows
