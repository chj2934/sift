"""FIRST EPSS enrichment — add exploit-prediction scores to existing `cve` notes.

EPSS estimates the probability a CVE will be exploited in the next 30 days.
High EPSS + in your target's stack = worth reading first.

A refresh touches only notes whose score actually changed: the file is rewritten in
place (whatever the user renamed it to), ``ingested`` is bumped only for those, and
they are re-embedded in batches. ``high-epss`` is removed again when a score falls
below the threshold. A failed API batch is retried and then skipped - the scores the
other batches returned are still applied.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from sift.config import get_settings
from sift.index.store import Store
from sift.ingest.base import maintain_index
from sift.pipeline import index_notes
from sift.vault.notes import Note, describe_error, load_note, save_note, write_lock

log = logging.getLogger(__name__)

API = "https://api.first.org/data/v1/epss"
BATCH = 100
ATTEMPTS = 4  # per API batch: 1 + 3 retries, backing off 1, 2, 4 s
INDEX_BATCH = 200
HIGH_EPSS = 0.5
HIGH_EPSS_TAG = "high-epss"

Scores = dict[str, dict[str, float]]


@dataclass
class EpssResult:
    notes: int = 0  # CVE notes found
    scored: int = 0  # ...with a score from the API
    changed: int = 0  # rewritten because a score or the tag changed
    unchanged: int = 0
    failed_batches: int = 0  # API batches given up on (their CVEs keep their old scores)
    errors: int = 0  # notes that could not be read, saved or indexed
    indexed_chunks: int = 0


def _retryable(r: httpx.Response) -> bool:
    return r.status_code == 429 or r.status_code >= 500


def _fetch_scores(
    cves: list[str],
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Scores, int]:
    """(scores by CVE id, number of batches that failed). Never raises for a batch."""
    out: Scores = {}
    failed = 0
    own = client is None
    c = client or httpx.Client(timeout=60, follow_redirects=True)
    try:
        for i in range(0, len(cves), BATCH):
            chunk = cves[i : i + BATCH]
            rows = None
            for attempt in range(ATTEMPTS):
                try:
                    r = c.get(API, params={"cve": ",".join(chunk)})
                    if _retryable(r) and attempt < ATTEMPTS - 1:
                        sleep(2**attempt)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    rows = data.get("data", []) if isinstance(data, dict) else None
                    if rows is None:
                        raise ValueError("response has no 'data'")
                    break
                except (httpx.TransportError, ValueError) as exc:
                    if attempt < ATTEMPTS - 1:
                        sleep(2**attempt)
                        continue
                    log.warning("epss: batch %d failed: %s", i // BATCH, describe_error(exc))
                except httpx.HTTPStatusError as exc:
                    log.warning("epss: batch %d failed: %s", i // BATCH, describe_error(exc))
                    break
            if rows is None:
                failed += 1
                continue
            for row in rows:
                try:
                    out[str(row["cve"])] = {
                        "epss": float(row.get("epss", 0)),
                        "percentile": float(row.get("percentile", 0)),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
    finally:
        if own:
            c.close()
    return out, failed


def apply_score(note: Note, score: dict[str, float]) -> bool:
    """Write `score` into the note's frontmatter; False when nothing would change. Pure."""
    epss = round(score["epss"], 5)
    pct = round(score["percentile"], 5)
    want_tag = score["epss"] >= HIGH_EPSS
    has_tag = HIGH_EPSS_TAG in note.meta.tags
    extra = note.meta.extra
    if extra.get("epss") == epss and extra.get("epss_percentile") == pct and want_tag == has_tag:
        return False
    extra["epss"] = epss
    extra["epss_percentile"] = pct
    if want_tag and not has_tag:
        note.meta.tags.append(HIGH_EPSS_TAG)
    elif has_tag and not want_tag:
        note.meta.tags = [t for t in note.meta.tags if t != HIGH_EPSS_TAG]
    return True


def _cve_rows(vault: Path) -> list[Any]:
    from sift.vault.catalog import fresh_catalog

    return [r for r in fresh_catalog(vault).rows("cve") if r.id.upper().startswith("CVE-")]


def enrich_notes(
    *,
    vault: Path | None = None,
    store: Any = None,
    fetch: Callable[[list[str]], tuple[Scores, int]] = _fetch_scores,
) -> EpssResult:
    """Apply current EPSS scores to every CVE note; see the module docstring."""
    vault = Path(vault) if vault is not None else get_settings().resolved_vault()
    res = EpssResult()
    rows = _cve_rows(vault)
    res.notes = len(rows)
    if not rows:
        log.warning("epss: no cve notes to enrich; run `sift ingest kev` / `nvd` first")
        return res

    scores, res.failed_batches = fetch(sorted({r.id for r in rows}))
    store = store if store is not None else Store()
    pending: list[Note] = []

    def flush(batch: Iterable[Note]) -> None:
        batch = list(batch)
        if not batch:
            return
        try:
            res.indexed_chunks += index_notes(batch, store)
        except Exception as exc:  # noqa: BLE001 - saved; `sift reindex` picks them up
            res.errors += len(batch)
            log.warning(
                "epss: index batch of %d notes failed (they are saved; run `sift reindex`): %s",
                len(batch),
                describe_error(exc),
            )

    for row in rows:
        score = scores.get(row.id)
        if not score:
            continue
        res.scored += 1
        try:
            with write_lock(vault):  # read-modify-write: an MCP edit cannot slip between
                note = load_note(row.path)
                if note.meta.id != row.id:
                    continue  # the file changed under the catalog; the next run sees it
                if not apply_score(note, score):
                    res.unchanged += 1
                    continue
                note.meta.ingested = datetime.now(UTC)
                save_note(vault, note, stamp=False, existing=row.path, rename=False)
        except Exception as exc:  # noqa: BLE001 - one bad note never stops the run
            res.errors += 1
            log.warning("epss: could not update %s: %s", row.path.name, describe_error(exc))
            continue
        res.changed += 1
        pending.append(note)
        if len(pending) >= INDEX_BATCH:
            flush(pending)
            pending.clear()
    flush(pending)
    if res.changed:
        maintain_index(store, "epss")
    if res.failed_batches:
        log.warning(
            "epss: %d API batches failed; their CVEs keep their previous scores",
            res.failed_batches,
        )
    return res


def enrich() -> int:
    """Number of CVE notes whose EPSS data changed (see `enrich_notes` for details)."""
    return enrich_notes().changed
