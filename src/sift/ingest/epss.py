"""FIRST EPSS enrichment — add exploit-prediction scores to existing `cve` notes.

EPSS estimates the probability a CVE will be exploited in the next 30 days.
High EPSS + in your target's stack = worth reading first.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from sift.config import get_settings
from sift.index.store import Store
from sift.pipeline import index_note
from sift.vault.notes import iter_notes, save_note

API = "https://api.first.org/data/v1/epss"
BATCH = 100


def _fetch_scores(cves: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        for i in range(0, len(cves), BATCH):
            chunk = cves[i : i + BATCH]
            r = c.get(API, params={"cve": ",".join(chunk)})
            r.raise_for_status()
            for row in r.json().get("data", []):
                out[row["cve"]] = {
                    "epss": float(row.get("epss", 0)),
                    "percentile": float(row.get("percentile", 0)),
                }
    return out


def enrich() -> int:
    s = get_settings()
    vault = s.resolved_vault()
    store = Store()

    cve_notes = [n for n in iter_notes(vault, note_type="cve")]
    ids = [n.meta.id for n in cve_notes if n.meta.id.upper().startswith("CVE-")]
    if not ids:
        print("  (no cve notes to enrich; run `sift ingest kev` / `nvd` first)")
        return 0

    scores = _fetch_scores(ids)
    updated = 0
    for note in cve_notes:
        sc = scores.get(note.meta.id)
        if not sc:
            continue
        note.meta.extra["epss"] = round(sc["epss"], 5)
        note.meta.extra["epss_percentile"] = round(sc["percentile"], 5)
        if sc["epss"] >= 0.5 and "high-epss" not in note.meta.tags:
            note.meta.tags.append("high-epss")
        note.meta.ingested = datetime.now(UTC)
        save_note(vault, note, stamp=False)
        index_note(note, store)
        updated += 1
    store.ensure_fts()
    return updated
