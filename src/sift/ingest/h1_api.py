"""HackerOne Hacker API v1.

Two sources:
  * ``my_reports()``  -> every report *you* have submitted (resolved, duplicate,
    informative, …). Stored as PRIVATE notes (``source: hackerone-mine``). The
    vault is gitignored, but these never leave your machine regardless.
  * ``hacktivity()``  -> the public disclosed-activity feed, filtered with a
    Lucene query string. Good for ongoing ingestion of fresh writeups.

Auth: HTTP Basic with ``H1_API_USERNAME`` / ``H1_API_TOKEN``.
Token: https://hackerone.com/settings/api_token/edit

Re-running ``my_reports`` updates only what the API knows better than the note - the
report's state, triage and close dates, the state tags - and keeps the body and any
tags or links the user added in Obsidian (`merge_my_report`). It used to overwrite
the whole note, annotations included. ``hacktivity`` skips reports already stored.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import httpx

from sift.config import get_settings
from sift.ingest.base import clean_text, extract_cwes
from sift.ingest.existing import merge_into_existing, stored_ids, union
from sift.vault.notes import IdConflict, Note
from sift.vault.schema import Frontmatter

# The body the API-less fallback writes; a later run with a real body replaces it.
_NO_BODY = "_(no writeup body returned by API)_"

BASE = "https://api.hackerone.com/v1"


def _client() -> httpx.Client:
    s = get_settings()
    if not (s.h1_api_username and s.h1_api_token):
        raise RuntimeError(
            "HackerOne API not configured — set H1_API_USERNAME and H1_API_TOKEN in .env"
        )
    return httpx.Client(
        base_url=BASE,
        auth=(s.h1_api_username, s.h1_api_token),
        headers={"Accept": "application/json"},
        timeout=60,
        follow_redirects=True,
    )


def _paginate(client: httpx.Client, path: str, params: dict) -> Iterator[dict]:
    params = {**params, "page[number]": 1, "page[size]": 100}
    while True:
        r = client.get(path, params=params)
        if r.status_code == 429:
            time.sleep(10)
            continue
        r.raise_for_status()
        payload = r.json()
        yield from payload.get("data", [])
        nxt = (payload.get("links") or {}).get("next")
        if not nxt:
            return
        params["page[number]"] += 1
        time.sleep(1.0)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _rel_attr(item: dict, rel: str, key: str) -> str | None:
    node = ((item.get("relationships") or {}).get(rel) or {}).get("data") or {}
    return (node.get("attributes") or {}).get(key)


# --------------------------------------------------------------------------- #
# your own reports
# --------------------------------------------------------------------------- #
def _my_report_to_note(item: dict) -> Note | None:
    attrs = item.get("attributes", {})
    rid = item.get("id")
    if not rid:
        return None
    state = (attrs.get("state") or "").lower()
    body = clean_text(attrs.get("vulnerability_information")) or _NO_BODY
    program = _rel_attr(item, "program", "name") or _rel_attr(item, "program", "handle")
    weakness = _rel_attr(item, "weakness", "name")
    weakness_cwe = _rel_attr(item, "weakness", "external_id")

    cwes = extract_cwes(weakness_cwe, body)
    tags = ["hackerone", "mine", state or "unknown"]
    if state == "duplicate":
        tags.append("dupe")
    if weakness:
        tags.append(weakness.lower())

    meta = Frontmatter(
        id=f"h1mine-{rid}",
        type="report",
        title=clean_text(attrs.get("title")) or f"My report {rid}",
        source="hackerone-mine",
        url=f"https://hackerone.com/reports/{rid}",
        created=_parse_date(attrs.get("created_at")),
        cwe=cwes,
        program=program,
        tags=tags,
        extra={
            "state": state,
            "weakness": weakness,
            "triaged_at": attrs.get("triaged_at"),
            "closed_at": attrs.get("closed_at"),
        },
    )
    return Note(meta=meta, body=body)


def _state_tags(meta: Frontmatter) -> set[str]:
    """The tags `_my_report_to_note` derives from the report's state."""
    state = str(meta.extra.get("state") or "") if isinstance(meta.extra, dict) else ""
    tags = {state or "unknown"}
    if state == "duplicate":
        tags.add("dupe")
    return tags


def merge_my_report(stored: Note, incoming: Note) -> Note:
    """Refresh a stored h1-mine note from the API without touching what the user wrote.

    Updated: ``extra`` (state, triaged_at, closed_at, weakness; None never erases),
    the state tags (a report that moved from triaged to resolved loses ``triaged``),
    cwe, program and url. Kept: the body - unless it is still the no-body placeholder
    - the title, and every tag or link the user added. Raises `IdConflict` for a note
    that is not this report from this source.
    """
    s, i = stored.meta, incoming.meta
    if s.id != i.id or (s.source or "") != (i.source or ""):
        raise IdConflict(
            i.id, stored.path or Path(f"{s.id}.md"), "source",
            existing_source=s.source, existing_url=s.url,
        )  # fmt: skip
    stale = _state_tags(s) - _state_tags(i)
    meta = s.model_copy(deep=True)
    meta.tags = union([t for t in s.tags if t not in stale], i.tags)
    extra = dict(s.extra) if isinstance(s.extra, dict) else {}
    extra.update({k: v for k, v in (i.extra or {}).items() if v is not None})
    meta.extra = extra
    meta.cwe = union(s.cwe, i.cwe)
    meta.program = i.program or s.program
    meta.url = i.url or s.url
    meta.created = s.created or i.created
    meta.ingested = None  # stamped by the writer only if something changed
    keep_body = stored.body.strip() and stored.body.strip() != _NO_BODY
    body = stored.body if keep_body else incoming.body
    return Note(meta=meta, body=body, path=stored.path)


def my_reports() -> Iterator[Note]:
    vault = get_settings().resolved_vault()
    with _client() as client:
        for item in _paginate(client, "/hackers/me/reports", {}):
            note = _my_report_to_note(item)
            if note:
                yield merge_into_existing(vault, note, merge_my_report)


# --------------------------------------------------------------------------- #
# public hacktivity feed
# --------------------------------------------------------------------------- #
DEFAULT_HACKTIVITY_QUERY = "disclosed:true AND severity_rating:(high OR critical)"


def _hacktivity_to_note(item: dict) -> Note | None:
    attrs = item.get("attributes", {})
    rid = item.get("id")
    if not rid:
        return None
    body = clean_text(attrs.get("vulnerability_information")) or clean_text(attrs.get("title"))
    if not body:
        return None
    program = _rel_attr(item, "program", "name") or _rel_attr(item, "program", "handle")
    weakness = _rel_attr(item, "weakness", "name")
    sev = attrs.get("severity_rating")

    meta = Frontmatter(
        id=f"h1act-{rid}",
        type="report",
        title=clean_text(attrs.get("title")) or f"Hacktivity {rid}",
        source="hackerone-hacktivity",
        url=f"https://hackerone.com/reports/{rid}",
        created=_parse_date(
            attrs.get("disclosed_at") or attrs.get("latest_disclosable_activity_at")
        ),
        cwe=extract_cwes(body, weakness),
        severity=sev.lower() if sev else None,
        program=program,
        tags=["hackerone", "disclosed", "hacktivity"] + ([weakness.lower()] if weakness else []),
        extra={"total_awarded_amount": attrs.get("total_awarded_amount"), "weakness": weakness},
    )
    return Note(meta=meta, body=body)


def hacktivity(
    *, query: str | None = None, limit: int | None = 500, refresh: bool = False
) -> Iterator[Note]:
    """The newest disclosed reports matching `query`. ``limit`` caps the reports
    considered, newest first, whether new or already stored - so a re-run looks at
    the same window and stores what is new in it, rather than paging ever deeper. A
    stored report is skipped unless ``refresh``."""
    query = query or DEFAULT_HACKTIVITY_QUERY
    have = set() if refresh else stored_ids(get_settings().resolved_vault(), prefix="h1act-")
    seen = 0
    with _client() as client:
        for item in _paginate(
            client,
            "/hackers/hacktivity",
            {"queryString": query, "sort": "-disclosed_at"},
        ):
            rid = item.get("id") if isinstance(item, dict) else None
            if rid and f"h1act-{rid}" in have:
                seen += 1
            else:
                note = _hacktivity_to_note(item)
                if not note:
                    continue
                yield note
                seen += 1
            if limit and seen >= limit:
                return
