"""HackerOne Hacker API v1.

Two sources:
  * ``my_reports()``  -> every report *you* have submitted (resolved, duplicate,
    informative, …). Stored as PRIVATE notes (``source: hackerone-mine``). The
    vault is gitignored, but these never leave your machine regardless.
  * ``hacktivity()``  -> the public disclosed-activity feed, filtered with a
    Lucene query string. Good for ongoing ingestion of fresh writeups.

Auth: HTTP Basic with ``H1_API_USERNAME`` / ``H1_API_TOKEN``.
Token: https://hackerone.com/settings/api_token/edit
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date

import httpx

from sift.config import get_settings
from sift.ingest.base import clean_text, extract_cwes
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

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
    body = (
        clean_text(attrs.get("vulnerability_information")) or "_(no writeup body returned by API)_"
    )
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


def my_reports() -> Iterator[Note]:
    with _client() as client:
        for item in _paginate(client, "/hackers/me/reports", {}):
            note = _my_report_to_note(item)
            if note:
                yield note


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


def hacktivity(*, query: str | None = None, limit: int | None = 500) -> Iterator[Note]:
    query = query or DEFAULT_HACKTIVITY_QUERY
    seen = 0
    with _client() as client:
        for item in _paginate(
            client,
            "/hackers/hacktivity",
            {"queryString": query, "sort": "-disclosed_at"},
        ):
            note = _hacktivity_to_note(item)
            if not note:
                continue
            yield note
            seen += 1
            if limit and seen >= limit:
                return
