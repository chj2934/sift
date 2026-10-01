"""PentesterLand's curated bug bounty writeup index -> `writeup` notes.

~6,400 entries going back to 2010, with structured metadata (programs, bug classes,
bounty, publication date) that maps cleanly onto our frontmatter.

Depth is the point here, but so is restraint: pre-2024 writeups are overwhelmingly
techniques the reasoning model already knows, and every one of them costs a gate
call. `since_year` defaults to something recent for that reason - widen it
deliberately, not by accident.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date
from urllib.parse import urlparse

import httpx
from slugify import slugify

from sift.config import get_settings
from sift.ingest.base import clean_text
from sift.ingest.medium import fetch_via_feed, is_medium
from sift.ingest.research import BODY_CHARS, extract_article, extraction_failed
from sift.ingest.top10 import looks_like_prose
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

INDEX_URL = "https://pentester.land/writeups.json"
UA = "sift-research-ingest/0.1 (personal bug-bounty memory)"

# Default cutoff. Older material is mostly well-trodden and would burn gate budget.
DEFAULT_SINCE_YEAR = 2024


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").removeprefix("www.")


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        y, m, d = (int(p) for p in raw.split("-")[:3])
        return date(y, m, d)
    except (ValueError, TypeError):
        return None


def _to_note(entry: dict, body_text: str = "") -> Note | None:
    """Pure — separated from I/O so tests can exercise it against a fixture."""
    links = entry.get("Links") or []
    if not links:
        return None
    link = (links[0].get("Link") or "").strip()
    title = clean_text(links[0].get("Title"))
    if not link or not title:
        return None

    programs = [clean_text(p) for p in (entry.get("Programs") or []) if p]
    bugs = [clean_text(b) for b in (entry.get("Bugs") or []) if b]
    authors = [clean_text(a) for a in (entry.get("Authors") or []) if a]
    published = _parse_date(entry.get("PublicationDate"))

    bounty_raw = (entry.get("Bounty") or "").strip()
    bounty = None
    if bounty_raw and bounty_raw != "-":
        digits = "".join(c for c in bounty_raw if c.isdigit() or c == ".")
        try:
            bounty = float(digits) if digits else None
        except ValueError:
            bounty = None

    # The article text has to be here: the novelty gate cannot judge a technique from
    # a title and a bug-class list, and a note it can't judge is worse than no note.
    body = "\n".join(
        [
            f"**Bugs:** {', '.join(bugs)}" if bugs else "",
            f"**Programs:** {', '.join(programs)}" if programs else "",
            f"**Authors:** {', '.join(authors)}" if authors else "",
            f"**Bounty:** {bounty_raw}" if bounty_raw and bounty_raw != "-" else "",
            "",
            body_text.strip(),
            "",
            "---",
            f"Source: {link}",
        ]
    ).strip()

    host = _host(link)
    meta = Frontmatter(
        id=f"writeup-{slugify(title, max_length=90)}",
        type="writeup",
        title=title,
        source=host or "pentester.land",
        url=link,
        created=published,
        tags=sorted({"writeup", "pentesterland", *(host and [host] or [])}),
        program=programs[0] if programs else None,
        bounty=bounty,
        extra={"bug_classes": bugs, "authors": authors} if (bugs or authors) else {},
    )
    return Note(meta=meta, body=body)


def source(
    *,
    limit: int | None = None,
    since_year: int = DEFAULT_SINCE_YEAR,
    refresh: bool = False,
    delay: float = 0.8,
) -> Iterator[Note]:
    vault = get_settings().resolved_vault()
    skipped_unfetchable = 0
    recovered_via_feed = 0

    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": UA}) as client:
        try:
            r = client.get(INDEX_URL, timeout=120)
            r.raise_for_status()
            rows = r.json().get("data", [])
        except (httpx.HTTPError, ValueError) as exc:
            print(f"  ! writeups: could not fetch index: {exc}")
            return

        # Newest first, so a --limit takes the most recent rather than an arbitrary slice.
        rows.sort(key=lambda e: e.get("PublicationDate") or "", reverse=True)

        seen = 0
        for entry in rows:
            published = _parse_date(entry.get("PublicationDate"))
            if since_year and (published is None or published.year < since_year):
                continue

            stub = _to_note(entry)
            if stub is None:
                continue
            slug = slugify(stub.meta.id, max_length=80)
            if not refresh and (vault / "writeup" / f"{slug}.md").exists():
                continue

            raw_html = ""
            body_text = ""
            try:
                a = client.get(stub.meta.url)
                a.raise_for_status()
                raw_html = a.text
                body_text = extract_article(raw_html)[:BODY_CHARS]
            except httpx.HTTPError:
                # Medium 403s every non-browser client - 28 of 30 fetch failures in a
                # sample of this corpus. Rather than spoof a browser to defeat that,
                # go through the feed Medium publishes for the purpose.
                if is_medium(stub.meta.url):
                    body_text = fetch_via_feed(client, stub.meta.url)[:BODY_CHARS]
                    if body_text:
                        recovered_via_feed += 1
                if not body_text:
                    skipped_unfetchable += 1
                    continue
            finally:
                time.sleep(delay)

            # Too short, binary (PDF), or only boilerplate (JS-rendered page). The
            # ratio guard only applies when we actually fetched the page ourselves.
            if len(body_text) < 400 or not looks_like_prose(body_text):
                skipped_unfetchable += 1
                continue
            if raw_html and extraction_failed(body_text, raw_html):
                skipped_unfetchable += 1
                continue

            note = _to_note(entry, body_text=body_text)
            if note is None:
                continue
            yield note
            seen += 1
            if limit and seen >= limit:
                break

    if recovered_via_feed:
        print(f"  writeups: recovered {recovered_via_feed} Medium posts via their RSS feeds")
    if skipped_unfetchable:
        print(f"  writeups: skipped {skipped_unfetchable} entries with no fetchable body")
