"""PentesterLand's curated bug bounty writeup index -> `writeup` notes.

~6,400 entries going back to 2010, with structured metadata (programs, bug classes,
bounty, publication date) that maps cleanly onto our frontmatter.

Depth is the point here, but so is restraint: pre-2024 writeups are overwhelmingly
techniques the reasoning model already knows, and every one of them costs a gate
call. `since_year` defaults to something recent for that reason - widen it
deliberately, not by accident. ``since`` (a date) overrides it when given.

An entry already in the vault (same article URL, among writeup notes) is not fetched
or yielded again unless ``refresh``. Medium-hosted entries go straight to Medium's own
feed (one download per feed per run): every direct request to Medium is a guaranteed
403, and the feed is the sanctioned channel.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import date
from urllib.parse import urlparse

import httpx
from slugify import slugify

from sift.config import get_settings
from sift.ingest.base import KnownNotes, canonical_url, clean_text, safe_get
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

log = logging.getLogger(__name__)

INDEX_URL = "https://pentester.land/writeups.json"
UA = "sift-research-ingest/0.1 (personal bug-bounty memory)"

# Default cutoff. Older material is mostly well-trodden and would burn gate budget.
DEFAULT_SINCE_YEAR = 2024


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").removeprefix("www.")
    except ValueError:  # "http://[::1/x": one malformed link must not end the run
        return ""


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


def _index_rows(client: httpx.Client) -> list[dict] | None:
    r = safe_get(client, INDEX_URL, what="writeups index", timeout=120)
    if r is None:
        return None
    try:
        data = r.json()
    except ValueError as exc:
        log.warning("writeups: index is not JSON: %s", type(exc).__name__)
        return None
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        log.warning("writeups: index has no 'data' list")
        return None
    return [r for r in rows if isinstance(r, dict)]


def source(
    *,
    limit: int | None = None,
    since_year: int = DEFAULT_SINCE_YEAR,
    refresh: bool = False,
    delay: float = 0.8,
    since: date | None = None,
) -> Iterator[Note]:
    """Writeups published since ``since`` (or Jan 1 of ``since_year``) that are not in
    the vault yet. ``limit`` counts notes yielded (new ones, unless ``refresh``)."""
    from sift.ingest.article import fetch_article

    vault = get_settings().resolved_vault()
    known = KnownNotes(vault)
    feed_cache: dict = {}
    skipped_unfetchable = 0
    recovered_via_feed = 0

    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": UA}) as client:
        rows = _index_rows(client)
        if rows is None:
            return

        # Newest first, so a --limit takes the most recent rather than an arbitrary slice.
        rows.sort(key=lambda e: str(e.get("PublicationDate") or ""), reverse=True)

        seen_urls: set[str] = set()
        yielded = 0
        for entry in rows:
            published = _parse_date(entry.get("PublicationDate"))
            if since is not None:
                if published is None or published < since:
                    continue
            elif since_year and (published is None or published.year < since_year):
                continue

            stub = _to_note(entry)
            if stub is None:
                continue
            cu = canonical_url(stub.meta.url)
            if cu in seen_urls:
                continue
            seen_urls.add(cu)
            if not refresh and known.has(stub.meta):
                continue  # already have it - don't fetch it again

            # Too short, binary (PDF), only boilerplate (JS-rendered page), or a Medium
            # post its feed no longer carries: all skipped, never stored as a stub.
            res = fetch_article(client, stub.meta.url, feed_cache=feed_cache)
            if res.fetched and delay:
                time.sleep(delay)
            if not res.ok:
                skipped_unfetchable += 1
                log.debug("writeups: skipping %s (%s)", stub.meta.url, res.reason)
                continue
            if res.via == "medium-feed":
                recovered_via_feed += 1

            note = _to_note(entry, body_text=res.text)
            if note is None:
                continue
            yield note
            known.add(note.meta)
            yielded += 1
            if limit and yielded >= limit:
                break

    if recovered_via_feed:
        log.info("writeups: recovered %d Medium posts via their RSS feeds", recovered_via_feed)
    if skipped_unfetchable:
        log.info("writeups: skipped %d entries with no fetchable body", skipped_unfetchable)
