"""Recent security research from RSS/Atom feeds -> `writeup` notes.

The point of this source (unlike the bulk disclosed-report corpus) is *freshness*:
post-training-cutoff techniques the reasoning model can't already know. Defaults
to PortSwigger Research; add more feeds with ``SIFT_RESEARCH_FEEDS`` (comma-sep).

By default a note carries the feed's summary/excerpt plus a link. ``--fetch-body``
also pulls the article page and does a crude HTML->text extraction.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from datetime import date
from urllib.parse import urlparse

import httpx
from slugify import slugify

from sift.config import get_settings
from sift.ingest.base import clean_text
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

DEFAULT_FEEDS = ("https://portswigger.net/research/rss",)

_TAG_RE = re.compile(r"(?s)<[^>]+>")
_DROP_RE = re.compile(r"(?is)<(script|style|nav|footer|header|form)\b.*?</\1>")
_WS_RE = re.compile(r"[ \t]*\n[ \t]*")


def _feeds() -> list[str]:
    extra = [u.strip() for u in get_settings().research_feeds.split(",") if u.strip()]
    seen: dict[str, None] = {}
    for u in (*DEFAULT_FEEDS, *extra):
        seen.setdefault(u, None)
    return list(seen)


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").removeprefix("www.")


def _entry_date(entry) -> date | None:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            try:
                return date(t.tm_year, t.tm_mon, t.tm_mday)
            except ValueError:
                return None
    return None


def _html_to_text(raw: str) -> str:
    raw = _DROP_RE.sub(" ", raw)
    text = html.unescape(_TAG_RE.sub(" ", raw))
    text = _WS_RE.sub("\n", re.sub(r"[ \t]{2,}", " ", text))
    return clean_text(text)


def _fetch_body(client: httpx.Client, url: str) -> str:
    try:
        r = client.get(url)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"  ! research: body fetch failed for {url}: {exc}")
        return ""
    return _html_to_text(r.text)[:8000]


def _to_note(entry, feed_url: str, *, body_text: str = "") -> Note | None:
    title = clean_text(entry.get("title"))
    link = (entry.get("link") or "").strip()
    if not title or not link:
        return None

    summary = _html_to_text(entry.get("summary") or "")
    parts = [p for p in (body_text.strip(), summary.strip()) if p]
    body = (parts[0] if parts else "") + f"\n\n---\nSource: {link}"

    host = _host(link) or _host(feed_url)
    meta = Frontmatter(
        id=f"research-{slugify(title, max_length=90)}",
        type="writeup",
        title=title,
        source=host or "research-feed",
        url=link,
        created=_entry_date(entry),
        tags=["research", host] if host else ["research"],
    )
    return Note(meta=meta, body=body.strip())


def source(*, limit: int | None = None, fetch_body: bool = False) -> Iterator[Note]:
    import feedparser  # deferred: keeps the dep out of the hot import path

    vault = get_settings().resolved_vault()
    seen = 0
    with httpx.Client(
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": "sift-research-ingest/0.1 (personal bug-bounty memory)"},
    ) as client:
        for feed_url in _feeds():
            try:
                raw = client.get(feed_url)
                raw.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  ! research: could not fetch {feed_url}: {exc}")
                continue
            parsed = feedparser.parse(raw.content)
            for entry in parsed.entries:
                title = clean_text(entry.get("title"))
                if not title:
                    continue
                slug = slugify(f"research-{slugify(title, max_length=90)}", max_length=80)
                if (vault / "writeup" / f"{slug}.md").exists():
                    continue  # already have it — don't re-fetch the body
                body_text = ""
                if fetch_body and entry.get("link"):
                    body_text = _fetch_body(client, entry["link"])
                note = _to_note(entry, feed_url, body_text=body_text)
                if not note:
                    continue
                yield note
                seen += 1
                if limit and seen >= limit:
                    return
