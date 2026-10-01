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

# High-signal security research blogs. These publish novel technique rather than
# commentary, so they clear the novelty gate far more often than aggregators do.
# All verified live 2026-09; dead feeds (NCC, Synacktiv 403, Securitum, YesWeHack,
# Ambionics) are deliberately absent rather than left to fail silently every run.
DEFAULT_FEEDS = (
    "https://portswigger.net/research/rss",
    "https://googleprojectzero.blogspot.com/feeds/posts/default",
    "https://www.assetnote.io/resources/research/rss.xml",
    "https://blog.assetnote.io/feed.xml",
    "https://blog.doyensec.com/atom.xml",
    "https://blog.orange.tw/atom.xml",
    "https://blog.trailofbits.com/feed/",
    "https://blog.includesecurity.com/feed/",
    "https://www.sonarsource.com/blog/rss.xml",
    "https://labs.watchtowr.com/rss/",
    "https://starlabs.sg/blog/index.xml",
    "https://blog.slonser.info/index.xml",
    "https://labs.detectify.com/feed/",
    "https://rhinosecuritylabs.com/feed/",
    # Noisier aggregator, but genuine bug bounty content and the body fetch works.
    "https://blog.intigriti.com/feed/",
    # Web3 / wallet-adjacent. Added for the ENS Immunefi competition, where the KB had
    # zero technique notes covering wallets, signing or account abstraction. Chosen for
    # appsec density, not chain coverage: pure smart-contract feeds (rekt, Ethereum
    # blog, SlowMist) are a different discipline from web app hunting and were skipped.
    "https://www.zellic.io/blog/rss.xml",
    "https://medium.com/feed/immunefi",
    "https://blog.openzeppelin.com/rss.xml",
    "https://spaceraccoon.dev/feed.xml",
    "https://blog.quarkslab.com/feeds/all.rss.xml",
    # Medium. Every article fetch 403s a non-browser client, but the feed's `content`
    # block carries ~4k chars of the real post, so _to_note falls back to that rather
    # than spoofing a browser to defeat the block. Noisy (a lot of CTF/course content),
    # which is what the gate is for.
    "https://infosecwriteups.com/feed",
)

# Full article text, not an excerpt: the novelty gate has to see the actual technique
# to judge it. The old 8k cap truncated long research posts mid-technique.
BODY_CHARS = 40000

# Below this share of the raw HTML, the extractor found navigation rather than an
# article: the page is JS-rendered (SPA) or an error page. Measured on this corpus,
# real articles land at 0.06-0.37 while known failures sit at 0.002-0.009, so 0.02
# separates them with a wide margin. No server-side extractor fixes a JS-rendered
# page - trafilatura was tried and did no better - so these are skipped, not stored.
MIN_EXTRACT_RATIO = 0.02
# ...unless enough text came out anyway, which a long article inside a heavy SPA
# bundle can do.
EXTRACT_RATIO_EXEMPT_CHARS = 5000

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


def extract_article(raw: str) -> str:
    """Article text from a page, boilerplate stripped.

    trafilatura understands article structure; the regex stripper below does not, and
    leaves every navigation menu in the note ("Skip to main content Outcomes Partners
    Company Resource Hub..." opened every Assetnote note). Measured on this corpus it
    returns ~80% of the regex's character count with the nav removed, and payload
    strings survive intact.

    It is not a fix for JS-rendered pages - nothing server-side is - so `extraction_failed`
    still guards the result. The regex remains the fallback for pages trafilatura
    declines to parse, and for small HTML fragments like RSS content blocks.
    """
    try:
        import trafilatura
    except ImportError:  # pragma: no cover - dependency is declared, but degrade gracefully
        return _html_to_text(raw)

    fallback = _html_to_text(raw)
    try:
        got = trafilatura.extract(raw, include_tables=True, include_comments=False)
    except Exception:  # trafilatura raises on some malformed documents
        return fallback
    if not got:
        return fallback
    got = clean_text(got)
    # Guard against over-stripping: if it discarded most of the page, trust the regex.
    return got if len(got) >= 0.4 * len(fallback) else fallback


def extraction_failed(text: str, raw: str) -> bool:
    """True when the extractor got boilerplate instead of the article."""
    if not raw:
        return True
    if len(text) >= EXTRACT_RATIO_EXEMPT_CHARS:
        return False
    return (len(text) / len(raw)) < MIN_EXTRACT_RATIO


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
    text = extract_article(r.text)[:BODY_CHARS]
    # Same guards the other sources use: a PDF or undecodable response yields binary,
    # and a JS-rendered page yields the nav menu. Either would become note text and
    # cost a gate call while teaching nothing.
    from sift.ingest.top10 import looks_like_prose

    if text and not looks_like_prose(text):
        print(f"  ! research: body is not text, keeping excerpt only for {url}")
        return ""
    if extraction_failed(text, r.text):
        print(f"  ! research: extraction got boilerplate ({len(text)} chars), skipping body for {url}")
        return ""
    return text


def _to_note(entry, feed_url: str, *, body_text: str = "") -> Note | None:
    title = clean_text(entry.get("title"))
    link = (entry.get("link") or "").strip()
    if not title or not link:
        return None

    # Feeds carry two text fields and they are not equivalent. `content` is the full
    # (or near-full) post body; `summary` is usually a teaser. Medium in particular
    # 403s every article fetch but serves ~4k chars of real article text in the feed's
    # content block - the sanctioned channel, so use it rather than defeating the block.
    feed_text = ""
    content = entry.get("content")
    if content:
        raw = content[0].get("value", "") if isinstance(content, list) else str(content)
        feed_text = _html_to_text(raw)
    summary = _html_to_text(entry.get("summary") or "")
    if len(summary) > len(feed_text):
        feed_text = summary

    parts = [p for p in (body_text.strip(), feed_text.strip()) if p]
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


def source(
    *, limit: int | None = None, fetch_body: bool = False, refresh: bool = False
) -> Iterator[Note]:
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
                if not refresh and (vault / "writeup" / f"{slug}.md").exists():
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
