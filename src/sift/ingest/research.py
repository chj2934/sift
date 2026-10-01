"""Recent security research from RSS/Atom feeds -> `writeup` notes.

The point of this source (unlike the bulk disclosed-report corpus) is *freshness*:
post-training-cutoff techniques the reasoning model can't already know. Defaults
to PortSwigger Research; add more feeds with ``SIFT_RESEARCH_FEEDS`` (comma-sep).

The horizon defaults to ``SIFT_MODEL_CUTOFF`` (CLAUDE.md): an entry published before
it is something the model trained on, so it is skipped before any fetch. Undated
entries are kept. Pass an older ``since`` to go deeper.

By default a note carries the feed's summary/excerpt plus a link. ``--fetch-body``
also pulls the article page and extracts its text (`sift.ingest.article`).

An entry already in the vault (same article URL, among research notes) is not fetched
or yielded again unless ``refresh``. A refresh never replaces a longer stored body with
a shorter one - a feed teaser, or the excerpt left after a failed body fetch - only the
frontmatter is refreshed.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
from slugify import slugify

from sift.config import get_settings
from sift.ingest.base import KnownNotes, canonical_url, clean_text, id_family, safe_get
from sift.vault.notes import Note, load_note, locate_note
from sift.vault.schema import Frontmatter

log = logging.getLogger(__name__)

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
    try:
        return (urlparse(url).hostname or "").removeprefix("www.")
    except ValueError:  # "http://[::1/x": one malformed link must not end the run
        return ""


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
    """The article's full text, or "" (the note then keeps the feed excerpt).

    Same guards as every other source (`sift.ingest.article`): a PDF or undecodable
    response yields binary, and a JS-rendered page yields the nav menu. Either would
    become note text and cost a gate call while teaching nothing. Unlike writeups,
    there is no length floor: a short body still beats no body here, because the
    excerpt is the fallback rather than nothing.
    """
    from sift.ingest.article import fetch_article  # deferred: article imports this module

    res = fetch_article(client, url, min_chars=1, medium_via_feed=False)
    if res.reason:
        log.warning("research: no body for %s (%s); keeping the feed excerpt", url, res.reason)
        return ""
    return res.text


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


def keep_longer_stored_body(vault: Path, known: KnownNotes, note: Note) -> bool:
    """Give `note` the stored body of the same article when that one is longer.

    A refresh re-yields notes already in the vault; without this, the feed teaser (or
    the excerpt left after a failed `--fetch-body`) silently replaced a full article
    - 7,575 chars became 88 in a measured case. Only the body is kept; frontmatter is
    still refreshed. Returns whether the stored body was kept.
    """
    cu = canonical_url(note.meta.url)
    if not cu:
        return False
    for note_id in known.ids_for_url(note.meta.url, family=id_family(note.meta.id)):
        path = locate_note(vault, note_id)
        if path is None:
            continue
        try:
            stored = load_note(path)
        except Exception as exc:  # noqa: BLE001 - unreadable: run_source decides what to do
            log.debug("research: could not read %s: %s", path, exc)
            continue
        if canonical_url(stored.meta.url) != cu:
            continue
        if len(stored.body.strip()) > len(note.body.strip()):
            note.body = stored.body
            return True
        return False
    return False


def source(
    *,
    limit: int | None = None,
    fetch_body: bool = False,
    refresh: bool = False,
    since: date | None = None,
) -> Iterator[Note]:
    """Feed entries published on/after ``since`` (default ``SIFT_MODEL_CUTOFF``).

    ``limit`` counts notes yielded, i.e. new ones unless ``refresh``. With the
    horizon in place, ``refresh`` only reaches entries newer than it: pass an older
    ``since`` too to upgrade pre-cutoff excerpts.
    """
    import feedparser  # deferred: keeps the dep out of the hot import path

    from sift.ingest.medium import is_medium

    settings = get_settings()
    vault = settings.resolved_vault()
    horizon = since or settings.model_cutoff
    known = KnownNotes(vault)
    seen_urls: set[str] = set()  # one article carried by two feeds in this run
    yielded = 0
    old = 0
    with httpx.Client(
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": "sift-research-ingest/0.1 (personal bug-bounty memory)"},
    ) as client:
        for feed_url in _feeds():
            raw = safe_get(client, feed_url, what="research feed")
            if raw is None:
                continue
            parsed = feedparser.parse(raw.content)
            for entry in parsed.entries:
                stub = _to_note(entry, feed_url)  # no network: id, title and url
                if stub is None:
                    continue
                published = _entry_date(entry)
                if published is not None and published < horizon:
                    old += 1  # pre-cutoff: the model already knows it
                    continue
                cu = canonical_url(stub.meta.url)
                if cu in seen_urls:
                    continue
                seen_urls.add(cu)
                if not refresh and known.has(stub.meta):
                    continue  # already have it - don't re-fetch the body
                body_text = ""
                # Medium 403s every article fetch; the feed's content block is the
                # sanctioned copy and `_to_note` already uses it.
                if fetch_body and not is_medium(stub.meta.url):
                    body_text = _fetch_body(client, stub.meta.url)
                note = _to_note(entry, feed_url, body_text=body_text) if body_text else stub
                if note is None:
                    continue
                if refresh:
                    keep_longer_stored_body(vault, known, note)
                yield note
                known.add(note.meta)
                yielded += 1
                if limit and yielded >= limit:
                    return
    if old:
        log.info("research: skipped %d entries published before %s", old, horizon.isoformat())
