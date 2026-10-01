"""Recover Medium-hosted articles through Medium's own RSS feeds.

Medium returns 403 to every non-browser HTTP client, and that is a deliberate access
control, so this does not spoof a browser to get around it. Instead it uses the feed
Medium publishes for exactly this purpose - the same channel already used for
infosecwriteups.com - which serves the full post body.

Measured on this corpus: 42% of recent PentesterLand links fail, and 28 of 30 failures
are Medium. Feeds return complete articles (8-10k chars), not excerpts.

**Known limit:** a Medium feed carries only the author's ~10 most recent posts. This
recovers current material reliably and older archives not at all. It is a fix for
ongoing ingestion, not a backfill.

A run that meets many posts from one publication should pass a `FeedCache` (one dict
per run, never module-global): each feed is then downloaded and parsed once, and a
feed that failed is not asked again.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse

import httpx

from sift.ingest.base import clean_text, safe_get
from sift.ingest.research import _entry_date, _html_to_text

log = logging.getLogger(__name__)

# Medium publication domains that are not *.medium.com. Their feed is /feed on the
# same host.
_PUBLICATION_HOSTS = frozenset(
    {
        "infosecwriteups.com",
        "medium.com",
        "betterprogramming.pub",
        "levelup.gitconnected.com",
        "systemweakness.com",
        "osintteam.blog",
        "cybersecuritywriteups.com",
    }
)

# Medium post URLs end in a 8-16 char hex id: /some-title-slug-1a3436277906
_POST_ID = re.compile(r"-([0-9a-f]{8,16})/?$", re.IGNORECASE)
# https://medium.com/@author/slug  or  https://author.medium.com/slug
_AT_AUTHOR = re.compile(r"^/@([^/]+)/")

# A feed entry shorter than this is a teaser, not the article: better to skip than to
# gate on a stub.
MIN_FEED_CHARS = 400


@dataclass(frozen=True)
class FeedPost:
    """One post as a Medium feed carries it."""

    link: str
    title: str
    text: str
    published: date | None


# feed url -> lookup key -> post. An empty dict records a feed that could not be
# fetched, so it is not requested again in the same run.
FeedCache = dict[str, dict[str, FeedPost]]


def _parse(url: str):
    """urlparse, or None for a link it rejects ("http://[::1/x")."""
    try:
        parsed = urlparse(url)
        parsed.hostname  # noqa: B018 - raises for a malformed netloc
    except ValueError:
        return None
    return parsed


def is_medium(url: str) -> bool:
    parsed = _parse(url)
    host = ((parsed.hostname if parsed else "") or "").removeprefix("www.")
    return host.endswith(".medium.com") or host in _PUBLICATION_HOSTS


def post_id(url: str) -> str | None:
    parsed = _parse(url)
    m = _POST_ID.search(parsed.path) if parsed else None
    return m.group(1).lower() if m else None


def feed_urls_for(url: str) -> list[str]:
    """Candidate feeds that might carry this post, best first."""
    parsed = _parse(url)
    if parsed is None:
        return []
    host = (parsed.hostname or "").removeprefix("www.")
    out: list[str] = []

    if host.endswith(".medium.com"):
        author = host.removesuffix(".medium.com")
        out.append(f"https://medium.com/feed/@{author}")
    elif host == "medium.com":
        m = _AT_AUTHOR.match(parsed.path)
        if m:
            out.append(f"https://medium.com/feed/@{m.group(1)}")
        else:
            # medium.com/<publication>/<slug>
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2:
                out.append(f"https://medium.com/feed/{parts[0]}")
    elif host in _PUBLICATION_HOSTS:
        out.append(f"https://{host}/feed")

    return out


def _url_key(url: str) -> str:
    return "url:" + url.strip().split("#")[0].split("?")[0].rstrip("/")


def _parse_feed(content: bytes) -> dict[str, FeedPost]:
    """Lookup table for one feed: each usable post under its post id and its URL.
    Pure, so tests can run it on a fixture."""
    import feedparser

    posts: dict[str, FeedPost] = {}
    for entry in feedparser.parse(content).entries:
        link = (entry.get("link") or "").strip()
        if not link:
            continue
        content_block = entry.get("content")
        raw = (
            content_block[0].get("value", "")
            if isinstance(content_block, list) and content_block
            else entry.get("summary", "")
        )
        text = _html_to_text(raw)
        if len(text) < MIN_FEED_CHARS:
            continue
        post = FeedPost(link, clean_text(entry.get("title")), text, _entry_date(entry))
        pid = post_id(link)
        if pid:
            posts.setdefault(f"id:{pid}", post)
        posts.setdefault(_url_key(link), post)
    return posts


def feed_lookup(
    client: httpx.Client, url: str, cache: FeedCache | None = None
) -> tuple[FeedPost | None, bool]:
    """(the post, whether a network request was made). The post is None when no
    candidate feed carries it any more.

    Matches on the post id when both sides have one, else on the URL without query
    or fragment. A feed that fails (403, 404, timeout, a link httpx cannot parse) is
    skipped, and remembered in `cache` so it is not requested again.
    """
    target = post_id(url)
    fetched = False
    for feed_url in feed_urls_for(url):
        if cache is not None and feed_url in cache:
            posts = cache[feed_url]
        else:
            fetched = True
            r = safe_get(client, feed_url, what="medium feed")
            posts = _parse_feed(r.content) if r is not None else {}
            if cache is not None:
                cache[feed_url] = posts
        post = (posts.get(f"id:{target}") if target else None) or posts.get(_url_key(url))
        if post is not None:
            return post, fetched
    return None, fetched


def fetch_via_feed(client: httpx.Client, url: str, cache: FeedCache | None = None) -> str:
    """Article text for a Medium post, or "" if the feed no longer carries it."""
    post, _fetched = feed_lookup(client, url, cache)
    return post.text if post is not None else ""
