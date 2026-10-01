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
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from sift.ingest.research import _html_to_text

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


def is_medium(url: str) -> bool:
    host = (urlparse(url).hostname or "").removeprefix("www.")
    return host.endswith(".medium.com") or host in _PUBLICATION_HOSTS


def post_id(url: str) -> str | None:
    m = _POST_ID.search(urlparse(url).path)
    return m.group(1).lower() if m else None


def feed_urls_for(url: str) -> list[str]:
    """Candidate feeds that might carry this post, best first."""
    parsed = urlparse(url)
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


def fetch_via_feed(client: httpx.Client, url: str) -> str:
    """Article text for a Medium post, or "" if the feed no longer carries it."""
    import feedparser

    target = post_id(url)
    for feed_url in feed_urls_for(url):
        try:
            r = client.get(feed_url)
            r.raise_for_status()
        except httpx.HTTPError:
            continue
        parsed = feedparser.parse(r.content)
        for entry in parsed.entries:
            link = (entry.get("link") or "").strip()
            # Match on the post id when both sides have one; fall back to exact URL.
            if target and post_id(link) == target:
                pass
            elif link.split("?")[0].rstrip("/") != url.split("?")[0].rstrip("/"):
                continue
            content = entry.get("content")
            raw = (
                content[0].get("value", "")
                if isinstance(content, list) and content
                else entry.get("summary", "")
            )
            text = _html_to_text(raw)
            if len(text) >= 400:
                return text
    return ""
