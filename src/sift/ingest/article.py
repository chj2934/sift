"""Fetch one article page and decide whether its text is worth storing.

research, writeups, top10 and single-URL capture each used to carry their own copy of
this sequence, and the copies drifted: research had no length floor and no Medium
handling, top10 printed, writeups slept after requests it never made. Every rule in
CLAUDE.md's "things that silently corrupt the corpus" that concerns one page lives here
now:

* **Compression** - httpx decodes ``br`` and ``zstd`` because brotli and zstandard are
  hard dependencies; nothing here asks for an encoding it cannot decode.
* **Extraction** - `research.extract_article` (trafilatura, regex fallback).
* **Binary** - `top10.looks_like_prose`, script-agnostic on purpose.
* **JS-rendered pages** - `research.extraction_failed` (body under 2% of the HTML).
* **Medium** - only through Medium's own feeds (`medium.feed_lookup`). There is no
  direct request to a Medium host, and no browser spoofing to get past its 403.

`fetch_article` never raises for a page that cannot be had: it returns a
`FetchResult` whose ``reason`` says why, and the caller decides whether that is a log
line, a counter or an error for the user. Nothing here prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

from sift.ingest.base import FETCH_ERRORS

# Shorter than this, a "full article" fetch got a stub, a cookie wall or an error page.
MIN_ARTICLE_CHARS = 400
# Decoded (decompressed) bytes read from one page before giving up. Real articles are
# well under 1 MB; the cap is what keeps a zip bomb or a video link from filling RAM.
DEFAULT_MAX_BYTES = 10_000_000

# Not an article whatever it says: refused before the body is read.
_BINARY_TYPES = (
    "application/pdf",
    "application/zip",
    "application/octet-stream",
    "application/x-",
    "image/",
    "audio/",
    "video/",
    "font/",
)


@dataclass(frozen=True)
class FetchResult:
    """What one article fetch produced. ``reason`` is None when ``text`` is usable."""

    url: str
    text: str = ""
    html: str = ""
    reason: str | None = None
    via: str = "direct"  # "direct" or "medium-feed"
    fetched: bool = False  # a network request was made (callers rate-limit on this)
    final_url: str = ""  # after redirects
    title: str = ""  # the feed's title, for a Medium post
    published: date | None = None  # the feed's date, for a Medium post

    @property
    def ok(self) -> bool:
        return self.reason is None


class _Rejected(Exception):
    """The response is not worth reading (binary type, too large)."""


def check_article_text(
    text: str, html: str = "", *, min_chars: int = MIN_ARTICLE_CHARS
) -> str | None:
    """Why `text` is not worth storing, or None. Pure.

    In order: nothing or too little text, binary (a PDF or an undecoded response),
    boilerplate (a JS-rendered page). The ratio test only applies when `html` is the
    page we fetched ourselves; a feed's content block has no page to compare with.
    """
    from sift.ingest.research import extraction_failed
    from sift.ingest.top10 import looks_like_prose

    if not text:
        return "no text extracted"
    if len(text) < min_chars:
        return "too short"
    if not looks_like_prose(text):
        return "binary/not text"
    if html and extraction_failed(text, html):
        return "js-rendered or boilerplate"
    return None


def http_reason(exc: BaseException) -> str:
    """A short, content-free reason for a failed request ("http 403", "timeout")."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.TooManyRedirects):
        return "too many redirects"
    if isinstance(exc, (httpx.InvalidURL, httpx.UnsupportedProtocol, ValueError)):
        return "invalid url"
    return f"fetch failed ({type(exc).__name__})"


def _download(client: httpx.Client, url: str, *, max_bytes: int) -> tuple[str, str]:
    """(page text, final url). Streams so an oversized body is abandoned early."""
    with client.stream("GET", url) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype.startswith(_BINARY_TYPES):
            raise _Rejected("binary/not text")
        declared = r.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_bytes:
            raise _Rejected("too large")
        buf = bytearray()
        for chunk in r.iter_bytes():  # decompressed: the cap applies to what we keep
            buf += chunk
            if len(buf) > max_bytes:
                raise _Rejected("too large")
        encoding = r.charset_encoding or "utf-8"
        try:
            page = buf.decode(encoding, errors="replace")
        except LookupError:  # a charset Python does not know
            page = buf.decode("utf-8", errors="replace")
        return page, str(r.url)


def fetch_article(
    client: httpx.Client,
    url: str,
    *,
    min_chars: int = MIN_ARTICLE_CHARS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    feed_cache: dict | None = None,
    medium_via_feed: bool = True,
    max_chars: int | None = None,
) -> FetchResult:
    """Fetch `url` and extract its article text, applying every corpus guard.

    Medium URLs go to Medium's feeds only (`medium_via_feed=False` makes the caller
    responsible for not requesting them at all). `feed_cache` is a per-run
    `medium.FeedCache`. Text is cut to `max_chars` (default `research.BODY_CHARS`)
    before the checks, as the batch sources always did.
    """
    from sift.ingest.medium import feed_lookup, is_medium
    from sift.ingest.research import BODY_CHARS, extract_article

    limit = BODY_CHARS if max_chars is None else max_chars
    if medium_via_feed and is_medium(url):
        post, fetched = feed_lookup(client, url, feed_cache)
        if post is None:
            return FetchResult(
                url, reason="medium post not in author feed", via="medium-feed", fetched=fetched
            )
        text = post.text[:limit]
        return FetchResult(
            url,
            text=text,
            reason=check_article_text(text, min_chars=min_chars),
            via="medium-feed",
            fetched=fetched,
            final_url=post.link,
            title=post.title,
            published=post.published,
        )
    try:
        page, final_url = _download(client, url, max_bytes=max_bytes)
    except _Rejected as exc:
        return FetchResult(url, reason=str(exc), fetched=True)
    except FETCH_ERRORS as exc:
        return FetchResult(url, reason=http_reason(exc), fetched=True)
    text = extract_article(page)[:limit]
    return FetchResult(
        url,
        text=text,
        html=page,
        reason=check_article_text(text, page, min_chars=min_chars),
        fetched=True,
        final_url=final_url,
    )
