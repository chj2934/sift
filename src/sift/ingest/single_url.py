"""Capture one article URL as a `writeup` note, without saving it.

The batch sources (research, writeups, top10) are the only way an article reaches the
vault with its verbatim text, published date and URL; an agent that reads a fresh
writeup mid-session could otherwise only paraphrase it through `remember`.
`fetch_url_note` applies the same rules as the batch sources (`sift.ingest.article`):
trafilatura extraction, the binary and JS-rendered-page guards, br/zstd decoding, and
Medium only through Medium's own feeds - no browser spoofing.

It is reachable from the MCP server with a URL the agent chose, and in a bug-bounty
session the agent reads attacker-controlled pages that can steer it at internal
addresses. So the fetch is refused for anything but public http(s) hosts - on the
first request *and on every redirect hop* - and the body is capped. (The hostname is
resolved for the check and again by the connection, so a DNS answer that changes in
between is not caught; the cap and the http(s)-only rule still hold.)

What a caller does with the note - see `capture`-style tools - is up to it:

* `existing_ids(url)` first: the article may already be in the vault (any source);
* `is_pre_cutoff(note)`: SIFT_MODEL_CUTOFF says the model already knows it;
* `resolve_id(note)` before saving: a different article may hold its title-derived id;
* then `vault.notes.write_note` and `pipeline.index_note`.

Nothing here prints; a refusal or a failed fetch raises `CaptureError` with a short,
content-free reason.
"""

from __future__ import annotations

import ipaddress
import socket
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse, urlsplit

import httpx
from slugify import slugify

from sift.config import get_settings
from sift.ingest.article import FetchResult, fetch_article
from sift.ingest.base import KnownNotes, clean_text
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

# Honest about what it is: sift does not pretend to be a browser (CLAUDE.md, Medium).
UA = "sift-capture/0.1 (personal bug-bounty memory)"
DEFAULT_TIMEOUT = 20.0
MAX_BYTES = 5_000_000
MAX_REDIRECTS = 5


class CaptureError(Exception):
    """The URL was refused or yielded no storable article. ``reason`` is short and never
    quotes page content. Deliberately not a ValueError, so no fetch helper's
    error handling can swallow a refusal raised on a redirect hop."""

    def __init__(self, reason: str, url: str = "") -> None:
        self.reason = reason
        self.url = url
        super().__init__(f"{reason}: {url}" if url else reason)


def _resolve(host: str, port: int) -> list[str]:
    """Every address `host` resolves to (patched in tests: no network)."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def blocked_reason(url: str) -> str | None:
    """Why `url` must not be fetched, or None.

    Only http(s), no credentials in the URL, and every address the host resolves to
    must be globally routable: loopback, private, link-local (169.254.169.254, the
    cloud metadata service), CGNAT, multicast, reserved and unspecified addresses are
    refused, including IPv4-mapped IPv6 forms of them.
    """
    try:
        parts = urlsplit(str(url or "").strip())
        scheme = parts.scheme.lower()
        host = parts.hostname
        port = parts.port
    except ValueError:
        return "invalid url"
    if scheme not in ("http", "https"):
        return "only http(s) urls can be captured"
    if not host:
        return "url has no host"
    if parts.username or parts.password:
        return "urls with credentials are not fetched"
    try:
        addrs = _resolve(host, port or (443 if scheme == "https" else 80))
    except (OSError, UnicodeError):
        return "host does not resolve"
    if not addrs:
        return "host does not resolve"
    for raw in addrs:
        try:
            ip = ipaddress.ip_address(raw.split("%", 1)[0])
        except ValueError:
            return "host resolves to an unparseable address"
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            return f"refusing non-public address {ip}"
    return None


def _guard_request(request: httpx.Request) -> None:
    """httpx request hook: runs for the first request and for every redirect."""
    why = blocked_reason(str(request.url))
    if why:
        raise CaptureError(why, str(request.url))


def _client(*, timeout: float, transport: httpx.BaseTransport | None, guard: bool) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        max_redirects=MAX_REDIRECTS,
        headers={"User-Agent": UA},
        event_hooks={"request": [_guard_request]} if guard else {},
        transport=transport,
    )


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def article_note(url: str, res: FetchResult) -> Note:
    """The writeup note for a successful fetch. Pure (no I/O)."""
    from sift.ingest.top10 import _article_title, find_article_date

    host = _host(url)
    if res.via == "medium-feed":
        title = res.title or host
        published = res.published
    else:
        title = _article_title(res.html, "", url)
        published = find_article_date(res.html, max_year=datetime.now(UTC).year)
    title = clean_text(title)[:180] or host or "captured article"
    extra: dict = {"captured_via": "single-url", "fetched_via": res.via}
    if res.final_url and res.final_url != url:
        extra["final_url"] = res.final_url
    meta = Frontmatter(
        # The batch writeup scheme, so `KnownNotes` and later batch runs treat this as
        # the same article (run `resolve_id` before saving).
        id=f"writeup-{slugify(title, max_length=90)}",
        type="writeup",
        title=title,
        source=host or "web",
        url=url,
        created=published,
        tags=sorted({"writeup", *([host] if host else [])}),
        extra=extra,
    )
    body = (res.text.strip() + f"\n\n---\nSource: {url}").strip()
    return Note(meta=meta, body=body)


def fetch_url_note(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.BaseTransport | None = None,
    guard: bool = True,
) -> Note:
    """Fetch and extract the article at `url` and return it as an unsaved `writeup`.

    Raises `CaptureError` when the URL is refused (not public http(s), here or on a
    redirect hop), the fetch fails ("http 403", "timeout"), or the page is not a
    storable article ("too short", "binary/not text", "js-rendered or boilerplate",
    "medium post not in author feed"). Never stores a stub or a link-only note.
    `transport` is for tests; `guard=False` only for trusted callers (tests, a CLI
    the user drives).
    """
    url = str(url or "").strip()
    if guard:
        why = blocked_reason(url)
        if why:
            raise CaptureError(why, url)
    with _client(timeout=timeout, transport=transport, guard=guard) as client:
        res = fetch_article(client, url, max_bytes=MAX_BYTES)
    if not res.ok:
        raise CaptureError(res.reason or "fetch failed", url)
    return article_note(url, res)


def is_pre_cutoff(note: Note) -> bool:
    """True when the article is dated before SIFT_MODEL_CUTOFF, i.e. the reasoning
    model trained on it (CLAUDE.md). Undated articles are not pre-cutoff."""
    created = note.meta.created
    return created is not None and created < get_settings().model_cutoff


def existing_ids(url: str, *, vault: Path | None = None) -> tuple[str, ...]:
    """Ids of notes (any source) that already hold this article, by canonical URL."""
    return KnownNotes(vault).ids_for_url(url)


def resolve_id(note: Note, *, vault: Path | None = None) -> str:
    """The id `note` should be saved under: its own, an existing writeup's for the
    same URL, or a URL-hashed one when a different article holds its title id."""
    return KnownNotes(vault).resolve_id(note.meta)
