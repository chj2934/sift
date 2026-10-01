"""Mine PortSwigger's "Top 10 Web Hacking Techniques" pages for the research they link.

The index posts themselves are worthless as technique notes - they get dropped by the
prefilter. Their value is the *links*: each year's nomination list is a community-
curated set of research already filtered for "novel, practical, re-applicable", which
is almost exactly the novelty gate's own criterion. That makes this the highest
pass-rate discovery source available, far better than scraping aggregators.

Fetches each linked article and stores it as a `writeup` note for the gate to judge.
"""

from __future__ import annotations

import html
import re
import time
from collections.abc import Iterator
from datetime import date
from urllib.parse import urlparse

import httpx
from slugify import slugify

from sift.config import get_settings
from sift.ingest.base import clean_text
from sift.ingest.research import BODY_CHARS, extract_article, extraction_failed
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

UA = "sift-research-ingest/0.1 (personal bug-bounty memory)"
BASE = "https://portswigger.net/research/top-10-web-hacking-techniques-of-{year}"

# The project runs annually from 2006, but pre-2021 winners are canon the reasoning
# model knows cold - ingesting them burns gate budget to confirm what it already knows.
DEFAULT_YEARS = (2021, 2022, 2023, 2024, 2025)

# Social, sharing, forms and video - never the research itself. PortSwigger's own
# domain is excluded because those articles arrive via the research RSS feed instead.
_DENY_HOSTS = frozenset(
    {
        "portswigger.net", "twitter.com", "x.com", "t.co", "bsky.app", "linkedin.com",
        "docs.google.com", "forms.gle", "youtube.com", "youtu.be", "api.whatsapp.com",
        "infosec.exchange", "mastodon.social", "discord.com", "discord.gg",
        "reddit.com", "facebook.com", "news.ycombinator.com", "web.archive.org",
    }
)

# href plus its anchor text - the anchor is usually the research's real title.
_ANCHOR_RE = re.compile(
    r"<a\b[^>]*href=[\"'](https?://[^\"'\s>]+)[\"'][^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"(?s)<[^>]+>")
_WS_RE = re.compile(r"\s{2,}")


def _collapse_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def looks_like_prose(text: str) -> bool:
    """Reject binary masquerading as text.

    Two sources of binary end up here: PDFs linked from nomination lists, and - until
    brotli/zstandard were added as dependencies - undecoded compressed responses.

    Deliberately script-agnostic. An earlier ASCII-word version scored a Japanese
    writeup at 0.167 and would have discarded it, and payload-dense articles are the
    ones most worth keeping, so this counts *character classes* rather than English
    words: real text in any script is almost entirely letters, digits, spaces and
    punctuation, while binary is full of control and unassigned code points.
    """
    if not text:
        return False
    sample = text[:20000]
    good = sum(1 for c in sample if c.isalnum() or c.isspace() or c in "-_.,:;!?'\"()[]{}/<>@#$%&*+=|\\~`^")
    return (good / len(sample)) > 0.90


# Publication date, in descending order of trustworthiness. Without this, every note
# got 1 Jan of its nomination year - so a February 2020 article was stamped 2024 and
# scored as recent by SIFT_RECENCY_WEIGHT.
_DATE_PATTERNS = (
    re.compile(r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r'<meta[^>]+name=["\'](?:date|pubdate|publish[-_]?date)["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r'["\']datePublished["\']\s*:\s*["\']([^"\']+)', re.I),
    re.compile(r'<time[^>]+datetime=["\']([^"\']+)', re.I),
)
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").removeprefix("www.")


def article_date(page_html: str, fallback_year: int) -> date:
    """Real publication date if the page exposes one, else 1 Jan of the nomination year."""
    for pattern in _DATE_PATTERNS:
        for raw in pattern.findall(page_html)[:3]:
            m = _ISO_DATE.search(raw)
            if not m:
                continue
            try:
                found = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
            # Sanity-bound it: the web is full of stray dates in unrelated markup.
            if 1995 <= found.year <= fallback_year + 1:
                return found
    return date(fallback_year, 1, 1)


# Author bio / index pages linked from nomination lists - real sites, no technique.
_NON_ARTICLE_PATH = re.compile(
    r"^/?(about|home|index|archive|tags?|categories|authors?|contact|feed|rss)/?$",
    re.IGNORECASE,
)


def _is_denied_host(host: str) -> bool:
    """Match the domain and any subdomain - `uk.linkedin.com` must not slip past a
    `linkedin.com` entry."""
    return any(host == d or host.endswith("." + d) for d in _DENY_HOSTS)


def _is_research_link(url: str) -> bool:
    host = _host(url)
    if not host or _is_denied_host(host):
        return False
    path = urlparse(url).path or "/"
    if _NON_ARTICLE_PATH.match(path):
        return False
    # github.com is source, not prose - repos are PoCs and blob URLs are code files
    # (a linked `solution.py` is not a writeup). gist.github.com and *.github.io are
    # separate hosts and stay allowed, since those often are the writeup itself.
    return host != "github.com"


def _index_urls(years: tuple[int, ...]) -> list[str]:
    out = []
    for y in years:
        out.append(BASE.format(year=y))
        out.append(BASE.format(year=y) + "-nominations-open")
    return out


def _extract_links(page_html: str) -> dict[str, str]:
    """url -> anchor text. Pure, so tests can run it on a fixture."""
    found: dict[str, str] = {}
    for url, inner in _ANCHOR_RE.findall(page_html):
        if not _is_research_link(url):
            continue
        # Tags become spaces, so inline markup ("CVE-1234 <b>deep dive</b>") would
        # otherwise leave doubled spaces in the title.
        text = _collapse_ws(clean_text(html.unescape(_TAG_RE.sub(" ", inner))))
        # Longest anchor text wins - the same URL often appears as both a bare link
        # and a titled one.
        if len(text) > len(found.get(url, "")):
            found[url] = text
    return found


# Nomination lists link research from inside sentences ("an earlier post",
# "already inspired a follow-up"). The target is real research, but the anchor is
# prose, not a title - detectable because it opens with a lowercase word.
_PROSE_ANCHOR = re.compile(r"^[a-z]")


def _article_title(page_html: str, fallback: str, url: str) -> str:
    """The page's own <title> beats anchor text almost always - anchors are prose as
    often as they are titles. The anchor is only a fallback."""
    m = _TITLE_RE.search(page_html)
    if m:
        title = _collapse_ws(clean_text(html.unescape(_TAG_RE.sub(" ", m.group(1)))))
        if len(title) > 10:
            return title[:180]
    if fallback and len(fallback) > 12 and not _PROSE_ANCHOR.match(fallback):
        return fallback
    return fallback or _host(url)


def _to_note(
    url: str, title: str, body_text: str, year: int, published: date | None = None
) -> Note | None:
    title = clean_text(title)
    if not title or not url:
        return None
    host = _host(url)
    body = (body_text.strip() + f"\n\n---\nSource: {url}").strip()
    meta = Frontmatter(
        id=f"top10-{slugify(title, max_length=90)}",
        type="writeup",
        title=title,
        source=host or "top10",
        url=url,
        created=published or date(year, 1, 1),
        tags=sorted({"writeup", "top10", f"top10-{year}", *(host and [host] or [])}),
        extra={"discovered_via": "portswigger-top10", "nomination_year": year},
    )
    return Note(meta=meta, body=body)


def source(
    *,
    years: tuple[int, ...] = DEFAULT_YEARS,
    limit: int | None = None,
    delay: float = 1.0,
    refresh: bool = False,
) -> Iterator[Note]:
    vault = get_settings().resolved_vault()
    seen_urls: set[str] = set()
    written = 0

    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": UA}) as client:
        for year in years:
            links: dict[str, str] = {}
            for index_url in _index_urls((year,)):
                try:
                    r = client.get(index_url)
                    if r.status_code != 200:
                        continue
                except httpx.HTTPError as exc:
                    print(f"  ! top10: {index_url}: {exc}")
                    continue
                for url, text in _extract_links(r.text).items():
                    if len(text) > len(links.get(url, "")):
                        links[url] = text
                time.sleep(delay)

            print(f"  top10 {year}: {len(links)} research links")

            for url, anchor in links.items():
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                probe_title = anchor or _host(url)
                slug = slugify(f"top10-{slugify(probe_title, max_length=90)}", max_length=80)
                if not refresh and (vault / "writeup" / f"{slug}.md").exists():
                    continue

                try:
                    a = client.get(url)
                    a.raise_for_status()
                except httpx.HTTPError as exc:
                    print(f"  ! top10: {url}: {exc}")
                    continue
                finally:
                    time.sleep(delay)

                body_text = extract_article(a.text)[:BODY_CHARS]
                if len(body_text) < 400:
                    continue
                if not looks_like_prose(body_text):  # PDF or other binary
                    print(f"  ! top10: not text, skipping {url}")
                    continue
                if extraction_failed(body_text, a.text):  # JS-rendered or error page
                    print(f"  ! top10: only boilerplate extracted, skipping {url}")
                    continue

                note = _to_note(
                    url,
                    _article_title(a.text, anchor, url),
                    body_text,
                    year,
                    published=article_date(a.text, year),
                )
                if note is None:
                    continue
                yield note
                written += 1
                if limit and written >= limit:
                    return
