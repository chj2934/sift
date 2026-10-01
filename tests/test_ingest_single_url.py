"""`fetch_url_note`: one article, the batch sources' rules, and an SSRF guard.

Offline: a MockTransport serves pages and `_resolve` is patched, so no DNS either.
"""

from __future__ import annotations

import pytest

PUBLIC = "93.184.216.34"  # a globally routable address


def _page(title="Cookie sandwich: reading HttpOnly cookies", words=60, date="2026-08-01"):
    para = "<p>" + "The quoted cookie value swallows the next cookie header. " * words + "</p>"
    return (
        f"<html><head><title>{title}</title>"
        f'<meta property="article:published_time" content="{date}T10:00:00Z"></head>'
        f"<body><article><h1>{title}</h1>{para}</article></body></html>"
    )


@pytest.fixture
def dns(monkeypatch):
    """host -> addresses; anything unknown resolves to a public address."""
    from sift.ingest import single_url

    table = {"localhost": ["127.0.0.1"], "metadata.test": ["169.254.169.254"]}

    def resolve(host, port):
        if host in table:
            return table[host]
        if host.replace(".", "").isdigit() or ":" in host:
            return [host]
        return [PUBLIC]

    monkeypatch.setattr(single_url, "_resolve", resolve)
    return table


def _transport(routes, seen=None):
    import httpx

    def handler(request):
        url = str(request.url)
        if seen is not None:
            seen.append(url)
        status, body, headers = routes.get(url, (404, "nope", {}))
        content = body if isinstance(body, bytes) else body.encode("utf-8")
        return httpx.Response(
            status, content=content, headers={"content-type": "text/html", **headers}
        )

    return httpx.MockTransport(handler)


def test_a_good_article_becomes_an_unsaved_writeup(vault_path, dns):
    from sift.ingest.single_url import fetch_url_note

    url = "https://blog.test/cookie-sandwich"
    note = fetch_url_note(url, transport=_transport({url: (200, _page(), {})}))

    assert note.meta.type == "writeup" and note.meta.url == url
    assert note.meta.id == "writeup-cookie-sandwich-reading-httponly-cookies"
    assert note.meta.title == "Cookie sandwich: reading HttpOnly cookies"
    assert note.meta.created.isoformat() == "2026-08-01"
    assert note.meta.extra["captured_via"] == "single-url"
    assert "swallows the next cookie" in note.body and note.body.endswith(f"Source: {url}")
    assert note.path is None and not list(vault_path.rglob("*.md")), "nothing is saved"


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://127.0.0.1/admin", "non-public"),
        ("http://localhost:8080/", "non-public"),
        ("http://169.254.169.254/latest/meta-data/", "non-public"),
        ("http://metadata.test/", "non-public"),
        ("http://10.0.0.5/", "non-public"),
        ("http://[::ffff:127.0.0.1]/", "non-public"),
        ("file:///etc/passwd", "http(s)"),
        ("ftp://blog.test/x", "http(s)"),
        ("https://user:pw@blog.test/x", "credentials"),
    ],
)
def test_internal_and_non_http_urls_are_refused_before_any_request(dns, url, reason):
    from sift.ingest.single_url import CaptureError, fetch_url_note

    seen: list[str] = []
    with pytest.raises(CaptureError) as err:
        fetch_url_note(url, transport=_transport({}, seen))
    assert reason in err.value.reason
    assert seen == []


def test_a_redirect_to_an_internal_address_is_refused(dns):
    from sift.ingest.single_url import CaptureError, fetch_url_note

    url = "https://blog.test/innocent"
    seen: list[str] = []
    routes = {url: (302, "", {"location": "http://169.254.169.254/latest/meta-data/"})}
    with pytest.raises(CaptureError) as err:
        fetch_url_note(url, transport=_transport(routes, seen))
    assert "non-public" in err.value.reason
    assert seen == [url], "the internal hop was never requested"


@pytest.mark.parametrize(
    ("body", "headers", "reason"),
    [
        (_page(words=2), {}, "too short"),
        (b"%PDF-1.7 binary", {"content-type": "application/pdf"}, "binary"),
        (
            "<html><body><nav>Home About</nav>"
            + "<script>"
            + "x" * 300_000
            + "</script>"
            + "<p>"
            + "short bit of prose here. " * 20
            + "</p></body></html>",
            {},
            "js-rendered",
        ),
        ("x" * 6_000_000, {}, "too large"),
    ],
    ids=["stub", "pdf", "spa", "huge"],
)
def test_pages_that_are_not_articles_are_refused(dns, body, headers, reason):
    from sift.ingest.single_url import CaptureError, fetch_url_note

    url = "https://blog.test/page"
    with pytest.raises(CaptureError) as err:
        fetch_url_note(url, transport=_transport({url: (200, body, headers)}))
    assert reason in err.value.reason


def test_http_errors_are_reported_not_raised_raw(dns):
    from sift.ingest.single_url import CaptureError, fetch_url_note

    url = "https://blog.test/gone"
    with pytest.raises(CaptureError) as err:
        fetch_url_note(url, transport=_transport({url: (403, "no", {})}))
    assert err.value.reason == "http 403"


def test_medium_goes_through_the_feed_only(dns):
    from sift.ingest.single_url import CaptureError, fetch_url_note

    url = "https://medium.com/@a/target-59152daaf413"
    feed = (
        '<rss version="2.0"><channel><item><title>Target post</title>'
        f"<link>{url}</link><pubDate>Sat, 01 Aug 2026 00:00:00 GMT</pubDate>"
        "<description><![CDATA[<p>" + "the real technique " * 40 + "</p>]]></description>"
        "</item></channel></rss>"
    )
    seen: list[str] = []
    note = fetch_url_note(
        url, transport=_transport({"https://medium.com/feed/@a": (200, feed, {})}, seen)
    )

    assert note.meta.title == "Target post" and note.meta.created.isoformat() == "2026-08-01"
    assert note.meta.extra["fetched_via"] == "medium-feed"
    assert seen == ["https://medium.com/feed/@a"], "the article itself is never requested"

    with pytest.raises(CaptureError) as err:
        fetch_url_note("https://medium.com/@a/other-111111111111", transport=_transport({}))
    assert err.value.reason == "medium post not in author feed"


def test_pre_cutoff_and_dedupe_helpers(vault_path, dns, monkeypatch):
    from sift.config import get_settings
    from sift.ingest.single_url import existing_ids, fetch_url_note, is_pre_cutoff, resolve_id
    from sift.vault.notes import save_note

    monkeypatch.setenv("SIFT_MODEL_CUTOFF", "2026-04-01")
    get_settings.cache_clear()

    old_url = "https://blog.test/old"
    old = fetch_url_note(
        old_url, transport=_transport({old_url: (200, _page(date="2025-12-12"), {})})
    )
    assert is_pre_cutoff(old)

    url = "https://blog.test/cookie-sandwich"
    note = fetch_url_note(url, transport=_transport({url: (200, _page(), {})}))
    assert not is_pre_cutoff(note) and existing_ids(url) == ()
    save_note(vault_path, note)
    assert existing_ids("http://www.blog.test/cookie-sandwich/?utm_source=x") == (note.meta.id,)

    other_url = "https://elsewhere.test/cookie-sandwich"
    other = fetch_url_note(other_url, transport=_transport({other_url: (200, _page(), {})}))
    assert other.meta.id == note.meta.id  # same title
    assert resolve_id(other) != note.meta.id, "a different article gets its own id"


def test_a_failed_capture_writes_nothing_to_stdout(dns, capfd):
    from sift.ingest.single_url import CaptureError, fetch_url_note

    with pytest.raises(CaptureError):
        fetch_url_note("https://blog.test/x", transport=_transport({}))
    assert capfd.readouterr().out == ""
