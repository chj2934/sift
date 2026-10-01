"""Medium feed fallback.

Medium 403s every non-browser client. That is a deliberate access control, so the
recovery path goes through the feeds Medium publishes rather than pretending to be a
browser. Measured: 18 of 28 recent Medium posts recovered this way, lifting overall
PentesterLand coverage from 52% to 79%.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.parametrize(
    "url",
    [
        "https://medium.com/@a13h1/1500-cr-lf-injection-59152daaf413",
        "https://mikko-kenttala.medium.com/zero-click-calendar-invite-53c90d2e6f01",
        "https://infosecwriteups.com/forced-sso-session-fixation-5d3b457b79cb",
        "https://systemweakness.com/some-post-abcdef123456",
    ],
)
def test_recognises_medium_hosts(url):
    from sift.ingest.medium import is_medium

    assert is_medium(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://portswigger.net/research/something",
        "https://blog.doyensec.com/2024/07/02/cspt2csrf.html",
        "https://notmedium.com/@user/post-123456789abc",
    ],
)
def test_ignores_non_medium_hosts(url):
    from sift.ingest.medium import is_medium

    assert not is_medium(url)


def test_post_id_extraction():
    from sift.ingest.medium import post_id

    assert post_id("https://medium.com/@a13h1/1500-cr-lf-injection-59152daaf413") == "59152daaf413"
    assert post_id("https://x.medium.com/title-abc123def456/") == "abc123def456"
    assert post_id("https://medium.com/@a/no-hex-suffix") is None


def test_feed_url_for_subdomain_author():
    from sift.ingest.medium import feed_urls_for

    assert feed_urls_for("https://mikko-kenttala.medium.com/zero-click-53c90d2e6f01") == [
        "https://medium.com/feed/@mikko-kenttala"
    ]


def test_feed_url_for_at_author():
    from sift.ingest.medium import feed_urls_for

    assert feed_urls_for("https://medium.com/@a13h1/1500-cr-lf-59152daaf413") == [
        "https://medium.com/feed/@a13h1"
    ]


def test_feed_url_for_publication_host():
    from sift.ingest.medium import feed_urls_for

    assert feed_urls_for("https://infosecwriteups.com/forced-sso-5d3b457b79cb") == [
        "https://infosecwriteups.com/feed"
    ]


# --- feed matching ---------------------------------------------------------------

_FEED = """<?xml version="1.0"?><rss version="2.0"
 xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
<item><title>Other post</title><link>https://medium.com/@a/other-111111111111</link>
<content:encoded><![CDATA[<p>{other}</p>]]></content:encoded></item>
<item><title>Target post</title><link>https://medium.com/@a/target-59152daaf413</link>
<content:encoded><![CDATA[<p>{target}</p>]]></content:encoded></item>
</channel></rss>"""


class _FeedClient:
    def __init__(self, body: str, status: int = 200):
        self._body = body.encode()
        self._status = status
        self.requested: list[str] = []

    def get(self, url):
        self.requested.append(url)
        import httpx

        return httpx.Response(self._status, content=self._body, request=httpx.Request("GET", url))


def test_matches_the_right_entry_by_post_id():
    from sift.ingest.medium import fetch_via_feed

    body = _FEED.format(other="wrong article " * 60, target="the real technique " * 60)
    client = _FeedClient(body)
    text = fetch_via_feed(client, "https://medium.com/@a/target-59152daaf413")

    assert "the real technique" in text
    assert "wrong article" not in text
    assert client.requested == ["https://medium.com/feed/@a"]


def test_returns_empty_when_post_has_scrolled_out_of_the_feed():
    """The known limit: feeds carry only recent posts, so older articles are lost."""
    from sift.ingest.medium import fetch_via_feed

    body = _FEED.format(other="x " * 300, target="y " * 300)
    assert fetch_via_feed(_FeedClient(body), "https://medium.com/@a/gone-aaaaaaaaaaaa") == ""


def test_short_entries_are_rejected():
    """A teaser is not an article - better to skip than to gate on a stub."""
    from sift.ingest.medium import fetch_via_feed

    body = _FEED.format(other="x", target="too short")
    assert fetch_via_feed(_FeedClient(body), "https://medium.com/@a/target-59152daaf413") == ""


def test_feed_failure_is_not_fatal():
    from sift.ingest.medium import fetch_via_feed

    assert fetch_via_feed(_FeedClient("", status=403), "https://medium.com/@a/x-59152daaf413") == ""


def test_no_feed_candidates_for_unknown_shape():
    from sift.ingest.medium import feed_urls_for

    assert feed_urls_for("https://medium.com/") == []


def test_a_cached_feed_is_downloaded_once_per_run():
    """N posts from one author used to cost N feed downloads (plus N guaranteed 403s)."""
    from sift.ingest.medium import feed_lookup

    body = _FEED.format(other="other article " * 60, target="the real technique " * 60)
    client = _FeedClient(body)
    cache: dict = {}

    first, fetched1 = feed_lookup(client, "https://medium.com/@a/target-59152daaf413", cache)
    second, fetched2 = feed_lookup(client, "https://medium.com/@a/other-111111111111", cache)

    assert "the real technique" in first.text and "other article" in second.text
    assert first.title == "Target post"
    assert (fetched1, fetched2) == (True, False)
    assert client.requested == ["https://medium.com/feed/@a"]


def test_a_failed_feed_is_remembered_in_the_cache():
    from sift.ingest.medium import fetch_via_feed

    client = _FeedClient("", status=403)
    cache: dict = {}
    for n in range(3):
        assert fetch_via_feed(client, f"https://medium.com/@a/p{n}-59152daaf41{n}", cache) == ""
    assert client.requested == ["https://medium.com/feed/@a"]


def test_unparseable_links_are_not_medium_and_do_not_raise():
    from sift.ingest.medium import feed_urls_for, is_medium, post_id

    assert not is_medium("http://[::1/x")
    assert post_id("http://[::1/x") is None
    assert feed_urls_for("http://[::1/x") == []


def test_labels_json_fixture_still_parses():
    """Guards the eval set against an accidental rewrite."""
    from sift.config import PROJECT_ROOT

    raw = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "gate_labels.json").read_text("utf-8"))
    assert len(raw) == 40
