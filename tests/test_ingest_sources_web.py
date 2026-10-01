"""research / writeups / top10 against a mock web.

Covers the "already have it" checks (they probed `<slug>.md` after filenames became
titles, never matched, and re-fetched and re-embedded everything on every run), the
excerpt guard, the model-cutoff horizon, malformed links, Medium via its feed, and
two different articles that share a title.

No network: every `httpx.Client` a source builds is routed through a MockTransport.
Index writes are stubbed; the vault is the conftest temp dir.
"""

from __future__ import annotations

import json

import pytest

FEED = "https://feed.test/rss"


class Web:
    """URL -> canned response, plus a log of what was requested."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, object, dict]] = {}
        self.requested: list[str] = []

    def route(self, url: str, body: object = "", status: int = 200, **headers: str) -> None:
        self.routes[url] = (status, body, headers)

    def handler(self, request):
        import httpx

        url = str(request.url)
        self.requested.append(url)
        hit = self.routes.get(url)
        if hit is None:
            return httpx.Response(404, text="not found")
        status, body, headers = hit
        content = body.encode("utf-8") if isinstance(body, str) else body
        return httpx.Response(
            status,
            content=content,
            headers={"content-type": "text/html; charset=utf-8", **headers},
        )

    def count(self, url: str) -> int:
        return sum(1 for u in self.requested if u == url)


@pytest.fixture
def web(monkeypatch):
    import httpx

    w = Web()
    real = httpx.Client

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(w.handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    return w


@pytest.fixture
def index(monkeypatch):
    """Stub the index: run_source's writes are what is under test."""
    from sift.ingest import base

    state: dict = {"batches": []}

    class _Store:
        def indexed_ids(self):
            return {i for batch in state["batches"] for i in batch}

        def optimize(self):
            return {"error": None}

    store = _Store()

    def fake_index(notes, _store):
        state["batches"].append([n.meta.id for n in notes])
        return len(notes)

    monkeypatch.setattr(base, "index_notes", fake_index)
    monkeypatch.setattr(base, "Store", lambda: store)
    return state


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setenv("SIFT_MODEL_CUTOFF", "2026-04-01")
    monkeypatch.setenv("SIFT_RESEARCH_FEEDS", "")
    from sift.config import get_settings

    get_settings.cache_clear()


def _article(title: str, word: str = "smuggling") -> str:
    para = f"<p>This {word} technique abuses a parser differential in the proxy layer. " * 6
    return (
        f"<html><head><title>{title}</title>"
        '<meta property="article:published_time" content="2026-08-01T10:00:00Z"></head>'
        f"<body><article><h1>{title}</h1>{para * 3}</article></body></html>"
    )


def _rss(*items: tuple[str, str, str | None, str]) -> str:
    out = []
    for title, link, when, desc in items:
        date_tag = f"<pubDate>{when}</pubDate>" if when else ""
        out.append(
            f"<item><title>{title}</title><link>{link}</link>{date_tag}"
            f"<description><![CDATA[{desc}]]></description></item>"
        )
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>{"".join(out)}</channel></rss>'


AUG = "Sat, 01 Aug 2026 00:00:00 GMT"


def _files(vault_path, folder="writeup"):
    return sorted((vault_path / folder).glob("*.md"))


def _research(monkeypatch):
    from sift.ingest import research

    monkeypatch.setattr(research, "_feeds", lambda: [FEED])
    return research


# --- research ---------------------------------------------------------------------


def test_research_second_run_fetches_and_writes_nothing(web, index, monkeypatch, vault_path):
    from sift.ingest.base import run_source

    research = _research(monkeypatch)
    post = "https://blog.test/desync"
    web.route(FEED, _rss(("Desync attacks", post, AUG, "teaser")))
    web.route(post, _article("Desync attacks"))

    first = run_source("research", research.source(fetch_body=True))
    second = run_source("research", research.source(fetch_body=True))

    assert first.written == 1
    assert second.seen == 0 and second.written == 0 and second.updated == 0
    assert web.count(post) == 1, "the article was fetched again"
    assert len(_files(vault_path)) == 1


def test_a_refresh_without_body_never_replaces_the_full_text(web, index, monkeypatch, vault_path):
    from sift.ingest.base import run_source
    from sift.vault.notes import load_note

    research = _research(monkeypatch)
    post = "https://blog.test/desync"
    web.route(FEED, _rss(("Desync attacks", post, AUG, "short teaser")))
    web.route(post, _article("Desync attacks"))
    run_source("research", research.source(fetch_body=True))
    full = load_note(_files(vault_path)[0]).body
    assert len(full) > 1000

    res = run_source("research", research.source(refresh=True))

    assert load_note(_files(vault_path)[0]).body == full
    assert res.written == 0 and len(_files(vault_path)) == 1


def test_a_refresh_whose_body_fetch_fails_keeps_the_full_text(web, index, monkeypatch, vault_path):
    from sift.ingest.base import run_source
    from sift.vault.notes import load_note

    research = _research(monkeypatch)
    post = "https://blog.test/desync"
    web.route(FEED, _rss(("Desync attacks", post, AUG, "short teaser")))
    web.route(post, _article("Desync attacks"))
    run_source("research", research.source(fetch_body=True))
    full = load_note(_files(vault_path)[0]).body

    web.route(post, "upstream down", status=500)
    run_source("research", research.source(refresh=True, fetch_body=True))

    assert load_note(_files(vault_path)[0]).body == full


def test_research_skips_pre_cutoff_entries_but_keeps_undated_ones(web, index, monkeypatch):
    from datetime import date

    research = _research(monkeypatch)
    web.route(
        FEED,
        _rss(
            ("Old post", "https://blog.test/old", "Fri, 12 Dec 2025 00:00:00 GMT", "x"),
            ("New post", "https://blog.test/new", AUG, "y"),
            ("Undated post", "https://blog.test/undated", None, "z"),
        ),
    )

    titles = [n.meta.title for n in research.source()]
    assert titles == ["New post", "Undated post"]

    deeper = [n.meta.title for n in research.source(since=date(2025, 1, 1))]
    assert deeper == ["Old post", "New post", "Undated post"]


def test_query_identified_posts_are_distinct_articles(web, index, monkeypatch, vault_path):
    """WordPress serves posts as /?p=123: dropping the query would make the second
    post look like the first and skip it forever."""
    from sift.ingest.base import run_source

    research = _research(monkeypatch)
    web.route(
        FEED,
        _rss(
            ("Release notes", "https://blog.test/?p=1", AUG, "first post body"),
            ("Release notes", "https://blog.test/?p=2", AUG, "second post body"),
        ),
    )

    res = run_source("research", research.source())

    files = _files(vault_path)
    assert res.written == 2 and res.collisions == 1 and len(files) == 2
    again = run_source("research", research.source())
    assert again.seen == 0 and len(_files(vault_path)) == 2


def test_medium_links_are_never_fetched_for_a_body(web, index, monkeypatch):
    research = _research(monkeypatch)
    post = "https://medium.com/@a/post-59152daaf413"
    web.route(FEED, _rss(("Medium post", post, AUG, "the feed's own copy " * 30)))

    notes = list(research.source(fetch_body=True))

    assert len(notes) == 1 and "the feed's own copy" in notes[0].body
    assert web.count(post) == 0


# --- same title, different articles (title-derived ids) ----------------------------


def test_same_title_research_articles_both_survive_with_stable_ids(
    web, index, monkeypatch, vault_path
):
    from sift.ingest.base import run_source
    from sift.vault.notes import load_note

    research = _research(monkeypatch)
    a, b = "https://vendor-a.test/advisory", "https://vendor-b.test/advisory"
    web.route(FEED, _rss(("Security Advisory", a, AUG, "vendor A text")))
    run_source("research", research.source())

    # B appears later: it must not be skipped as "already have it", nor overwrite A.
    web.route(
        FEED,
        _rss(("Security Advisory", a, AUG, "vendor A text"), ("Security Advisory", b, AUG, "B")),
    )
    res = run_source("research", research.source())
    assert res.written == 1 and res.collisions == 1

    notes = {load_note(p).meta.url: load_note(p).meta.id for p in _files(vault_path)}
    assert notes[a] == "research-security-advisory", "the first article keeps the legacy id"
    assert notes[b].startswith("research-security-advisory-") and len(notes[b]) <= 78

    again = run_source("research", research.source(refresh=True))
    after = {load_note(p).meta.url: load_note(p).meta.id for p in _files(vault_path)}
    assert after == notes and again.written == 0


def test_a_tracking_variant_of_a_stored_url_is_the_same_article(
    web, index, monkeypatch, vault_path
):
    from sift.ingest.base import run_source

    research = _research(monkeypatch)
    web.route(FEED, _rss(("Desync attacks", "https://blog.test/desync", AUG, "x")))
    run_source("research", research.source())

    web.route(
        FEED,
        _rss(("Desync attacks", "http://www.blog.test/desync/?utm_source=rss#top", AUG, "x")),
    )
    res = run_source("research", research.source())

    assert res.seen == 0 and len(_files(vault_path)) == 1


def test_same_title_writeups_get_distinct_ids(vault_path, index):
    """Through the real `_to_note` id scheme, not hand-built ids."""
    from sift.ingest.base import run_source
    from sift.ingest.writeups import _to_note

    def entry(link):
        return {
            "Links": [{"Title": "Account Takeover", "Link": link}],
            "PublicationDate": "2026-08-01",
        }

    first = _to_note(entry("https://a.test/ato"), body_text="first writeup " * 50)
    second = _to_note(entry("https://b.test/ato"), body_text="second writeup " * 50)
    assert first.meta.id == second.meta.id  # the bug: the title is the only identity

    res = run_source("writeups", [first, second])

    files = _files(vault_path)
    assert res.written == 2 and res.collisions == 1 and len(files) == 2


# --- writeups -----------------------------------------------------------------------


def _pl_index(*rows: tuple[str, str, str]) -> str:
    data = [
        {"Links": [{"Title": t, "Link": link}], "PublicationDate": d, "Bugs": ["XSS"]}
        for t, link, d in rows
    ]
    return json.dumps({"data": data})


def test_writeups_second_run_fetches_nothing(web, index, vault_path):
    from sift.ingest.base import run_source
    from sift.ingest.writeups import INDEX_URL, source

    post = "https://blog.test/xss"
    web.route(
        INDEX_URL,
        _pl_index(("Stored XSS in X", post, "2026-08-01")),
        **{"content-type": "application/json"},
    )
    web.route(post, _article("Stored XSS in X", "xss"))

    first = run_source("writeups", source(delay=0))
    second = run_source("writeups", source(delay=0))

    assert first.written == 1 and second.seen == 0
    assert web.count(post) == 1


def test_one_malformed_link_does_not_abort_the_run(web, index, vault_path):
    """An IDNA-invalid host or an unparseable link used to escape the generator, leave
    saved notes unindexed and recur on every run at the same entry."""
    from sift.ingest.base import load_state, run_source
    from sift.ingest.writeups import INDEX_URL, source

    good1, good2 = "https://blog.test/one", "https://blog.test/two"
    web.route(
        INDEX_URL,
        _pl_index(
            ("Good one", good1, "2026-08-04"),
            ("Emoji host", "https://\U0001f4a9.la/p", "2026-08-03"),
            ("Broken link", "http://[::1/x", "2026-08-02"),
            ("Good two", good2, "2026-08-01"),
        ),
    )
    web.route(good1, _article("Good one"))
    web.route(good2, _article("Good two"))

    res = run_source("writeups", source(delay=0))

    assert res.complete and res.written == 2
    assert sorted(i for b in index["batches"] for i in b) == [
        "writeup-good-one",
        "writeup-good-two",
    ]
    assert load_state()["writeups"]["complete"] is True


def test_medium_posts_use_one_feed_request_and_no_article_requests(web, index):
    from sift.ingest.writeups import INDEX_URL, source

    posts = [f"https://medium.com/@a/post-{n}{'0' * 11}" for n in range(1, 4)]
    feed_items = "".join(
        f"<item><title>Post {n}</title><link>{u}</link>"
        f"<description><![CDATA[<p>{'medium article text ' * 40}</p>]]></description></item>"
        for n, u in enumerate(posts, 1)
    )
    web.route(
        "https://medium.com/feed/@a", f'<rss version="2.0"><channel>{feed_items}</channel></rss>'
    )
    web.route(
        INDEX_URL, _pl_index(*[(f"Post {n}", u, "2026-08-01") for n, u in enumerate(posts, 1)])
    )

    notes = list(source(delay=0))

    assert len(notes) == 3
    assert web.count("https://medium.com/feed/@a") == 1
    assert not any(web.count(u) for u in posts), "Medium articles 403; only the feed is used"


def test_a_failing_medium_feed_is_requested_once_per_run(web, index):
    from sift.ingest.writeups import INDEX_URL, source

    posts = [f"https://medium.com/@b/post-{n}{'0' * 11}" for n in range(1, 4)]
    web.route("https://medium.com/feed/@b", "forbidden", status=403)
    web.route(
        INDEX_URL, _pl_index(*[(f"Post {n}", u, "2026-08-01") for n, u in enumerate(posts, 1)])
    )

    assert list(source(delay=0)) == []
    assert web.count("https://medium.com/feed/@b") == 1


# --- top10 --------------------------------------------------------------------------


def test_top10_second_run_fetches_no_article(web, index, vault_path):
    from sift.ingest import top10
    from sift.ingest.base import run_source

    art = "https://research.test/fontleak"
    page = top10.BASE.format(year=2025)
    # The anchor text differs from the page <title>: the skip must not depend on it.
    web.route(page, f'<a href="{art}">an earlier post</a>')
    web.route(art, _article("Fontleak: stealing text with CSS fonts"))

    first = run_source("top10", top10.source(years=(2025,), delay=0))
    second = run_source("top10", top10.source(years=(2025,), delay=0))

    assert first.written == 1 and second.seen == 0
    assert web.count(art) == 1


def test_top10_still_records_an_article_another_source_holds(web, index, vault_path):
    """Cross-source dedup would lose the nomination tags; only top10 notes count."""
    from sift.ingest import top10
    from sift.ingest.base import run_source
    from sift.ingest.research import _to_note

    art = "https://research.test/fontleak"
    entry = {"title": "Fontleak", "link": art, "summary": "x" * 50}
    run_source("research", [_to_note(entry, FEED)])
    web.route(top10.BASE.format(year=2025), f'<a href="{art}">Fontleak deep dive</a>')
    web.route(art, _article("Fontleak: stealing text with CSS fonts"))

    notes = list(top10.source(years=(2025,), delay=0))
    assert len(notes) == 1 and "top10-2025" in notes[0].meta.tags


def test_top10_skips_unparseable_links_without_aborting(web, index):
    from sift.ingest import top10

    art = "https://research.test/real"
    web.route(
        top10.BASE.format(year=2025),
        f'<a href="http://[::1/x">broken</a><a href="{art}">A real research post</a>',
    )
    web.route(art, _article("A real research post"))

    notes = list(top10.source(years=(2025,), delay=0))
    assert [n.meta.url for n in notes] == [art]


def test_sources_print_nothing(web, index, monkeypatch, capfd):
    """C1: diagnostics go through logging (stderr), never stdout."""
    from sift.ingest.base import run_source
    from sift.ingest.writeups import INDEX_URL, source

    research = _research(monkeypatch)
    web.route(FEED, _rss(("Post", "https://blog.test/p", AUG, "x")))
    web.route("https://blog.test/p", "down", status=500)
    web.route(INDEX_URL, _pl_index(("Gone", "https://blog.test/gone", "2026-08-01")))

    run_source("research", research.source(fetch_body=True))
    run_source("writeups", source(delay=0))

    assert capfd.readouterr().out == ""
