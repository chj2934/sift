"""A tombstoned url blocks the same article however the next feed spells it.

The ledger matched urls by its own rule (scheme and host case, trailing slash), while
ingest decides "same article" by `vault.notes.canonical_url`, which also ignores
``www.`` and tracking parameters. So an article forgotten as
``https://www.blog.test/post/?utm_source=rss`` came back the next time a feed listed
it as ``https://blog.test/post`` - a pruned or forgotten note resurrected by a bulk
re-ingest, the thing the ledger exists to stop.
"""

from __future__ import annotations

import pytest

FORGOTTEN = "https://www.blog.test/post/?utm_source=rss&utm_medium=feed#comments"


@pytest.mark.parametrize(
    "spelling",
    [
        "https://blog.test/post",
        "http://blog.test/post/",
        "https://WWW.Blog.test/post?utm_campaign=x",
        "https://blog.test/post?fbclid=abc",
    ],
)
def test_a_forgotten_url_matches_every_spelling_of_the_article(spelling):
    from sift.tombstones import load_tombstones, record_tombstones

    record_tombstones(urls=[FORGOTTEN], reason="test")
    assert load_tombstones().has_url(spelling)


@pytest.mark.parametrize(
    "other",
    [
        "https://blog.test/other-post",
        "https://blog.test/post?p=2",  # a query that names another page
        "https://other.test/post",
    ],
)
def test_other_articles_are_not_blocked(other):
    """Control: the wider match is still one article, not its neighbours."""
    from sift.tombstones import load_tombstones, record_tombstones

    record_tombstones(urls=[FORGOTTEN], reason="test")
    assert not load_tombstones().has_url(other)


def test_the_tombstone_rule_is_the_ingest_identity_rule():
    from sift.tombstones import normalize_url
    from sift.vault.notes import canonical_url

    for url in (FORGOTTEN, "https://x.test/a?b=2&a=1", "http://[::1/x", "", None):
        assert normalize_url(url) == canonical_url(url)


def test_a_bulk_reingest_does_not_resurrect_a_forgotten_article(vault_path, monkeypatch):
    from sift.ingest import base
    from sift.tombstones import record_tombstones
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    class _Store:
        def indexed_ids(self):
            return set()

        def optimize(self):
            return {"error": None}

    monkeypatch.setattr(base, "index_notes", lambda notes, _store: len(notes))
    monkeypatch.setattr(base, "Store", _Store)

    record_tombstones(urls=[FORGOTTEN], reason="forget_note: noise")

    def note(title, url):
        meta = Frontmatter(
            id=f"research-{title.lower().replace(' ', '-')}",
            type="writeup",
            title=title,
            source="research-feed",
            url=url,
        )
        return Note(meta=meta, body="feed excerpt")

    res = base.run_source(
        "research",
        [
            note("Forgotten post", "https://blog.test/post"),
            note("New post", "https://blog.test/new"),
        ],
    )

    assert res.tombstoned == 1 and res.written == 1
    assert [p.stem for p in (vault_path / "writeup").glob("*.md")] == ["New post"]
