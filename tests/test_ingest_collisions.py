"""Filename collisions must not silently lose notes.

Filenames are note titles, so two notes sharing a title want the same file. Before
this guard the second save overwrote the first while `written` counted both: a
research ingest reported "492 new notes, 0 errors" and left 488 files on disk.
`save_note` now resolves the clash to `Title (2).md`, Obsidian's own convention.

The first four tests use URL-derived ids, so they guard only the filename half. The
real research / writeups / top10 sources derive the id from the title alone, which
defeated that guard: two articles sharing a title shared an id, and the second was
"the same note" and overwrote the first with collisions=0. The tests further down
build notes through the sources' own `_to_note`, so the fixtures cannot drift from
the production id scheme again.
"""

from __future__ import annotations


def _note(title: str, url: str, body: str = "body text here, long enough to index"):
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    return Note(
        # id derives from the URL so two notes can legitimately share a title;
        # same id means same note, which save_note rightly treats as a rewrite.
        meta=Frontmatter(
            id=f"research-{url.rsplit('/', 1)[-1]}",
            type="writeup",
            title=title,
            url=url,
            source="example.com",
        ),
        body=body,
    )


# Same title, different notes - the realistic clash now that filenames are titles.
SHARED_TITLE = "Advisory: Pre-Auth RCE"


def test_colliding_titles_both_survive(vault_path, monkeypatch):
    from sift.ingest import base

    # Index writes need a real embedder; the disk behaviour is what's under test.
    monkeypatch.setattr(base, "index_notes", lambda notes, store: len(notes))
    monkeypatch.setattr(base, "Store", lambda: object())

    notes = [
        _note(SHARED_TITLE, "https://example.com/a"),
        _note(SHARED_TITLE, "https://example.com/b"),
    ]
    # Precondition: these really do collide, otherwise the test proves nothing.
    from sift.vault.notes import note_path

    assert note_path(vault_path, notes[0].meta) == note_path(vault_path, notes[1].meta)

    res = base.run_source("test", notes, reindex_fts=False)

    on_disk = list((vault_path / "writeup").glob("*.md"))
    assert res.written == 2
    assert res.collisions == 1
    assert len(on_disk) == 2, "a collision silently overwrote a note"


def test_written_count_matches_files_on_disk(vault_path, monkeypatch):
    """The property that actually broke: reported writes must equal real files."""
    from sift.ingest import base

    monkeypatch.setattr(base, "index_notes", lambda notes, store: len(notes))
    monkeypatch.setattr(base, "Store", lambda: object())

    notes = [_note(SHARED_TITLE, f"https://example.com/{i}") for i in range(5)]
    res = base.run_source("test", notes, reindex_fts=False)

    on_disk = list((vault_path / "writeup").glob("*.md"))
    assert len(on_disk) == res.written == 5
    assert res.collisions == 4


def test_distinct_titles_do_not_report_collisions(vault_path, monkeypatch):
    from sift.ingest import base

    monkeypatch.setattr(base, "index_notes", lambda notes, store: len(notes))
    monkeypatch.setattr(base, "Store", lambda: object())

    notes = [
        _note("Cookie sandwich", "https://a.tld/1"),
        _note("Desync endgame", "https://a.tld/2"),
    ]
    res = base.run_source("test", notes, reindex_fts=False)

    assert res.collisions == 0
    assert len(list((vault_path / "writeup").glob("*.md"))) == 2


def test_rerun_reuses_the_same_files(vault_path, monkeypatch):
    """A note keeps its file across runs: save_note reuses a path already holding the
    same id, so re-ingesting does not multiply copies."""
    from sift.ingest import base

    monkeypatch.setattr(base, "index_notes", lambda notes, store: len(notes))
    monkeypatch.setattr(base, "Store", lambda: object())

    for _ in range(2):
        base.run_source(
            "test",
            [
                _note(SHARED_TITLE, "https://example.com/a"),
                _note(SHARED_TITLE, "https://example.com/b"),
            ],
            reindex_fts=False,
        )

    assert len(list((vault_path / "writeup").glob("*.md"))) == 2


# --------------------------------------------------------------------------------
# Production ids: research / writeups / top10 derive the id from the title alone
# --------------------------------------------------------------------------------


def _stub_index(monkeypatch) -> list[list[str]]:
    """Stub out embedding and the store; returns the ids of each indexed batch."""
    from sift.ingest import base

    batches: list[list[str]] = []

    def fake_index(notes, store):
        batches.append([n.meta.id for n in notes])
        return len(notes)

    monkeypatch.setattr(base, "index_notes", fake_index)
    monkeypatch.setattr(base, "Store", lambda: object())
    return batches


def _research(title: str, link: str, text: str = "Feed text of the post, long enough."):
    from sift.ingest import research

    entry = {"title": title, "link": link, "summary": text}
    note = research._to_note(entry, "https://feed.tld/rss")
    assert note is not None
    return note


def _writeup(title: str, link: str):
    from sift.ingest import writeups

    entry = {"Links": [{"Link": link, "Title": title}], "Bugs": ["IDOR"], "Authors": ["someone"]}
    note = writeups._to_note(entry, body_text="Article text " * 20)
    assert note is not None
    return note


def _top10(title: str, link: str):
    from sift.ingest import top10

    note = top10._to_note(link, title, "Article text " * 40, 2025)
    assert note is not None
    return note


def _files(vault_path):
    from sift.vault.notes import load_note

    return {p.name: load_note(p) for p in sorted((vault_path / "writeup").rglob("*.md"))}


def test_same_title_research_articles_from_two_vendors_both_survive(vault_path, monkeypatch):
    from sift.ingest import base
    from sift.ingest.base import url_note_id

    _stub_index(monkeypatch)
    a = _research("Security Advisory", "https://vendor-a.tld/advisories/1", "Vendor A text.")
    b = _research("Security Advisory", "https://vendor-b.tld/advisories/7", "Vendor B text.")
    # Precondition: the production scheme really gives both articles one id.
    assert a.meta.id == b.meta.id == "research-security-advisory"

    res = base.run_source("research", [a, b], reindex_fts=False)

    files = _files(vault_path)
    assert len(files) == 2, "the second article overwrote the first"
    assert res.written == 2 and res.collisions == 1 and res.id_conflicts == 0
    by_url = {n.meta.url: n for n in files.values()}
    # The first article keeps the legacy id; the second gets a stable URL-hashed one.
    assert by_url["https://vendor-a.tld/advisories/1"].meta.id == "research-security-advisory"
    hashed = by_url["https://vendor-b.tld/advisories/7"].meta.id
    assert hashed == url_note_id("research-security-advisory", "https://vendor-b.tld/advisories/7")
    assert "Vendor A text." in by_url["https://vendor-a.tld/advisories/1"].body
    assert "Vendor B text." in by_url["https://vendor-b.tld/advisories/7"].body


def test_one_blog_reusing_a_title_is_not_one_document(vault_path, monkeypatch):
    """Same host, same title, different URL: `same_document` alone calls that one
    document (the titles match), so the ingest layer must split it by URL."""
    from sift.ingest import base

    _stub_index(monkeypatch)
    jan = _research("Release notes", "https://blog.tld/2026/01/release-notes", "January.")
    feb = _research("Release notes", "https://blog.tld/2026/02/release-notes", "February.")
    assert jan.meta.id == feb.meta.id and jan.meta.source == feb.meta.source

    res = base.run_source("research", [jan, feb], reindex_fts=False)

    bodies = sorted(n.body for n in _files(vault_path).values())
    assert len(bodies) == 2 and res.collisions == 1
    assert any("January." in b for b in bodies) and any("February." in b for b in bodies)


def test_rerun_keeps_both_files_and_ids_stable_in_any_order(vault_path, monkeypatch):
    from sift.ingest import base

    batches = _stub_index(monkeypatch)
    links = ("https://vendor-a.tld/advisories/1", "https://vendor-b.tld/advisories/7")

    first_run = [_research("Security Advisory", u) for u in links]
    base.run_source("research", first_run, reindex_fts=False)
    first = {n.meta.url: n.meta.id for n in _files(vault_path).values()}

    # Reversed: B arrives first, with the bare title id that A holds.
    again = base.run_source(
        "research", [_research("Security Advisory", u) for u in reversed(links)], reindex_fts=False
    )

    assert {n.meta.url: n.meta.id for n in _files(vault_path).values()} == first
    assert again.written == 0 and again.updated == 0 and again.unchanged == 2
    assert again.collisions == 0, "a known clash is not a new collision"
    assert len(batches) == 1, "an unchanged re-run re-embedded notes"


def test_url_variants_resolve_to_the_existing_note(vault_path, monkeypatch):
    from sift.ingest import base

    _stub_index(monkeypatch)
    first = _research("Cookie sandwich", "https://blog.tld/post/42")
    base.run_source("research", [first], reindex_fts=False)
    for variant in (
        "https://blog.tld/post/42/",
        "http://www.blog.tld/post/42?utm_source=rss&utm_medium=feed",
        "https://blog.tld/post/42#comments",
    ):
        res = base.run_source(
            "research", [_research("Cookie sandwich", variant)], reindex_fts=False
        )
        assert res.written == 0 and res.collisions == 0, variant
    files = _files(vault_path)
    assert len(files) == 1
    assert [n.meta.id for n in files.values()] == ["research-cookie-sandwich"]


def test_query_parameters_that_name_the_article_keep_articles_apart(vault_path, monkeypatch):
    """WordPress serves posts as `/?p=123`: dropping the query would merge them."""
    from sift.ingest import base

    _stub_index(monkeypatch)
    notes = [_research("Weekly roundup", f"https://news.tld/?p={n}") for n in (1, 2)]
    res = base.run_source("research", notes, reindex_fts=False)
    assert len(_files(vault_path)) == 2 and res.written == 2


def test_titles_slugifying_to_one_id_across_runs_get_distinct_ids(vault_path, monkeypatch):
    """'Release Notes!' and 'Release Notes?' are different filenames but one id. On
    the second run nothing run-local remembers the first, so the clash must be seen
    from the vault."""
    from sift.ingest import base

    _stub_index(monkeypatch)
    a = _research("Release Notes!", "https://a.tld/notes")
    b = _research("Release Notes?", "https://b.tld/notes")
    assert a.meta.id == b.meta.id

    base.run_source("research", [a], reindex_fts=False)
    res = base.run_source("research", [b], reindex_fts=False)

    ids = sorted(n.meta.id for n in _files(vault_path).values())
    assert len(ids) == 2 and ids[0] != ids[1] and res.collisions == 1


def test_a_stale_resolver_is_backed_by_the_on_disk_check(vault_path, monkeypatch):
    """Even if the URL index misses the clash (another process wrote the first article
    a moment ago), the file found by id is checked for its URL before it is reused."""
    from sift.ingest import base

    _stub_index(monkeypatch)
    first = _research("Release notes", "https://blog.tld/1")
    base.run_source("research", [first], reindex_fts=False)
    monkeypatch.setattr(base.KnownNotes, "_resolve", lambda self, meta: (meta.id, ""))

    second = _research("Release notes", "https://blog.tld/2")
    res = base.run_source("research", [second], reindex_fts=False)

    assert len(_files(vault_path)) == 2 and res.written == 1 and res.collisions == 1


def test_writeups_and_top10_ids_are_disambiguated_too(vault_path, monkeypatch):
    from sift.ingest import base

    _stub_index(monkeypatch)
    w = [_writeup("Account Takeover", f"https://author{i}.tld/ato") for i in (1, 2)]
    t = [_top10("Account Takeover via OAuth", f"https://research{i}.tld/oauth") for i in (1, 2)]
    assert w[0].meta.id == w[1].meta.id and t[0].meta.id == t[1].meta.id

    res_w = base.run_source("writeups", w, reindex_fts=False)
    res_t = base.run_source("top10", t, reindex_fts=False)

    ids = [n.meta.id for n in _files(vault_path).values()]
    assert len(ids) == 4 and len(set(ids)) == 4
    assert res_w.collisions == 1 and res_t.collisions == 1


def test_a_note_without_url_is_still_rewritten_in_place(vault_path, monkeypatch):
    """Remember-style notes carry no URL: there is nothing to split them by, so a
    title-derived id keeps today's upsert-by-id behaviour."""
    from sift.ingest import base
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    _stub_index(monkeypatch)

    def note(body):
        meta = Frontmatter(id="research-own-notes", type="writeup", title="Own notes", source="me")
        return Note(meta=meta, body=body)

    base.run_source("test", [note("first version")], reindex_fts=False)
    res = base.run_source("test", [note("second version")], reindex_fts=False)

    files = _files(vault_path)
    assert res.updated == 1 and res.written == 0 and len(files) == 1
    assert "second version" in next(iter(files.values())).body


def test_skip_check_does_not_hide_a_same_title_article(vault_path, monkeypatch):
    """The source-side "already have it?" check: an id match alone must not count."""
    from sift.ingest import base

    _stub_index(monkeypatch)
    a = _research("Security Advisory", "https://vendor-a.tld/advisories/1")
    b = _research("Security Advisory", "https://vendor-b.tld/advisories/7")
    base.run_source("research", [a], reindex_fts=False)

    known = base.KnownNotes(vault_path)
    assert known.has(_research("Security Advisory", "https://vendor-a.tld/advisories/1").meta)
    assert known.has(_research("Security Advisory", "http://vendor-a.tld/advisories/1/").meta)
    assert not known.has(b.meta), "B would be skipped forever as already ingested"
    assert not base.have_note(vault_path, b.meta)
    assert base.have_note(vault_path, a.meta)


def test_an_existing_same_id_pair_updates_the_right_copy(vault_path, monkeypatch):
    """The real vault already holds writeup pairs sharing one title-derived id (written
    before upsert-by-id). Re-ingesting one article updates its own file; it neither
    forks a third copy nor touches its twin."""
    from sift.ingest import base

    _stub_index(monkeypatch)
    d = vault_path / "writeup"
    d.mkdir()
    for name, url, body in (
        ("Account Takeover.md", "https://a.tld/ato", "first article"),
        ("Account Takeover (2).md", "https://b.tld/ato", "second article"),
    ):
        text = (
            f"---\nid: research-account-takeover\ntype: writeup\ntitle: Account Takeover\n"
            f"source: {url.split('/')[2]}\nurl: {url}\n---\n\n{body}\n"
        )
        (d / name).write_text(text, encoding="utf-8")
    twin = (d / "Account Takeover.md").read_bytes()

    res = base.run_source(
        "research",
        [_research("Account Takeover", "https://b.tld/ato", "edited")],
        reindex_fts=False,
    )

    files = _files(vault_path)
    assert sorted(files) == ["Account Takeover (2).md", "Account Takeover.md"]
    assert "edited" in files["Account Takeover (2).md"].body
    assert (d / "Account Takeover.md").read_bytes() == twin
    assert res.updated == 1 and res.written == 0 and res.collisions == 0


def test_hashed_ids_survive_the_legacy_slug_cut():
    from sift.ingest.base import url_note_id
    from sift.vault.notes import legacy_slug

    base_id = "research-" + "a-very-long-title-word-" * 6
    nid = url_note_id(base_id, "https://x.tld/p")
    assert len(nid) <= 78
    assert nid == url_note_id(base_id, "http://www.x.tld/p/?utm_source=rss")  # canonical
    assert nid != url_note_id(base_id, "https://x.tld/q")
    assert legacy_slug(nid).endswith(nid[-8:]), "the hash must survive the 80-char slug"
