"""Filename collisions must not silently lose notes.

Filenames are note titles, so two notes sharing a title want the same file. Before
this guard the second save overwrote the first while `written` counted both: a
research ingest reported "492 new notes, 0 errors" and left 488 files on disk.
`save_note` now resolves the clash to `Title (2).md`, Obsidian's own convention.
"""

from __future__ import annotations


def _note(title: str, url: str, body: str = "body text here, long enough to index"):
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    return Note(
        # id derives from the URL so two notes can legitimately share a title;
        # same id means same note, which save_note rightly treats as a rewrite.
        meta=Frontmatter(id=f"research-{url.rsplit('/', 1)[-1]}", type="writeup",
                         title=title, url=url, source="example.com"),
        body=body,
    )


# Same title, different notes - the realistic clash now that filenames are titles.
SHARED_TITLE = "Advisory: Pre-Auth RCE"


def test_colliding_titles_both_survive(vault_path, monkeypatch):
    from sift.ingest import base

    # Index writes need a real embedder; the disk behaviour is what's under test.
    monkeypatch.setattr(base, "index_notes", lambda notes, store: len(notes))
    monkeypatch.setattr(base, "Store", lambda: object())

    notes = [_note(SHARED_TITLE, "https://example.com/a"), _note(SHARED_TITLE, "https://example.com/b")]
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

    notes = [_note("Cookie sandwich", "https://a.tld/1"), _note("Desync endgame", "https://a.tld/2")]
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
            [_note(SHARED_TITLE, "https://example.com/a"), _note(SHARED_TITLE, "https://example.com/b")],
            reindex_fts=False,
        )

    assert len(list((vault_path / "writeup").glob("*.md"))) == 2
