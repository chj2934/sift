"""`run_source` writes a note to the file the source chose (``note.path``).

A source-side merge (KEV/NVD, h1-mine) loads one twin of an id, merges into it and
yields the result with ``note.path`` set to that twin. `run_source` used to re-locate
the id with `locate_note(meta=...)` and pass that as ``existing``, which beats
``note.path`` in `write_note` - so with two same-document twins the merge computed
from twin A was written over twin B, and B's own text was lost.
"""

from __future__ import annotations

import pytest

TITLE = "Shared title"
NID = "h1-77"


@pytest.fixture
def index(monkeypatch):
    """Stub the index: the disk behaviour is under test."""
    from sift.ingest import base

    batches: list[list[str]] = []

    class _Store:
        def indexed_ids(self):
            return {i for b in batches for i in b}

        def optimize(self):
            return {"error": None}

    def fake_index(notes, _store):
        batches.append([n.meta.id for n in notes])
        return len(notes)

    store = _Store()
    monkeypatch.setattr(base, "index_notes", fake_index)
    monkeypatch.setattr(base, "Store", lambda: store)
    return batches


def _meta(note_id=NID, title=TITLE):
    from sift.vault.schema import Frontmatter

    return Frontmatter(id=note_id, type="report", title=title, source="h1")


def _twins(vault_path):
    """Two files carrying one id and the same document: B sits at the title path, so
    `locate_note` ranks it first; A has another name."""
    from sift.vault.notes import Note

    folder = vault_path / "report"
    folder.mkdir(parents=True)
    a = folder / "Other copy.md"
    b = folder / f"{TITLE}.md"
    a.write_text(Note(meta=_meta(), body="twin A text").render(), encoding="utf-8")
    b.write_text(Note(meta=_meta(), body="twin B text").render(), encoding="utf-8")
    return a, b


def test_a_note_is_written_to_the_twin_the_source_chose(vault_path, index):
    from sift.ingest.base import run_source
    from sift.vault.notes import Note, load_note, locate_note

    a, b = _twins(vault_path)
    assert locate_note(vault_path, NID, meta=_meta()) == b, "precondition: B ranks first"
    b_before = b.read_bytes()

    merged = Note(meta=_meta(), body="twin A text\n\nmerged update", path=a)
    res = run_source("h1", [merged])

    assert res.updated == 1 and res.written == 0 and res.id_conflicts == 0
    assert load_note(a).body == "twin A text\n\nmerged update"
    assert b.read_bytes() == b_before, "the other twin is never written over"


def test_without_a_path_the_ranked_carrier_is_updated(vault_path, index):
    """Control: the same note with no chosen file goes where the ranking says (B), so
    the test above is about ``note.path``, not about which twin is writable."""
    from sift.ingest.base import run_source
    from sift.vault.notes import Note, load_note

    a, b = _twins(vault_path)
    a_before = a.read_bytes()

    res = run_source("h1", [Note(meta=_meta(), body="update without a path")])

    assert res.updated == 1
    assert load_note(b).body == "update without a path"
    assert a.read_bytes() == a_before


def test_a_path_that_does_not_carry_the_id_is_ignored(vault_path, index):
    """A stale ``note.path`` (the file now holds another id) is not trusted: the note
    goes to the file that carries its id, and the other file is untouched."""
    from sift.ingest.base import run_source
    from sift.vault.notes import Note, load_note

    _a, b = _twins(vault_path)
    other = vault_path / "report" / "Unrelated.md"
    other.write_text(
        Note(meta=_meta("h1-999", "Unrelated"), body="unrelated").render(), encoding="utf-8"
    )
    other_before = other.read_bytes()

    res = run_source("h1", [Note(meta=_meta(), body="routed by id", path=other)])

    assert res.updated == 1
    assert load_note(b).body == "routed by id"
    assert other.read_bytes() == other_before


def test_unchanged_is_judged_against_the_chosen_file(vault_path, index):
    """Re-yielding twin A as it is must count as unchanged and write nothing, even
    though twin B (ranked first) has different text."""
    from sift.ingest.base import run_source
    from sift.vault.notes import load_note

    a, b = _twins(vault_path)
    before = {p: p.read_bytes() for p in (a, b)}

    res = run_source("h1", [load_note(a)])

    assert res.unchanged == 1 and res.updated == 0
    assert {p: p.read_bytes() for p in (a, b)} == before
