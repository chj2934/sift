"""Ingest -> vault -> index -> search, with the fake embedder and a stubbed feed."""

from __future__ import annotations

import hashlib
import re

import pytest

DIM = 64


class FakeEmbedder:
    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def _vec(self, text: str):
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed(self, texts, *, kind="passage", batch_size=32):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_DIM", str(DIM))
    from sift.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: FakeEmbedder())
    yield


_KEV_SAMPLE = [
    {
        "cveID": "CVE-2021-44228",
        "vendorProject": "Apache",
        "product": "Log4j2",
        "vulnerabilityName": "Apache Log4j2 Remote Code Execution",
        "dateAdded": "2021-12-10",
        "shortDescription": "JNDI features do not protect against attacker-controlled LDAP lookups, allowing remote code execution.",
        "requiredAction": "Apply updates.",
        "cwes": ["CWE-502", "CWE-917"],
    },
    {
        "cveID": "CVE-2022-22965",
        "vendorProject": "VMware",
        "product": "Spring Framework",
        "vulnerabilityName": "Spring4Shell",
        "dateAdded": "2022-04-04",
        "shortDescription": "Spring MVC data binding on JDK 9+ allows remote code execution via class loader manipulation.",
        "requiredAction": "Apply updates.",
        "cwes": ["CWE-94"],
    },
]


def test_kev_ingest_and_search(monkeypatch):
    from sift.ingest import kev
    from sift.ingest.base import load_state, run_source
    from sift.pipeline import search

    monkeypatch.setattr(kev, "fetch", lambda: _KEV_SAMPLE)

    res = run_source("kev", kev.source())
    assert res.written == 2
    assert res.errors == 0
    assert res.indexed_chunks >= 2

    hits = search("log4j jndi ldap remote code execution", k=2).hits
    assert hits and hits[0].note_id == "CVE-2021-44228"

    filtered = search("remote code execution", k=5, filters={"cwe": "CWE-94"}).hits
    assert [h.note_id for h in filtered] == ["CVE-2022-22965"]

    assert "kev" in load_state()


def test_reindex_is_idempotent(monkeypatch, vault_path):
    from sift.index.store import Store
    from sift.ingest import kev
    from sift.ingest.base import run_source
    from sift.pipeline import reindex

    monkeypatch.setattr(kev, "fetch", lambda: _KEV_SAMPLE)
    run_source("kev", kev.source())
    n1 = Store().count()

    reindex(force=False)
    reindex(force=False)
    assert Store().count() == n1  # no duplicate chunks


# --- index_notes: one commit, emptied notes, twins, renames -------------------------


def _note(nid: str, title: str, body: str, **meta):
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    return Note(meta=Frontmatter(id=nid, type="technique", title=title, **meta), body=body)


def _paths_by_id():
    from sift.index.store import Store, norm_path

    out: dict[str, set[str]] = {}
    for nid, path, _m in Store().indexed_files():
        out.setdefault(nid, set()).add(norm_path(path))
    return out


def test_index_notes_is_one_commit_for_the_whole_batch(vault_path):
    """It used to delete per note (N commits) and then add (one more)."""
    from sift.index.store import Store
    from sift.pipeline import index_note, index_notes
    from sift.vault.notes import save_note

    index_note(_note("seed", "Seed", "creates the table"))
    v0 = Store().table().version

    notes = [_note(f"n{i}", f"Note {i}", f"body number {i} about ssrf") for i in range(5)]
    for n in notes:
        save_note(vault_path, n, stamp=False)
    index_notes(notes)
    assert Store().table().version == v0 + 1

    index_notes(notes)  # re-indexing the same notes replaces, never duplicates
    assert Store().table().version == v0 + 2
    assert sum(1 for nid, _p, _m in Store().indexed_files() if nid.startswith("n")) == 5


def test_index_note_clears_a_note_whose_body_emptied(vault_path):
    from sift.pipeline import index_note
    from sift.vault.notes import save_note

    note = _note("idea-x", "An idea", "something worth indexing")
    save_note(vault_path, note, stamp=False)
    assert index_note(note) > 0
    assert "idea-x" in _paths_by_id()

    note.body = "  "
    save_note(vault_path, note, stamp=False)
    assert index_note(note) == 0
    assert "idea-x" not in _paths_by_id()


def test_indexing_one_twin_leaves_the_other_twins_rows(vault_path):
    """KEV and NVD records of one CVE share an id. Ingesting one used to delete the
    other's rows (delete by id), so search flipped between them."""
    from sift.index.store import norm_path
    from sift.pipeline import index_notes, reindex
    from sift.vault.notes import load_note

    nvd = vault_path / "cve" / "CVE-2024-1 nvd.md"
    kev = vault_path / "cve" / "CVE-2024-1 kev.md"
    nvd.parent.mkdir(parents=True)
    nvd.write_text(_note("CVE-2024-1", "NVD", "cvss vector").render(), encoding="utf-8")
    kev.write_text(_note("CVE-2024-1", "KEV", "known exploited").render(), encoding="utf-8")
    reindex(force=True)
    assert _paths_by_id()["CVE-2024-1"] == {norm_path(nvd), norm_path(kev)}

    index_notes([load_note(kev)])
    assert _paths_by_id()["CVE-2024-1"] == {norm_path(nvd), norm_path(kev)}


def test_a_retitled_note_leaves_no_rows_under_its_old_filename(vault_path):
    from sift.index.store import norm_path
    from sift.pipeline import index_note
    from sift.vault.notes import save_note

    note = _note("tech-retitle", "Old title", "body text about cache poisoning")
    old = save_note(vault_path, note, stamp=False)
    index_note(note)

    note.meta.title = "New title"
    new = save_note(vault_path, note, stamp=False)  # renamed in its folder
    assert new != old and not old.exists()
    index_note(note)

    assert _paths_by_id()["tech-retitle"] == {norm_path(new)}


# --- chunks fit the embedder window, and are chunked once ---------------------------


class _WindowEmbedder:
    """An embedder with the token API: one token per whitespace word, a 64-token window."""

    model_name = "window-fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = "passage: "
    max_seq_length = 64
    num_special_tokens = 2

    def __init__(self):
        self.passages: list[str] = []

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    def embed(self, texts, *, kind="passage", batch_size=32):
        texts = list(texts)
        if kind == "passage":
            self.passages.extend(texts)
        return [FakeEmbedder()._vec(t) for t in texts]

    def embed_query(self, text):
        return FakeEmbedder()._vec(text)


def test_every_indexed_chunk_fits_the_embedder_window(vault_path, monkeypatch):
    from sift.index.store import Store
    from sift.pipeline import index_note, reindex
    from sift.vault.notes import save_note

    emb = _WindowEmbedder()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: emb)
    curl = "\n".join(f"# step {i}\ncurl -sk https://a.test/x?id={i} -H 'X: y'" for i in range(60))
    body = f"## PoC\n\n```bash\n{curl}\n```\n\n## Notes\n\n" + "word " * 400
    title = "A fairly long title for a proof of concept note"
    note = _note("tech-window", title, body)
    save_note(vault_path, note, stamp=False)

    for run in (lambda: index_note(note), lambda: reindex(force=True)):
        emb.passages.clear()
        run()
        rows = (
            Store()
            .table()
            .search()
            .select(["note_id", "title", "heading", "text"])
            .limit(1000)
            .to_list()
        )
        rows = [r for r in rows if r["note_id"] == "tech-window"]
        assert len(rows) > 5, "the long body was not split"
        assert len(rows) == len(emb.passages), "rows and embedded passages are misaligned"
        for r in rows:
            header = f"{r['title']}\n{r['heading']}\n" if r["heading"] else f"{r['title']}\n"
            used = emb.num_special_tokens + emb.count_tokens(
                emb.passage_prefix + header + r["text"]
            )
            assert used <= emb.max_seq_length, (used, r["heading"], r["text"][:60])
        assert {"PoC", "Notes"} <= {r["heading"] for r in rows}  # '# step' is code


def test_rows_refuse_a_vector_count_that_does_not_match_the_chunks():
    from sift.pipeline import _chunk, _rows_for_note

    note = _note("n", "T", "## A\n\nfirst\n\n## B\n\nsecond")
    chunks = _chunk(note, FakeEmbedder())
    assert len(chunks) == 2
    with pytest.raises(ValueError):
        _rows_for_note(note, chunks=chunks, vectors=[[0.0] * DIM])


# --- recency for undated user notes ---------------------------------------------------


def test_undated_user_notes_get_their_capture_date_and_bulk_notes_stay_neutral():
    from datetime import UTC, datetime

    from sift.pipeline import _created_ts

    idea = _note(
        "idea-x-20261001120000000",
        "Idea",
        "b",
        source="sift-capture-idea",
        ingested=datetime(2026, 9, 30, tzinfo=UTC),
        extra={"captured": "2026-09-01T10:00:00+00:00"},
    )
    assert _created_ts(idea) == datetime(2026, 9, 1, tzinfo=UTC).timestamp()

    remembered = _note(
        "find-x", "Mine", "b", source="sift-remember", ingested=datetime(2026, 8, 2, tzinfo=UTC)
    )
    assert _created_ts(remembered) == datetime(2026, 8, 2, tzinfo=UTC).timestamp()

    distilled = _note(
        "tech-y", "Distilled", "b", source="distill", ingested=datetime(2026, 8, 2, tzinfo=UTC)
    )
    assert _created_ts(distilled) == 0.0


# --- search: warnings and id-seeded link expansion ------------------------------------


def test_search_reports_warnings_and_expands_links_by_note_id(vault_path):
    from sift.pipeline import reindex, search
    from sift.vault.notes import save_note

    save_note(vault_path, _note("tech-a", "Alpha", "prototype pollution gadget chain [[Beta]]"))
    save_note(vault_path, _note("tech-b", "Beta", "unrelated words entirely"))
    reindex(force=True)

    res = search("prototype pollution gadget", k=1, expand_links=True)
    assert [h.note_id for h in res.hits] == ["tech-a"]
    assert res.warnings == []
    assert [(x["note_id"], x["title"]) for x in res.linked] == [("tech-b", "Beta")]


# --- the chunker record ---------------------------------------------------------------


def test_index_notes_records_the_chunker_only_for_a_table_it_started(monkeypatch):
    """An index built by an older chunker carries no record; an ingest or MCP write
    into it must not stamp one, or the "run reindex --force" warning would vanish."""
    from sift.index.store import Store
    from sift.pipeline import index_meta, index_note
    from sift.vault.chunk import CHUNKER_VERSION

    index_note(_note("first", "First", "body one"))
    assert index_meta()["chunker_version"] == CHUNKER_VERSION  # positive control

    (Store().db_path / "index_meta.json").unlink()  # as if built before the record existed
    index_note(_note("second", "Second", "body two"))
    assert index_meta() is None

    monkeypatch.setattr(Store, "count", lambda self: 0)  # a read error reads as "empty"
    index_note(_note("third", "Third", "body three"))
    assert index_meta() is None
