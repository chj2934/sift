"""`ingest.base.run_source`: abort handling, tombstones, unchanged re-ingests, id
conflicts, merges, `_state.json` and the no-stdout rule.

Index writes are stubbed (the disk behaviour is under test); one test at the end runs
the real index with a fake embedder to check the "unchanged costs nothing" claim
end to end.
"""

from __future__ import annotations

import hashlib
import logging
import re

import pytest


class _Store:
    """A stand-in index: records what run_source asks of it."""

    def __init__(self, ids=None, *, optimize_error: Exception | None = None):
        self.ids = set(ids or ())
        self.optimized = 0
        self.optimize_error = optimize_error

    def indexed_ids(self):
        return set(self.ids)

    def optimize(self):
        self.optimized += 1
        if self.optimize_error:
            raise self.optimize_error
        return {"ran": False, "error": None}


@pytest.fixture
def index(monkeypatch):
    """Stub index_notes and Store; returns a dict with the batches and the store."""
    from sift.ingest import base

    state: dict = {"batches": [], "store": _Store(), "fail": None}

    def fake_index(notes, store):
        if state["fail"]:
            raise state["fail"]
        state["batches"].append([n.meta.id for n in notes])
        store.ids.update(n.meta.id for n in notes)
        return len(notes)

    monkeypatch.setattr(base, "index_notes", fake_index)
    monkeypatch.setattr(base, "Store", lambda: state["store"])
    return state


def _note(note_id, title=None, *, body="body text", source="h1", url=None, note_type="report"):
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    meta = Frontmatter(
        id=note_id, type=note_type, title=title or note_id.upper(), source=source, url=url
    )
    return Note(meta=meta, body=body)


def _indexed(index) -> list[str]:
    return [i for batch in index["batches"] for i in batch]


# --- aborts --------------------------------------------------------------------------


@pytest.mark.parametrize("error", [ConnectionError("HTTP 502 on page 7"), KeyboardInterrupt()])
def test_a_source_that_dies_still_indexes_what_was_saved(vault_path, index, error):
    from sift.ingest.base import load_state, run_source

    def source():
        yield _note("h1-1")
        yield _note("h1-2")
        yield _note("h1-3")
        raise error

    with pytest.raises(type(error)):
        run_source("h1", source(), flush_every=200)

    assert sorted(_indexed(index)) == ["h1-1", "h1-2", "h1-3"]
    entry = load_state()["h1"]
    assert entry["complete"] is False and entry["written"] == 3
    assert type(error).__name__ in entry["aborted"]
    # Maintenance runs for the rows written - unless the user pressed Ctrl-C.
    assert index["store"].optimized == (0 if isinstance(error, KeyboardInterrupt) else 1)


def test_a_complete_run_after_an_aborted_one_is_recorded_as_complete(vault_path, index):
    from sift.ingest.base import load_state, run_source

    def broken():
        yield _note("h1-1")
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        run_source("h1", broken())
    aborted = load_state()["h1"]
    assert "last_complete" not in aborted

    run_source("h1", [_note("h1-2")])
    entry = load_state()["h1"]
    assert entry["complete"] is True and entry["last_complete"] == entry["last_run"]


def test_one_bad_note_does_not_stop_the_run(vault_path, index):
    from sift.ingest.base import run_source

    res = run_source("h1", [_note("h1-1"), object(), _note("h1-2")])  # a source bug
    assert res.errors == 1 and res.written == 2 and res.complete


def test_an_index_failure_is_counted_and_the_run_still_recorded(vault_path, index):
    from sift.ingest.base import load_state, run_source

    index["fail"] = RuntimeError("commit conflict")
    res = run_source("h1", [_note("h1-1"), _note("h1-2")])
    assert res.written == 2 and res.errors == 2
    assert load_state()["h1"]["errors"] == 2


# --- tombstones ----------------------------------------------------------------------


def test_tombstoned_notes_are_not_resurrected(vault_path, index):
    from sift.ingest.base import run_source
    from sift.tombstones import record_tombstones

    record_tombstones(ids=["CVE-2020-0001"], source="nvd")
    record_tombstones(urls=["https://hackerone.com/reports/9"])  # a "forget": every source

    res = run_source(
        "nvd",
        [
            _note("CVE-2020-0001", source="nvd", note_type="cve"),
            _note("h1-9", url="http://hackerone.com/reports/9/"),
            _note("CVE-2020-0002", source="nvd", note_type="cve"),
        ],
    )

    assert res.tombstoned == 2 and res.written == 1
    assert _indexed(index) == ["CVE-2020-0002"]
    assert not list((vault_path / "cve").glob("CVE-2020-0001*"))


def test_a_tombstone_is_scoped_to_its_source(vault_path, index):
    """An NVD prune says nothing about the same CVE arriving from KEV."""
    from sift.ingest.base import run_source
    from sift.tombstones import record_tombstones

    record_tombstones(ids=["CVE-2020-0001"], source="nvd")
    res = run_source("kev", [_note("CVE-2020-0001", source="cisa-kev", note_type="cve")])
    assert res.written == 1 and res.tombstoned == 0


def test_tombstones_can_be_bypassed_explicitly(vault_path, index):
    from sift.ingest.base import run_source
    from sift.tombstones import record_tombstones

    record_tombstones(ids=["h1-1"])
    res = run_source("h1", [_note("h1-1")], skip_tombstoned=False)
    assert res.written == 1


# --- unchanged, updated, renamed -------------------------------------------------------


def test_an_unchanged_reingest_is_not_rewritten_or_reembedded(vault_path, index):
    from sift.ingest.base import run_source

    run_source("h1", [_note("h1-1"), _note("h1-2")])
    path = next((vault_path / "report").glob("H1-1.md"))
    before = path.read_bytes()

    res = run_source("h1", [_note("h1-1"), _note("h1-2")])

    assert (res.written, res.updated, res.unchanged) == (0, 0, 2)
    assert path.read_bytes() == before, "the ingested stamp alone must not rewrite a note"
    assert len(index["batches"]) == 1
    assert index["store"].optimized == 1, "no maintenance after a run that wrote nothing"


def test_a_changed_note_is_updated_in_place(vault_path, index):
    from sift.ingest.base import run_source

    run_source("h1", [_note("h1-1", body="old")])
    res = run_source("h1", [_note("h1-1", body="new")])
    assert (res.written, res.updated, res.unchanged) == (0, 1, 0)
    assert _indexed(index) == ["h1-1", "h1-1"]


def test_unchanged_but_missing_from_the_index_is_indexed(vault_path, index):
    """A note saved by a run whose index batch never happened is repaired by the next
    ingest, without being rewritten."""
    from sift.ingest.base import run_source

    index["fail"] = RuntimeError("index down")
    run_source("h1", [_note("h1-1")])
    index["fail"] = None

    res = run_source("h1", [_note("h1-1")])
    assert res.unchanged == 1 and res.written == 0
    assert _indexed(index) == ["h1-1"]


def test_a_retitled_note_keeps_one_file(vault_path, index):
    from sift.ingest.base import run_source

    run_source("h1", [_note("h1-1", "Old title")])
    res = run_source("h1", [_note("h1-1", "New title")])

    names = [p.name for p in (vault_path / "report").glob("*.md")]
    assert names == ["New title.md"]
    assert res.updated == 1 and res.collisions == 0


# --- id conflicts and merges -----------------------------------------------------------


def test_a_different_source_under_the_same_id_is_refused_not_clobbered(vault_path, index):
    from sift.ingest.base import run_source

    run_source("nvd", [_note("CVE-2024-1", "NVD title", body="NVD body", source="nvd")])
    nvd_file = next((vault_path / "report").glob("*.md"))
    before = nvd_file.read_bytes()

    res = run_source("kev", [_note("CVE-2024-1", "KEV title", body="KEV body", source="cisa-kev")])

    assert res.id_conflicts == 1 and res.written == 0 and res.updated == 0
    assert nvd_file.read_bytes() == before
    assert len(list((vault_path / "report").glob("*.md"))) == 1


def test_a_merge_combines_two_sources_into_one_note(vault_path, index):
    from sift.ingest.base import run_source
    from sift.vault.notes import load_note

    run_source("nvd", [_note("CVE-2024-1", "NVD title", body="## NVD\nscore", source="nvd")])

    def merge(existing, incoming):
        existing.meta.tags = sorted({*existing.meta.tags, "kev"})
        existing.body = existing.body.strip() + "\n\n## CISA KEV\nrequired action"
        return existing

    res = run_source("kev", [_note("CVE-2024-1", "KEV title", source="cisa-kev")], merge=merge)
    files = list((vault_path / "report").glob("*.md"))
    assert res.updated == 1 and res.id_conflicts == 0 and len(files) == 1
    note = load_note(files[0])
    assert "kev" in note.meta.tags and "## NVD" in note.body and "## CISA KEV" in note.body

    again = run_source("kev", [_note("CVE-2024-1", "KEV title", source="cisa-kev")], merge=merge)
    assert again.updated == 1  # this merge appends; a None return would skip instead


def test_a_merge_returning_none_leaves_the_note_alone(vault_path, index):
    from sift.ingest.base import run_source

    run_source("h1", [_note("h1-1", body="keep me")])
    path = next((vault_path / "report").glob("*.md"))
    before = path.read_bytes()
    res = run_source("h1", [_note("h1-1", body="other")], merge=lambda old, new: None)
    assert res.unchanged == 1 and path.read_bytes() == before


def test_keep_longer_body_never_downgrades_a_full_text_note(vault_path, index):
    from sift.ingest.base import keep_longer_body, run_source
    from sift.vault.notes import load_note

    url = "https://blog.tld/post"
    full = _note("research-post", "Post", body="full article " * 50, source="blog.tld", url=url)
    run_source("research", [full])
    teaser = _note("research-post", "Post", body="teaser", source="blog.tld", url=url)
    teaser.meta.tags = ["research"]

    res = run_source("research", [teaser], merge=keep_longer_body)

    note = load_note(next((vault_path / "report").glob("*.md")))
    assert res.updated == 1 and note.body.startswith("full article")
    assert note.meta.tags == ["research"], "frontmatter is still refreshed"


def test_keep_longer_body_refuses_to_graft_another_document(vault_path, index):
    from sift.ingest.base import keep_longer_body, run_source

    run_source("x", [_note("x-1", body="long body " * 20, source="a")])
    res = run_source("x", [_note("x-1", body="short", source="b")], merge=keep_longer_body)
    assert res.id_conflicts == 1 and res.updated == 0


# --- _state.json -----------------------------------------------------------------------


def test_state_keeps_every_source_and_records_the_counts(vault_path, index):
    from sift.ingest.base import load_state, run_source

    run_source("a", [_note("a-1")])
    run_source("b", [_note("b-1"), _note("b-2")])
    run_source("a", [_note("a-1")])

    state = load_state()
    assert set(state) == {"a", "b"}
    assert state["b"]["written"] == 2 and state["a"]["unchanged"] == 1
    assert not list(vault_path.glob("*.tmp")) and not list(vault_path.glob(".*.tmp"))


def test_a_corrupt_state_file_is_moved_aside_not_overwritten(vault_path, index):
    from sift.ingest.base import load_state, run_source

    state_file = vault_path / "_state.json"
    state_file.write_text('{"kev": {"last_run": "2026-01-01', encoding="utf-8")  # torn write
    assert load_state() == {}

    run_source("h1", [_note("h1-1")])

    aside = list(vault_path.glob("_state.json.corrupt-*"))
    assert len(aside) == 1 and aside[0].read_text(encoding="utf-8").startswith('{"kev"')
    assert set(load_state()) == {"h1"}


def test_load_state_tolerates_a_bom_and_never_writes(vault_path):
    from sift.ingest.base import load_state

    state_file = vault_path / "_state.json"
    state_file.write_bytes(b'\xef\xbb\xbf{"kev": {"written": 3}}')
    assert load_state() == {"kev": {"written": 3}}
    state_file.write_text("[1, 2]", encoding="utf-8")
    assert load_state() == {}
    assert state_file.read_text(encoding="utf-8") == "[1, 2]"


def test_concurrent_record_runs_do_not_lose_entries(vault_path):
    """Two ingests finishing together used to read the same state and each write back
    only its own key."""
    import threading

    from sift.ingest.base import IngestResult, load_state, record_run

    def worker(name):
        for i in range(15):
            record_run(name, IngestResult(source=name, written=i))

    threads = [threading.Thread(target=worker, args=(f"src{n}",)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    state = load_state()
    assert sorted(state) == ["src0", "src1", "src2", "src3"]
    assert all(entry["written"] == 14 for entry in state.values())


# --- reporting -------------------------------------------------------------------------


def test_progress_reports_every_note_the_source_yielded(vault_path, index):
    from sift.ingest.base import run_source

    calls = []
    run_source("h1", [_note("h1-1"), _note("h1-2")], on_progress=lambda n, t: calls.append((n, t)))
    run_source("h1", [_note("h1-1")], on_progress=lambda n, t: calls.append((n, t)))
    assert calls == [(1, "H1-1"), (2, "H1-2"), (1, "H1-1")], "an unchanged note still counts"


def test_nothing_reaches_stdout(vault_path, index, capfd, caplog):
    """K1: run_source is plumbing; its diagnostics go through logging (stderr)."""
    from sift.ingest.base import run_source

    caplog.set_level(logging.INFO, logger="sift")
    clash = [_note("x-1", "Same", source="a"), _note("x-2", "Same", source="a")]
    run_source("x", [*clash, object()])
    run_source("kev", [_note("x-1", "Same", source="b")])  # an id conflict

    assert capfd.readouterr().out == ""
    text = caplog.text
    assert "title clash" in text and "failed on" in text and "not overwritten" in text


def test_maintenance_failure_never_fails_the_run(vault_path, index):
    from sift.ingest.base import load_state, run_source

    index["store"] = _Store(optimize_error=RuntimeError("commit conflict"))
    res = run_source("h1", [_note("h1-1")])
    assert res.written == 1 and load_state()["h1"]["complete"] is True


def test_summary_line():
    from sift.ingest.base import IngestResult

    res = IngestResult(source="x", written=3, updated=1, unchanged=40, indexed_chunks=12)
    assert res.summary() == "3 new, 1 updated, 40 unchanged, 12 chunks, 0 errors"
    res.collisions, res.complete, res.aborted = 2, False, "KeyboardInterrupt"
    assert "2 collisions" in res.summary() and "INCOMPLETE (KeyboardInterrupt)" in res.summary()


# --- end to end with the real index ----------------------------------------------------

DIM = 32


class _FakeEmbedder:
    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def __init__(self):
        self.passages = 0

    def _vec(self, text):
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed(self, texts, *, kind="passage", batch_size=32):
        self.passages += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def embed_one(self, text):
        return self._vec(text)


def test_reingesting_unchanged_notes_embeds_nothing(vault_path, monkeypatch):
    from sift.config import get_settings

    monkeypatch.setenv("SIFT_EMBED_DIM", str(DIM))
    get_settings.cache_clear()
    fake = _FakeEmbedder()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: fake)

    from sift.index.store import Store
    from sift.ingest.base import run_source

    notes = lambda: [_note(f"h1-{i}", body=f"report number {i} text") for i in range(3)]  # noqa: E731
    first = run_source("h1", notes())
    embedded, rows = fake.passages, Store().count()
    assert first.written == 3 and embedded > 0 and rows > 0

    again = run_source("h1", notes())
    assert again.unchanged == 3 and again.indexed_chunks == 0
    assert fake.passages == embedded and Store().count() == rows
