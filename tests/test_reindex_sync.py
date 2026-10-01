"""pipeline.sync: picking up Obsidian edits in a long-lived process (the MCP server).

The index never saw edits made during a session: a no-op `sift reindex` cost ~10 s
(full parse, column scan, FTS rebuild) and the server never ran one.
"""

from __future__ import annotations

import os
import time

import pytest

DIM = 32


class FakeEmbedder:
    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def __init__(self):
        self.embedded = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in text.lower().split():
            v[sum(map(ord, tok)) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts, *, kind="passage", batch_size=32):
        texts = list(texts)
        self.embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_DIM", str(DIM))
    from sift import config, pipeline

    config.get_settings.cache_clear()
    embedder = FakeEmbedder()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: embedder)
    monkeypatch.setattr(pipeline, "_SYNC_STATES", {})  # no state carried between tests
    return embedder


def _write(vault, nid: str, body: str = "plain body words"):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    return save_note(
        vault, Note(meta=Frontmatter(id=nid, type="technique", title=nid), body=body), stamp=False
    )


def _bump(path, secs: float = 10.0) -> None:
    t = time.time() + secs
    os.utime(path, (t, t))


def _ids():
    from sift.index.store import Store

    return Store().indexed_ids()


def _baseline(vault, n: int = 3):
    from sift.pipeline import reindex, sync

    paths = [_write(vault, f"note-{i}") for i in range(n)]
    reindex(force=True)
    first = sync(min_interval=0)  # first sync of a process: compares with the index
    assert first is not None and first.notes == 0 and first.unchanged == n
    return paths


def test_a_no_op_sync_never_touches_the_index(vault_path, fake, monkeypatch):
    from sift.index.store import Store
    from sift.pipeline import sync

    _baseline(vault_path)

    def boom(self):
        raise AssertionError("a no-op sync read the index")

    monkeypatch.setattr(Store, "indexed_files", boom)
    fake.embedded = 0
    assert sync(min_interval=0) is None
    assert fake.embedded == 0


def test_an_outside_edit_is_picked_up(vault_path, fake):
    from sift.pipeline import search, sync

    paths = _baseline(vault_path)
    paths[1].write_text(
        paths[1].read_text(encoding="utf-8").replace("plain body words", "zebra crossing exploit"),
        encoding="utf-8",
    )
    _bump(paths[1])

    stats = sync(min_interval=0)
    assert stats is not None and stats.notes == 1 and stats.unchanged == 2
    hits = search("zebra crossing exploit", k=1).hits
    assert hits and hits[0].note_id == "note-1" and "zebra" in hits[0].excerpt


def test_an_outside_delete_is_reaped_and_nothing_is_optimized(vault_path, fake, monkeypatch):
    from sift.index.store import Store
    from sift.pipeline import sync

    paths = _baseline(vault_path)
    monkeypatch.setattr(
        Store, "optimize", lambda self, *a, **k: pytest.fail("sync must never optimize")
    )
    paths[0].unlink()

    stats = sync(min_interval=0)
    assert stats is not None and stats.removed == 1
    assert _ids() == {"note-1", "note-2"}


def test_sync_is_debounced(vault_path, fake):
    from sift.pipeline import sync

    paths = _baseline(vault_path)
    _write(vault_path, "note-0", body="changed text")
    _bump(paths[0])

    assert sync(min_interval=3600) is None  # the baseline sync ran a moment ago
    assert sync(min_interval=0).notes == 1


def test_a_large_change_is_left_for_the_cli(vault_path, fake):
    """A background sync must never embed a whole vault (a moved vault, a git checkout)."""
    from sift.pipeline import sync

    _baseline(vault_path, n=1)
    for i in range(3):
        _write(vault_path, f"new-{i}")
    fake.embedded = 0

    stats = sync(min_interval=0, max_notes=2)
    assert stats.deferred == 3 and stats.notes == 0
    assert fake.embedded == 0 and not any(i.startswith("new-") for i in _ids())

    assert sync(min_interval=0, max_notes=2) is None  # reported once, not every call


def test_the_servers_own_writes_do_not_trigger_a_rescan(vault_path, fake, monkeypatch):
    """remember/capture_idea index their note themselves; sync must not redo it."""
    from sift.index.store import Store
    from sift.pipeline import index_note, sync
    from sift.vault.notes import load_note

    _baseline(vault_path)
    path = _write(vault_path, "remembered", body="saved by the mcp server")
    index_note(load_note(path))
    assert "remembered" in _ids()

    def boom(self):
        raise AssertionError("sync rescanned the index for the server's own write")

    monkeypatch.setattr(Store, "indexed_files", boom)
    assert sync(min_interval=0) is None


def test_a_failing_sync_is_logged_never_raised(vault_path, fake, monkeypatch, caplog):
    from sift.index.store import Store
    from sift.pipeline import sync

    _write(vault_path, "note-0")

    def broken(self):
        raise OSError("disk on fire")

    monkeypatch.setattr(Store, "indexed_files", broken)
    with caplog.at_level("WARNING", logger="sift.pipeline"):
        assert sync(min_interval=0) is None
    assert any("index sync failed" in r.getMessage() for r in caplog.records)


def test_sync_in_background_indexes_and_prints_nothing(vault_path, fake, capfd):
    from sift.pipeline import sync_in_background

    paths = _baseline(vault_path)
    _write(vault_path, "note-2", body="background edit")
    _bump(paths[2])

    t = sync_in_background(min_interval=0)
    assert t is not None
    t.join(timeout=60)
    assert not t.is_alive()

    from sift.index.store import Store

    rows = Store().table().search().select(["note_id", "text"]).limit(100).to_list()
    assert any("background edit" in r["text"] for r in rows if r["note_id"] == "note-2")
    assert capfd.readouterr().out == ""
