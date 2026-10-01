"""Store write paths: every batch is ONE commit, nothing fails silently, and a cached
table handle never serves a stale or dropped table.

Each commit writes a manifest listing every fragment; one commit per note is what
grew the real index to 53k versions / 29 GB. These tests count table versions, so a
regression back to per-note commits fails loudly.
"""

from __future__ import annotations

import pytest

DIM = 8


def _vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


def _rows(nid: str, n: int = 1, *, path: str | None = None, text: str = "alpha beta", quality=50):
    from sift.index.store import ChunkRow

    path = path if path is not None else f"/vault/technique/{nid}.md"
    return [
        ChunkRow(
            note_id=nid,
            slug=nid,
            type="technique",
            title=nid,
            heading="",
            text=f"{text} chunk{i}",
            chunk_index=i,
            vector=_vec(i),
            path=path,
            quality=quality,
        )
        for i in range(n)
    ]


def _store():
    from sift.index.store import Store

    return Store(dim=DIM)


def _ids(store) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in store.table().search().select(["note_id"]).limit(0).to_list():
        out[r["note_id"]] = out.get(r["note_id"], 0) + 1
    return out


def _texts(store, nid: str) -> list[str]:
    rows = store.table().search().where(f"note_id = '{nid}'").limit(0).to_list()
    return sorted(r["text"] for r in rows)


# ---- delete_notes ---------------------------------------------------------


def test_delete_notes_is_one_commit_and_doubles_quotes():
    store = _store()
    ids = [f"n{i}" for i in range(30)] + ["it's", "a'b'c"]
    store.add_chunks([r for nid in ids for r in _rows(nid, 2)])

    v0 = store.table().version
    gone = store.delete_notes([*ids[:20], "it's", "a'b'c", "it's", "never-indexed"])

    assert store.table().version == v0 + 1, "a batch delete must be a single commit"
    assert gone == 22 * 2
    left = _ids(store)
    assert "it's" not in left and "a'b'c" not in left
    assert set(left) == set(ids[20:30])


def test_delete_notes_past_one_in_list_is_still_one_commit():
    store = _store()
    ids = [f"bulk-{i}" for i in range(1_200)]  # > the 500 literals of one IN list
    store.add_chunks([r for nid in ids for r in _rows(nid)])

    v0 = store.table().version
    assert store.delete_notes(ids) == 1_200
    assert store.table().version == v0 + 1
    assert store.count() == 0


def test_delete_of_ids_that_are_not_indexed_commits_nothing():
    """A zero-match delete still commits a version; most ids in an ingest flush are new."""
    store = _store()
    store.add_chunks(_rows("kept"))
    v0 = store.table().version

    assert store.delete_notes(["brand-new", "also-new"]) == 0
    assert store.delete_notes([]) == 0
    assert store.table().version == v0
    assert _ids(store) == {"kept": 1}


def test_delete_without_a_table_creates_nothing():
    from sift.index.store import TABLE

    store = _store()
    assert store.delete_notes(["x"]) == 0
    assert store.delete_paths(["/nowhere.md"]) == 0
    assert TABLE not in store._table_names(), "a delete must not create the table"


def test_delete_failures_propagate(monkeypatch):
    """Swallowing a failed delete let the caller add a second copy of every chunk."""
    store = _store()
    store.add_chunks(_rows("a"))
    tbl = store.table()

    def boom(*_a, **_k):
        raise OSError("table locked")

    monkeypatch.setattr(tbl, "delete", boom)
    with pytest.raises(OSError, match="locked"):
        store.delete_notes(["a"])
    with pytest.raises(OSError, match="locked"):
        store.delete_note("a")


def test_delete_paths_keeps_a_sibling_file_sharing_the_id():
    store = _store()
    store.add_chunks(_rows("CVE-2024-1", 2, path="/vault/cve/kev.md"))
    store.add_chunks(_rows("CVE-2024-1", 3, path="/vault/cve/nvd.md"))

    assert store.delete_paths(["/vault/cve/kev.md"]) == 2
    assert [p for (_nid, p, _m) in store.indexed_files()] == ["/vault/cve/nvd.md"]


# ---- replace_notes / upsert_note ------------------------------------------


def test_replace_notes_is_one_atomic_commit_and_drops_old_chunks():
    store = _store()
    store.add_chunks(_rows("a", 4, text="old"))
    store.add_chunks(_rows("b", 2, text="other"))

    v0 = store.table().version
    written = store.replace_notes(_rows("a", 2, text="new"))

    assert written == 2
    assert store.table().version == v0 + 1, "replace must not be delete-commit + add-commit"
    assert _texts(store, "a") == ["new chunk0", "new chunk1"]
    assert _ids(store) == {"a": 2, "b": 2}


def test_replace_notes_batch_of_new_notes_is_one_commit():
    store = _store()
    store.add_chunks(_rows("seed"))
    v0 = store.table().version

    batch = [r for i in range(50) for r in _rows(f"new-{i}", 3)]
    store.replace_notes(batch)

    assert store.table().version == v0 + 1
    assert store.count() == 1 + 150


def test_replace_notes_with_no_rows_clears_an_emptied_note():
    store = _store()
    store.add_chunks(_rows("emptied", 3))
    store.add_chunks(_rows("kept", 1))

    v0 = store.table().version
    assert store.replace_notes([], ["emptied"]) == 0
    assert store.table().version == v0 + 1
    assert _ids(store) == {"kept": 1}

    # Nothing to clear: no commit at all.
    store.replace_notes([], ["never-indexed"])
    assert store.table().version == v0 + 1


def test_replace_notes_scope_by_id_or_by_path_only():
    """Default scope is id OR path: a renamed note's old-path rows go. note_ids=[]
    scopes by path alone, so re-indexing one of two files that share an id (KEV and
    NVD twins of one CVE) leaves the other file's rows alone."""
    store = _store()
    store.add_chunks(_rows("CVE-1", 2, path="/vault/cve/kev.md", text="kev"))
    store.add_chunks(_rows("CVE-1", 2, path="/vault/cve/nvd.md", text="nvd"))

    store.replace_notes(_rows("CVE-1", 1, path="/vault/cve/kev.md", text="kev2"), note_ids=[])
    assert _texts(store, "CVE-1") == ["kev2 chunk0", "nvd chunk0", "nvd chunk1"]

    store.replace_notes(_rows("CVE-1", 1, path="/vault/cve/renamed.md", text="moved"))
    assert _texts(store, "CVE-1") == ["moved chunk0"], "default scope clears the id everywhere"


def test_replace_notes_clears_a_file_whose_id_was_edited():
    store = _store()
    store.add_chunks(_rows("old-id", 2, path="/vault/technique/note.md"))

    store.replace_notes(_rows("new-id", 1, path="/vault/technique/note.md"))

    assert _ids(store) == {"new-id": 1}, "the file now carries another id; its old rows must go"


def test_replace_notes_same_file_twice_in_one_batch_keeps_the_later_copy():
    """merge_insert rejects repeated chunk ids ("ambiguous merge insert") when they
    already exist, and inserts them twice when they do not."""
    store = _store()
    store.add_chunks(_rows("dup", 3, text="indexed"))

    batch = _rows("dup", 3, text="first") + _rows("other", 1) + _rows("dup", 2, text="second")
    store.replace_notes(batch)

    assert _texts(store, "dup") == ["second chunk0", "second chunk1"]
    assert _ids(store) == {"dup": 2, "other": 1}


def test_interleaved_files_in_one_batch_keep_every_chunk():
    """Copies are told apart by a repeated chunk index, not by row adjacency: rows of
    two files arriving interleaved are one copy each, not superseded copies."""
    from sift.index.store import _last_copy_per_file

    a, b = _rows("a", 3), _rows("b", 2)
    batch = [a[0], b[0], a[1], b[1], a[2]]
    assert _last_copy_per_file(batch) == batch

    store = _store()
    store.replace_notes(batch)
    assert _ids(store) == {"a": 3, "b": 2}


def test_two_spellings_of_one_path_are_one_file():
    """Chunk ids hash the canonical path, so a second spelling of the same file must be
    treated as a newer copy, or merge_insert sees one id twice."""
    import os

    from sift.index.store import _last_copy_per_file, norm_path

    upper = "/VAULT/technique/n.md"
    if norm_path(upper) != norm_path("/vault/technique/n.md"):
        pytest.skip("case-sensitive filesystem: these are two files")
    first = _rows("n", 2, path="/vault/technique/n.md", text="first")
    second = _rows("n", 2, path=upper.replace("/", os.sep), text="second")
    assert _last_copy_per_file(first + second) == second

    store = _store()
    store.replace_notes(first + second)
    assert _texts(store, "n") == ["second chunk0", "second chunk1"]


def test_upsert_note_is_one_commit_and_can_clear():
    store = _store()
    store.add_chunks(_rows("seed"))

    v0 = store.table().version
    store.upsert_note(_rows("a", 4))
    assert store.table().version == v0 + 1

    store.upsert_note(_rows("a", 2, text="shorter"))
    assert store.table().version == v0 + 2
    assert _texts(store, "a") == ["shorter chunk0", "shorter chunk1"]

    store.upsert_note([], note_id="a")
    assert "a" not in _ids(store)


# ---- chunk ids --------------------------------------------------------------


def test_chunk_ids_are_unique_per_file_not_per_note_id():
    from sift.index.store import ChunkRow, norm_path

    def row(path, i=0):
        return ChunkRow(
            note_id="CVE-2024-1",
            slug="s",
            type="cve",
            title="t",
            heading="",
            text="x",
            chunk_index=i,
            vector=_vec(0),
            path=path,
        )

    kev, nvd = row("/vault/cve/kev.md"), row("/vault/cve/nvd.md")
    assert kev.chunk_id != nvd.chunk_id
    assert kev.chunk_id == row("/vault/cve/kev.md").chunk_id, "must be stable across runs"
    assert kev.chunk_id != row("/vault/cve/kev.md", 1).chunk_id
    assert kev.chunk_id.startswith("CVE-2024-1::") and kev.chunk_id.endswith("::0")
    assert kev.to_record()["id"] == kev.chunk_id
    if norm_path("/vault/cve/KEV.md") == norm_path("/vault/cve/kev.md"):  # Windows
        assert row("/vault/cve/KEV.md").chunk_id == kev.chunk_id
    assert row("").chunk_id == "CVE-2024-1::0"  # no path: the old format


def test_two_files_sharing_an_id_can_be_indexed_in_one_batch():
    store = _store()
    batch = _rows("CVE-2024-1", 2, path="/vault/cve/kev.md") + _rows(
        "CVE-2024-1", 2, path="/vault/cve/nvd.md"
    )
    store.replace_notes(batch)
    store.replace_notes(batch)  # again, now that the rows exist: not "ambiguous"

    assert store.count() == 4
    assert sorted(p for (_n, p, _m) in store.indexed_files()) == [
        "/vault/cve/kev.md",
        "/vault/cve/nvd.md",
    ]


def test_cwe_list_written_with_commas_is_stored_as_tokens():
    from sift.index.store import ChunkRow

    rec = ChunkRow(
        note_id="n",
        slug="n",
        type="cve",
        title="t",
        heading="",
        text="x",
        chunk_index=0,
        vector=_vec(0),
        cwe=["CWE-79, cwe-89", "352"],
    ).to_record()
    assert rec["cwe_str"] == "CWE-79 CWE-89 CWE-352"


# ---- the cached table handle ---------------------------------------------


def test_drop_invalidates_the_cached_handle():
    store = _store()
    store.add_chunks(_rows("old", 3))
    store.drop()

    store.add_chunks(_rows("new", 2))  # must land in the recreated table

    assert store.count() == 2
    assert _store().count() == 2


def test_cached_handle_sees_writes_from_other_stores():
    """remember (one Store) then search_memory (another) must see the write at once."""
    reader = _store()
    reader.add_chunks(_rows("first"))
    assert reader.count() == 1

    writer = _store()
    writer.add_chunks(_rows("second", 2))
    assert reader.count() == 3

    writer.upsert_note(_rows("first", 1, text="edited"))
    assert _texts(reader, "first") == ["edited chunk0"]


def test_reads_do_not_create_the_table():
    from sift.index.store import TABLE

    store = _store()
    assert store.count() == 0
    assert store.note_mtimes() == {}
    assert store.indexed_files() == []
    assert store.rescore({"x": 50}) == 0
    assert TABLE not in store._table_names()


# ---- what is indexed ------------------------------------------------------


def test_indexed_views_carry_path_and_both_files_of_a_shared_id():
    from sift.index.store import ChunkRow

    store = _store()
    rows = _rows("CVE-1", 2, path="/vault/cve/kev.md") + _rows("CVE-1", 1, path="/vault/cve/nvd.md")
    rows += [
        ChunkRow(
            note_id="t1",
            slug="t1",
            type="technique",
            title="t",
            heading="",
            text="x",
            chunk_index=0,
            vector=_vec(0),
            path="/vault/technique/t1.md",
            mtime=123.5,
        )
    ]
    store.add_chunks(rows)

    files = sorted(store.indexed_files())
    assert [(n, p) for n, p, _ in files] == [
        ("CVE-1", "/vault/cve/kev.md"),
        ("CVE-1", "/vault/cve/nvd.md"),
        ("t1", "/vault/technique/t1.md"),
    ]
    assert store.note_index()["t1"] == (123.5, "/vault/technique/t1.md")
    assert store.note_mtimes()["t1"] == 123.5
    assert set(store.path_state()) == {p for _, p, _ in files}
    assert store.path_state()["/vault/technique/t1.md"] == ("t1", 123.5)
    assert store.indexed_ids() == {"CVE-1", "t1"}


# ---- rescore ----------------------------------------------------------------


def test_rescore_changes_quality_without_reembedding():
    store = _store()
    store.add_chunks(_rows("a", 2) + _rows("b", 2) + _rows("c", 1))
    before = {r["id"]: list(r["vector"]) for r in store.table().search().limit(0).to_list()}

    v0 = store.table().version
    changed = store.rescore({"a": 80, "b": 50, "c": 80, "missing": 10, "q": 300})

    assert changed == 2, "b already scores 50; missing is not indexed"
    assert store.table().version == v0 + 1, "one commit per distinct new score"
    rows = store.table().search().limit(0).to_list()
    assert {r["note_id"]: r["quality"] for r in rows} == {"a": 80, "b": 50, "c": 80}
    assert {r["id"]: list(r["vector"]) for r in rows} == before

    assert store.rescore({"a": 80}) == 0, "an unchanged score costs no commit"
    assert store.table().version == v0 + 1
