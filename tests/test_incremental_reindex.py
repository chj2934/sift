"""Incremental reindex.

Adding one note used to re-embed all 9,000+ notes — minutes of GPU work to index a
single file. Chunk rows already carry the source file's mtime, so an incremental pass
skips everything untouched. `--force` still rebuilds everything.
"""

from __future__ import annotations

import os
import time

import pytest

DIM = 64


class FakeEmbedder:
    """Counts how many passages were embedded — the thing under test."""

    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def __init__(self):
        self.embedded = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for i, tok in enumerate(text.split()):
            v[hash(tok) % DIM] += 1.0
            if i > 200:
                break
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts, *, kind="passage", batch_size=32):
        self.embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_DIM", str(DIM))
    from sift import config

    config.get_settings.cache_clear()
    embedder = FakeEmbedder()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: embedder)
    return embedder


def _write(vault, name: str, body: str = "some technique body text here"):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    return save_note(
        vault,
        Note(meta=Frontmatter(id=name, type="technique", title=name), body=body),
        stamp=False,
    )


def test_second_reindex_embeds_nothing(vault_path, fake):
    from sift.pipeline import reindex

    for i in range(4):
        _write(vault_path, f"note-{i}")

    first = reindex(force=True)
    assert first.notes == 4
    assert fake.embedded > 0

    fake.embedded = 0
    second = reindex()

    assert second.unchanged == 4
    assert second.notes == 0
    assert fake.embedded == 0, "unchanged notes were re-embedded"


def test_only_the_changed_note_is_reembedded(vault_path, fake):
    from sift.pipeline import reindex

    for i in range(4):
        _write(vault_path, f"note-{i}")
    reindex(force=True)

    path = _write(vault_path, "note-2", body="rewritten body with different words")
    future = time.time() + 10
    os.utime(path, (future, future))

    fake.embedded = 0
    stats = reindex()

    assert stats.notes == 1, "expected exactly the edited note to be reindexed"
    assert stats.unchanged == 3
    assert fake.embedded > 0


def test_a_new_note_is_picked_up(vault_path, fake):
    """capture_idea writing a note mid-hunt must land in the index on the next pass."""
    from sift.pipeline import reindex

    _write(vault_path, "existing")
    reindex(force=True)

    _write(vault_path, "brand-new")
    stats = reindex()

    assert stats.notes == 1
    assert stats.unchanged == 1


def test_force_ignores_mtimes_entirely(vault_path, fake):
    from sift.pipeline import reindex

    for i in range(3):
        _write(vault_path, f"note-{i}")
    reindex(force=True)

    fake.embedded = 0
    stats = reindex(force=True)

    assert stats.unchanged == 0, "--force must rebuild everything"
    assert stats.notes == 3
    assert fake.embedded > 0


def test_search_still_finds_an_unchanged_note_after_incremental(vault_path, fake):
    """The skip must not drop rows — the whole point is the index stays complete."""
    from sift.pipeline import reindex, search

    _write(vault_path, "cookie-sandwich", body="quoted cookie value parsing httponly bypass")
    _write(vault_path, "other", body="completely unrelated content about pdfs")
    reindex(force=True)
    reindex()  # incremental pass skips both

    hits = search("quoted cookie value parsing", k=2).hits
    assert any(h.note_id == "cookie-sandwich" for h in hits)


# --- reaping: deleted files, emptied notes, edited ids (K3) ---------------------------


def _bump(path, secs: float = 10.0) -> None:
    t = time.time() + secs
    os.utime(path, (t, t))


def _raw(vault, rel: str, note_id: str, title: str, body: str = "raw body text here"):
    """Write a note file directly, bypassing save_note's upsert-by-id."""
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    note = Note(meta=Frontmatter(id=note_id, type="technique", title=title), body=body)
    p.write_text(note.render(), encoding="utf-8")
    return p


def _indexed():
    """{note_id: {normalised stored path, ...}} straight from the table."""
    from sift.index.store import Store, norm_path

    out: dict[str, set[str]] = {}
    for nid, path, _mtime in Store().indexed_files():
        out.setdefault(nid, set()).add(norm_path(path))
    return out


def test_a_deleted_file_is_reaped_without_force(vault_path, fake):
    from sift.index.store import Store
    from sift.pipeline import reindex

    _write(vault_path, "keep-me")
    gone = _write(vault_path, "delete-me")
    reindex(force=True)
    assert {"keep-me", "delete-me"} <= _indexed().keys()  # positive control
    before = Store().count()

    gone.unlink()
    fake.embedded = 0
    stats = reindex()

    assert "delete-me" not in _indexed()
    assert "keep-me" in _indexed()
    assert Store().count() < before
    assert stats.removed == 1 and stats.reap_refused == 0
    assert fake.embedded == 0, "reaping must not re-embed anything"


def test_an_emptied_note_loses_its_rows(vault_path, fake):
    from sift.pipeline import reindex

    path = _write(vault_path, "emptied", body="text that will be deleted")
    _write(vault_path, "stays")
    reindex(force=True)
    assert "emptied" in _indexed()

    _write(vault_path, "emptied", body="   ")
    _bump(path)
    stats = reindex()

    assert "emptied" not in _indexed()
    assert "stays" in _indexed()
    assert stats.skipped == 1 and stats.removed == 1


def test_an_id_edited_in_place_replaces_the_old_id(vault_path, fake):
    from sift.pipeline import reindex

    path = _write(vault_path, "old-id")
    reindex(force=True)
    assert "old-id" in _indexed()

    text = path.read_text(encoding="utf-8").replace("id: old-id", "id: new-id")
    path.write_text(text, encoding="utf-8")
    _bump(path)
    reindex()

    ids = _indexed()
    assert "old-id" not in ids
    assert "new-id" in ids


def test_a_renamed_file_moves_its_rows_instead_of_duplicating_them(vault_path, fake):
    from sift.index.store import Store, norm_path
    from sift.pipeline import reindex

    old = _write(vault_path, "renamed-note", body="body about request smuggling")
    reindex(force=True)
    n = Store().count()

    new = old.with_name("A new name.md")
    os.replace(old, new)  # keeps the mtime: only the path changed
    reindex()

    assert _indexed()["renamed-note"] == {norm_path(new)}
    assert Store().count() == n


def test_a_note_that_fails_to_parse_keeps_its_rows(vault_path, fake, caplog):
    """A half-typed YAML block in Obsidian must not drop the note from search. Paired
    with a positive control in the same run: a deleted file IS reaped."""
    from sift.pipeline import reindex

    broken = _write(vault_path, "being-edited")
    gone = _write(vault_path, "really-deleted")
    reindex(force=True)

    broken.write_text("---\nid: [unclosed\ntype: technique\n---\nbody\n", encoding="utf-8")
    _bump(broken)
    gone.unlink()
    stats = reindex()

    ids = _indexed()
    assert "being-edited" in ids, "a parse error deleted index rows"
    assert "really-deleted" not in ids
    assert stats.unreadable == 1 and stats.removed == 1


def test_a_missing_vault_reaps_nothing(vault_path, fake, tmp_path):
    from sift.index.store import Store
    from sift.pipeline import reindex

    for i in range(3):
        _write(vault_path, f"note-{i}")
    reindex(force=True)
    n = Store().count()

    os.replace(vault_path, tmp_path / "unmounted")  # every stored path is now gone
    stats = reindex()

    assert Store().count() == n, "an unmounted vault wiped the index"
    assert stats.reap_refused == 3 and stats.removed == 0

    # Positive control: the operator can still confirm a real mass delete.
    stats = reindex(allow_mass_reap=True)
    assert Store().count() == 0 and stats.removed == 3


def test_a_mass_delete_is_refused_but_a_small_one_is_not(vault_path, fake, monkeypatch):
    from sift import pipeline

    monkeypatch.setattr(pipeline, "_REAP_FLOOR", 2)
    paths = [_write(vault_path, f"note-{i}") for i in range(5)]
    pipeline.reindex(force=True)

    for p in paths[:3]:  # 3 > max(2, 5 // 5): looks like an unmounted folder
        p.unlink()
    stats = pipeline.reindex()
    assert stats.reap_refused == 3
    assert {f"note-{i}" for i in range(5)} <= _indexed().keys()

    stats = pipeline.reindex(allow_mass_reap=True)
    assert stats.removed == 3
    assert set(_indexed()) == {"note-3", "note-4"}

    paths[3].unlink()  # 1 <= the floor: reaped by a routine pass
    stats = pipeline.reindex()
    assert stats.removed == 1 and set(_indexed()) == {"note-4"}


# --- files sharing an id (KEV/NVD twins) -------------------------------------------


def test_two_files_sharing_an_id_are_both_indexed_and_converge(vault_path, fake):
    """The index used to hold one twin per id and swap it (re-embedding both) on every
    incremental pass. Keyed by file, both stay indexed and a second pass embeds 0."""
    from sift.index.store import norm_path
    from sift.pipeline import reindex

    a = _raw(vault_path, "cve/CVE-2024-0001 (NVD).md", "CVE-2024-0001", "NVD record", "cvss data")
    b = _raw(vault_path, "cve/CVE-2024-0001 (KEV).md", "CVE-2024-0001", "KEV record", "kev data")
    first = reindex(force=True)
    assert _indexed()["CVE-2024-0001"] == {norm_path(a), norm_path(b)}
    assert first.duplicate_ids == 1

    fake.embedded = 0
    again = reindex()
    assert again.notes == 0 and again.unchanged == 2 and fake.embedded == 0

    _bump(b)
    edited = reindex()
    assert edited.notes == 1
    assert _indexed()["CVE-2024-0001"] == {norm_path(a), norm_path(b)}, "a twin was clobbered"


# --- write cost ------------------------------------------------------------------


def test_an_incremental_pass_is_one_commit_and_a_no_op_is_none(vault_path, fake):
    from sift.index.store import Store
    from sift.pipeline import reindex

    paths = [_write(vault_path, f"note-{i}") for i in range(6)]
    reindex(force=True)
    v0 = Store().table().version

    reindex()
    assert Store().table().version == v0, "a no-op reindex committed"

    for p in paths[:3]:
        _write(vault_path, p.stem, body="new words for this note")
        _bump(p)
    paths[5].unlink()
    stats = reindex()

    assert stats.notes == 3 and stats.removed == 1
    assert Store().table().version == v0 + 1, "edits and the reap should be one commit"


def test_force_leaves_a_keyword_index_behind(vault_path, fake):
    from sift.index.store import Store
    from sift.pipeline import reindex

    _write(vault_path, "fts-note", body="unique keyword zanzibar")
    reindex(force=True)
    assert Store().has_fts()


# --- mtime race, progress, staleness, stdout ----------------------------------------


def test_an_edit_saved_while_embedding_is_picked_up_next_pass(vault_path, fake):
    """The mtime used to be read at flush, after embedding, so an edit saved meanwhile
    was stored as 'indexed' with the old text and skipped forever."""
    from sift.pipeline import reindex

    path = _write(vault_path, "racy", body="ONE first version")
    reindex(force=True)
    _write(vault_path, "racy", body="TWO second version")
    _bump(path, 20)

    real_embed = fake.embed
    fired = []

    def embed_and_edit(texts, **kw):
        if not fired:
            fired.append(1)
            _write(vault_path, "racy", body="THREE saved during embedding")
            _bump(path, 40)
        return real_embed(texts, **kw)

    fake.embed = embed_and_edit
    racing = reindex()
    assert racing.requeued == 1 and racing.notes == 0

    fake.embed = real_embed
    after = reindex()
    assert after.notes == 1, "the edit made during embedding was never indexed"

    from sift.index.store import Store

    texts = Store().table().search().select(["note_id", "text"]).limit(100).to_list()
    assert any("THREE" in r["text"] for r in texts if r["note_id"] == "racy")


def test_progress_ends_at_the_walk_total(vault_path, fake):
    from sift.pipeline import count_notes, reindex

    paths = [_write(vault_path, f"note-{i}") for i in range(5)]
    reindex(force=True)
    _write(vault_path, "note-2", body="changed")
    _bump(paths[2])

    seen: list[int] = []
    stats = reindex(on_progress=seen.append)
    assert seen[-1] == count_notes() == stats.walked == 5


def test_count_notes_matches_the_walk_iter_notes_uses(vault_path, fake):
    """count_notes globbed the type folders while iter_notes rglobbed the vault."""
    from sift.pipeline import count_notes
    from sift.vault.notes import iter_notes

    _write(vault_path, "plain")
    _raw(vault_path, "technique/sub/nested.md", "nested", "Nested note")
    _raw(vault_path, "Loose note.md", "loose", "Loose note")
    _raw(vault_path, ".trash/deleted.md", "deleted", "Deleted note")
    _raw(vault_path, "_templates/tpl.md", "tpl", "Template")

    assert count_notes() == sum(1 for _ in iter_notes(vault_path)) == 3


def test_the_index_records_its_chunker_and_reports_a_stale_one(vault_path, fake, caplog):
    import json

    from sift.index.store import Store
    from sift.pipeline import index_meta, reindex
    from sift.vault.chunk import CHUNKER_VERSION

    _write(vault_path, "note-0")
    reindex(force=True)
    meta = index_meta()
    assert meta["chunker_version"] == CHUNKER_VERSION and meta["embed_model"] == "fake"
    assert reindex().stale_index is None

    meta["chunker_version"] = "chars-800"
    (Store().db_path / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with caplog.at_level("WARNING", logger="sift.pipeline"):
        stats = reindex()
    assert "chars-800" in (stats.stale_index or "")
    assert any("reindex --force" in r.getMessage() for r in caplog.records)


def test_reindex_never_writes_to_stdout(vault_path, fake, capfd):
    """pipeline is reachable from the MCP server, whose stdout is the JSON-RPC wire."""
    from sift.pipeline import reindex

    _write(vault_path, "fine")
    (vault_path / "technique" / "empty.md").write_text("", encoding="utf-8")
    (vault_path / "technique" / "no frontmatter.md").write_text("# just text\n", encoding="utf-8")
    reindex(force=True)
    reindex()

    assert capfd.readouterr().out == ""
