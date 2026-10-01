"""`prune.apply_prune` / `restore_quarantine`: the I/O half of `sift prune --yes`."""

from __future__ import annotations

from datetime import date

import pytest


class FakeStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.batches: list[list[str]] = []
        self.fail = fail

    def delete_notes(self, ids) -> int:
        if self.fail:
            raise RuntimeError("table locked")
        self.batches.append(list(ids))
        return 2 * len(self.batches[-1])  # pretend two chunks per note


class PathStore(FakeStore):
    """The current Store, which can also delete one file's rows by stored `path`."""

    def __init__(self, stored_paths) -> None:
        super().__init__()
        self.stored = list(stored_paths)
        self.deleted_paths: list[list[str]] = []

    def path_state(self):
        return {p: ("some-id", 0.0) for p in self.stored}

    def delete_paths(self, paths) -> int:
        self.deleted_paths.append(list(paths))
        return 3 * len(self.deleted_paths[-1])


def _write(
    vault,
    *,
    nid,
    ntype="cve",
    source="nvd",
    title=None,
    sub=None,
    created=date(2019, 1, 1),
    url=None,
    tags=None,
    extra=None,
):
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    meta = Frontmatter(
        id=nid,
        type=ntype,
        title=title or nid,
        source=source,
        created=created,
        url=url,
        tags=tags or [],
        extra=extra or {},
    )
    note = Note(meta=meta, body=f"body of {nid}")
    folder = vault / ntype / sub if sub else vault / ntype
    folder.mkdir(parents=True, exist_ok=True)
    note.path = folder / f"{meta.title}.md"
    note.path.write_text(note.render(), encoding="utf-8")
    return note


def test_quarantine_dir_is_under_the_db_parent_and_outside_the_vault(vault_path):
    from sift.config import get_settings
    from sift.prune import quarantine_dir

    q = quarantine_dir("20261001-120000")
    assert q == get_settings().resolved_db().parent / "pruned" / "20261001-120000"
    assert not q.resolve().is_relative_to(vault_path.resolve())


def test_apply_prune_moves_tombstones_and_unindexes(vault_path):
    from sift.prune import apply_prune
    from sift.tombstones import load_tombstones

    url = "https://nvd.nist.gov/vuln/detail/CVE-2019-1"
    a = _write(vault_path, nid="CVE-2019-1", title="Same", sub="2019", url=url)
    b = _write(vault_path, nid="CVE-2019-2", title="Same", sub="2020")  # same basename
    c = _write(
        vault_path,
        nid="h1-9",
        ntype="report",
        source="hackerone-public",
        url="https://hackerone.com/reports/9",
    )
    before = {n.meta.id: n.path.read_bytes() for n in (a, b, c)}
    store = FakeStore()

    res = apply_prune([a, b, c], vault=vault_path, store=store)

    assert sorted(res.moved) == ["CVE-2019-1", "CVE-2019-2", "h1-9"]
    assert not res.failed and not res.refused and res.index_error is None
    q = res.quarantine
    assert not q.resolve().is_relative_to(vault_path.resolve())
    for n in (a, b, c):
        assert not n.path.exists()  # out of the vault...
        moved = q / n.path.relative_to(vault_path)  # ...at its vault-relative path
        assert moved.read_bytes() == before[n.meta.id]
    assert store.batches == [["CVE-2019-1", "CVE-2019-2", "h1-9"]]  # one batched delete
    assert res.rows_deleted == 6
    assert res.unindexed == ["CVE-2019-1", "CVE-2019-2", "h1-9"]

    t = load_tombstones()
    assert res.tombstoned == 3
    assert t.has_id("CVE-2019-1", source="nvd") and t.has_url(url, source="nvd")
    assert t.has_id("h1-9", source="hackerone-public")
    # The same CVE arriving from CISA KEV is not blocked by an nvd verdict.
    assert not t.has_id("CVE-2019-1", source="cisa-kev")


def test_apply_prune_refuses_anything_classify_would_keep(vault_path):
    from sift.prune import apply_prune
    from sift.tombstones import tombstones_path

    protected = [
        _write(vault_path, nid="tool-acme", ntype="tool", source="nvd"),
        _write(vault_path, nid="h1mine-1", ntype="report", source="hackerone-mine", tags=["mine"]),
        _write(
            vault_path,
            nid="repo-x-20260901123456789",
            ntype="report",
            source="hackerone-public",
            created=None,
        ),
        _write(vault_path, nid="local-notes", ntype="report", source="my-notes"),
        _write(vault_path, nid="CVE-2010-1", source="nvd", created=None),  # undated
        _write(vault_path, nid="CVE-2010-2", source="nvd", extra={"authored_via": "sift-remember"}),
    ]
    store = FakeStore()
    res = apply_prune(protected, vault=vault_path, store=store)

    assert res.moved == [] and len(res.refused) == len(protected)
    assert all(n.path.exists() for n in protected)
    assert store.batches == []
    assert not tombstones_path().exists()
    assert not res.quarantine.exists()  # nothing to move, so no folder either


def test_kept_twin_keeps_its_chunks(vault_path):
    """KEV and NVD notes share a CVE id. Dropping the NVD twin must not delete the
    chunks of the KEV twin that stays."""
    from sift.prune import apply_prune

    nvd = _write(vault_path, nid="CVE-2019-5", title="CVE-2019-5 nvd")
    store = FakeStore()
    res = apply_prune([nvd], vault=vault_path, keep_ids={"CVE-2019-5"}, store=store)
    assert res.moved == ["CVE-2019-5"]
    assert res.unindexed == [] and store.batches == []


def test_departed_twin_loses_only_its_own_rows_by_path(vault_path):
    import os

    from sift.prune import apply_prune

    nvd = _write(vault_path, nid="CVE-2019-5", title="CVE-2019-5 nvd")
    kev_path = str(vault_path / "cve" / "CVE-2019-5 kev.md")
    # The index stores paths as written; Windows may spell the same file differently.
    stored_nvd = str(nvd.path).upper() if os.name == "nt" else str(nvd.path)
    store = PathStore([stored_nvd, kev_path])
    res = apply_prune([nvd], vault=vault_path, keep_ids={"CVE-2019-5"}, store=store)

    assert store.batches == []  # never by id: the KEV twin keeps its chunks
    assert store.deleted_paths == [[stored_nvd]]  # the stored spelling is passed back
    assert res.unindexed == [] and res.unindexed_paths == [stored_nvd]
    assert res.rows_deleted == 3


def _lock(monkeypatch, prune, note):
    """Make every rename of `note`'s file fail like a Windows sharing violation."""
    real_rename = prune.os.rename

    def rename(src, dst):
        if str(src) == str(note.path):
            raise PermissionError(13, "The process cannot access the file", str(src))
        return real_rename(src, dst)

    monkeypatch.setattr(prune.os, "rename", rename)
    monkeypatch.setattr(prune, "_MOVE_ATTEMPTS", 2)


def test_failed_move_leaves_the_note_indexed_and_untombstoned(vault_path, monkeypatch):
    import sift.prune as prune
    from sift.tombstones import load_tombstones

    locked = _write(vault_path, nid="CVE-2019-11")
    ok = _write(vault_path, nid="CVE-2019-10")
    _lock(monkeypatch, prune, locked)
    store = FakeStore()
    res = prune.apply_prune([locked, ok], vault=vault_path, store=store)

    assert res.moved == ["CVE-2019-10"]  # one failure doesn't stop the loop
    assert len(res.failed) == 1 and "CVE-2019-11.md" in res.failed[0]
    assert locked.path.exists()
    assert store.batches == [["CVE-2019-10"]]  # the locked note keeps its chunks
    t = load_tombstones()
    assert t.has_id("CVE-2019-10") and not t.has_id("CVE-2019-11")


def test_id_still_on_disk_after_a_failed_move_keeps_its_chunks(vault_path, monkeypatch):
    import sift.prune as prune

    locked = _write(vault_path, nid="CVE-2019-11")
    twin = _write(vault_path, nid="CVE-2019-11", title="CVE-2019-11 copy")  # same id
    _lock(monkeypatch, prune, locked)
    store = FakeStore()
    res = prune.apply_prune([locked, twin], vault=vault_path, store=store)

    assert res.moved == ["CVE-2019-11"] and not twin.path.exists()
    assert locked.path.exists()
    assert store.batches == [] and res.unindexed == []


def test_moves_and_tombstones_happen_under_the_vault_write_lock(vault_path, monkeypatch):
    import contextlib

    import sift.prune as prune
    import sift.tombstones as tombstones
    from sift.vault import notes

    held = {"depth": 0, "vaults": []}

    @contextlib.contextmanager
    def fake_lock(vault):
        held["vaults"].append(vault)
        held["depth"] += 1
        try:
            yield
        finally:
            held["depth"] -= 1

    seen: list[tuple[str, int]] = []
    real_move, real_record = prune._move, tombstones.record_note_tombstones

    def move(src, dst):
        seen.append(("move", held["depth"]))
        return real_move(src, dst)

    def record(notes_, **kw):
        seen.append(("tombstone", held["depth"]))
        return real_record(notes_, **kw)

    monkeypatch.setattr(notes, "write_lock", fake_lock)
    monkeypatch.setattr(prune, "_move", move)
    monkeypatch.setattr(tombstones, "record_note_tombstones", record)

    res = prune.apply_prune(
        [_write(vault_path, nid="CVE-2019-13")], vault=vault_path, store=FakeStore()
    )
    assert res.moved == ["CVE-2019-13"]
    assert seen == [("move", 1), ("tombstone", 1)]
    assert held["vaults"] == [vault_path.resolve()] and held["depth"] == 0

    seen.clear()
    back = prune.restore_quarantine(res.quarantine, vault=vault_path)
    assert back.restored and seen == [("move", 1)]


def test_transient_lock_is_retried(vault_path, monkeypatch):
    import sift.prune as prune

    note = _write(vault_path, nid="CVE-2019-12")
    real_rename = prune.os.rename
    calls = {"n": 0}

    def rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "sharing violation", str(src))
        return real_rename(src, dst)

    monkeypatch.setattr(prune.os, "rename", rename)
    res = prune.apply_prune([note], vault=vault_path, store=FakeStore())
    assert res.moved == ["CVE-2019-12"] and not res.failed and calls["n"] == 2


def test_already_gone_file_is_unindexed_but_not_tombstoned(vault_path):
    from sift.prune import apply_prune
    from sift.tombstones import load_tombstones

    note = _write(vault_path, nid="CVE-2019-20")
    note.path.unlink()
    store = FakeStore()
    res = apply_prune([note], vault=vault_path, store=store)
    assert res.already_gone == ["CVE-2019-20"] and res.moved == []
    assert store.batches == [["CVE-2019-20"]]  # clears its orphan chunks
    assert not load_tombstones().has_id("CVE-2019-20")


def test_empty_drop_does_nothing(vault_path):
    from sift.prune import apply_prune

    store = FakeStore()
    res = apply_prune([], vault=vault_path, store=store)
    assert res.moved == [] and store.batches == []
    assert not res.quarantine.exists()


def test_index_failure_is_reported_not_raised(vault_path):
    from sift.prune import apply_prune
    from sift.tombstones import load_tombstones

    note = _write(vault_path, nid="CVE-2019-30")
    res = apply_prune([note], vault=vault_path, store=FakeStore(fail=True))
    assert res.index_error and "table locked" in res.index_error
    assert res.moved == ["CVE-2019-30"] and not note.path.exists()
    assert res.unindexed == []
    assert load_tombstones().has_id("CVE-2019-30")


def test_quarantine_inside_the_vault_is_refused(vault_path):
    from sift.prune import apply_prune

    note = _write(vault_path, nid="CVE-2019-50")
    with pytest.raises(ValueError, match="inside the vault"):
        apply_prune([note], vault=vault_path, dest=vault_path / "pruned", store=FakeStore())
    assert note.path.exists()


def test_two_prunes_in_one_second_get_separate_folders(vault_path, tmp_path, monkeypatch):
    import sift.prune as prune

    base = tmp_path / "data" / "pruned" / "20261001-120000"
    monkeypatch.setattr(prune, "quarantine_dir", lambda stamp=None: base)
    first = prune.apply_prune(
        [_write(vault_path, nid="CVE-2019-60")], vault=vault_path, store=FakeStore()
    )
    second = prune.apply_prune(
        [_write(vault_path, nid="CVE-2019-61")], vault=vault_path, store=FakeStore()
    )
    assert first.quarantine == base.resolve()
    assert second.quarantine == base.resolve().with_name(base.name + "-2")
    assert (first.quarantine / "cve" / "CVE-2019-60.md").exists()
    assert (second.quarantine / "cve" / "CVE-2019-61.md").exists()


def test_restore_quarantine_round_trip(vault_path):
    from sift.prune import apply_prune, restore_quarantine
    from sift.tombstones import load_tombstones

    a = _write(vault_path, nid="CVE-2019-70", url="https://nvd.nist.gov/vuln/detail/CVE-2019-70")
    b = _write(vault_path, nid="CVE-2019-71", sub="deep")
    c = _write(vault_path, nid="CVE-2019-72")
    original = {n.meta.id: n.path.read_bytes() for n in (a, b, c)}
    res = apply_prune([a, b, c], vault=vault_path, store=FakeStore())
    assert len(res.moved) == 3

    # The vault reused one path in the meantime: that note must stay in quarantine.
    c.path.write_text("---\nid: other\ntype: cve\ntitle: other\n---\n\nnew\n", "utf-8")

    back = restore_quarantine(res.quarantine, vault=vault_path)
    assert sorted(back.restored) == sorted(str(n.path.relative_to(vault_path)) for n in (a, b))
    assert back.conflicts == [str(c.path.relative_to(vault_path))]
    assert a.path.read_bytes() == original["CVE-2019-70"]
    assert b.path.read_bytes() == original["CVE-2019-71"]
    assert "new" in c.path.read_text("utf-8")  # never overwritten
    t = load_tombstones()
    assert not t.has_id("CVE-2019-70") and not t.has_id("CVE-2019-71")
    assert not t.has_url("https://nvd.nist.gov/vuln/detail/CVE-2019-70")
    assert t.has_id("CVE-2019-72")  # still in quarantine, still tombstoned
    assert back.untombstoned == 2
    # Emptied folders are gone; the conflicting file keeps its folder.
    assert not (res.quarantine / "cve" / "deep").exists()
    assert (res.quarantine / c.path.relative_to(vault_path)).exists()


def test_restore_refuses_a_path_overlapping_the_vault(vault_path):
    from sift.prune import restore_quarantine

    empty = vault_path / "inbox"
    empty.mkdir()
    for wrong in (vault_path, empty, vault_path.parent):
        with pytest.raises(ValueError, match="overlaps the vault"):
            restore_quarantine(wrong, vault=vault_path)
    assert empty.is_dir()  # its empty folders were not "cleaned up"
