"""Index and vault maintenance commands: reindex, compact, prune, doctor, status, trash.

The embedder is the conftest `fake_embedder` (no model), the index a temp LanceDB.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from datetime import date, timedelta

import pytest


def _run(*args: str):
    from typer.testing import CliRunner

    from sift.cli import app

    return CliRunner().invoke(app, list(args))


def _flat(text: str) -> str:
    return " ".join(text.split())


def _note(vault, note_id, *, type="technique", title=None, body=None, **meta):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    fm = Frontmatter(id=note_id, type=type, title=title or note_id, **meta)
    text = body or f"{note_id} body text about a technique worth indexing"
    return save_note(vault, Note(meta=fm, body=text), stamp=False)


def _indexed_ids() -> set[str]:
    from sift.index.store import Store

    return Store().indexed_ids()


# --------------------------------------------------------------------------- #
# reindex
# --------------------------------------------------------------------------- #
class _Bar:
    def __init__(self) -> None:
        self.total = None
        self.updates: list[dict] = []

    def update(self, _task, **kw) -> None:
        self.updates.append(kw)


@pytest.fixture
def bar(monkeypatch):
    rec = _Bar()

    @contextmanager
    def fake_bar(description, *, total=None):
        rec.total = total
        yield rec, 1

    monkeypatch.setattr("sift.cli._bar", fake_bar)
    return rec


def test_reindex_progress_ends_at_the_total(vault_path, fake_embedder, bar):
    for name in ("a", "b", "c"):
        _note(vault_path, f"note-{name}")
    res = _run("reindex")
    assert res.exit_code == 0, res.output
    assert bar.total == 3
    assert bar.updates[-1] == {"total": 3, "completed": 3}
    assert "done 3 notes" in _flat(res.stdout)
    assert _indexed_ids() == {"note-a", "note-b", "note-c"}


def test_reindex_reaps_a_deleted_note(vault_path, fake_embedder, bar):
    paths = [_note(vault_path, f"note-{n}") for n in ("a", "b", "c")]
    assert _run("reindex").exit_code == 0
    assert "note-b" in _indexed_ids()  # positive control
    paths[1].unlink()
    res = _run("reindex")
    assert res.exit_code == 0, res.output
    assert "removed the index rows of 1 deleted or emptied note(s)" in _flat(res.stdout)
    assert _indexed_ids() == {"note-a", "note-c"}


def test_reindex_force_refuses_a_width_mismatch_before_dropping(
    vault_path, fake_embedder, bar, monkeypatch
):
    from sift import config
    from sift.index.store import Store

    _note(vault_path, "note-a")
    assert _run("reindex").exit_code == 0
    rows = Store().count()
    assert rows > 0

    monkeypatch.setenv("SIFT_EMBED_DIM", "32")  # the fake embedder makes 64-dim vectors
    config.get_settings.cache_clear()
    res = _run("reindex", "--force")
    assert res.exit_code == 1
    assert "nothing was dropped" in _flat(res.stderr)

    monkeypatch.setenv("SIFT_EMBED_DIM", "64")
    config.get_settings.cache_clear()
    assert Store().count() == rows


def test_reindex_force_rebuilds_when_the_width_fits(vault_path, fake_embedder, bar):
    _note(vault_path, "note-a")
    _note(vault_path, "note-b")
    res = _run("reindex", "--force")
    assert res.exit_code == 0, res.output
    assert _indexed_ids() == {"note-a", "note-b"}


def test_reindex_reports_a_refused_mass_reap(vault_path, fake_embedder, bar, monkeypatch):
    from sift.pipeline import ReindexStats

    seen: dict = {}

    def fake_reindex(**kwargs):
        seen.update(kwargs)
        return ReindexStats(walked=0, reap_refused=60, stale_index="the index records no chunker")

    monkeypatch.setattr("sift.pipeline.reindex", fake_reindex)
    res = _run("reindex")
    assert res.exit_code == 0
    assert seen["allow_mass_reap"] is False and seen["force"] is False
    err = _flat(res.stderr)
    assert "kept the index rows of 60 missing note(s)" in err
    assert "--allow-mass-reap" in err
    assert "sift reindex --force" in err

    assert _run("reindex", "--allow-mass-reap").exit_code == 0
    assert seen["allow_mass_reap"] is True


def test_reindex_rescore_fixes_stored_quality_without_embedding(vault_path, fake_embedder, bar):
    from sift.index.store import Store
    from sift.quality import score_note
    from sift.vault.notes import load_note

    p = _note(vault_path, "note-a")
    _note(vault_path, "note-b")
    assert _run("reindex").exit_code == 0
    assert Store().rescore({"note-a": 1}) == 1  # simulate a score from older rules
    embedded = fake_embedder.embedded

    res = _run("reindex", "--rescore")
    assert res.exit_code == 0, res.output
    m = re.search(r"rescored (\d+) of 2 notes", _flat(res.stdout))
    assert m and int(m.group(1)) >= 1
    assert fake_embedder.embedded == embedded  # nothing re-embedded

    note = load_note(p)
    tbl = Store().table().search().select(["note_id", "quality"]).limit(0).to_arrow()
    stored = {
        q
        for nid, q in zip(tbl["note_id"].to_pylist(), tbl["quality"].to_pylist(), strict=True)
        if nid == "note-a"
    }
    assert stored == {score_note(note.meta, note.body)}


def test_rescore_and_force_do_not_combine():
    res = _run("reindex", "--rescore", "--force")
    assert res.exit_code == 1
    assert "don't combine" in res.stderr


# --------------------------------------------------------------------------- #
# compact
# --------------------------------------------------------------------------- #
def test_compact_reports_before_and_after(vault_path, fake_embedder, bar):
    for n in ("a", "b"):
        _note(vault_path, f"note-{n}")
    assert _run("reindex").exit_code == 0
    res = _run("compact")
    assert res.exit_code == 0, res.output
    out = _flat(res.stdout)
    assert "fragments" in out and "disk" in out and "took" in out
    assert _indexed_ids() == {"note-a", "note-b"}


def test_compact_with_no_index():
    res = _run("compact")
    assert res.exit_code == 0, res.output
    assert "nothing to compact" in res.stdout


def test_compact_refuses_a_short_retention_without_the_flag(monkeypatch):
    called = []
    monkeypatch.setattr(
        "sift.index.store.Store.optimize", lambda self, *a, **k: called.append(1) or {}
    )
    res = _run("compact", "--retain-minutes", "5")
    assert res.exit_code == 1
    assert "--unsafe-zero-retention" in _flat(res.stderr)
    assert called == []


def test_compact_unsafe_flag_passes_zero_retention(monkeypatch):
    seen: dict = {}

    def optimize(self, retain=None, **kwargs):
        seen.update(retain=retain, **kwargs)
        info = {"fragments": 3, "version": 9, "rows": 10, "disk_bytes": 4096}
        return {"before": info, "after": dict(info, fragments=1), "seconds": 0.1, "error": None}

    monkeypatch.setattr("sift.index.store.Store.optimize", optimize)
    res = _run("compact", "--unsafe-zero-retention")
    assert res.exit_code == 0, res.output
    assert seen["retain"] == timedelta(0)
    assert seen["allow_unsafe_retain"] is True
    assert seen["force"] is True and seen["measure_disk"] is True
    assert "every sift process" in _flat(res.stderr)

    res = _run("compact")
    assert res.exit_code == 0
    assert seen["retain"] == timedelta(minutes=60)
    assert seen["allow_unsafe_retain"] is False


def test_compact_failure_exits_1(monkeypatch):
    monkeypatch.setattr(
        "sift.index.store.Store.optimize",
        lambda self, *a, **k: {"before": {"fragments": 2}, "after": {}, "error": "commit conflict"},
    )
    res = _run("compact")
    assert res.exit_code == 1
    assert "commit conflict" in res.stderr


# --------------------------------------------------------------------------- #
# prune
# --------------------------------------------------------------------------- #
def _prunable(vault):
    """An old, low-signal NVD CVE (dropped) and a user's finding (kept)."""
    old = _note(
        vault,
        "CVE-2019-0001",
        type="cve",
        title="CVE-2019-0001 test",
        source="nvd",
        created=date(2019, 5, 1),
        severity="medium",
        url="https://nvd.nist.gov/vuln/detail/CVE-2019-0001",
    )
    mine = _note(vault, "finding-mine", type="finding", title="My finding")
    return old, mine


def test_prune_dry_run_changes_nothing(vault_path, monkeypatch):
    old, mine = _prunable(vault_path)
    monkeypatch.setattr("sift.prune.apply_prune", lambda *a, **k: pytest.fail("not a dry run"))
    res = _run("prune")
    assert res.exit_code == 0, res.output
    assert "dry run" in res.stdout
    assert "drop 1" in _flat(res.stdout)
    assert old.exists() and mine.exists()


def test_prune_yes_quarantines_tombstones_and_restores(vault_path, monkeypatch):
    from sift.tombstones import load_tombstones

    def no_rebuild(*a, **k):
        raise AssertionError("prune must not rebuild the index")

    monkeypatch.setattr("sift.pipeline.reindex", no_rebuild)
    old, mine = _prunable(vault_path)
    rel = old.relative_to(vault_path)

    res = _run("prune", "--yes")
    assert res.exit_code == 0, res.output
    assert "moved 1 note(s)" in _flat(res.stdout)
    assert not old.exists() and mine.exists()
    quarantined = list((vault_path.parent / "pruned").rglob("*.md"))
    assert [p.name for p in quarantined] == [old.name]
    assert load_tombstones().has_id("CVE-2019-0001")
    folder = quarantined[0].parents[len(rel.parts) - 1]

    back = _run("prune", "--restore", str(folder))
    assert back.exit_code == 0, back.output
    assert "restored 1 note(s)" in _flat(back.stdout)
    assert (vault_path / rel).exists()
    assert not load_tombstones().has_id("CVE-2019-0001")


def test_prune_passes_the_kept_ids(vault_path, monkeypatch):
    from sift.prune import PruneResult

    seen: dict = {}

    def apply(drop, *, vault, keep_ids):
        seen.update(drop=[n.meta.id for n in drop], keep_ids=set(keep_ids))
        return PruneResult(quarantine=vault.parent / "q", moved=["CVE-2019-0001"], tombstoned=1)

    monkeypatch.setattr("sift.prune.apply_prune", apply)
    _prunable(vault_path)
    res = _run("prune", "--yes")
    assert res.exit_code == 0, res.output
    assert seen["drop"] == ["CVE-2019-0001"]
    # A dropped NVD twin must not delete the chunks of a kept twin sharing its id.
    assert "finding-mine" in seen["keep_ids"]


def test_prune_reports_an_index_error(vault_path, monkeypatch):
    from sift.prune import PruneResult

    monkeypatch.setattr(
        "sift.prune.apply_prune",
        lambda drop, **k: PruneResult(
            quarantine=vault_path.parent / "q", moved=["x"], index_error="table locked"
        ),
    )
    _prunable(vault_path)
    res = _run("prune", "--yes")
    assert res.exit_code == 1
    assert "table locked" in res.stderr
    assert "sift reindex" in _flat(res.stderr)


def test_prune_restore_needs_a_folder():
    res = _run("prune", "--restore", "no/such/folder")
    assert res.exit_code == 1
    assert "no such quarantine folder" in res.stderr


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def test_status_counts_from_the_catalog(vault_path, monkeypatch):
    from sift.ingest.base import IngestResult, record_run

    _note(vault_path, "note-a")
    _note(vault_path, "finding-1", type="finding")
    (vault_path / "report").mkdir(exist_ok=True)
    (vault_path / "report" / "empty.md").write_bytes(b"")
    record_run("research", IngestResult("research", written=2), complete=False, aborted="timeout")

    def no_parse(*a, **k):
        raise AssertionError("status must not parse the whole vault")

    monkeypatch.setattr("sift.vault.notes.iter_notes", no_parse)
    res = _run("status")
    assert res.exit_code == 0, res.output
    out = _flat(res.stdout)
    assert re.search(r"technique\W+1\b", out)
    assert re.search(r"finding\W+1\b", out)
    assert re.search(r"total\W+2\b", out)
    assert "1 note file(s) can't be read" in out
    assert "no index yet" in out
    assert "incomplete" in out and "timeout" in out


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_doctor_reports_without_changing_anything(vault_path):
    _write(
        vault_path / "writeup" / "A.md",
        "---\nid: dup-id\ntype: writeup\ntitle: A\nsource: blog-a\n"
        "url: https://a.example/x\n---\n\nbody one\n",
    )
    _write(
        vault_path / "writeup" / "B.md",
        "---\nid: dup-id\ntype: writeup\ntitle: B\nsource: blog-b\n"
        "url: https://b.example/y\n---\n\nbody two\n",
    )
    (vault_path / "report").mkdir()
    (vault_path / "report" / "empty.md").write_bytes(b"")
    before = {p: p.stat().st_mtime_ns for p in vault_path.rglob("*")}

    res = _run("doctor", "--no-index")
    assert res.exit_code == 0, res.output
    out = res.stdout
    assert "dup-id" in out
    assert "writeup/A.md" in out and "writeup/B.md" in out
    assert "blog-a" in out and "blog-b" in out
    assert "differ" in out
    assert "report/empty.md" in out
    assert {p: p.stat().st_mtime_ns for p in vault_path.rglob("*")} == before


def test_doctor_marks_identical_bodies(vault_path):
    for name in ("A", "B"):
        _write(
            vault_path / "writeup" / f"{name}.md",
            f"---\nid: same\ntype: writeup\ntitle: {name}\n---\n\nthe same body\n",
        )
    res = _run("doctor", "--no-index")
    assert res.exit_code == 0
    assert "identical" in res.stdout


def test_doctor_compares_the_index_with_the_vault(vault_path, fake_embedder, bar):
    _note(vault_path, "note-a")
    gone = _note(vault_path, "note-b")
    assert _run("reindex").exit_code == 0

    clean = _run("doctor")
    assert clean.exit_code == 0, clean.output
    assert "index matches the vault" in _flat(clean.stdout)  # positive control

    gone.unlink()
    _note(vault_path, "note-c")  # on disk, not indexed yet
    res = _run("doctor")
    out = _flat(res.stdout)
    assert "1 indexed file(s) are no longer on disk" in out
    assert "note-b" in out
    assert "1 note file(s) are not indexed yet" in out
    assert "note-b" in _indexed_ids()  # report only: doctor removed nothing


# --------------------------------------------------------------------------- #
# trash
# --------------------------------------------------------------------------- #
def test_trash_restore_undoes_a_forget(vault_path):
    from sift.tombstones import load_tombstones, record_tombstones
    from sift.vault.notes import delete_note, load_note

    p = _note(vault_path, "finding-x", type="finding", title="Finding X", url="https://ex.com/x")
    rel = p.relative_to(vault_path)
    assert delete_note("finding-x", vault=vault_path, reason="dead end")
    record_tombstones(ids=["finding-x"], urls=["https://ex.com/x"])
    assert not p.exists()

    listed = _run("trash", "list")
    assert listed.exit_code == 0
    assert "finding-x" in listed.stdout and "dead end" in listed.stdout

    res = _run("trash", "restore", "finding-x")
    assert res.exit_code == 0, res.output
    back = vault_path / rel
    assert back.exists()
    extra = load_note(back).meta.extra
    assert "deleted" not in extra and "deleted_reason" not in extra
    assert not load_tombstones().has_id("finding-x")
    assert not list((vault_path / ".trash").rglob("*.md"))
    assert "sift reindex" in _flat(res.stdout)

    again = _run("trash", "restore", "finding-x")
    assert again.exit_code == 1
    assert "no trashed note" in again.stderr


def test_trash_restore_refuses_when_the_id_is_back(vault_path):
    from sift.vault.notes import delete_note

    _note(vault_path, "finding-y", type="finding", title="Finding Y")
    delete_note("finding-y", vault=vault_path, reason="oops")
    _note(vault_path, "finding-y", type="finding", title="Finding Y again")

    res = _run("trash", "restore", "finding-y")
    assert res.exit_code == 1
    assert "in the vault again" in _flat(res.stderr)
    assert len(list((vault_path / ".trash").rglob("*.md"))) == 1  # left where it was
