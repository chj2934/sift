from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading


def _note(nid, *, url=None, source=None, ntype="cve"):
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    return Note(meta=Frontmatter(id=nid, type=ntype, title=nid, url=url, source=source), body="b")


def test_empty_when_there_is_no_ledger():
    from sift.tombstones import load_tombstones, tombstones_path

    assert not tombstones_path().exists()
    t = load_tombstones()
    assert len(t) == 0
    assert not t.has_id("CVE-2019-1")
    assert not t.has_url("https://nvd.nist.gov/vuln/detail/CVE-2019-1")
    assert not t.has_id(None) and not t.has_url(None) and not t.has_url("")


def test_record_then_load():
    from sift.tombstones import load_tombstones, record_tombstones

    assert record_tombstones(["CVE-2019-1", "h1-5"], ["https://example.com/a"]) == 3
    t = load_tombstones()
    assert len(t) == 3
    assert t.has_id("CVE-2019-1") and t.has_id("h1-5")
    assert not t.has_id("CVE-2019-2")
    assert t.has_url("https://example.com/a")
    # Recorded without a source (a user "forget"): blocks every source.
    assert t.has_id("CVE-2019-1", source="cisa-kev")
    assert t.has_url("https://example.com/a", source="research-feed")


def test_ledger_lives_under_the_db_dir_not_the_vault(vault_path):
    from sift.config import get_settings
    from sift.tombstones import record_tombstones, tombstones_path

    record_tombstones(["x"])
    p = tombstones_path()
    assert p.is_file()
    assert p.parent == get_settings().resolved_db()
    assert not p.resolve().is_relative_to(vault_path.resolve())
    assert list(vault_path.rglob("*")) == []


def test_source_scoped_tombstone_only_blocks_its_own_source():
    """An NVD CVE pruned as low-signal must come back when CISA KEV catalogues it:
    both sources use the CVE id and the same NVD url."""
    from sift.tombstones import load_tombstones, record_tombstones

    url = "https://nvd.nist.gov/vuln/detail/CVE-2019-1234"
    record_tombstones(["CVE-2019-1234"], [url], source="nvd", reason="sift prune")
    t = load_tombstones()
    assert t.has_id("CVE-2019-1234", source="nvd")
    assert t.has_url(url, source="nvd")
    assert not t.has_id("CVE-2019-1234", source="cisa-kev")
    assert not t.has_url(url, source="cisa-kev")
    assert t.has_id("CVE-2019-1234")  # an unscoped question sees any tombstone


def test_record_note_tombstones_keeps_id_url_and_source_together():
    from sift.tombstones import load_tombstones, record_note_tombstones, tombstones_path

    notes = [
        _note("CVE-2019-7", url="https://nvd.nist.gov/vuln/detail/CVE-2019-7", source="nvd"),
        _note(
            "h1-77",
            url="https://hackerone.com/reports/77",
            source="hackerone-public",
            ntype="report",
        ),
        _note("CVE-2018-1", source="nvd"),  # no url
    ]
    assert record_note_tombstones(notes, reason="sift prune") == 3
    rows = [json.loads(line) for line in tombstones_path().read_text("utf-8").splitlines()]
    assert rows[0]["id"] == "CVE-2019-7" and rows[0]["source"] == "nvd"
    assert rows[0]["url"].endswith("CVE-2019-7") and rows[0]["reason"] == "sift prune"
    assert rows[0]["ts"]
    assert "url" not in rows[2]
    t = load_tombstones()
    assert t.has_url("https://hackerone.com/reports/77", source="hackerone-public")
    assert not t.has_url("https://hackerone.com/reports/77", source="hackerone-hacktivity")


def test_url_matching_is_normalised_but_keeps_the_query():
    from sift.tombstones import load_tombstones, record_tombstones

    record_tombstones(urls=["http://Example.COM/post/1/#comments"])
    record_tombstones(urls=["https://example.com/item?id=1"])
    t = load_tombstones()
    assert t.has_url("https://example.com/post/1")
    assert t.has_url("https://EXAMPLE.com/post/1/")
    assert not t.has_url("https://example.com/post/2")
    assert t.has_url("https://example.com/item?id=1")
    assert not t.has_url("https://example.com/item?id=2")


def test_unparseable_url_is_matched_verbatim_without_raising():
    from sift.tombstones import load_tombstones, record_tombstones

    record_tombstones(urls=["http://[::1/x"])
    assert load_tombstones().has_url("http://[::1/x")


def test_recording_twice_is_idempotent_per_source():
    from sift.tombstones import record_tombstones, tombstones_path

    assert record_tombstones(["a"], source="nvd") == 1
    assert record_tombstones(["a"], source="nvd") == 0
    assert record_tombstones(["a"], source="hackerone-public") == 1  # a different scope
    assert len(tombstones_path().read_text("utf-8").splitlines()) == 2


def test_unreadable_and_partial_lines_are_skipped_but_never_destroyed():
    from sift.tombstones import load_tombstones, record_tombstones, tombstones_path

    p = tombstones_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # A torn last line with no newline, as an interrupted append would leave it.
    p.write_text('{"id": "good"}\nnot json at all\n[1, 2]\n{"id": "tru', encoding="utf-8")
    t = load_tombstones()
    assert t.has_id("good") and len(t) == 1

    assert record_tombstones(["new"]) == 1
    lines = p.read_text("utf-8").splitlines()
    assert lines[:4] == ['{"id": "good"}', "not json at all", "[1, 2]", '{"id": "tru']
    assert json.loads(lines[4])["id"] == "new"  # not glued onto the torn line
    t = load_tombstones()
    assert t.has_id("good") and t.has_id("new")


def test_writes_leave_no_temp_files():
    from sift.tombstones import record_tombstones, remove_tombstones, tombstones_path

    record_tombstones(["a"])
    record_tombstones(["b"])
    remove_tombstones(ids=["a"])
    left = [p.name for p in tombstones_path().parent.iterdir() if p.name.endswith(".tmp")]
    assert left == []


def test_remove_tombstones_clears_every_source():
    from sift.tombstones import load_tombstones, record_tombstones, remove_tombstones

    record_tombstones(["a", "b"], ["https://x.test/p"], source="nvd")
    record_tombstones(["a"], source="cisa-kev")
    assert remove_tombstones(ids=["a"], urls=["http://x.test/p/"]) == 3
    t = load_tombstones()
    assert not t.has_id("a") and t.has_id("b") and not t.has_url("https://x.test/p")
    assert remove_tombstones(ids=["missing"]) == 0
    assert remove_tombstones() == 0


def test_concurrent_threads_lose_no_tombstones():
    from sift.tombstones import load_tombstones, record_tombstones

    def worker(tag: str) -> None:
        for i in range(15):
            record_tombstones([f"{tag}-{i}"], source="nvd")

    threads = [threading.Thread(target=worker, args=(f"t{n}",)) for n in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    t = load_tombstones()
    assert len(t) == 90
    assert all(t.has_id(f"t{n}-{i}") for n in range(6) for i in range(15))


def test_concurrent_processes_lose_no_tombstones():
    from sift.tombstones import load_tombstones

    code = textwrap.dedent(
        """
        import sys
        from sift.tombstones import record_tombstones
        tag = sys.argv[1]
        for i in range(20):
            record_tombstones([f"{tag}-{i}"], source="nvd")
        """
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", code, f"p{n}"], env=dict(os.environ))
        for n in range(3)
    ]
    for p in procs:
        assert p.wait(timeout=120) == 0
    t = load_tombstones()
    assert len(t) == 60
    assert all(t.has_id(f"p{n}-{i}") for n in range(3) for i in range(20))
