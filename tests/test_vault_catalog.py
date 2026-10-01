"""The vault catalog: id / slug / filename -> file, and listing rows, without
re-parsing 13.8k notes per call (list_notes, stats and get_note paid 2-5 s each)."""

from __future__ import annotations

import pytest


def _put(vault, rel, note_id, *, title=None, body="body", extra="", note_type=None):
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    ntype = note_type or rel.split("/")[0]
    fm = f"id: {note_id}\ntype: {ntype}\ntitle: {title or note_id}\n{extra}"
    path.write_text(f"---\n{fm}---\n\n{body}\n", encoding="utf-8")
    return path


def _age(*paths, seconds=600):
    """Push mtimes out of the racy window, as for any note not touched just now."""
    import os
    import time

    t = time.time() - seconds
    for p in paths:
        os.utime(p, (t, t))


@pytest.fixture
def count_parses(monkeypatch):
    from sift.vault import catalog as catalog_mod

    calls: list = []
    real = catalog_mod.load_note

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(catalog_mod, "load_note", counting)
    return calls


def test_rows_carry_listing_fields(vault_path):
    from sift.vault.catalog import fresh_catalog

    _put(vault_path, "finding/Idea.md", "idea-1", title="Idea",
         extra="program: acme\nseverity: High\ncreated: 2024-05-01\ntags: [jwt]\n"
               "extra:\n  status: hypothesis\n",
         body="see [[Other note#H]]")  # fmt: skip
    (row,) = fresh_catalog(vault_path).rows()
    assert (row.id, row.type, row.title, row.rel) == (
        "idea-1",
        "finding",
        "Idea",
        "finding/Idea.md",
    )
    assert (row.program, row.severity, row.status, row.created) == (
        "acme",
        "high",
        "hypothesis",
        "2024-05-01",
    )
    assert row.tags == ("jwt",) and row.links == ("other-note",)
    assert row.slug == "idea-1" and row.stem == "Idea" and row.folder == "finding"


def test_unchanged_files_are_not_reparsed(vault_path, count_parses):
    from sift.vault.catalog import fresh_catalog

    paths = [_put(vault_path, f"report/R{i}.md", f"r-{i}") for i in range(5)]
    _age(*paths)
    cat = fresh_catalog(vault_path)
    assert len(count_parses) == 5  # positive control: the first walk parses each once
    gen = cat.generation
    for _ in range(3):
        cat.refresh()
    assert len(count_parses) == 5
    assert cat.generation == gen


def test_a_quick_same_size_rewrite_is_still_seen(vault_path):
    """Windows keeps one mtime across quick same-size rewrites; such a file is read
    again on the next walk (git's racy-clean rule)."""
    import os

    from sift.vault.catalog import fresh_catalog

    p = _put(vault_path, "report/R.md", "r-1", title="AAAA")
    cat = fresh_catalog(vault_path)
    st = p.stat()
    p.write_text(p.read_text(encoding="utf-8").replace("AAAA", "BBBB"), encoding="utf-8")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert (p.stat().st_mtime_ns, p.stat().st_size) == (st.st_mtime_ns, st.st_size)
    cat.refresh()
    assert cat.by_id("r-1")[0].title == "BBBB"


def test_changes_deletions_and_older_mtimes_are_picked_up(vault_path):
    import os

    from sift.vault.catalog import fresh_catalog

    a = _put(vault_path, "report/A.md", "a")
    b = _put(vault_path, "report/B.md", "b")
    _age(a, b)
    cat = fresh_catalog(vault_path)
    assert [r.id for r in cat.rows()] == ["a", "b"]

    b.unlink()
    old = a.stat().st_mtime - 100
    _put(vault_path, "report/A.md", "a", title="A retitled with an older mtime")
    os.utime(a, (old, old))
    cat.ensure_fresh(max_age=3600)  # folder mtime changed: walks despite max_age
    assert [r.title for r in cat.rows()] == ["A retitled with an older mtime"]
    assert cat.by_id("b") == ()


def test_unreadable_files_are_listed_and_reported_once(vault_path, capfd):
    import logging

    from sift.vault.catalog import fresh_catalog

    records = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    lg = logging.getLogger("sift.vault")
    lg.addHandler(handler)
    try:
        _put(vault_path, "report/Good.md", "good")
        (vault_path / "report" / "empty.md").write_bytes(b"")
        cat = fresh_catalog(vault_path)
        cat.refresh()
        cat.refresh(force=True)
    finally:
        lg.removeHandler(handler)
    assert [r.id for r in cat.rows()] == ["good"]
    assert [(p.name, why) for p, why in cat.skipped()] == [("empty.md", "empty file")]
    assert len([r for r in records if "empty.md" in r.getMessage()]) == 1
    assert capfd.readouterr().out == ""


def test_lookup_prefers_the_exact_id_and_reports_ambiguity(vault_path):
    from slugify import slugify

    from sift.vault.catalog import fresh_catalog

    # B's id slugifies to A's id: the exact id must still win (get_note returned B).
    _put(vault_path, "technique/A.md", "tech-foo", title="A")
    _put(vault_path, "technique/B.md", "Tech Foo", title="B")
    base = "research-" + "a-very-long-shared-article-title-prefix-" * 3
    _put(vault_path, "writeup/First.md", base + "first", title="First")
    _put(vault_path, "writeup/Second.md", base + "second", title="Second")
    cat = fresh_catalog(vault_path)

    assert [r.title for r in cat.lookup("tech-foo")] == ["A"]
    assert [r.title for r in cat.lookup("Tech Foo")] == ["B"]
    assert [r.title for r in cat.lookup(slugify(base + "second"))] == ["Second"]
    legacy = slugify(base + "first", max_length=80)
    assert sorted(r.title for r in cat.lookup(legacy)) == ["First", "Second"]  # ambiguous
    assert sorted(r.title for r in cat.by_legacy_slug(legacy)) == ["First", "Second"]
    assert [r.title for r in cat.lookup("technique/a.md")] == ["A"]  # filename tier
    assert [r.title for r in cat.by_filename("SECOND")] == ["Second"]
    assert cat.lookup("nothing-like-it") == () and cat.lookup("") == ()


def test_by_path_only_returns_catalogued_note_files(vault_path, tmp_path):
    from sift.vault.catalog import fresh_catalog

    p = _put(vault_path, "report/R.md", "r")
    _put(vault_path, ".trash/report/Old.md", "old")
    outside = tmp_path / "outside.md"
    outside.write_text("---\nid: x\ntype: report\ntitle: x\n---\n", encoding="utf-8")
    cat = fresh_catalog(vault_path)

    assert cat.by_path(p).id == "r"
    assert cat.by_path("report/R.md").id == "r"
    assert cat.by_path(vault_path / ".trash" / "report" / "Old.md") is None
    assert cat.by_path(outside) is None
    assert cat.by_path("../outside.md") is None


def test_rows_match_iter_notes_order_and_scope(vault_path):
    from sift.vault.catalog import fresh_catalog
    from sift.vault.notes import iter_notes

    _put(vault_path, "technique/a.md", "t-a")
    _put(vault_path, "technique/a/b.md", "t-ab")
    _put(vault_path, "technique/c.md", "t-b")
    _put(vault_path, "report/z.md", "r-z")
    _put(vault_path, "Loose.md", "loose", note_type="finding")
    cat = fresh_catalog(vault_path)

    assert [r.id for r in cat.rows()] == [n.meta.id for n in iter_notes(vault_path)]
    assert [r.id for r in cat.rows("technique")] == [
        n.meta.id for n in iter_notes(vault_path, note_type="technique")
    ]
    assert cat.counts_by_type() == {"technique": 3, "report": 1, "finding": 1}
    assert len(cat) == 5
    assert [r.id for r in cat.rows()] == ["loose", "r-z", "t-ab", "t-a", "t-b"]  # a/b.md < a.md


def test_duplicate_ids_are_listed_never_dropped(vault_path):
    from sift.vault.catalog import fresh_catalog

    _put(vault_path, "cve/CVE-1 nvd.md", "CVE-1")
    _put(vault_path, "cve/CVE-1 kev.md", "CVE-1")
    cat = fresh_catalog(vault_path)
    dups = cat.duplicate_ids()
    assert list(dups) == ["CVE-1"]
    assert sorted(p.name for p in dups["CVE-1"]) == ["CVE-1 kev.md", "CVE-1 nvd.md"]
    assert len(cat.by_id("CVE-1")) == 2


def test_sift_writes_update_the_catalog_without_a_walk(vault_path, monkeypatch):
    from sift.vault import catalog as catalog_mod
    from sift.vault.catalog import fresh_catalog
    from sift.vault.notes import Note, delete_note, save_note
    from sift.vault.schema import Frontmatter

    cat = fresh_catalog(vault_path)
    path = save_note(vault_path, Note(Frontmatter(id="n1", type="finding", title="One"), "b"))
    cat.ensure_fresh()
    walks = []
    real = catalog_mod.walk_note_entries
    monkeypatch.setattr(
        catalog_mod, "walk_note_entries", lambda *a, **k: walks.append(1) or real(*a, **k)
    )

    gen = cat.generation
    save_note(vault_path, Note(Frontmatter(id="n1", type="finding", title="One renamed"), "b"))
    assert [r.title for r in cat.by_id("n1")] == ["One renamed"]
    assert cat.by_path(path) is None and cat.generation > gen
    assert cat.ensure_fresh(max_age=3600) == cat.generation and walks == []

    delete_note("n1", vault=vault_path)
    assert cat.by_id("n1") == ()


def test_the_cache_file_is_reused_across_processes(vault_path, count_parses):
    from sift.config import get_settings
    from sift.vault.catalog import clear_catalogs, fresh_catalog

    paths = [_put(vault_path, f"report/R{i}.md", f"r-{i}") for i in range(4)]
    _age(*paths)
    fresh_catalog(vault_path).persist()
    cache = list((get_settings().resolved_db() / "_sift").glob("catalog-*.json"))
    assert len(cache) == 1
    assert not list(vault_path.rglob("*.json"))  # never inside the vault
    assert len(count_parses) == 4

    clear_catalogs()  # a new process
    cat = fresh_catalog(vault_path)
    assert len(count_parses) == 4  # nothing re-parsed
    assert [r.id for r in cat.rows()] == ["r-0", "r-1", "r-2", "r-3"]

    _put(vault_path, "report/R1.md", "r-1", title="changed")
    clear_catalogs()
    cat = fresh_catalog(vault_path)
    assert len(count_parses) == 5  # only the changed file
    assert cat.by_id("r-1")[0].title == "changed"


@pytest.mark.parametrize("damage", ["garbage", "other-vault", "escape", "version"])
def test_a_bad_cache_file_is_ignored(vault_path, count_parses, damage):
    import json

    from sift.config import get_settings
    from sift.vault.catalog import clear_catalogs, fresh_catalog, get_catalog

    p = _put(vault_path, "report/R.md", "r")
    _age(p)
    fresh_catalog(vault_path).persist()
    (cache,) = (get_settings().resolved_db() / "_sift").glob("catalog-*.json")
    data = json.loads(cache.read_text(encoding="utf-8"))
    if damage == "garbage":
        cache.write_text("{not json", encoding="utf-8")
    else:
        if damage == "other-vault":
            data["vault"] = str(vault_path.parent / "elsewhere")
        elif damage == "escape":
            data["rows"][0][0] = "../outside.md"
        else:
            data["version"] = 999
        cache.write_text(json.dumps(data), encoding="utf-8")

    clear_catalogs()
    cat = get_catalog(vault_path)
    assert cat.by_id("r") == ()  # nothing served from a cache it does not trust
    cat.ensure_fresh()
    assert [r.id for r in cat.rows()] == ["r"]
    assert len(count_parses) == 2  # rebuilt from the vault


def test_concurrent_lookups_and_saves_stay_consistent(vault_path):
    from concurrent.futures import ThreadPoolExecutor

    from sift.vault.catalog import fresh_catalog
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    def save(i):
        save_note(vault_path, Note(Frontmatter(id=f"c-{i}", type="report", title=f"C {i}"), "b"))

    def read(_i):
        cat = fresh_catalog(vault_path, max_age=0.5)
        rows = cat.rows()
        assert len({r.id for r in rows}) == len(rows)
        return len(rows)

    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(save if i % 2 else read, i) for i in range(80)]
        for f in futures:
            f.result()
    cat = fresh_catalog(vault_path)
    assert sorted(r.id for r in cat.rows()) == sorted(f"c-{i}" for i in range(1, 80, 2))
