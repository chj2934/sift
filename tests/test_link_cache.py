"""Link-graph cache.

A cold build parses every note: about 5 s over 13,856 notes (2026-10-01). Until the
cache went per-file it was keyed on (note count, newest mtime), so every write -
remember, capture_idea, resolve_idea, an Obsidian save - threw the whole graph away
and the next `search_memory(expand_links=True)` re-parsed the vault. It also kept
every note body resident (~129 MB) just to resolve links, and it missed renames.

The graph now holds one slim record per file, derived from the vault catalog, which
revalidates each file by (mtime_ns, size) and is persisted, so a write costs one parse
and a new process does not re-parse the vault for the graph. A stale graph would be
worse than a slow one, so most of these tests are about invalidation: a note written
mid-hunt must be visible to the very next query.
"""

from __future__ import annotations

import os
import threading
import time


def _write(vault, name: str, body: str = "hello", links: str = ""):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    note = Note(
        meta=Frontmatter(id=name, type="technique", title=name, links=[links] if links else []),
        body=body,
    )
    return save_note(vault, note)


def _put(path, note_id: str, title: str | None = None, body: str = "body") -> None:
    """Write a note file by hand, so the test controls the exact filename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: {note_id}\ntype: technique\ntitle: {title or note_id}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _age(*paths, seconds: int = 3600) -> None:
    """Backdate files out of the racy window, so only real changes cause a re-parse."""
    t = time.time() - seconds
    for p in paths:
        os.utime(p, (t, t))


def _count_parses(monkeypatch) -> list[str]:
    """Record every file parsed to build the graph: the vault catalog reads the files,
    and the graph derives its records from the catalog's rows."""
    from sift.vault import catalog

    parsed: list[str] = []
    real = catalog.load_note

    def counting(path):
        parsed.append(os.path.basename(path))
        return real(path)

    monkeypatch.setattr(catalog, "load_note", counting)
    return parsed


def test_cache_returns_the_same_object_when_nothing_changed(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    first = build_link_index(vault_path)
    second = build_link_index(vault_path)
    assert first is second, "identical vault state should not be re-parsed"


def test_new_note_invalidates_the_cache(vault_path):
    """A capture_idea write mid-hunt must show up on the next query."""
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    assert "alpha" in build_link_index(vault_path)

    _write(vault_path, "beta")
    refreshed = build_link_index(vault_path)
    assert "beta" in refreshed, "cache served a stale graph after a new note"


def test_deleted_note_invalidates_the_cache(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    path = _write(vault_path, "doomed")
    assert "doomed" in build_link_index(vault_path)

    path.unlink()
    assert "doomed" not in build_link_index(vault_path)


def test_edited_note_invalidates_the_cache(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    _write(vault_path, "beta")
    assert build_link_index(vault_path)["beta"].all_links() == []

    # Bump mtime deterministically rather than relying on filesystem granularity.
    path = _write(vault_path, "beta", links="alpha")
    future = time.time() + 10
    os.utime(path, (future, future))

    assert "alpha" in build_link_index(vault_path)["beta"].all_links()


def test_same_size_rewrite_inside_one_timestamp_tick_is_seen(vault_path):
    """NTFS keeps the same mtime_ns for 58% of rapid same-size rewrites, so
    (mtime_ns, size) alone would serve the old links. A file modified close to the
    moment it was read is re-checked on the next call (git's racy-clean rule)."""
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    path = vault_path / "technique" / "beta.md"
    _put(path, "beta", body="see [[aaaa]]")
    st = os.stat(path)
    assert build_link_index(vault_path)["beta"].all_links() == ["aaaa"]

    _put(path, "beta", body="see [[bbbb]]")  # same size...
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # ...and the same mtime_ns
    assert os.stat(path).st_size == st.st_size

    assert build_link_index(vault_path)["beta"].all_links() == ["bbbb"]


def test_fingerprint_ignores_underscore_files(vault_path):
    """_state.json and _rejects.jsonl churn constantly; they must not bust the cache."""
    from sift.index.graph import _fingerprint

    _write(vault_path, "alpha")
    before = _fingerprint(vault_path)
    (vault_path / "technique" / "_scratch.md").write_text("noise", encoding="utf-8")
    assert _fingerprint(vault_path) == before


def test_underscore_file_causes_no_reparse(vault_path, monkeypatch):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _age(_write(vault_path, "alpha"))
    first = build_link_index(vault_path)
    parsed = _count_parses(monkeypatch)
    (vault_path / "technique" / "_scratch.md").write_text("noise", encoding="utf-8")
    assert build_link_index(vault_path) is first
    assert parsed == []


def test_use_cache_false_always_rebuilds(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    a = build_link_index(vault_path, use_cache=False)
    b = build_link_index(vault_path, use_cache=False)
    assert a is not b


def test_use_cache_false_leaves_the_shared_cache_alone(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _age(_write(vault_path, "alpha"))
    cached = build_link_index(vault_path)
    fresh = build_link_index(vault_path, use_cache=False)
    assert fresh is not cached and fresh.keys() == cached.keys()
    assert build_link_index(vault_path) is cached


# --- incremental refresh ------------------------------------------------------------


def test_one_write_reparses_exactly_one_file(vault_path, monkeypatch):
    """The whole point: a write mid-session costs one parse, not the whole vault."""
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _age(*[_write(vault_path, f"note-{i}") for i in range(5)])
    before = build_link_index(vault_path)

    parsed = _count_parses(monkeypatch)
    _write(vault_path, "fresh", body="see [[note-1]]")
    after = build_link_index(vault_path)

    assert parsed == ["fresh.md"]
    assert after is not before, "a changed vault must produce a new index object"
    assert "fresh" in after and "fresh" not in before
    assert after["fresh"].all_links() == ["note-1"]


def test_unchanged_vault_parses_nothing(vault_path, monkeypatch):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _age(*[_write(vault_path, f"note-{i}") for i in range(3)])
    first = build_link_index(vault_path)
    parsed = _count_parses(monkeypatch)
    assert build_link_index(vault_path) is first
    assert parsed == []


def test_a_new_process_builds_the_graph_without_parsing_the_vault(vault_path, monkeypatch):
    """The graph used to keep its own parse cache, so every new MCP server re-parsed
    the whole vault for it (~5-9 s over 13.8k notes) right after the catalog had been
    loaded from its cache file. Built from the catalog, a new process reads the cache
    file and stat-walks; it parses nothing that did not change."""
    from sift.index.graph import build_link_index, clear_link_index_cache
    from sift.vault.catalog import clear_catalogs, get_catalog

    clear_link_index_cache()
    _age(*[_write(vault_path, f"note-{i}", body=f"see [[note-{i + 1}]]") for i in range(5)])
    before = build_link_index(vault_path)
    get_catalog(vault_path).persist()

    # A new process: no graph, no in-memory catalog; the catalog's cache file remains.
    clear_link_index_cache()
    clear_catalogs()
    from sift.vault import notes

    reads: list[str] = []  # every note file read, whoever reads it
    real = notes.read_note_text
    monkeypatch.setattr(
        notes, "read_note_text", lambda p: reads.append(os.path.basename(p)) or real(p)
    )
    after = build_link_index(vault_path)

    assert reads == []
    assert after is not before and after.keys() == before.keys()
    assert after["note-1"].all_links() == ["note-2"]


def test_rename_with_preserved_mtime_yields_the_new_path(vault_path):
    """A rename keeps the mtime and the note count, so the old (count, newest mtime)
    fingerprint served the cached graph with a `path` that no longer existed."""
    from sift.index.graph import _fingerprint, build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "anchor")
    old = _write(vault_path, "alpha")
    _age(old)
    assert build_link_index(vault_path)["alpha"].path == old
    fp = _fingerprint(vault_path)

    mtime_ns = os.stat(old).st_mtime_ns
    new = old.with_name("alpha renamed in Obsidian.md")
    old.rename(new)
    assert os.stat(new).st_mtime_ns == mtime_ns, "precondition: a rename keeps the mtime"

    assert _fingerprint(vault_path) != fp, "fingerprint missed a rename"
    rec = build_link_index(vault_path)["alpha"]
    assert rec.path == new and rec.path.exists()


def test_older_mtime_swap_is_detected(vault_path):
    """Delete one note and restore another with an older mtime (git restore,
    copy -p): the count and the newest mtime are unchanged, and the old fingerprint
    missed it."""
    from sift.index.graph import _fingerprint, build_link_index, clear_link_index_cache

    clear_link_index_cache()
    newest = _write(vault_path, "newest")
    gone = _write(vault_path, "gone")
    _age(gone, seconds=7200)
    _age(newest, seconds=60)
    assert "gone" in build_link_index(vault_path)
    fp = _fingerprint(vault_path)

    gone.unlink()
    restored = _write(vault_path, "restored")
    _age(restored, seconds=7200)

    assert _fingerprint(vault_path) != fp
    idx = build_link_index(vault_path)
    assert "restored" in idx and "gone" not in idx


# --- unreadable files and stdout ----------------------------------------------------


def test_zero_byte_file_is_skipped_and_never_printed(vault_path, monkeypatch, capsys, caplog):
    """A 0-byte file (an interrupted save) is not a note. The vault layer reports it
    once, like iter_notes does (`sift doctor` and `stats` list it), never on stdout;
    the graph neither holds it nor re-reads it on every call."""
    import logging

    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    empty = vault_path / "technique" / "empty.md"
    empty.write_bytes(b"")
    _age(empty)
    parsed = _count_parses(monkeypatch)
    caplog.set_level(logging.WARNING)

    first = build_link_index(vault_path)
    second = build_link_index(vault_path)

    assert first is second and "alpha" in first
    assert all(rec.path != empty for rec in first.values())
    assert parsed.count("empty.md") <= 1
    assert capsys.readouterr().out == ""
    assert len([r for r in caplog.records if "empty.md" in r.getMessage()]) <= 1


def test_unreadable_note_is_parsed_once_and_reported_once_never_on_stdout(
    vault_path, monkeypatch, capsys, caplog
):
    """stdout is the MCP JSON-RPC channel; a print there corrupts the protocol. And a
    broken file must not be re-parsed and re-reported on every query."""
    import logging

    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _age(_write(vault_path, "alpha"))
    broken = vault_path / "technique" / "Landing page.md"
    broken.write_text("# no frontmatter here\n\njust a page\n", encoding="utf-8")
    _age(broken)
    parsed = _count_parses(monkeypatch)
    # Root level: the warning comes from the vault package's reporter when it has one.
    caplog.set_level(logging.WARNING)

    first = build_link_index(vault_path)
    second = build_link_index(vault_path)

    assert first is second
    assert parsed.count("Landing page.md") == 1
    assert capsys.readouterr().out == ""
    warnings = [r for r in caplog.records if "Landing page.md" in r.getMessage()]
    assert len(warnings) == 1, warnings
    assert "alpha" in first


def test_the_warning_never_quotes_frontmatter(vault_path, caplog):
    """A pydantic or YAML message quotes the offending values; the log names the file
    and the error class only."""
    import logging

    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    bad = vault_path / "technique" / "Bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        "---\nid: tech-bad\ntype: not-a-real-type-SECRET-VALUE\ntitle: Bad\n---\n\nbody\n",
        encoding="utf-8",
    )
    caplog.set_level(logging.WARNING)

    assert "tech-bad" not in build_link_index(vault_path)
    messages = [r.getMessage() for r in caplog.records if "Bad.md" in r.getMessage()]
    assert len(messages) == 1, messages
    assert "SECRET-VALUE" not in messages[0]


def test_concurrent_cold_builds_share_one_parse_pass(vault_path, monkeypatch):
    """FastMCP runs sync tools in a threadpool, so parallel expanding searches each
    paid a full cold build. They must wait for one build and share it."""
    from sift.index.graph import build_link_index, clear_link_index_cache

    _age(*[_write(vault_path, f"note-{i}") for i in range(20)])
    clear_link_index_cache()
    parsed = _count_parses(monkeypatch)

    barrier = threading.Barrier(4)
    results: list = []
    errors: list = []

    def worker():
        try:
            barrier.wait(timeout=10)
            results.append(build_link_index(vault_path))
        except Exception as exc:  # pragma: no cover - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors
    assert len(results) == 4
    assert all(r is results[0] for r in results)
    assert len(parsed) == 20, f"{len(parsed)} parses for 20 notes"


# --- what the graph holds and where it looks ------------------------------------------


def test_records_are_slim_and_read_like_notes(vault_path):
    """No body is kept: the old cache held 37M characters of note text just to answer
    link lookups. pipeline.search still reads .meta.title/.type/.url and .path."""
    from sift.index.graph import build_link_index, clear_link_index_cache
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    clear_link_index_cache()
    path = save_note(
        vault_path,
        Note(
            meta=Frontmatter(
                id="tech-x", type="technique", title="X marks", url="https://example.com/x"
            ),
            body="A long body. " * 200 + "[[Elsewhere]]",
        ),
    )
    rec = build_link_index(vault_path)["tech-x"]
    assert not hasattr(rec, "body")
    assert rec.meta.title == "X marks" and rec.meta.type == "technique"
    assert rec.meta.url == "https://example.com/x" and rec.meta.id == "tech-x"
    assert rec.note_id == "tech-x" and rec.slug == "tech-x" and rec.path == path
    assert rec.all_links() == ["elsewhere"]


def test_walk_skips_what_is_not_a_live_note(vault_path):
    """Obsidian's .trash holds deleted notes with valid frontmatter (duplicate ids that
    sort first), _templates holds placeholders, README is documentation, and a save's
    temp file is not a note."""
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _put(vault_path / "technique" / "live.md", "live")
    _put(vault_path / "technique" / "sub" / "nested.md", "nested")  # parity with rglob
    _put(vault_path / ".trash" / "technique" / "trashed.md", "trashed")
    _put(vault_path / ".obsidian" / "x.md", "obsidian-config")
    _put(vault_path / "_templates" / "template.md", "template")
    _put(vault_path / "technique" / "Readme.md", "readme")
    _put(vault_path / "technique" / "_draft.md", "draft")
    _put(vault_path / "technique" / ".live.md.1234.tmp", "tempfile")
    _put(vault_path / "technique" / ".hidden.md", "hidden")

    idx = build_link_index(vault_path)
    assert "live" in idx and "nested" in idx
    for absent in (
        "trashed",
        "obsidian-config",
        "template",
        "readme",
        "draft",
        "tempfile",
        "hidden",
    ):
        assert absent not in idx, absent


def test_claimant_order_matches_sorted_path_order(vault_path):
    """Which note wins a contested key depends on order, so the graph must see files in
    exactly the order `sorted(vault.rglob("*.md"))` gives iter_notes (and get_note's
    first match): the claimants of one shared id come out in that order."""
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    names = [
        "b.md",
        "B2.md",
        "a b.md",
        "a-b.md",
        "a.md",
        "a (2).md",
        "A.MD.md",
        "z.md",
        "a/z.md",
        "ab/c.md",
        "a b/d.md",
        "Ünïcode.md",
        "~tilde.md",
        "1.md",
    ]
    for i, name in enumerate(names):
        _put(vault_path / "technique" / name, "shared-id", title=f"Copy {i}")

    claimants = build_link_index(vault_path).ambiguous["shared-id"]
    assert [rec.path for rec in claimants] == sorted(vault_path.rglob("*.md"))


def test_graph_holds_exactly_the_notes_iter_notes_reads(vault_path):
    """One scope for walk and parse. A note iter_notes indexes but the graph skips can
    never be expanded to; one the graph holds but iter_notes skips (a template, a
    trashed copy) is expanded to and then cannot be fetched."""
    import json

    from sift.index.graph import build_link_index, clear_link_index_cache
    from sift.vault.notes import iter_notes

    clear_link_index_cache()
    _put(vault_path / "technique" / "live.md", "live")
    _put(vault_path / "technique" / "deep" / "er" / "nested.md", "nested")
    _put(vault_path / "report" / "Report.md", "report-1")
    # Obsidian's core Templates plugin names its folder here; no underscore, no dot.
    _put(vault_path / "Templates" / "Finding template.md", "template-placeholder")
    (vault_path / ".obsidian").mkdir()
    (vault_path / ".obsidian" / "templates.json").write_text(
        json.dumps({"folder": "Templates"}), encoding="utf-8"
    )
    _put(vault_path / ".trash" / "technique" / "old.md", "trashed")
    _put(vault_path / "_templates" / "t.md", "underscore-template")
    _put(vault_path / "technique" / "README.md", "readme")
    _put(vault_path / "technique" / "_draft.md", "draft")
    (vault_path / "technique" / "empty.md").write_bytes(b"")
    (vault_path / "technique" / "Landing.md").write_text("# no frontmatter\n", encoding="utf-8")

    graph_paths = sorted({rec.path for rec in build_link_index(vault_path).values()})
    note_paths = sorted(n.path for n in iter_notes(vault_path))
    assert graph_paths == note_paths
    assert {p.name for p in graph_paths} >= {"live.md", "nested.md", "Report.md"}


def test_missing_vault_gives_an_empty_index(tmp_path):
    from sift.index.graph import build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    idx = build_link_index(tmp_path / "no-such-vault")
    assert len(idx) == 0
    assert expand(["anything"], idx) == []


# --- wikilink alias resolution ---------------------------------------------------
# Every cross-link in the first batch of technique notes was broken: they were written
# from the title ("[[cookie-sandwich...]]") while the note's slug carries an id prefix
# ("tech-cookie-sandwich..."). 0 of 5 resolved, so the link graph returned nothing.


def test_wikilink_without_the_id_prefix_resolves(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    _write(vault_path, "tech-cookie-sandwich", body="see [[unicode-overflow]]")
    _write(vault_path, "tech-unicode-overflow", body="the target")

    idx = build_link_index(vault_path)
    assert "unicode-overflow" in idx, "alias not registered"
    # ...and expansion must emit the CANONICAL slug, not the alias, so callers can
    # pass it straight to get_note.
    assert expand(["tech-cookie-sandwich"], idx) == ["tech-unicode-overflow"]


def test_a_real_slug_is_never_shadowed_by_an_alias(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    real = _write(vault_path, "unicode-overflow", body="the genuine article")
    _write(vault_path, "tech-unicode-overflow", body="would alias to the same key")

    idx = build_link_index(vault_path)
    assert idx["unicode-overflow"].path == real, "alias overwrote a real note"


def test_unknown_wikilinks_are_ignored(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    _write(vault_path, "tech-alpha", body="see [[does-not-exist]]")
    idx = build_link_index(vault_path)
    assert expand(["tech-alpha"], idx) == []
