"""Index maintenance: FTS is created once and never rebuilt per write; optimize()
compacts only when it pays, never prunes versions a live reader may need, and never
turns a committed ingest into a failure. Nothing here may write to stdout - the MCP
server's stdout is the JSON-RPC wire.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import timedelta

import pytest

DIM = 8


def _vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


def _rows(nid: str, n: int = 1, *, text: str = "alpha beta"):
    from sift.index.store import ChunkRow

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
            path=f"/vault/technique/{nid}.md",
        )
        for i in range(n)
    ]


def _store():
    from sift.index.store import Store

    return Store(dim=DIM)


def _fragments(store) -> int:
    return store.index_info()["fragments"]


def _fts(store, word: str) -> list[str]:
    q = store.table().search(word, query_type="fts", fts_columns="search_text")
    return sorted({r["note_id"] for r in q.limit(50).to_list()})


@pytest.fixture
def store_log(caplog):
    """Capture the store's logger even if a CLI test switched off propagation."""
    lg = logging.getLogger("sift.index.store")
    lg.addHandler(caplog.handler)
    old = lg.level
    lg.setLevel(logging.INFO)
    yield caplog
    lg.removeHandler(caplog.handler)
    lg.setLevel(old)


# ---- ensure_fts -------------------------------------------------------------


def test_ensure_fts_creates_once_and_does_not_rebuild():
    store = _store()
    store.add_chunks(_rows("a"))
    assert not store.has_fts()

    assert store.ensure_fts() is True
    assert store.has_fts()
    v = store.table().version
    assert store.ensure_fts() is True
    assert store.table().version == v, "an existing index must not be rebuilt"

    assert store.ensure_fts(force=True) is True
    assert store.table().version > v, "force=True rebuilds"


def test_keyword_search_covers_rows_added_after_the_index_without_a_rebuild():
    store = _store()
    store.add_chunks(_rows("a"))
    store.ensure_fts()
    store.add_chunks(_rows("late", text="zebracorn"))

    assert _fts(store, "zebracorn") == ["late"]
    assert store.index_info()["fts_unindexed_rows"] >= 1


def test_ensure_fts_after_drop_recreates_the_index():
    store = _store()
    store.add_chunks(_rows("a"))
    store.ensure_fts()
    store.drop()
    store.add_chunks(_rows("b"))

    assert not store.has_fts()
    assert store.ensure_fts()
    assert store.has_fts()


def test_failed_fts_build_is_logged_not_printed(monkeypatch, capsys, store_log):
    store = _store()
    store.add_chunks(_rows("a"))
    tbl = store.table()

    def boom(*_a, **_k):
        raise RuntimeError("tokenizer exploded")

    monkeypatch.setattr(tbl, "create_index", boom)
    monkeypatch.setattr(tbl, "create_fts_index", boom)

    assert store.ensure_fts() is False
    assert capsys.readouterr().out == ""
    assert any("FTS index build failed" in r.getMessage() for r in store_log.records)


# ---- optimize ---------------------------------------------------------------


def test_routine_optimize_below_thresholds_skips_compaction_but_creates_fts():
    store = _store()
    for i in range(3):
        store.add_chunks(_rows(f"n{i}"))
    frags = _fragments(store)

    report = store.optimize()

    assert report["ran"] is False and report["reason"] == "below thresholds"
    assert report["error"] is None
    assert _fragments(store) == frags
    assert store.has_fts(), "optimize must leave a usable FTS index (e.g. after drop)"


def test_optimize_compacts_past_the_fragment_threshold(monkeypatch):
    from sift.index.store import Store

    monkeypatch.setattr(Store, "COMPACT_MIN_FRAGMENTS", 5)
    store = _store()
    store.add_chunks(_rows("first"))
    store.ensure_fts()
    for i in range(11):
        store.add_chunks(_rows(f"n{i}", 2))
    rows = store.count()
    assert _fragments(store) == 12

    report = store.optimize()

    assert report["ran"] is True and report["error"] is None
    assert report["before"]["fragments"] == 12
    assert report["after"]["fragments"] <= 2
    assert store.count() == rows, "compaction must keep every row"
    assert report["after"]["fts_unindexed_rows"] == 0, "new rows folded into the FTS index"
    assert _fts(store, "chunk1") == sorted(f"n{i}" for i in range(11))


def test_optimize_runs_when_many_rows_are_outside_the_fts_index(monkeypatch):
    from sift.index.store import Store

    monkeypatch.setattr(Store, "COMPACT_MIN_UNINDEXED", 10)
    store = _store()
    store.add_chunks(_rows("seed"))
    store.ensure_fts()
    store.add_chunks([r for i in range(4) for r in _rows(f"n{i}", 3)])

    report = store.optimize()

    assert report["ran"] is True
    assert report["after"]["fts_unindexed_rows"] == 0


def test_force_compacts_regardless_and_can_rebuild_fts():
    store = _store()
    for i in range(4):
        store.add_chunks(_rows(f"n{i}"))
    store.ensure_fts()

    report = store.optimize(force=True, rebuild_fts=True, measure_disk=True)

    assert report["ran"] is True and report["reason"] == "forced"
    assert report["after"]["fragments"] == 1
    assert report["before"]["disk_bytes"] > 0 and report["after"]["disk_bytes"] > 0
    assert _fts(store, "alpha") == ["n0", "n1", "n2", "n3"]


def test_a_huge_backlog_is_left_for_sift_compact(monkeypatch, store_log):
    """The first cleanup of a long-neglected table takes minutes: an explicit step."""
    from sift.index.store import Store

    monkeypatch.setattr(Store, "COMPACT_MAX_ROUTINE_FRAGMENTS", 3)
    store = _store()
    for i in range(6):
        store.add_chunks(_rows(f"n{i}"))

    report = store.optimize()

    assert report["ran"] is False
    assert "sift compact" in report["reason"]
    assert _fragments(store) == 6
    assert any("sift compact" in r.getMessage() for r in store_log.records)
    assert store.optimize(force=True)["ran"] is True


@pytest.mark.parametrize("retain", [timedelta(0), timedelta(minutes=5)])
def test_optimize_refuses_a_retention_that_breaks_live_readers(retain):
    store = _store()
    store.add_chunks(_rows("a"))
    with pytest.raises(ValueError, match="MCP server"):
        store.optimize(retain)
    with pytest.raises(ValueError):
        store.optimize(retain, force=True)


def test_unsafe_retention_needs_the_explicit_flag():
    store = _store()
    for i in range(3):
        store.add_chunks(_rows(f"n{i}"))
    report = store.optimize(timedelta(0), force=True, allow_unsafe_retain=True)
    assert report["ran"] is True and report["error"] is None
    assert len(store.table().list_versions()) <= 3


def test_a_failed_optimize_never_fails_the_caller(monkeypatch, capsys, store_log):
    """Compaction runs after the data is committed; an ingest must still report success."""
    store = _store()
    for i in range(3):
        store.add_chunks(_rows(f"n{i}"))
    tbl = store.table()

    def boom(**_k):
        raise RuntimeError("commit conflict with a concurrent writer")

    monkeypatch.setattr(tbl, "optimize", boom)
    report = store.optimize(force=True)

    assert report["ran"] is False
    assert "commit conflict" in report["error"]
    assert store.count() == 3
    assert capsys.readouterr().out == ""
    assert any("index maintenance failed" in r.getMessage() for r in store_log.records)


def test_optimize_without_a_table_is_a_no_op():
    from sift.index.store import TABLE

    store = _store()
    report = store.optimize(force=True)
    assert report["reason"] == "no index yet" and report["ran"] is False
    assert TABLE not in store._table_names()


def test_optimize_reports_a_dimension_mismatch_instead_of_raising():
    from sift.index.store import Store

    _store().add_chunks(_rows("a"))
    report = Store(dim=DIM * 2).optimize(force=True)
    assert "reindex --force" in report["error"]


def test_index_info_reports_counts():
    store = _store()
    assert store.index_info() == {"rows": 0, "fragments": 0, "version": 0, "exists": False}
    store.add_chunks(_rows("a", 3))
    store.ensure_fts()

    info = store.index_info(disk=True)

    assert info["exists"] and info["rows"] == 3 and info["fragments"] == 1
    assert info["fts"] is True and info["fts_unindexed_rows"] == 0
    assert info["version"] == store.table().version
    assert info["disk_bytes"] > 0


# ---- filters (pure) -----------------------------------------------------------


def test_where_doubles_quotes_in_every_clause():
    from sift.index.store import Store

    where = Store._where(
        {"type": "Re'port", "program": " O'Brien ", "severity": "Hi'gh", "cwe": "cwe-79"}
    )
    assert "type = 're''port'" in where
    assert "lower(program) = 'o''brien'" in where
    assert "severity = 'hi''gh'" in where
    assert "(' ' || cwe_str || ' ') LIKE '% CWE-79 %'" in where
    assert Store._where({}) is None and Store._where(None) is None


@pytest.mark.parametrize(
    "raw,token",
    [
        ("CWE-79", "CWE-79"),
        ("cwe-79", "CWE-79"),
        ("79", "CWE-79"),
        (" CWE 79 ", "CWE-79"),
        ("CWE_78", "CWE-78"),
        ("CWE787", "CWE-787"),
        ("NVD-CWE-Other", "NVD-CWE-OTHER"),
    ],
)
def test_cwe_filter_normalises_to_one_exact_token(raw, token):
    from sift.index.store import Store

    assert Store._where({"cwe": raw}) == f"(' ' || cwe_str || ' ') LIKE '% {token} %'"


@pytest.mark.parametrize("raw", ["CWE-7%", "CWE-79' OR 1=1 --", "79_", "XSS (stored)", "CWE-"])
def test_malformed_cwe_filter_is_an_error_not_an_empty_answer(raw):
    from sift.index.store import Store

    with pytest.raises(ValueError, match="cwe filter"):
        Store._where({"cwe": raw})


# ---- cold start ---------------------------------------------------------------


def test_importing_the_store_does_not_load_lancedb():
    """The MCP server imports this module at startup; lancedb loads on first use."""
    code = "import sys, sift.index.store; print('lancedb' in sys.modules, 'pyarrow' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"]
