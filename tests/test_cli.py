"""The CLI through Typer's CliRunner: logging, `sift mcp`, ingest wiring, search.

Offline: every network source and model is stubbed. Index-side commands (reindex,
compact, prune, doctor, status, trash) are in test_cli_index.py, distill in
test_cli_distill.py.
"""

from __future__ import annotations

import logging
import os
from datetime import date

import pytest


def _run(*args: str):
    from typer.testing import CliRunner

    from sift.cli import app

    return CliRunner().invoke(app, list(args))


def _flat(text: str) -> str:
    """Rich wraps at 80 columns when not on a terminal; compare on collapsed spaces."""
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# logging: stderr, never stdout (stdout is the MCP wire)
# --------------------------------------------------------------------------- #
def test_cli_logging_goes_to_stderr_never_stdout(capsys):
    from sift import cli

    cli._configure_logging(serving=False)
    log = logging.getLogger("sift.anything")
    log.info("progress line")
    log.warning("something odd")
    out, err = capsys.readouterr()
    assert out == ""
    assert "progress line" in err
    assert "warning: something odd" in err


def test_serving_logs_warnings_only(capsys):
    from sift import cli

    cli._configure_logging(serving=True)
    log = logging.getLogger("sift.anything")
    log.info("chatty")
    log.warning("worth knowing")
    out, err = capsys.readouterr()
    assert out == ""
    assert "chatty" not in err
    assert "sift WARNING sift.anything: worth knowing" in err


def test_logging_setup_does_not_stack_handlers():
    from sift import cli

    for _ in range(3):
        cli._configure_logging(serving=False)
    own = [h for h in logging.getLogger("sift").handlers if getattr(h, cli._OWN_HANDLER, False)]
    assert len(own) == 1


def test_every_command_gets_the_stderr_handler():
    from sift import cli

    res = _run("trash", "list")
    assert res.exit_code == 0, res.output
    assert "trash is empty" in res.stdout
    lg = logging.getLogger("sift")
    assert any(getattr(h, cli._OWN_HANDLER, False) for h in lg.handlers)
    assert lg.level == logging.INFO
    assert lg.propagate is False


# --------------------------------------------------------------------------- #
# sift mcp
# --------------------------------------------------------------------------- #
def _clear_env(monkeypatch, *names: str) -> None:
    """Unset, and restore to unset afterwards (delenv alone records nothing when the
    variable is absent, so a value the code under test sets would leak)."""
    for name in names:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_mcp_command_runs_the_one_server_entry_point(monkeypatch):
    import sys

    import sift.mcp_server as server

    seen: dict = {"calls": 0}

    def fake_main() -> None:
        seen["calls"] += 1
        # The import-time redirect must be over: the SDK finds the wire via sys.stdout.
        seen["stdout_is_stderr"] = sys.stdout is sys.stderr
        seen["check"] = os.environ.get("FASTMCP_CHECK_FOR_UPDATES")
        seen["banner"] = os.environ.get("FASTMCP_SHOW_SERVER_BANNER")
        seen["level"] = logging.getLogger("sift").level

    monkeypatch.setattr(server, "main", fake_main)
    _clear_env(monkeypatch, "FASTMCP_CHECK_FOR_UPDATES", "FASTMCP_SHOW_SERVER_BANNER")
    res = _run("mcp")
    assert res.exit_code == 0, res.output
    assert seen["calls"] == 1
    assert seen["stdout_is_stderr"] is False
    assert seen["check"] == "off"  # no PyPI request before serving
    assert seen["banner"] == "false"
    assert seen["level"] == logging.WARNING  # stderr is Claude Code's error log
    assert res.stdout == ""


def test_mcp_command_keeps_an_explicit_fastmcp_setting(monkeypatch):
    import sift.mcp_server as server

    monkeypatch.setattr(server, "main", lambda: None)
    monkeypatch.setenv("FASTMCP_CHECK_FOR_UPDATES", "stable")
    _clear_env(monkeypatch, "FASTMCP_SHOW_SERVER_BANNER")
    assert _run("mcp").exit_code == 0
    assert os.environ["FASTMCP_CHECK_FOR_UPDATES"] == "stable"


# --------------------------------------------------------------------------- #
# ingest wiring
# --------------------------------------------------------------------------- #
def _recording_source(calls: dict, name: str):
    def source(**kwargs):
        calls[name] = kwargs
        return iter(())

    return source


def test_ingest_google_runs_every_source_when_one_fails(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        "sift.ingest.chromium_fixes.source", _recording_source(calls, "chromium-fixes")
    )
    monkeypatch.setattr(
        "sift.ingest.chrome_releases.source", _recording_source(calls, "chrome-releases")
    )
    # chromium-docs stays real: with SIFT_CHROMIUM_SRC unset it fails first.
    res = _run("ingest", "google")
    assert res.exit_code == 1
    assert set(calls) == {"chromium-fixes", "chrome-releases"}  # the later ones still ran
    # Called as plain functions: an omitted option would arrive as a truthy OptionInfo.
    assert calls["chromium-fixes"]["refresh"] is False
    assert calls["chrome-releases"]["ledger"] is True
    out = _flat(res.output)
    assert "chromium-docs failed" in out
    assert "SIFT_CHROMIUM_SRC is not set" in out
    assert "failed: chromium-docs" in out


def test_ingest_google_exits_0_when_every_source_runs(monkeypatch):
    calls: dict = {}
    for module, name in (
        ("chromium_docs", "chromium-docs"),
        ("chromium_fixes", "chromium-fixes"),
        ("chrome_releases", "chrome-releases"),
    ):
        monkeypatch.setattr(f"sift.ingest.{module}.source", _recording_source(calls, name))
    res = _run("ingest", "google", "--since", "2026-05-01")
    assert res.exit_code == 0, res.output
    assert len(calls) == 3
    assert calls["chromium-docs"]["since"] == date(2026, 5, 1)


@pytest.mark.parametrize(
    ("module", "args", "expected"),
    [
        (
            "research",
            ["ingest", "research", "--since", "2025-01-02", "--refresh"],
            {"since": date(2025, 1, 2), "refresh": True},
        ),
        ("research", ["ingest", "research"], {"since": None, "refresh": False}),
        (
            "writeups",
            ["ingest", "writeups", "--refresh", "--since", "2025-03-04"],
            {"refresh": True, "since": date(2025, 3, 4), "since_year": 2024},
        ),
        ("writeups", ["ingest", "writeups"], {"refresh": False, "since": None}),
        ("top10", ["ingest", "top10", "--refresh"], {"refresh": True}),
        ("h1_public", ["ingest", "h1-public", "--refresh"], {"refresh": True}),
        ("h1_public", ["ingest", "h1-public"], {"refresh": False, "limit": None}),
        ("chromium_fixes", ["ingest", "chromium-fixes", "--refresh"], {"refresh": True}),
    ],
)
def test_ingest_options_reach_the_source(monkeypatch, module, args, expected):
    calls: dict = {}
    monkeypatch.setattr(f"sift.ingest.{module}.source", _recording_source(calls, module))
    res = _run(*args)
    assert res.exit_code == 0, res.output
    for key, value in expected.items():
        assert calls[module][key] == value, key


def test_h1_mine_passes_refresh_to_hacktivity(monkeypatch):
    seen: dict = {}

    def hacktivity(**kwargs):
        seen.update(kwargs)
        return iter(())

    monkeypatch.setattr("sift.ingest.h1_api.my_reports", lambda: iter(()))
    monkeypatch.setattr("sift.ingest.h1_api.hacktivity", hacktivity)
    res = _run("ingest", "h1-mine", "--hacktivity", "--refresh", "--limit", "7")
    assert res.exit_code == 0, res.output
    assert seen == {"query": None, "limit": 7, "refresh": True}


def test_a_bad_since_date_exits_1():
    res = _run("ingest", "research", "--since", "yesterday")
    assert res.exit_code == 1
    assert "YYYY-MM-DD" in res.stderr


def test_ingest_summary_and_exit_code(monkeypatch):
    from sift.ingest.base import IngestResult

    monkeypatch.setattr("sift.ingest.kev.source", lambda **k: iter(()))
    result = IngestResult("kev", written=3, updated=1, unchanged=40, indexed_chunks=12)
    monkeypatch.setattr("sift.ingest.base.run_source", lambda *a, **k: result)
    ok = _run("ingest", "kev")
    assert ok.exit_code == 0, ok.output
    assert "3 new, 1 updated, 40 unchanged, 12 chunks, 0 errors" in _flat(ok.stdout)

    result.errors = 2  # a half-ingested corpus must not look like a clean run
    bad = _run("ingest", "kev")
    assert bad.exit_code == 1
    assert "2 errors" in _flat(bad.stdout)


def test_id_conflicts_are_reported(monkeypatch):
    from sift.ingest.base import IngestResult

    monkeypatch.setattr("sift.ingest.nvd.source", lambda **k: iter(()))
    result = IngestResult("nvd", written=1, id_conflicts=2)
    monkeypatch.setattr("sift.ingest.base.run_source", lambda *a, **k: result)
    res = _run("ingest", "nvd")
    assert res.exit_code == 0
    assert "2 id conflicts" in _flat(res.stdout)
    assert "2 note(s) not saved" in _flat(res.stderr)


def test_epss_reports_failed_batches(monkeypatch):
    from sift.ingest.epss import EpssResult

    r = EpssResult(notes=3, scored=2, changed=1, unchanged=1, failed_batches=1)
    monkeypatch.setattr("sift.ingest.epss.enrich_notes", lambda: r)
    res = _run("ingest", "epss")
    assert res.exit_code == 1
    assert "1 changed, 1 unchanged, 2/3 CVE notes scored" in _flat(res.stdout)
    assert "1 failed API batches" in _flat(res.stdout)

    r.failed_batches = 0
    assert _run("ingest", "epss").exit_code == 0


def test_ingest_notes_lists_untouched_files_and_exits_1(monkeypatch, vault_path):
    from sift.ingest.local_notes import BackfillResult

    res_obj = BackfillResult(fixed=1, indexed=2, problems=[(vault_path / "bad.md", "bad YAML")])
    monkeypatch.setattr("sift.ingest.local_notes.backfill_and_index", lambda **k: res_obj)
    res = _run("ingest", "notes")
    assert res.exit_code == 1
    assert "1 frontmatter backfilled, 2 indexed" in _flat(res.stdout)
    assert "1 file(s) left untouched" in _flat(res.stderr)
    assert "bad YAML" in res.stderr

    res_obj.problems = []
    assert _run("ingest", "notes").exit_code == 0


def test_ingest_url_uses_the_capture_tool(monkeypatch):
    import sift.mcp_server as server

    seen: dict = {}

    def capture(url, *, program=None, tags=None, force=False):
        seen.update(url=url, program=program, tags=tags, force=force)
        return {
            "saved": True,
            "title": "A fresh writeup",
            "note_id": "writeup-a-fresh-writeup",
            "path": "writeup/A fresh writeup.md",
            "chunks_indexed": 3,
        }

    monkeypatch.setattr(server, "capture_url", capture)
    res = _run(
        "ingest", "url", "https://example.com/post", "--program", "acme", "--tag", "x", "--tag", "y"
    )
    assert res.exit_code == 0, res.output
    assert seen == {
        "url": "https://example.com/post",
        "program": "acme",
        "tags": ["x", "y"],
        "force": False,
    }
    assert "saved" in res.stdout and "writeup-a-fresh-writeup" in res.stdout


def test_ingest_url_refusals_exit_1(monkeypatch):
    from fastmcp.exceptions import ToolError

    import sift.mcp_server as server

    monkeypatch.setattr(
        server,
        "capture_url",
        lambda url, **k: {"saved": False, "reason": "pre-cutoff", "hint": "pass force"},
    )
    res = _run("ingest", "url", "https://example.com/old")
    assert res.exit_code == 1
    assert "not saved: pre-cutoff" in _flat(res.stderr)

    def refuse(url, **k):
        raise ToolError("refusing to fetch: a private address")

    monkeypatch.setattr(server, "capture_url", refuse)
    res = _run("ingest", "url", "http://127.0.0.1/")
    assert res.exit_code == 1
    assert "private address" in res.stderr

    monkeypatch.setattr(
        server,
        "capture_url",
        lambda url, **k: {"saved": False, "existing": True, "note_id": "n1", "title": "T"},
    )
    assert _run("ingest", "url", "https://example.com/dup").exit_code == 0


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def _hit(**over):
    from sift.index.store import Hit

    base = dict(
        note_id="note-1",
        slug="note-1",
        type="technique",
        title="Cache key confusion",
        url="https://example.com/a",
        source="research",
        severity="",
        program="",
        path="",
        score=0.5,
        excerpt="the excerpt",
        heading="",
        matched_chunks=1,
        quality=70,
    )
    base.update(over)
    return Hit(**base)


def test_search_reports_a_bad_filter_and_exits_1(monkeypatch):
    def boom(*a, **k):
        raise ValueError("malformed cwe filter 'CWE-x'")

    monkeypatch.setattr("sift.pipeline.search", boom)
    res = _run("search", "xss", "--cwe", "CWE-x")
    assert res.exit_code == 1
    assert "malformed cwe filter" in res.stderr
    assert res.stdout == ""


def test_search_reports_a_model_mismatch(monkeypatch):
    from sift.index.store import IndexDimMismatch

    def boom(*a, **k):
        raise IndexDimMismatch("index was built with 768-dim vectors; run `sift reindex --force`")

    monkeypatch.setattr("sift.pipeline.search", boom)
    res = _run("search", "xss")
    assert res.exit_code == 1
    assert "sift reindex --force" in _flat(res.stderr)


def test_search_shows_warnings_and_note_ids(monkeypatch):
    from sift.pipeline import SearchResult

    monkeypatch.setattr(
        "sift.pipeline.search",
        lambda *a, **k: SearchResult(hits=[_hit()], linked=[], warnings=["fts search unavailable"]),
    )
    res = _run("search", "cache")
    assert res.exit_code == 0, res.output
    assert "fts search unavailable" in res.stderr
    assert "Cache key confusion" in res.stdout
    assert "id note-1" in res.stdout


def test_search_with_no_hits_exits_1(monkeypatch):
    from sift.pipeline import SearchResult

    monkeypatch.setattr(
        "sift.pipeline.search", lambda *a, **k: SearchResult(hits=[], linked=[], warnings=[])
    )
    res = _run("search", "nothing")
    assert res.exit_code == 1
    assert "no matches" in res.stdout


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #
def test_init_builds_the_vault_and_never_writes_the_repo_env(vault_path):
    from sift.config import PROJECT_ROOT

    real_env = PROJECT_ROOT / ".env"
    before = real_env.stat().st_mtime_ns if real_env.exists() else None
    res = _run("init")
    assert res.exit_code == 0, res.output
    assert (vault_path / "finding").is_dir()
    assert "wrote" not in res.stdout  # tests run with no env_file
    after = real_env.stat().st_mtime_ns if real_env.exists() else None
    assert before == after
