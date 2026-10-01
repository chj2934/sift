"""`sift distill apply` and `eval`: candidates from the export, problems surfaced,
and a gate that can't run reported cleanly. Offline: no API call is made."""

from __future__ import annotations

import json
from datetime import date

import pytest


def _run(*args: str):
    from typer.testing import CliRunner

    from sift.cli import app

    return CliRunner().invoke(app, list(args))


def _flat(text: str) -> str:
    return " ".join(text.split())


def _drop(url: str) -> dict:
    return {
        "url": url,
        "decision": "drop",
        "already_known": "a textbook technique",
        "reason": "known",
        "justification": "covered in training data",
    }


def _jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def exported(tmp_path):
    """A report-type export, as `sift distill export --type report` writes it."""
    from sift.distill.candidates import Candidate
    from sift.distill.manual import write_candidates

    cand = Candidate(
        title="An old report",
        url="https://hackerone.com/reports/1",
        text="report text " * 20,
        source="hackerone-public",
        created=date(2026, 5, 1),
    )
    path = tmp_path / "candidates.jsonl"
    write_candidates([cand], path)
    return path, cand


def test_apply_matches_against_the_export_without_type(tmp_path, exported):
    from sift.distill.rejects import load_rejects

    cpath, cand = exported
    verdicts = tmp_path / "verdicts.jsonl"
    _jsonl(verdicts, [_drop(cand.url)])
    res = _run("distill", "apply", str(verdicts), "--candidates", str(cpath))
    assert res.exit_code == 0, res.output
    assert "0 kept, 1 dropped, 0 skipped" in _flat(res.stdout)
    assert "matching verdicts against" in _flat(res.stderr)
    assert [r["url"] for r in load_rejects()] == [cand.url]


def test_apply_falls_back_to_the_vault_when_the_export_is_gone(tmp_path, exported):
    _cpath, cand = exported
    verdicts = tmp_path / "verdicts.jsonl"
    _jsonl(verdicts, [_drop(cand.url)])
    res = _run("distill", "apply", str(verdicts), "--candidates", str(tmp_path / "missing.jsonl"))
    # The vault has no writeup for this url, so the row can't be matched: reported.
    assert res.exit_code == 1
    err = _flat(res.stderr)
    assert "not found; matching against the vault's writeup notes" in err
    assert "0 kept, 0 dropped, 1 skipped" in _flat(res.stdout)


def test_apply_lists_bad_rows_and_exits_1(tmp_path, exported):
    cpath, cand = exported
    verdicts = tmp_path / "verdicts.jsonl"
    bad = dict(_drop(cand.url))
    del bad["justification"]
    _jsonl(verdicts, [bad])
    res = _run("distill", "apply", str(verdicts), "--candidates", str(cpath))
    assert res.exit_code == 1
    assert "verdicts.jsonl:1" in res.stderr
    assert "justification" in res.stderr


def test_apply_reports_drops_already_logged(tmp_path, exported):
    cpath, cand = exported
    verdicts = tmp_path / "verdicts.jsonl"
    _jsonl(verdicts, [_drop(cand.url)])
    assert _run("distill", "apply", str(verdicts), "--candidates", str(cpath)).exit_code == 0
    again = _run("distill", "apply", str(verdicts), "--candidates", str(cpath))
    assert again.exit_code == 0, again.output
    assert "1 drops already logged" in _flat(again.stdout)


def test_apply_needs_the_verdicts_file(tmp_path):
    res = _run("distill", "apply", str(tmp_path / "nope.jsonl"))
    assert res.exit_code == 1
    assert "no such file" in res.stderr


def test_eval_reports_a_gate_that_cannot_run(monkeypatch):
    from sift.distill.gate import GateConfigError

    calls = []

    def score_gate(*a, **k):
        calls.append(a[0])
        raise GateConfigError("HTTP 401 from the API: check the key")

    monkeypatch.setattr("sift.distill.evaluate.load_labels", lambda: [])
    monkeypatch.setattr(
        "sift.distill.evaluate.labelled_candidates", lambda labels, vault: [("cand", "keep")]
    )
    monkeypatch.setattr("sift.distill.evaluate.score_gate", score_gate)
    monkeypatch.setattr("sift.distill.gate._client", lambda: object())
    res = _run("distill", "eval")
    assert res.exit_code == 1
    assert "HTTP 401" in res.stderr
    assert calls == ["self-report"]  # stopped at the first design, not 0/0 for each
