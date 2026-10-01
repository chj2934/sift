"""`sift ingest epss`: only changed scores are written, in place, in batches.

Before: every run rewrote all ~6.2k CVE files (bumping `ingested`, so Obsidian and
sync tools saw 6k changes), re-embedded them one at a time, wrote a renamed note back
under its title-derived name (a duplicate), never removed `high-epss`, and one 5xx
batch threw away every score fetched so far.
"""

from __future__ import annotations

import pytest


def _cve(vault_path, cve, *, title=None, epss=None, tags=("cve",)):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    extra = {} if epss is None else {"epss": epss, "epss_percentile": 0.5}
    meta = Frontmatter(
        id=cve,
        type="cve",
        title=title or f"{cve} — thing",
        source="nvd",
        tags=list(tags),
        extra=extra,
    )
    return save_note(vault_path, Note(meta=meta, body=f"## Description\n{cve} body"))


@pytest.fixture
def store():
    class _Store:
        optimized = 0

        def optimize(self):
            self.optimized += 1
            return {"error": None}

    return _Store()


@pytest.fixture
def indexed(monkeypatch):
    from sift.ingest import epss

    batches: list[list[str]] = []
    monkeypatch.setattr(
        epss,
        "index_notes",
        lambda notes, store: batches.append([n.meta.id for n in notes]) or len(notes),
    )
    return batches


def _scores(**kv):
    def fetch(ids):
        return {k.replace("_", "-"): {"epss": v, "percentile": 0.9} for k, v in kv.items()}, 0

    return fetch


def test_a_second_run_with_the_same_scores_writes_nothing(vault_path, store, indexed):
    from sift.ingest.epss import enrich_notes

    a = _cve(vault_path, "CVE-2026-0001")
    fetch = _scores(CVE_2026_0001=0.7)

    first = enrich_notes(vault=vault_path, store=store, fetch=fetch)
    stamp = a.read_bytes()
    second = enrich_notes(vault=vault_path, store=store, fetch=fetch)

    assert (first.changed, second.changed, second.unchanged) == (1, 0, 1)
    assert a.read_bytes() == stamp, "an unchanged score must not bump `ingested`"
    assert indexed == [["CVE-2026-0001"]] and store.optimized == 1


def test_a_renamed_note_is_updated_in_place(vault_path, store, indexed):
    from sift.ingest.epss import enrich_notes
    from sift.vault.notes import load_note

    original = _cve(vault_path, "CVE-2026-0002")
    renamed = original.with_name("my name for it.md")
    original.rename(renamed)

    enrich_notes(vault=vault_path, store=store, fetch=_scores(CVE_2026_0002=0.2))

    assert sorted(p.name for p in (vault_path / "cve").glob("*.md")) == ["my name for it.md"]
    assert load_note(renamed).meta.extra["epss"] == 0.2


def test_high_epss_tag_follows_the_score_both_ways(vault_path, store, indexed):
    from sift.ingest.epss import enrich_notes
    from sift.vault.notes import load_note

    path = _cve(vault_path, "CVE-2026-0003")
    enrich_notes(vault=vault_path, store=store, fetch=_scores(CVE_2026_0003=0.8))
    assert "high-epss" in load_note(path).meta.tags

    enrich_notes(vault=vault_path, store=store, fetch=_scores(CVE_2026_0003=0.1))
    assert "high-epss" not in load_note(path).meta.tags


def test_a_failing_batch_keeps_the_other_batches_scores(monkeypatch):
    import httpx

    from sift.ingest import epss

    monkeypatch.setattr(epss, "BATCH", 1)

    def handler(request):
        cve = request.url.params["cve"]
        if cve == "CVE-2026-0002":
            return httpx.Response(503, text="busy")
        return httpx.Response(
            200, json={"data": [{"cve": cve, "epss": "0.3", "percentile": "0.6"}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        scores, failed = epss._fetch_scores(
            ["CVE-2026-0001", "CVE-2026-0002", "CVE-2026-0003"], client=client, sleep=lambda s: None
        )

    assert failed == 1
    assert sorted(scores) == ["CVE-2026-0001", "CVE-2026-0003"]


def test_a_transient_error_is_retried(monkeypatch):
    import httpx

    from sift.ingest import epss

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(
            200, json={"data": [{"cve": "CVE-2026-0001", "epss": "0.3", "percentile": "0.6"}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        scores, failed = epss._fetch_scores(["CVE-2026-0001"], client=client, sleep=lambda s: None)

    assert failed == 0 and scores["CVE-2026-0001"]["epss"] == 0.3 and calls["n"] == 2


def test_no_cve_notes_prints_nothing(vault_path, store, capfd):
    from sift.ingest.epss import enrich_notes

    res = enrich_notes(vault=vault_path, store=store, fetch=_scores())
    assert res.notes == 0 and capfd.readouterr().out == ""
