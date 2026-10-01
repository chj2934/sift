"""One note per CVE, whichever of KEV and NVD wrote it first.

Before: `ingest kev` after `ingest nvd` left two files with one id (63 pairs in the
real vault) and search flipped between them on every reindex. With upsert-by-id but
no merge, KEV's data was refused outright. A plain overwrite would have been worse:
it drops the `kev` tag prune and scoring key on, and every EPSS score.
"""

from __future__ import annotations

from datetime import date

import pytest

CVE = "CVE-2026-1234"

KEV_RECORD = {
    "cveID": CVE,
    "vendorProject": "Acme",
    "product": "Gateway",
    "vulnerabilityName": "Acme Gateway Path Traversal",
    "dateAdded": "2026-08-20",
    "shortDescription": "Acme Gateway contains a path traversal.",
    "requiredAction": "Apply mitigations per vendor instructions.",
    "dueDate": "2026-09-10",
    "knownRansomwareCampaignUse": "Known",
    "notes": "https://acme.test/advisory",
    "cwes": ["CWE-22"],
}


def _nvd_item(desc="Path traversal in Acme Gateway before 4.2 allows file read.", score=7.5):
    return {
        "cve": {
            "id": CVE,
            "published": "2026-08-01T00:00:00.000",
            "descriptions": [{"lang": "en", "value": desc}],
            "weaknesses": [{"description": [{"value": "CWE-22"}]}],
            "metrics": {
                "cvssMetricV31": [
                    {
                        "cvssData": {
                            "baseSeverity": "HIGH",
                            "baseScore": score,
                            "vectorString": "AV:N",
                        }
                    }
                ]
            },
            "references": [{"url": "https://acme.test/advisory"}],
        }
    }


@pytest.fixture
def index(monkeypatch):
    from sift.ingest import base

    state: dict = {"batches": []}

    class _Store:
        def indexed_ids(self):
            return {i for b in state["batches"] for i in b}

        def optimize(self):
            return {"error": None}

    store = _Store()

    def fake_index(notes, _store):
        state["batches"].append([n.meta.id for n in notes])
        return len(notes)

    monkeypatch.setattr(base, "index_notes", fake_index)
    monkeypatch.setattr(base, "Store", lambda: store)
    return state


@pytest.fixture
def feeds(monkeypatch):
    """Serve KEV records and NVD items to the real sources, offline."""
    import httpx

    from sift.ingest import kev, nvd

    served = {"kev": [KEV_RECORD], "nvd": [_nvd_item()]}
    monkeypatch.setattr(kev, "fetch", lambda: list(served["kev"]))

    def handler(request):
        vulns = served["nvd"]
        return httpx.Response(200, json={"vulnerabilities": vulns, "totalResults": len(vulns)})

    real = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )
    monkeypatch.setattr(nvd.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        nvd,
        "_windows",
        lambda since_year, today=None: iter([(date(2026, 8, 1), date(2026, 8, 31))]),
    )
    return served


def _run(name):
    from sift.ingest import kev, nvd
    from sift.ingest.base import run_source

    if name == "kev":
        return run_source("kev", kev.source())
    return run_source("nvd", nvd.source(since_year=2026, cwes=["CWE-22"]))


def _cve_files(vault_path):
    return sorted((vault_path / "cve").glob("*.md"))


def _the_note(vault_path):
    from sift.vault.notes import load_note

    files = _cve_files(vault_path)
    assert len(files) == 1, [f.name for f in files]
    return load_note(files[0])


# --- end to end ------------------------------------------------------------------------


def test_kev_after_nvd_merges_into_one_note(vault_path, index, feeds):
    _run("nvd")
    res = _run("kev")

    assert res.updated == 1 and res.id_conflicts == 0 and res.written == 0
    note = _the_note(vault_path)
    assert {"kev", "known-exploited", "cve"} <= set(note.meta.tags)
    assert "## Description" in note.body and "## Required action" in note.body
    assert "## CVSS" in note.body and "## Ransomware use" in note.body
    assert note.meta.severity == "high" and note.meta.program == "Acme"
    assert note.meta.created == date(2026, 8, 1), "NVD's published date wins"
    assert note.meta.source == "nvd" and note.meta.extra["sources"] == ["nvd", "cisa-kev"]
    assert note.meta.title.startswith(f"{CVE} — Path traversal"), "the stored title is kept"


def test_nvd_after_kev_merges_into_one_note(vault_path, index, feeds):
    _run("kev")
    res = _run("nvd")

    assert res.updated == 1 and res.id_conflicts == 0
    note = _the_note(vault_path)
    assert note.meta.source == "cisa-kev" and "kev" in note.meta.tags
    assert "## Description" in note.body and "## Required action" in note.body
    assert note.meta.created == date(2026, 8, 1)


def test_reruns_change_nothing(vault_path, index, feeds):
    _run("nvd")
    _run("kev")
    path = _cve_files(vault_path)[0]
    before = path.read_bytes()
    embedded = len(index["batches"])

    assert _run("kev").unchanged == 1
    assert _run("nvd").unchanged == 1
    assert path.read_bytes() == before and len(index["batches"]) == embedded


def test_an_nvd_edit_keeps_kev_epss_and_the_users_section(vault_path, index, feeds):
    from sift.vault.notes import load_note, save_note

    _run("nvd")
    _run("kev")
    note = _the_note(vault_path)
    note.meta.extra["epss"] = 0.91
    note.meta.tags.append("high-epss")
    note.body += "\n\n## My notes\nTried on target X: patched."
    save_note(vault_path, note)

    feeds["nvd"] = [_nvd_item(desc="Updated: traversal also reaches /etc.", score=9.1)]
    res = _run("nvd")

    assert res.updated == 1
    after = load_note(_cve_files(vault_path)[0])
    assert "Updated: traversal also reaches /etc." in after.body
    assert "before 4.2 allows file read" not in after.body
    assert after.body.count("## Description") == 1
    assert "## My notes\nTried on target X: patched." in after.body
    assert "## Required action" in after.body
    assert after.meta.extra["epss"] == 0.91 and "high-epss" in after.meta.tags
    assert after.meta.extra["cvss_score"] == 9.1 and "kev" in after.meta.tags


def test_passing_the_merge_to_run_source_as_well_is_harmless(vault_path, index, feeds):
    """The sources merge themselves; a CLI that also passes merge=kev.merge must not
    merge twice or flip-flop."""
    from sift.ingest import kev
    from sift.ingest.base import run_source

    _run("nvd")
    first = run_source("kev", kev.source(), merge=kev.merge)
    again = run_source("kev", kev.source(), merge=kev.merge)

    assert first.updated == 1 and again.unchanged == 1
    assert len(_cve_files(vault_path)) == 1


def test_a_users_note_under_a_cve_id_is_never_merged_into(vault_path, index, feeds):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    mine = Note(
        meta=Frontmatter(id=CVE, type="finding", title="My finding", source="manual"),
        body="my own notes",
    )
    path = save_note(vault_path, mine)
    before = path.read_bytes()

    res = _run("kev")

    assert res.id_conflicts == 1 and path.read_bytes() == before


@pytest.mark.parametrize("newer", ["A twin.md", "B twin.md"])
def test_existing_twins_are_reported_and_only_one_is_merged_into(
    vault_path, index, feeds, caplog, monkeypatch, newer
):
    """Same-source twins (an NVD retitle used to fork a second file): the merge reaches
    exactly one of them - the one the writer updates - built from that twin's own text,
    never from the other's; the other is left for the user to judge, and the run says
    so. Before, the merge was computed against one twin and written over the other."""
    import logging
    import os

    from sift.ingest import existing
    from sift.ingest.nvd import to_note as nvd_note
    from sift.vault.notes import load_note

    monkeypatch.setattr(existing, "_TWINS_REPORTED", set())  # once per id per process

    texts = {"A twin.md": "Twin A wording of the flaw.", "B twin.md": "Twin B wording of the flaw."}
    (vault_path / "cve").mkdir(parents=True)
    for name, desc in texts.items():
        (vault_path / "cve" / name).write_text(
            nvd_note(_nvd_item(desc=desc)).render(), encoding="utf-8"
        )
    older = next(n for n in texts if n != newer)
    os.utime(vault_path / "cve" / older, (1_700_000_000, 1_700_000_000))
    before = {n: (vault_path / "cve" / n).read_bytes() for n in texts}

    caplog.set_level(logging.WARNING, logger="sift")
    res = _run("kev")

    assert res.updated == 1 and res.id_conflicts == 0
    merged = [n for n in texts if (vault_path / "cve" / n).read_bytes() != before[n]]
    assert len(merged) == 1, "exactly one twin is written"
    (name,) = merged
    other = next(n for n in texts if n != name)
    note = load_note(vault_path / "cve" / name)
    assert "kev" in note.meta.tags and "## Required action" in note.body
    assert texts[name] in note.body and texts[other] not in note.body, "no cross-twin content"
    assert "carried by 2 files" in caplog.text and other in caplog.text

    again = _run("kev")
    assert again.unchanged == 1 and (vault_path / "cve" / other).read_bytes() == before[other]


# --- the pure merge ---------------------------------------------------------------------


def _kev_note():
    from sift.ingest.kev import to_note

    return to_note(KEV_RECORD)


def _nvd_note(**kw):
    from sift.ingest.nvd import to_note

    return to_note(_nvd_item(**kw))


def test_split_sections_reproduces_a_feed_body():
    from sift.ingest.cve_merge import _join, split_sections

    for note in (_kev_note(), _nvd_note()):
        pre, secs = split_sections(note.body)
        assert _join(pre, [b for _, b in secs]) == note.body


def test_headings_inside_code_fences_are_not_sections():
    from sift.ingest.cve_merge import split_sections

    body = "## Description\ntext\n```\n## not a heading\n```\n## CVSS\nhigh"
    _pre, secs = split_sections(body)
    assert [k for k, _ in secs] == ["description", "cvss"]


def test_merge_is_idempotent_both_ways():
    from sift.ingest.cve_merge import merge_cve

    nvd, kev = _nvd_note(), _kev_note()
    once = merge_cve(nvd, kev)
    twice = merge_cve(once, _kev_note())
    once.meta.ingested = twice.meta.ingested = None
    assert twice.render() == once.render()

    rev = merge_cve(_kev_note(), _nvd_note())
    assert merge_cve(rev, _nvd_note()).render() == rev.render()


def test_a_section_the_feed_no_longer_provides_is_kept():
    from sift.ingest.cve_merge import merge_body

    stored = "## Description\nold\n\n## CVSS\nhigh (7.5)\n\n## References\n- a"
    incoming = "## Description\nnew"
    assert (
        merge_body(stored, incoming, "nvd")
        == "## Description\nnew\n\n## CVSS\nhigh (7.5)\n\n## References\n- a"
    )


def test_kev_lead_line_is_replaced_and_user_text_above_kept():
    from sift.ingest.cve_merge import merge_body

    stored = "**Acme Old** — Old name\nmy remark\n\n## Summary\nx"
    incoming = "**Acme Gateway** — New name\n\n## Summary\ny"
    merged = merge_body(stored, incoming, "cisa-kev")
    assert merged == "**Acme Gateway** — New name\n\nmy remark\n\n## Summary\ny"
