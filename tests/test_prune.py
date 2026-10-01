from __future__ import annotations

from datetime import date

import pytest

DETAILED = (
    "## Summary\nSSRF to metadata.\n\n## Steps to reproduce\n1. do x\n2. do y\n\n"
    "```\nGET /x HTTP/1.1\n```\n\n## Impact\nATO.\n"
) * 2

# Drops only ever come from these; a note without a bulk source is kept (fail closed).
PUBLIC = "hackerone-public"
NVD = "nvd"


def _note(ntype, *, created=None, tags=None, body="body", source=None, nid=None, **extra):
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    return Note(
        meta=Frontmatter(
            id=nid or f"{ntype}-1",
            type=ntype,
            title="t",
            source=source,
            created=created,
            tags=tags or [],
            extra=extra,
        ),
        body=body,
    )


def _v(note, year=2025, bar=55):
    from sift.prune import classify

    return classify(note, keep_since_year=year, report_quality_bar=bar)


def test_recent_report_kept():
    assert _v(_note("report", created=date(2025, 6, 1), source=PUBLIC)).keep


def test_old_thin_report_dropped():
    assert not _v(
        _note("report", created=date(2019, 1, 1), body="broken, fix it", source=PUBLIC)
    ).keep


def test_old_bountied_substantial_report_kept():
    n = _note(
        "report",
        created=date(2019, 1, 1),
        body=DETAILED,
        has_bounty=True,
        vote_count=15,
        source=PUBLIC,
    )
    v = _v(n)
    assert v.keep and v.reason == "old but bountied+substantial"


def test_kev_cve_kept_even_if_old():
    assert _v(
        _note("cve", created=date(2010, 1, 1), tags=["kev", "known-exploited"], source="cisa-kev")
    ).keep


def test_high_epss_cve_kept():
    assert _v(
        _note("cve", created=date(2018, 1, 1), tags=["cve"], epss_percentile=0.95, source=NVD)
    ).keep


def test_old_low_signal_cve_dropped():
    assert not _v(
        _note("cve", created=date(2018, 1, 1), tags=["cve"], epss_percentile=0.2, source=NVD)
    ).keep


def test_recent_cve_needs_severity_or_epss():
    plain = _note("cve", created=date(2025, 3, 1), tags=["cve"], epss_percentile=0.2, source=NVD)
    assert not _v(plain).keep
    hi = _note("cve", created=date(2025, 3, 1), tags=["cve"], epss_percentile=0.2, source=NVD)
    hi.meta.severity = "high"
    assert _v(hi).keep


def test_manual_notes_always_kept():
    for t in ("technique", "target", "finding", "writeup", "reference", "tool"):
        assert _v(_note(t, created=None)).keep, t


# --------------------------------------------------------------------------- #
# fail closed: provenance, unknown types, undated (prune-deletes-user-authored-notes)
# --------------------------------------------------------------------------- #
def test_tool_note_is_kept_even_with_a_bulk_source():
    v = _v(_note("tool", created=date(2019, 1, 1), source=PUBLIC))
    assert v.keep and v.reason == "authored/curated"


def test_unknown_future_type_is_kept_not_dropped():
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    # A type added to the schema later, without prune being taught about it.
    meta = Frontmatter.model_construct(
        id="pb-1",
        type="playbook",
        title="t",
        source=PUBLIC,
        created=date(2019, 1, 1),
        tags=[],
        extra={},
        severity=None,
        bounty=None,
        url=None,
    )
    v = _v(Note(meta=meta, body="x"))
    assert v.keep and "not prunable" in v.reason


@pytest.mark.parametrize("ntype", ["report", "cve"])
def test_remembered_note_without_created_is_kept(ntype):
    n = _note(
        ntype, created=None, source="sift-remember", nid=f"{ntype[:4]}-thin-note-20260901123456789"
    )
    v = _v(n)
    assert v.keep and v.reason == "user-authored"


def test_own_h1_report_from_2019_is_kept():
    n = _note(
        "report",
        created=date(2019, 5, 1),
        source="hackerone-mine",
        nid="h1mine-123",
        tags=["hackerone", "mine", "resolved"],
        state="resolved",
    )
    v = _v(n)
    assert v.keep and v.reason == "user-authored"


def test_mine_tag_alone_protects_a_report():
    n = _note("report", created=date(2019, 5, 1), source=PUBLIC, tags=["mine"])
    assert _v(n).keep


def test_local_note_with_a_custom_source_is_kept():
    n = _note("report", created=date(2019, 1, 1), source="my-notes", body="broken")
    v = _v(n)
    assert v.keep and "non-bulk source" in v.reason


def test_note_with_no_source_is_kept():
    v = _v(_note("cve", created=date(2018, 1, 1), tags=["cve"], epss_percentile=0.2))
    assert v.keep and "non-bulk source" in v.reason


def test_remember_marker_beats_a_caller_supplied_bulk_source():
    n = _note("report", created=date(2019, 1, 1), source=NVD, authored_via="sift-remember")
    assert _v(n).keep


def test_remember_id_beats_a_caller_supplied_bulk_source():
    # remember(source="hackerone-public", url=...) recording where a finding came from,
    # saved before remember stamped `authored_via`.
    n = _note(
        "report", created=None, source=PUBLIC, nid="repo-ssrf-in-pdf-export-20260915101112131"
    )
    v = _v(n)
    assert v.keep and v.reason == "user-authored"


@pytest.mark.parametrize("ntype,source", [("report", PUBLIC), ("cve", NVD)])
def test_undated_bulk_note_is_kept_in_its_own_bucket(ntype, source):
    from sift.prune import UNDATED

    v = _v(_note(ntype, created=None, source=source, body="thin", tags=["cve"]))
    assert v.keep and v.reason == UNDATED


def test_undated_bountied_report_is_reported_as_undated_not_old():
    from sift.prune import UNDATED

    n = _note("report", created=None, body=DETAILED, has_bounty=True, vote_count=15, source=PUBLIC)
    assert _v(n).reason == UNDATED


def test_hacktivity_award_counts_as_bounty():
    n = _note(
        "report",
        created=date(2019, 1, 1),
        body=DETAILED,
        vote_count=15,
        source="hackerone-hacktivity",
        total_awarded_amount="5000",
    )
    v = _v(n, bar=62)
    assert v.keep and v.reason == "old but bountied+substantial"


def test_only_bulk_reports_and_cves_can_ever_be_dropped():
    """Exhaustive over types x sources: a drop needs a prunable type AND a bulk source."""
    from sift.prune import BULK_SOURCES
    from sift.vault.schema import NOTE_TYPES

    sources = [
        None,
        "manual",
        "sift-remember",
        "hackerone-mine",
        "my-notes",
        "chromium-src",
        *sorted(BULK_SOURCES),
    ]
    for ntype in NOTE_TYPES:
        for source in sources:
            v = _v(_note(ntype, created=date(2010, 1, 1), source=source, body="thin", tags=["cve"]))
            if not v.keep:
                assert ntype in ("report", "cve") and source in BULK_SOURCES, (ntype, source)
