from __future__ import annotations

from datetime import date

from sift.prune import classify
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

DETAILED = (
    "## Summary\nSSRF to metadata.\n\n## Steps to reproduce\n1. do x\n2. do y\n\n"
    "```\nGET /x HTTP/1.1\n```\n\n## Impact\nATO.\n"
) * 2


def _note(ntype, *, created=None, tags=None, body="body", **extra):
    return Note(
        meta=Frontmatter(
            id=f"{ntype}-1", type=ntype, title="t",
            created=created, tags=tags or [], extra=extra,
        ),
        body=body,
    )


def _v(note, year=2025, bar=55):
    return classify(note, keep_since_year=year, report_quality_bar=bar)


def test_recent_report_kept():
    assert _v(_note("report", created=date(2025, 6, 1))).keep


def test_old_thin_report_dropped():
    assert not _v(_note("report", created=date(2019, 1, 1), body="broken, fix it")).keep


def test_old_bountied_substantial_report_kept():
    n = _note("report", created=date(2019, 1, 1), body=DETAILED, has_bounty=True,
              vote_count=15)
    assert _v(n).keep


def test_kev_cve_kept_even_if_old():
    assert _v(_note("cve", created=date(2010, 1, 1), tags=["kev", "known-exploited"])).keep


def test_high_epss_cve_kept():
    assert _v(_note("cve", created=date(2018, 1, 1), tags=["cve"], epss_percentile=0.95)).keep


def test_old_low_signal_cve_dropped():
    assert not _v(_note("cve", created=date(2018, 1, 1), tags=["cve"],
                        epss_percentile=0.2)).keep


def test_recent_cve_needs_severity_or_epss():
    plain = _note("cve", created=date(2025, 3, 1), tags=["cve"], epss_percentile=0.2)
    assert not _v(plain).keep
    hi = _note("cve", created=date(2025, 3, 1), tags=["cve"], epss_percentile=0.2)
    hi.meta.severity = "high"
    assert _v(hi).keep


def test_manual_notes_always_kept():
    for t in ("technique", "target", "finding", "writeup"):
        assert _v(_note(t, created=None)).keep
