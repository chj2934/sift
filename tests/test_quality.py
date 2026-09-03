from __future__ import annotations

from sift.quality import score_note
from sift.vault.schema import Frontmatter


def _report(body: str, **extra_and_meta):
    meta_kw = {k: extra_and_meta.pop(k) for k in ("severity",) if k in extra_and_meta}
    return Frontmatter(
        id="h1-1", type="report", title="t", extra=extra_and_meta or {}, **meta_kw
    ), body


DETAILED = (
    "## Summary\nThe `next` parameter allows SSRF to the cloud metadata endpoint.\n\n"
    "## Steps to reproduce\n1. Send `GET /fetch?next=http://169.254.169.254/latest/meta-data/`.\n"
    "2. Observe the IAM credentials in the response.\n\n"
    "```\nGET /fetch?next=http://169.254.169.254/ HTTP/1.1\nAuthorization: Bearer x\n```\n\n"
    "## Impact\nFull account compromise via stolen credentials.\n"
) * 2


def test_thin_report_scores_low():
    meta, body = _report("It is broken. Please fix.")
    assert score_note(meta, body) < 45


def test_detailed_bountied_report_scores_high():
    meta, body = _report(DETAILED, has_bounty=True, vote_count=20, severity="high")
    assert score_note(meta, body) > 75


def test_dupe_is_penalised():
    meta_clean, body = _report(DETAILED, has_bounty=True, vote_count=8)
    meta_dupe, _ = _report(DETAILED, has_bounty=True, vote_count=8, is_dupe=True)
    assert score_note(meta_dupe, body) < score_note(meta_clean, body) - 20


def test_manual_note_defaults_high():
    meta = Frontmatter(id="t-1", type="technique", title="IDOR via node id")
    assert score_note(meta, "Short idea, few words.") >= 70


def test_kev_cve_beats_plain_cve():
    kev = Frontmatter(
        id="CVE-2024-1", type="cve", title="c", tags=["kev", "known-exploited"],
        extra={"epss_percentile": 0.95},
    )
    plain = Frontmatter(id="CVE-2024-2", type="cve", title="c", tags=["cve"], extra={})
    body = "## Description\nSome vuln."
    assert score_note(kev, body) > score_note(plain, body) + 20
