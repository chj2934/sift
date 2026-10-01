from __future__ import annotations

import time

import pytest

DETAILED = (
    "## Summary\nThe `next` parameter allows SSRF to the cloud metadata endpoint.\n\n"
    "## Steps to reproduce\n1. Send `GET /fetch?next=http://169.254.169.254/latest/meta-data/`.\n"
    "2. Observe the IAM credentials in the response.\n\n"
    "```\nGET /fetch?next=http://169.254.169.254/ HTTP/1.1\nAuthorization: Bearer x\n```\n\n"
    "## Impact\nFull account compromise via stolen credentials.\n"
) * 2


def _fm(**kw):
    from sift.vault.schema import Frontmatter

    kw.setdefault("title", "t")
    return Frontmatter(**kw)


def _score(meta, body: str) -> int:
    from sift.quality import score_note

    return score_note(meta, body)


def _report(body: str, **extra_and_meta):
    meta_kw = {k: extra_and_meta.pop(k) for k in ("severity",) if k in extra_and_meta}
    return _fm(id="h1-1", type="report", extra=extra_and_meta or {}, **meta_kw), body


def test_thin_report_scores_low():
    meta, body = _report("It is broken. Please fix.")
    assert _score(meta, body) < 45


def test_detailed_bountied_report_scores_high():
    meta, body = _report(DETAILED, has_bounty=True, vote_count=20, severity="high")
    assert _score(meta, body) > 75


def test_dupe_is_penalised():
    meta_clean, body = _report(DETAILED, has_bounty=True, vote_count=8)
    meta_dupe, _ = _report(DETAILED, has_bounty=True, vote_count=8, is_dupe=True)
    assert _score(meta_dupe, body) < _score(meta_clean, body) - 20


def test_manual_note_defaults_high():
    meta = _fm(id="t-1", type="technique", title="IDOR via node id")
    assert _score(meta, "Short idea, few words.") >= 70


def test_kev_cve_beats_plain_cve():
    kev = _fm(
        id="CVE-2024-1",
        type="cve",
        tags=["kev", "known-exploited"],
        extra={"epss_percentile": 0.95},
    )
    plain = _fm(id="CVE-2024-2", type="cve", tags=["cve"], extra={})
    body = "## Description\nSome vuln."
    assert _score(kev, body) > _score(plain, body) + 20


# --------------------------------------------------------------------------- #
# quality-ignores-provenance-and-tool
# --------------------------------------------------------------------------- #
def test_tool_note_clears_min_quality_70():
    meta = _fm(id="tool-acme", type="tool", title="Acme setup")
    assert _score(meta, "Cloned to G:/acme. pnpm. Sepolia, funded.") >= 70


def test_tool_note_ranks_with_the_other_authored_types():
    body = "Setup notes. " * 50
    assert _score(_fm(id="a", type="tool"), body) == _score(_fm(id="b", type="technique"), body)


def _own(state: str):
    """The shape h1_api.my_reports writes: no has_bounty, no is_dupe, no vote_count."""
    tags = ["hackerone", "mine", state] + (["dupe"] if state == "duplicate" else [])
    return _fm(
        id="h1mine-42",
        type="report",
        source="hackerone-mine",
        tags=tags,
        extra={"state": state, "weakness": "SSRF"},
    )


@pytest.mark.parametrize("state", ["resolved", "triaged", "informative", "new"])
def test_own_report_clears_min_quality_70(state):
    assert _score(_own(state), "It is broken.") >= 70


@pytest.mark.parametrize("state", ["duplicate", "not-applicable", "spam"])
def test_own_rejected_report_gets_no_floor(state):
    assert _score(_own(state), "It is broken.") < 70


def test_own_report_tagged_dupe_gets_no_floor_even_without_a_state():
    # A remembered report the user tagged "dupe" by hand: no h1 `state` field at all.
    meta = _fm(id="repo-dupe-idor-20260901123456789", type="report", tags=["dupe"])
    assert _score(meta, "It is broken.") < 70


def test_dupe_tag_and_state_are_penalised_like_is_dupe():
    clean = _fm(id="h1-9", type="report", source="hackerone-public")
    tagged = _fm(id="h1-9", type="report", source="hackerone-public", tags=["dupe"])
    stated = _fm(id="h1-9", type="report", source="hackerone-public", extra={"state": "duplicate"})
    base = _score(clean, DETAILED)
    assert _score(tagged, DETAILED) == base - 35
    assert _score(stated, DETAILED) == base - 35


def test_hacktivity_award_counts_as_bounty():
    from sift.quality import has_bounty

    def meta(amount):
        return _fm(
            id="h1act-1",
            type="report",
            source="hackerone-hacktivity",
            extra={"total_awarded_amount": amount, "weakness": "SSRF"},
        )

    paid, unpaid = meta("5000"), meta(None)
    assert has_bounty(paid) and not has_bounty(unpaid)
    assert _score(paid, DETAILED) == _score(unpaid, DETAILED) + 15
    assert has_bounty(meta(1500.0)) and has_bounty(meta("$1,500"))


def test_frontmatter_bounty_field_counts_as_bounty():
    from sift.quality import has_bounty

    assert has_bounty(_fm(id="r", type="report", bounty=500))
    assert not has_bounty(_fm(id="r", type="report", bounty=0))


def test_resolved_earns_a_bonus_but_is_not_a_bounty():
    from sift.quality import has_bounty

    resolved = _fm(id="h1-2", type="report", source="hackerone-public", extra={"state": "resolved"})
    plain = _fm(id="h1-2", type="report", source="hackerone-public")
    assert not has_bounty(resolved)  # a VDP resolves at $0
    assert _score(resolved, DETAILED) == _score(plain, DETAILED) + 8


@pytest.mark.parametrize(
    "junk", [None, "", "n/a", "TBD", float("nan"), float("inf"), [], {}, True, 10**400]
)
def test_junk_amounts_and_counts_never_crash_or_count(junk):
    from sift.quality import has_bounty

    meta = _fm(
        id="h1act-3",
        type="report",
        extra={
            "total_awarded_amount": junk,
            "vote_count": junk,
            "has_bounty": "false" if junk is None else None,
        },
    )
    assert 0 <= _score(meta, "x") <= 100
    assert not has_bounty(meta)


def test_string_vote_count_counts_like_a_number():
    a = _fm(id="h1-4", type="report", extra={"vote_count": "15"})
    b = _fm(id="h1-4", type="report", extra={"vote_count": 15})
    assert _score(a, DETAILED) == _score(b, DETAILED)


def test_remembered_report_clears_min_quality_70():
    meta = _fm(id="repo-thin-finding-20260901123456789", type="report", source="sift-remember")
    assert _score(meta, "short") >= 70


def test_remember_with_a_caller_source_is_still_user_authored():
    from sift.quality import is_user_authored

    meta = _fm(
        id="repo-ssrf-in-export-20260915101112131",
        type="report",
        source="hackerone-public",
        url="https://hackerone.com/reports/1",
    )
    assert is_user_authored(meta)
    assert _score(meta, "short") >= 70


def test_authored_via_marker_and_mine_tag_count():
    from sift.quality import AUTHORED_VIA, is_user_authored

    marked = _fm(id="x-1", type="report", source="nvd", extra={AUTHORED_VIA: "sift-remember"})
    tagged = _fm(id="x-2", type="report", source="nvd", tags=["Mine"])
    assert is_user_authored(marked) and is_user_authored(tagged)


def test_bulk_notes_are_not_user_authored():
    from sift.quality import is_user_authored

    bulk = [
        _fm(id="h1-123", type="report", source="hackerone-public", tags=["hackerone", "disclosed"]),
        _fm(id="h1act-9", type="report", source="hackerone-hacktivity"),
        _fm(id="CVE-2024-12345", type="cve", source="nvd", tags=["cve"]),
        _fm(
            id="CVE-2021-44228",
            type="cve",
            source="cisa-kev",
            tags=["kev", "known-exploited", "apache", "log4j2"],
        ),
        _fm(id="tech-ssrf-via-pdf", type="technique", source="nvd"),
    ]
    for meta in bulk:
        assert not is_user_authored(meta), meta.id


# --------------------------------------------------------------------------- #
# step-regex-quadratic
# --------------------------------------------------------------------------- #
def test_whitespace_only_lines_score_in_linear_time():
    # The old `^\s*\d+\.\s+\S` took ~10 s on this body (and ~38 s at twice the size).
    meta = _fm(id="h1-5", type="report")
    body = "   \n" * 20000
    start = time.perf_counter()
    _score(meta, body)
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize(
    "step,not_step",
    [
        ("\xa01. do x", "\xa0a. do x"),  # NBSP before the number (scraped HTML)
        ("  2. do y", "  b. do y"),  # indented
        ("3.\xa0do z", "c.\xa0do z"),  # NBSP after the dot
    ],
)
def test_numbered_steps_still_earn_the_bonus(step, not_step):
    meta = _fm(id="h1-6", type="report")
    filler = "Some report prose. " * 5 + "\n"
    assert _score(meta, filler + step) - _score(meta, filler + not_step) == 8


def test_bare_hash_lines_are_not_headings():
    meta = _fm(id="h1-7", type="report")
    real = "# a\n# b\n# c\n"
    bare = "#\na\n#\nb\n#\nc\n"
    assert _score(meta, real) - _score(meta, bare) == 5
