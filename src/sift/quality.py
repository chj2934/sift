"""Heuristic note-quality score (0-100).

A cheap, deterministic signal for *how much a note is worth retrieving* — used
to re-weight search results and as the filter for ``sift prune``. No network,
no LLM: everything comes from frontmatter the ingesters already set plus a few
structural checks on the body.

The score is intentionally blunt. It only needs to separate "detailed writeup
with a working PoC and a bounty" from "four-sentence dupe with no bounty".

It also owns the two provenance facts prune shares with it — :func:`is_user_authored`
and :func:`has_bounty` — so the scorer and the pruner can never disagree about whose
note something is or whether it was paid.
"""

from __future__ import annotations

import math
import re

from sift.vault.schema import NOTE_TYPES, Frontmatter

# Base score by note type. Hand-authored notes are trusted curation and start high;
# bulk-ingested reports/CVEs have to earn their place from the signals below.
_BASE = {
    "report": 40,
    "cve": 42,
    "technique": 78,
    "target": 78,
    "finding": 78,
    "writeup": 70,
    # Per-program setup the user wrote: clone path, chains, funding, submission
    # gotchas. CLAUDE.md calls it the note that saves the most time on return, so it
    # must clear `search_memory(min_quality=70)` on its own, like the other authored
    # types. Without an entry it fell back to the bulk-report base and topped out at 62.
    "tool": 78,
    # Primary vendor source. High, but under the authored notes: a severity guideline
    # settles an argument, it does not tell you where to look.
    "reference": 74,
}

# `[^\S\n]` is "whitespace except newline", so a numbered step and its text must share
# a line. The old `^\s*\d+\.\s+\S` let `\s*` run across every following blank line and
# backtrack from each line start: quadratic on bodies of whitespace-only lines (38 s at
# 156 KB of "   \n"). Unlike `[ \t]`, it still accepts NBSP, which HTML-to-text output
# leaves in front of scraped list items.
_STEP_RE = re.compile(
    r"steps?\s+to\s+reproduce|^[^\S\n]*\d+\.[^\S\n]+\S", re.IGNORECASE | re.MULTILINE
)
# Same reason: a bare `#` line followed by text on the next line is not a heading.
_HEADING_RE = re.compile(r"^#{1,6}[^\S\n]+\S", re.MULTILINE)
_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_HTTP_RE = re.compile(
    r"HTTP/\d|^(?:GET|POST|PUT|PATCH|DELETE)\s+/|\bcurl\s+-|Authorization:\s|\bBurp\b",
    re.IGNORECASE | re.MULTILINE,
)

# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
# Sources that only ever carry what the user wrote or submitted: their own HackerOne
# reports, notes saved from a session, and hand-written markdown (`sift ingest local`
# defaults `source` to "manual").
USER_SOURCES = frozenset({"hackerone-mine", "sift-remember", "sift-capture-idea", "manual"})
# Id prefixes only the user's own writers mint: h1_api.my_reports, capture_idea, and
# local_notes' frontmatter backfill.
_USER_ID_PREFIXES = ("h1mine-", "idea-", "local-")
# `remember` ids are "<type[:4]>-<slug>-<YYYYmmddHHMMSS><ms>". The id still says
# "remembered" when the caller overrode `source` to record where a finding came from.
_REMEMBER_PREFIXES = "|".join(sorted(re.escape(t[:4]) for t in NOTE_TYPES))
_REMEMBER_ID_RE = re.compile(rf"^(?:{_REMEMBER_PREFIXES})-.+-\d{{17}}$")
# Frontmatter `extra` key the MCP writers stamp. Unlike `source`, a caller can't override it.
AUTHORED_VIA = "authored_via"

# The user's own notes clear the documented `min_quality=70` filter without
# outranking strong curated material (authored types sit at 78).
USER_FLOOR = 72
# HackerOne states that mean the program accepted the report. Not the same as paid:
# a VDP resolves at $0, so this earns its own smaller bonus, never `has_bounty`.
_ACCEPTED_STATES = frozenset({"resolved", "triaged"})
# Outcomes that get no provenance floor: the user's own dupe must not outrank a
# bountied third-party writeup just because they wrote it.
_REJECTED_STATES = frozenset({"duplicate", "not-applicable", "spam"})


def as_float(value: object) -> float:
    """Best-effort number from an API or frontmatter field.

    Amounts and counts arrive as ints, floats, numeric strings ("5000", "$1,500") or
    null. Anything else - None, booleans, junk strings, NaN, infinities - is 0.0, so a
    malformed field can never crash scoring (a crash here fails a whole index batch).
    """
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        if isinstance(value, (int, float)):
            f = float(value)
        elif isinstance(value, str):
            f = float(value.strip().replace(",", "").lstrip("$") or "nan")
        else:
            return 0.0
    except (ValueError, OverflowError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _truthy(value: object) -> bool:
    """A flag that may have been hand-edited into YAML as a string."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    if isinstance(value, bool):
        return value
    return as_float(value) != 0.0


def _state(extra: dict) -> str:
    return str(extra.get("state") or "").strip().lower()


def _lower_tags(tags: list[str]) -> set[str]:
    return {str(t).strip().lower() for t in tags or ()}


def is_user_authored(meta: Frontmatter) -> bool:
    """True for anything the user wrote, submitted or saved from a session.

    Shared by scoring (a floor, so the user's own notes clear `min_quality=70`) and by
    prune (which never touches them). Several independent markers, because no single
    one is reliable: `remember` lets the caller override `source`, notes saved before
    the `authored_via` stamp don't carry it, and a hand-written note can carry any source.
    """
    if (meta.source or "").strip().lower() in USER_SOURCES:
        return True
    nid = meta.id or ""
    if nid.startswith(_USER_ID_PREFIXES) or _REMEMBER_ID_RE.match(nid):
        return True
    if "mine" in _lower_tags(meta.tags):
        return True
    return bool((meta.extra or {}).get(AUTHORED_VIA))


def _paid(extra: dict, bounty: object = None) -> bool:
    return (
        _truthy(extra.get("has_bounty"))  # h1_public
        or as_float(extra.get("total_awarded_amount")) > 0  # h1 hacktivity
        or as_float(extra.get("bounty_amount")) > 0
        or as_float(bounty) > 0  # the frontmatter `bounty` field
    )


def has_bounty(meta: Frontmatter) -> bool:
    """True when any source's bounty signal says the report was paid.

    Each ingester records payment its own way; reading only h1_public's `has_bounty`
    meant a $5,000 hacktivity report counted as unpaid. Accepted-but-unpaid states
    ("resolved", "triaged") deliberately do not count.
    """
    return _paid(meta.extra or {}, meta.bounty)


def _is_dupe(extra: dict, tags: list[str]) -> bool:
    # h1_public sets `is_dupe`; h1_api.my_reports only tags "dupe" and records the state.
    return (
        _truthy(extra.get("is_dupe")) or "dupe" in _lower_tags(tags) or _state(extra) == "duplicate"
    )


def _clamp(n: float, lo: float = 0.0, hi: float = 100.0) -> int:
    return int(max(lo, min(hi, round(n))))


def _report_signals(
    extra: dict, severity: str | None, tags: list[str], bounty: object = None
) -> float:
    s = 0.0
    if _paid(extra, bounty):
        s += 15
    if _state(extra) in _ACCEPTED_STATES:
        s += 8
    vc = as_float(extra.get("vote_count"))
    if vc > 0:
        s += min(20.0, 4.0 * math.log2(1 + vc))
    if _is_dupe(extra, tags):
        s -= 35
    s += {"critical": 12, "high": 8, "medium": 3}.get((severity or "").lower(), 0)
    return s


def _cve_signals(extra: dict, tags: list[str], severity: str | None) -> float:
    s = 0.0
    if any(t in ("kev", "known-exploited") for t in tags):
        s += 22
    pct = min(1.0, max(0.0, as_float(extra.get("epss_percentile"))))
    s += 15.0 * pct  # 0..15, weighted to the exploit-likely tail
    s += {"critical": 8, "high": 5}.get((severity or "").lower(), 0)
    return s


def _structure_signals(body: str) -> float:
    s = 0.0
    n = len(body)
    if n >= 4000:
        s += 14
    elif n >= 1500:
        s += 8
    elif n < 400:
        s -= 10
    fences = len(_FENCE_RE.findall(body)) // 2
    s += min(12.0, 4.0 * fences)
    if _STEP_RE.search(body):
        s += 8
    if len(_HEADING_RE.findall(body)) >= 3:
        s += 5
    if _HTTP_RE.search(body):
        s += 6
    return s


def score_note(meta: Frontmatter, body: str) -> int:
    """Return a 0-100 retrieval-worth score for a note."""
    body = body or ""
    score = float(_BASE.get(meta.type, 40))
    extra = meta.extra or {}

    if meta.type == "report":
        score += _report_signals(extra, meta.severity, meta.tags, meta.bounty)
        score += _structure_signals(body)
    elif meta.type == "cve":
        score += _cve_signals(extra, meta.tags, meta.severity)
        # CVE bodies are short by nature; only reward genuinely fleshed-out ones.
        if len(body) >= 1200:
            score += 5
    else:
        score += _structure_signals(body) * 0.5  # manual notes: light touch on top of high base

    if (
        is_user_authored(meta)
        and _state(extra) not in _REJECTED_STATES
        and not _is_dupe(extra, meta.tags)
    ):
        score = max(score, USER_FLOOR)

    return _clamp(score)
