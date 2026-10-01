"""Heuristic note-quality score (0-100).

A cheap, deterministic signal for *how much a note is worth retrieving* — used
to re-weight search results and as the filter for ``sift prune``. No network,
no LLM: everything comes from frontmatter the ingesters already set plus a few
structural checks on the body.

The score is intentionally blunt. It only needs to separate "detailed writeup
with a working PoC and a bounty" from "four-sentence dupe with no bounty".
"""

from __future__ import annotations

import math
import re

from sift.vault.schema import Frontmatter

# Base score by note type. Hand-authored notes are trusted curation and start high;
# bulk-ingested reports/CVEs have to earn their place from the signals below.
_BASE = {
    "report": 40,
    "cve": 42,
    "technique": 78,
    "target": 78,
    "finding": 78,
    "writeup": 70,
    # Primary vendor source. High, but under the authored notes: a severity guideline
    # settles an argument, it does not tell you where to look.
    "reference": 74,
}

_STEP_RE = re.compile(r"steps?\s+to\s+reproduce|^\s*\d+\.\s+\S", re.IGNORECASE | re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_HTTP_RE = re.compile(
    r"HTTP/\d|^(?:GET|POST|PUT|PATCH|DELETE)\s+/|\bcurl\s+-|Authorization:\s|\bBurp\b",
    re.IGNORECASE | re.MULTILINE,
)


def _clamp(n: float, lo: float = 0.0, hi: float = 100.0) -> int:
    return int(max(lo, min(hi, round(n))))


def _report_signals(extra: dict, severity: str | None) -> float:
    s = 0.0
    if extra.get("has_bounty"):
        s += 15
    vc = extra.get("vote_count") or 0
    if vc > 0:
        s += min(20.0, 4.0 * math.log2(1 + vc))
    if extra.get("is_dupe"):
        s -= 35
    s += {"critical": 12, "high": 8, "medium": 3}.get((severity or "").lower(), 0)
    return s


def _cve_signals(extra: dict, tags: list[str], severity: str | None) -> float:
    s = 0.0
    if any(t in ("kev", "known-exploited") for t in tags):
        s += 22
    pct = extra.get("epss_percentile")
    if isinstance(pct, (int, float)):
        s += 15.0 * float(pct)  # 0..15, weighted to the exploit-likely tail
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
        score += _report_signals(extra, meta.severity)
        score += _structure_signals(body)
    elif meta.type == "cve":
        score += _cve_signals(extra, meta.tags, meta.severity)
        # CVE bodies are short by nature; only reward genuinely fleshed-out ones.
        if len(body) >= 1200:
            score += 5
    else:
        score += _structure_signals(body) * 0.5  # manual notes: light touch on top of high base

    return _clamp(score)
