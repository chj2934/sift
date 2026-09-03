"""Decide which bulk-ingested notes still earn their place in the vault.

Rationale (see project notes): the reasoning model already trained on the public
disclosed-report and NVD corpus, so retrieving it back adds little. What stays is
what the model *can't reliably reproduce* (detailed, bountied writeups) or *can't
know* (recent disclosures, actively-exploited CVEs) — plus everything the user
authored or curated.
"""

from __future__ import annotations

from dataclasses import dataclass

from sift.quality import score_note
from sift.vault.notes import Note

# CVE frontmatter tags that mark real-world exploitation.
_KEV_TAGS = frozenset({"kev", "known-exploited"})
# EPSS percentile at/above which we keep an otherwise-old CVE.
_EPSS_KEEP = 0.88
# A recently-catalogued CVE is only worth keeping if it also carries an
# exploitability signal — otherwise the reasoning model already covers it.
_EPSS_RECENT = 0.60
_RECENT_SEVERITIES = frozenset({"high", "critical"})
# Always keep — hand-authored or freshness-sourced.
_KEEP_TYPES = frozenset({"technique", "target", "finding", "writeup"})


@dataclass
class Verdict:
    keep: bool
    reason: str  # short bucket, for the summary table


def _year(note: Note) -> int | None:
    d = note.meta.created
    return d.year if d else None


def classify(note: Note, *, keep_since_year: int, report_quality_bar: int) -> Verdict:
    t = note.meta.type
    if t in _KEEP_TYPES:
        return Verdict(True, "authored/curated")

    yr = _year(note)
    extra = note.meta.extra or {}

    if t == "report":
        if yr is not None and yr >= keep_since_year:
            return Verdict(True, f"report {keep_since_year}+")
        if extra.get("has_bounty") and score_note(note.meta, note.body) >= report_quality_bar:
            return Verdict(True, "old but bountied+substantial")
        return Verdict(False, "old thin/dupe report")

    if t == "cve":
        if _KEV_TAGS.intersection(note.meta.tags):
            return Verdict(True, "KEV / known-exploited")
        pct = extra.get("epss_percentile")
        pct = float(pct) if isinstance(pct, (int, float)) else 0.0
        if pct >= _EPSS_KEEP:
            return Verdict(True, f"EPSS pct >= {_EPSS_KEEP}")
        if yr is not None and yr >= keep_since_year:
            if (note.meta.severity or "") in _RECENT_SEVERITIES or pct >= _EPSS_RECENT:
                return Verdict(True, f"CVE {keep_since_year}+ w/ severity/EPSS")
            return Verdict(False, f"CVE {keep_since_year}+ but low-signal")
        return Verdict(False, "old low-signal CVE")

    return Verdict(False, f"unclassified {t}")
