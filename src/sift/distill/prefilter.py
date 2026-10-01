"""Free heuristic drops, applied before any model sees a candidate.

Every drop in the first hand-graded batch of 40 was structurally obvious: tool
announcements, "Top 10" index pages, and cheat-sheet update posts. Those cost real
money to send through the gate and always come back the same way, so catch them
here instead.

The bar is deliberately high: a false drop is invisible (the material never reaches
the gate and never enters the vault), while a false *keep* only costs one gate call.
So every rule must be one that cannot plausibly fire on genuine technique research.
`tests/test_prefilter.py` asserts the rules against the hand-graded set.
"""

from __future__ import annotations

import re

# Announcement posts for a tool or product. These describe software, not a technique.
_TOOL_ANNOUNCEMENT = re.compile(
    r"^\s*(introducing|announcing|meet)\s+\S",
    re.IGNORECASE,
)

# The annual index/nomination posts. Valuable as reading lists (see top10.py, which
# mines them for links) but never a technique in themselves.
_INDEX_POST = re.compile(
    r"top\s+10\s+web\s+hacking\s+techniques|nominations?\s+open|call\s+for\s+nominations",
    re.IGNORECASE,
)

# Conference schedule / preview posts - the techniques land in separate papers.
_CONFERENCE_TRAIL = re.compile(
    r"\b(previewing|coming to)\b.*\b(def\s?con|black\s?hat)\b|\bhat-trick\b",
    re.IGNORECASE,
)

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tool-announcement", _TOOL_ANNOUNCEMENT),
    ("index-post", _INDEX_POST),
    ("conference-preview", _CONFERENCE_TRAIL),
)


def prefilter_reason(title: str, url: str = "") -> str | None:
    """Return a drop reason, or None to pass the candidate through to the gate."""
    text = title or ""
    for name, pattern in _RULES:
        if pattern.search(text):
            return name
    return None


def should_gate(title: str, url: str = "") -> bool:
    return prefilter_reason(title, url) is None
