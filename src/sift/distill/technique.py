"""A passing candidate becomes a `technique` note.

The field that makes these worth retrieving is `when_to_try` — the trigger. Search
should hand back a procedure you can act on, not an 8,000-word article you re-read
every time.
"""

from __future__ import annotations

from slugify import slugify

from sift.distill.candidates import Candidate
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter


def build_note(
    candidate: Candidate,
    *,
    title: str,
    when_to_try: str,
    body_md: str,
    keep_reason: str,
    already_known: str = "",
    tags: list[str] | None = None,
    cwe: list[str] | None = None,
    gated_by: str = "in-session",
) -> Note:
    """Assemble the technique note. Pure — no disk or network, so it's testable."""
    base = slugify(title, max_length=70) or "technique"
    meta = Frontmatter(
        id=f"tech-{base}",
        type="technique",
        title=title,
        source=candidate.source,
        url=candidate.url or None,
        created=candidate.created,
        tags=sorted({"technique", keep_reason, *(tags or [])}),
        cwe=cwe or [],
        extra={
            "keep_reason": keep_reason,
            "when_to_try": when_to_try,
            # Kept so a later threshold review can see what the gate believed at the
            # time it decided — the reject log alone only covers the drops.
            "already_known": already_known,
            "gated_by": gated_by,
        },
    )
    parts = [f"**When to try:** {when_to_try}", "", body_md.strip()]
    if candidate.url:
        parts += ["", "---", f"Source: {candidate.url}"]
    return Note(meta=meta, body="\n".join(parts).strip())
