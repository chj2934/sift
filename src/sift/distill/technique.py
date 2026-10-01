"""A passing candidate becomes a `technique` note.

The field that makes these worth retrieving is `when_to_try` — the trigger. Search
should hand back a procedure you can act on, not an 8,000-word article you re-read
every time.
"""

from __future__ import annotations

import hashlib

from slugify import slugify

from sift.distill.candidates import Candidate
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter


def technique_id(title: str, disambiguator: str = "") -> str:
    """The note id for a technique title. Pure, so the collision rule is testable.

    The plain form is `tech-<slug of the title, cut at 70>`. It depends only on the
    title the judge chose, so two articles distilled under the same title - or two
    titles sharing their first 70 slug characters - would claim one id, and saving the
    second silently replaced the first note and its index rows.

    With a `disambiguator` (the article's `url_key`) the form is
    `tech-<slug cut at 61>-<sha1[:8]>`: at most 75 characters, so `Note.slug`'s
    80-character cut keeps the suffix, and deterministic, so re-applying the same
    verdict lands on the same note instead of minting another.
    """
    if not disambiguator:
        return f"tech-{slugify(title, max_length=70) or 'technique'}"
    digest = hashlib.sha1(disambiguator.encode("utf-8")).hexdigest()[:8]
    return f"tech-{slugify(title, max_length=61) or 'technique'}-{digest}"


def build_note(
    candidate: Candidate,
    *,
    title: str,
    when_to_try: str,
    body_md: str,
    keep_reason: str,
    already_known: str = "",
    tags: list[str] | str | None = None,
    cwe: list[str] | str | None = None,
    gated_by: str = "in-session",
    note_id: str | None = None,
) -> Note:
    """Assemble the technique note. Pure — no disk or network, so it's testable.

    `note_id` overrides the title-derived id; `apply_verdicts` passes it when the plain
    id is already taken by a note about a different article.
    """
    # A bare string is one tag, not an iterable of characters: `"xss"` once became the
    # tags 's' and 'x'.
    tags = [tags] if isinstance(tags, str) else list(tags or [])
    cwe = [cwe] if isinstance(cwe, str) else list(cwe or [])
    meta = Frontmatter(
        id=note_id or technique_id(title),
        type="technique",
        title=title,
        source=candidate.source,
        url=candidate.url or None,
        created=candidate.created,
        tags=sorted({"technique", keep_reason, *(t.strip() for t in tags if t and t.strip())}),
        cwe=cwe,
        # The distilled note is written from the gate excerpt; linking the source note
        # lets `search_memory(expand_links=True)` and `get_note` reach the full article.
        # The canonical slug, not a `[[title]]` wikilink: titles alias at 80 characters
        # in the link graph, so a long title would not resolve.
        links=[candidate.source_slug] if candidate.source_slug else [],
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
