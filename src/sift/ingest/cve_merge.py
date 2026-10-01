"""One note per CVE, whichever feeds describe it.

CISA KEV and NVD both key their notes on the CVE number. Before upsert-by-id, ``ingest
kev`` after ``ingest nvd`` left two files with one id (63 such pairs in the real vault)
and search flipped between them on every reindex; with upsert-by-id and no merge, the
second feed is refused (`IdConflict`) and its data never lands. A plain overwrite would
be worse: KEV would erase NVD's description and CVSS, NVD would erase the ``kev`` tag
that both prune and scoring key on, and either would erase what ``ingest epss`` wrote.

`merge_cve` combines them, and is also applied when a feed re-ingests its own record,
so an NVD description edit never wipes EPSS scores or a section the user added.

Rules (pure function, idempotent: merging the same record twice changes nothing):

* **Body.** Each feed owns its sections - KEV: the bold vendor/product lead line,
  ``## Summary``, ``## Required action``, ``## Ransomware use``, ``## Notes``; NVD:
  ``## Description``, ``## CVSS``, ``## References``. A section the incoming record
  provides replaces the stored one in place; a section it does not provide is kept,
  never deleted; a new section is appended. Sections nobody owns (the user's) are
  kept where they are.
* **Lists** (tags, cwe): order-preserving union.
* **extra**: updated with the incoming values, ignoring None (a missing CVSS never
  erases a stored one; ``epss`` survives because no feed sets it).
* **Scalars** (severity, program, url): the incoming value when it has one.
* **created**: NVD's published date when NVD contributes, else the earliest date.
* **title** and **source**: the stored ones - the filename and every ``[[link]]`` stay
  stable, and ``source`` is the file's identity. Every contributing feed is listed in
  ``extra.sources`` once there is more than one.

A stored note that is not a feed-written CVE note (a user's finding under a CVE id,
say) is never merged into: `merge_cve` raises `IdConflict`.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from sift.ingest.existing import same_file, union
from sift.vault.notes import IdConflict, Note

KEV_SOURCE = "cisa-kev"
NVD_SOURCE = "nvd"
CVE_SOURCES = frozenset({KEV_SOURCE, NVD_SOURCE})

# Section headings (casefolded) each feed writes.
SECTION_OWNERS: dict[str, frozenset[str]] = {
    KEV_SOURCE: frozenset({"summary", "required action", "ransomware use", "notes"}),
    NVD_SOURCE: frozenset({"description", "cvss", "references"}),
}

# KEV's lead line: "**Vendor Product** — Vulnerability name".
_KEV_LEAD = re.compile(r"^\*\*.*\*\*(\s+—\s+.*)?$")
_FENCE = ("```", "~~~")


def _heading_key(text: str) -> str:
    return " ".join(text.strip().rstrip(":").split()).casefold()


def split_sections(body: str) -> tuple[str, list[tuple[str, str]]]:
    """(preamble, [(key, block), ...]) for a note body.

    A block runs from its ``## `` heading line to the next one, trailing whitespace
    removed, so ``"\\n\\n".join([preamble, *blocks])`` reproduces a feed's own body.
    ``## `` lines inside fenced code are not headings. Pure.
    """
    preamble: list[str] = []
    sections: list[tuple[str, str]] = []
    key: str | None = None
    block: list[str] = []
    fence: str | None = None
    for line in body.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(_FENCE):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
        elif fence is None and line.startswith("## "):
            if key is not None:
                sections.append((key, "\n".join(block).rstrip()))
            key, block = _heading_key(line[3:]), [line]
            continue
        (block if key is not None else preamble).append(line)
    if key is not None:
        sections.append((key, "\n".join(block).rstrip()))
    return "\n".join(preamble).strip(), sections


def _join(preamble: str, blocks: list[str]) -> str:
    return "\n\n".join(p for p in [preamble, *blocks] if p.strip())


def merge_body(stored: str, incoming: str, incoming_source: str | None) -> str:
    """`stored` with the sections `incoming` provides put in place. Pure, idempotent."""
    s_pre, s_secs = split_sections(stored)
    i_pre, i_secs = split_sections(incoming)

    provided: dict[str, str] = {}
    for key, block in i_secs:
        provided.setdefault(key, block)

    blocks: list[str] = []
    placed: set[str] = set()
    for key, block in s_secs:
        if key in provided and key not in placed:
            blocks.append(provided[key])
            placed.add(key)
        else:
            # Unowned, or a second section under the same heading (the user's own
            # "## Notes", say): kept as it is. Only the first is the feed's.
            blocks.append(block)
    for key, block in i_secs:
        if key not in placed:
            blocks.append(block)
            placed.add(key)

    if incoming_source == KEV_SOURCE and i_pre:
        # Replace KEV's own lead line(s); keep anything else written up there.
        rest = "\n".join(ln for ln in s_pre.split("\n") if not _KEV_LEAD.match(ln.strip())).strip()
        preamble = _join(i_pre, [rest])
    else:
        preamble = s_pre or i_pre
    return _join(preamble, blocks)


def _sources(stored_extra: dict, stored_source: str | None) -> list[str]:
    listed = stored_extra.get("sources")
    if isinstance(listed, (list, tuple)):
        return union([str(s) for s in listed], [stored_source or ""])
    return union([stored_source or ""])


def _created(
    stored: date | None,
    incoming: date | None,
    incoming_source: str | None,
    stored_sources: list[str],
) -> date | None:
    if incoming_source == NVD_SOURCE and incoming:
        return incoming  # the publication date; prune keys its horizon on it
    if NVD_SOURCE in stored_sources and stored:
        return stored
    dates = [d for d in (stored, incoming) if d]
    return min(dates) if dates else None


def merge_cve(stored: Note, incoming: Note) -> Note:
    """`incoming` (a KEV or NVD record) merged into `stored` (the note carrying its id).

    Usable as `run_source(merge=...)` and by the sources themselves. Raises
    `IdConflict` when `stored` is not a feed-written CVE note.
    """
    if incoming.path is not None and same_file(incoming.path, stored.path):
        return incoming  # already merged against this very file (source-side merge)
    s, i = stored.meta, incoming.meta
    where = stored.path or Path(f"{s.id}.md")
    if s.id != i.id:
        raise IdConflict(i.id, where, "id", existing_source=s.source, existing_url=s.url)
    if s.type != "cve" or (s.source or "") not in CVE_SOURCES:
        raise IdConflict(i.id, where, "source", existing_source=s.source, existing_url=s.url)

    stored_extra = dict(s.extra) if isinstance(s.extra, dict) else {}
    stored_sources = _sources(stored_extra, s.source)
    sources = union(stored_sources, [i.source or ""])

    extra = dict(stored_extra)
    extra.update({k: v for k, v in (i.extra or {}).items() if v is not None})
    if len(sources) > 1:
        extra["sources"] = sources

    meta = s.model_copy(deep=True)
    meta.tags = union(s.tags, i.tags)
    meta.cwe = union(s.cwe, i.cwe)
    meta.extra = extra
    for field in ("severity", "program", "url"):
        value = getattr(i, field)
        if value:
            setattr(meta, field, value)
    meta.created = _created(s.created, i.created, i.source, stored_sources)
    meta.title = s.title or i.title
    meta.ingested = None  # stamped by the writer if this turns out to be a change

    body = merge_body(stored.body, incoming.body, i.source)
    return Note(meta=meta, body=body, path=stored.path)
