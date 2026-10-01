"""One input shape for the gate, whatever the source was.

Every ingest source (RSS research, writeup aggregators, H1 disclosures, structured
repos) normalizes to `Candidate` so `gate.judge` has a single type to reason about.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from sift.vault.notes import Note

# How much article text the gate sees. The gate only needs enough to recognize the
# technique, not the whole article — the full text is kept on the note itself.
GATE_TEXT_CHARS = 8000


def url_key(url: object) -> str:
    """Identity key for "is this the same article?": gating, dedup and verdict matching.

    Compare with it, never store it - reject rows and technique notes keep the raw url,
    and both sides of every comparison go through this function, so no stored data
    needs migrating when the rule changes.

    The rule is the one `collect()` has always deduplicated on - drop the query string
    and any trailing slash - plus the fragment and case-folding of scheme and host.
    Gating once compared raw urls against these keys, so a judged article whose url
    ended in '/' (most WordPress/Hugo blogs) or carried '?source=rss' was exported
    again in every batch.
    """
    if not isinstance(url, str):
        return ""
    key = url.strip().split("#", 1)[0].split("?", 1)[0].rstrip("/")
    scheme, sep, rest = key.partition("://")
    if not sep:
        return key
    host, slash, path = rest.partition("/")
    return f"{scheme.lower()}://{host.lower()}{slash}{path}"


@dataclass
class Candidate:
    """A piece of source material awaiting a keep/drop verdict."""

    title: str
    url: str
    text: str
    source: str
    created: date | None = None
    # Where the material lives in the vault. Lets an export point the judge at the full
    # article when the gate excerpt is truncated, and lets a technique note link back
    # to its source. Defaulted so existing positional constructors keep working.
    source_id: str = ""
    source_slug: str = ""
    path: str = ""

    def gate_text(self) -> str:
        """The excerpt shown to the gate."""
        return self.text[:GATE_TEXT_CHARS]

    @classmethod
    def from_note(cls, note: Note) -> Candidate:
        """Build a candidate from an already-ingested note (e.g. the writeup backlog)."""
        return cls(
            title=note.meta.title,
            url=note.meta.url or "",
            text=note.body,
            source=note.meta.source or "unknown",
            created=note.meta.created,
            source_id=note.meta.id,
            source_slug=note.slug,
            path=str(note.path) if note.path else "",
        )


def best_copies(
    notes: Iterable[Note],
    *,
    seen: set[str] | frozenset[str] = frozenset(),
    prefilter: bool = False,
) -> tuple[dict[str, Candidate], list[Candidate]]:
    """The copy of each article the gate should judge: the longest body per `url_key`.

    The same article legitimately arrives from several sources - the research feed, a
    Top-10 nomination and PentesterLand all carry Doyensec's CSPT2CSRF post (measured:
    68 duplicate copies across 63 URLs). Judging each copy costs money and yields
    duplicate technique notes, and the fullest copy is the best evidence.

    `seen` holds keys to leave out (already judged). With `prefilter`, structurally
    obvious drops are removed per copy *before* lengths are compared, which is what
    production gating does; `distill eval` scores the gate alone, so it passes neither.

    Returns `(best, no_url)`: `best` maps key -> candidate in first-seen order (the
    first copy wins a tie on length); `no_url` holds candidates without a url, which no
    verdict can ever be matched to.
    """
    from sift.distill.prefilter import prefilter_reason

    best: dict[str, Candidate] = {}
    no_url: list[Candidate] = []
    for note in notes:
        key = url_key(note.meta.url)
        if key and key in seen:
            continue
        if prefilter and prefilter_reason(note.meta.title, note.meta.url or ""):
            continue
        cand = Candidate.from_note(note)
        if not key:
            no_url.append(cand)
        elif key not in best or len(cand.text) > len(best[key].text):
            best[key] = cand
    return best, no_url
