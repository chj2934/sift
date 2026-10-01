"""One input shape for the gate, whatever the source was.

Every ingest source (RSS research, writeup aggregators, H1 disclosures, structured
repos) normalizes to `Candidate` so `gate.judge` has a single type to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sift.vault.notes import Note

# How much article text the gate sees. The gate only needs enough to recognize the
# technique, not the whole article — the full text is kept on the note itself.
GATE_TEXT_CHARS = 8000


@dataclass
class Candidate:
    """A piece of source material awaiting a keep/drop verdict."""

    title: str
    url: str
    text: str
    source: str
    created: date | None = None

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
        )
