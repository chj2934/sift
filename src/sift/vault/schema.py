"""Frontmatter schema for vault notes."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

NoteType = Literal[
    "report", "cve", "technique", "target", "finding", "writeup", "tool", "reference"
]

NOTE_TYPES: tuple[str, ...] = (
    "report",
    "cve",
    "technique",
    "target",
    "finding",
    "writeup",
    # Per-program tooling: what you installed, where you cloned it, which flags and
    # accounts you used. Cheap to write, and the thing you always wish you had when
    # coming back to a program months later.
    "tool",
    # Primary source material, quoted rather than distilled: the vendor's own security
    # documentation, severity guidelines, release notes and fix commits. Separate from
    # `technique` precisely because nothing here is summarised - the value is the
    # vendor's exact wording, pinned to a revision and citable back at triage.
    "reference",
)


class Frontmatter(BaseModel):
    """YAML frontmatter block on every note.

    Only ``id``, ``type`` and ``title`` are required. Everything else is optional
    and source-dependent.
    """

    id: str
    type: NoteType
    title: str

    source: str | None = None  # e.g. "hackerone-public", "nvd", "manual"
    url: str | None = None
    created: date | None = None  # when the underlying thing was created/disclosed
    ingested: datetime | None = None  # when sift wrote the note

    tags: list[str] = Field(default_factory=list)
    cwe: list[str] = Field(default_factory=list)  # e.g. ["CWE-79"]
    severity: str | None = None  # critical / high / medium / low / none / info
    program: str | None = None  # bug bounty program / vendor
    assets: list[str] = Field(default_factory=list)  # domains / components in scope
    bounty: float | None = None
    links: list[str] = Field(default_factory=list)  # slugs of related notes

    # Free-form extras kept verbatim (e.g. epss score, cvss vector).
    extra: dict = Field(default_factory=dict)

    @field_validator("tags", "cwe", "assets", "links", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> list:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)  # type: ignore[arg-type]

    @field_validator("cwe", mode="after")
    @classmethod
    def _normalize_cwe(cls, v: list[str]) -> list[str]:
        out = []
        for item in v:
            s = str(item).strip().upper()
            if s and not s.startswith("CWE-") and s.isdigit():
                s = f"CWE-{s}"
            if s:
                out.append(s)
        return out

    @field_validator("severity", mode="after")
    @classmethod
    def _normalize_severity(cls, v: str | None) -> str | None:
        if not v:
            return None
        return str(v).strip().lower()

    def to_yaml_dict(self) -> dict:
        """Ordered, YAML-friendly dict — omits empty optionals."""
        d: dict = {"id": self.id, "type": self.type, "title": self.title}
        for key in (
            "source",
            "url",
            "program",
            "severity",
            "bounty",
        ):
            val = getattr(self, key)
            if val not in (None, "", []):
                d[key] = val
        if self.created:
            d["created"] = self.created.isoformat()
        if self.ingested:
            d["ingested"] = self.ingested.isoformat(timespec="seconds")
        for key in ("cwe", "tags", "assets", "links"):
            val = getattr(self, key)
            if val:
                d[key] = val
        if self.extra:
            d["extra"] = self.extra
        return d
