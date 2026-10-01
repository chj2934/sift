"""Frontmatter schema for vault notes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

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

# A note without these is not a note: the loader refuses it rather than guessing.
REQUIRED_KEYS: frozenset[str] = frozenset({"id", "type", "title"})

_LIST_CONTAINERS = (list, tuple, set, frozenset)


class Frontmatter(BaseModel):
    """YAML frontmatter block on every note.

    Only ``id``, ``type`` and ``title`` are required. Everything else is optional
    and source-dependent.

    The vault is edited by hand in Obsidian, so validation is forgiving wherever that
    needs no guessing: numbers become strings (``title: 404``), a scalar becomes a
    one-item list (``cwe: 79``), ``type: Technique`` is lowercased and a date-time
    ``created`` keeps its date. One bad hand-typed property used to drop the whole note
    from search, listings and the next reindex.

    Keys sift does not model (Obsidian's ``aliases``, ``cssclasses``, a user's own
    ``status``) are kept and written back at the top level, where Obsidian reads them;
    re-saving a note used to delete them.
    """

    model_config = ConfigDict(extra="allow", coerce_numbers_to_str=True)

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

    # Values read from a file that failed validation for an *optional* field (or sat
    # under a non-string YAML key). The loader keeps them here instead of dropping the
    # note, and `to_yaml_dict` writes them back verbatim, so a typo in one property
    # never costs the user its value on the next save. See `vault.notes.parse_note_text`.
    _raw_invalid: dict = PrivateAttr(default_factory=dict)

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("tags", "cwe", "assets", "links", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> list[str]:
        if v is None:
            return []
        # A bare scalar is one item: `"xss"` once became the tags 's' and 'x', and
        # `cwe: 79` could not be iterated at all.
        items = list(v) if isinstance(v, _LIST_CONTAINERS) else [v]
        out: list[str] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, (Mapping, *_LIST_CONTAINERS)):
                # Not something to stringify silently; the loader keeps it verbatim.
                raise ValueError("list items must be plain values")
            s = str(item).strip()
            if s:
                out.append(s)
        return out

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

    @field_validator("created", mode="before")
    @classmethod
    def _coerce_created(cls, v: object) -> object:
        # Obsidian's date-time property type writes `2024-05-01T10:00`; YAML hands that
        # back as a datetime, which a `date` field rejects outright.
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return datetime.fromisoformat(s).date()
            except ValueError:
                return v  # not a date: left for the loader to keep verbatim
        return v

    @field_validator("ingested", mode="before")
    @classmethod
    def _coerce_ingested(cls, v: object) -> object:
        if isinstance(v, datetime):
            return v
        if isinstance(v, date):
            return datetime(v.year, v.month, v.day, tzinfo=UTC)
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("extra", mode="before")
    @classmethod
    def _coerce_extra(cls, v: object) -> object:
        return {} if v is None else v

    @field_validator("bounty", mode="before")
    @classmethod
    def _coerce_bounty(cls, v: object) -> object:
        if isinstance(v, str):
            s = v.replace(",", "").strip().strip("$").strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return v  # "lots": left for the loader to keep verbatim
        return v

    # ---- invalid values carried through a load/save round trip ----
    @property
    def invalid_fields(self) -> dict[Any, Any]:
        """Raw values the loader could not validate and kept verbatim (read-only copy)."""
        return dict(self._raw_invalid)

    def keep_invalid(self, raw: Mapping[Any, Any]) -> None:
        """Remember raw frontmatter values that failed validation, to write them back."""
        self._raw_invalid.update(raw)

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
        # Keys sift does not model stay at the top level, where Obsidian reads them.
        for key, val in (self.model_extra or {}).items():
            d.setdefault(key, val)
        # An invalid value goes back exactly as the user wrote it - unless the field
        # has since been given a valid value, which then wins.
        for key, val in self._raw_invalid.items():
            if key not in d:
                d[key] = val
        return d
