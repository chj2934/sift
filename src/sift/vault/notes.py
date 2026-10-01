"""Read / write / enumerate vault notes."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import frontmatter
import yaml
from slugify import slugify

from sift.vault.schema import Frontmatter

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]")


@dataclass
class Note:
    meta: Frontmatter
    body: str
    path: Path | None = None  # set once written / loaded

    # ---- derived ----
    @property
    def slug(self) -> str:
        return slugify(self.meta.id, max_length=80) or slugify(self.meta.title, max_length=80)

    @property
    def wikilinks(self) -> list[str]:
        """[[links]] found in the body, as slugs."""
        return [slugify(m.group(1).strip()) for m in WIKILINK_RE.finditer(self.body)]

    def all_links(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in [*self.meta.links, *self.wikilinks]:
            seen.setdefault(slugify(s), None)
        return list(seen)

    def render(self) -> str:
        fm = yaml.safe_dump(
            self.meta.to_yaml_dict(),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).strip()
        body = self.body.strip()
        return f"---\n{fm}\n---\n\n{body}\n"


# Characters Windows forbids in a filename, plus control chars. Everything else -
# spaces, case, accents, CJK - is kept, because the filename IS the note's name in
# Obsidian and `[[Cookie sandwich - reading HttpOnly cookies]]` should resolve there
# exactly as it reads.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WS = re.compile(r"\s+")
# Windows refuses these as filenames whatever the extension.
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
# Vault path + type dir eat ~60 chars of the 260-char Windows limit; leave room.
FILENAME_MAX = 150


def title_to_filename(title: str, fallback: str = "note") -> str:
    """A human-readable, filesystem-safe filename stem for a note title."""
    stem = _ILLEGAL.sub("-", title or "")
    stem = _WS.sub(" ", stem).strip().strip(".")
    if len(stem) > FILENAME_MAX:
        stem = stem[:FILENAME_MAX].rsplit(" ", 1)[0].strip() or stem[:FILENAME_MAX]
    if stem.upper() in _RESERVED:
        stem = f"{stem}-note"
    return stem or slugify(fallback, max_length=80) or "note"


def note_path(vault: Path, meta: Frontmatter) -> Path:
    """Where a note lives. The filename is its title, so the vault opens cleanly in
    Obsidian; `meta.id` remains the stable identity used by the index."""
    return vault / meta.type / f"{title_to_filename(meta.title, meta.id)}.md"


def _free_path(dest: Path, note_id: str) -> Path:
    """Resolve a filename clash.

    Since the filename is the title, two notes with the same title want the same
    file - and silently overwriting one was a real data-loss bug. Reuse the path when
    it already holds *this* note (a re-ingest), otherwise take the next free
    `Title (2).md`, which is Obsidian's own convention.
    """
    if not dest.exists():
        return dest
    try:
        existing = frontmatter.loads(dest.read_text(encoding="utf-8")).metadata
        if existing.get("id") == note_id:
            return dest  # same note, being rewritten
    except Exception:  # noqa: BLE001 - unreadable file, treat as occupied
        pass
    for n in range(2, 1000):
        candidate = dest.with_name(f"{dest.stem} ({n}){dest.suffix}")
        if not candidate.exists():
            return candidate
        try:
            existing = frontmatter.loads(candidate.read_text(encoding="utf-8")).metadata
            if existing.get("id") == note_id:
                return candidate
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"could not find a free filename for {dest}")


def save_note(vault: Path, note: Note, *, stamp: bool = True) -> Path:
    if stamp and note.meta.ingested is None:
        note.meta.ingested = datetime.now(timezone.utc)
    dest = note_path(vault, note.meta)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = _free_path(dest, note.meta.id)
    dest.write_text(note.render(), encoding="utf-8")
    note.path = dest
    return dest


def load_note(path: Path) -> Note:
    post = frontmatter.loads(path.read_text(encoding="utf-8"))
    meta = Frontmatter.model_validate(post.metadata)
    return Note(meta=meta, body=post.content, path=path)


def iter_notes(vault: Path, *, note_type: str | None = None) -> Iterator[Note]:
    root = vault / note_type if note_type else vault
    if not root.exists():
        return
    for path in sorted(root.rglob("*.md")):
        if path.name.startswith("_") or path.name == "README.md":
            continue
        try:
            yield load_note(path)
        except Exception as exc:  # noqa: BLE001 - surface but keep going
            print(f"  ! skipping unreadable note {path}: {exc}")
