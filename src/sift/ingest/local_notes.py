"""Index hand-authored markdown and backfill missing frontmatter.

Drop a `.md` file anywhere under the vault (e.g. `vault/inbox/idea.md`). If it
has no frontmatter, we infer `type` from its parent folder (default `finding`),
`title` from the first heading or filename, and a stable `id`, then rewrite it
as a proper note before indexing.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import frontmatter
from slugify import slugify

from sift.config import get_settings
from sift.index.store import Store
from sift.pipeline import index_note
from sift.vault.notes import Note, save_note
from sift.vault.schema import NOTE_TYPES, Frontmatter

_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _infer_type(path: Path, vault: Path) -> str:
    for part in path.relative_to(vault).parts[:-1]:
        if part in NOTE_TYPES:
            return part
    return "finding"


def _needs_backfill(meta: dict) -> bool:
    return not all(k in meta and meta[k] for k in ("id", "type", "title"))


def backfill_and_index() -> tuple[int, int]:
    s = get_settings()
    vault = s.resolved_vault()
    store = Store()
    fixed = 0
    indexed = 0

    for path in sorted(vault.rglob("*.md")):
        if path.name == "README.md" or path.name.startswith("_"):
            continue
        raw = path.read_text(encoding="utf-8")
        post = frontmatter.loads(raw)

        if _needs_backfill(post.metadata):
            body = post.content.strip() or raw.strip()
            m = _H1.search(body)
            title = post.metadata.get("title") or (
                m.group(1).strip() if m else path.stem.replace("-", " ").title()
            )
            ntype = post.metadata.get("type") or _infer_type(path, vault)
            nid = post.metadata.get("id") or f"local-{slugify(path.stem, max_length=60)}"
            meta = Frontmatter(
                id=nid,
                type=ntype if ntype in NOTE_TYPES else "finding",
                title=title,
                source=post.metadata.get("source", "manual"),
                tags=list(post.metadata.get("tags", []) or []),
                ingested=datetime.now(UTC),
            )
            note = Note(meta=meta, body=body)
            # Write to the canonical location; remove the old loose file if it moved.
            new_path = save_note(vault, note)
            if new_path.resolve() != path.resolve():
                path.unlink(missing_ok=True)
            fixed += 1
        else:
            note = Note(
                meta=Frontmatter.model_validate(post.metadata),
                body=post.content,
                path=path,
            )

        indexed += 1 if index_note(note, store) else 0

    store.ensure_fts()
    return fixed, indexed
