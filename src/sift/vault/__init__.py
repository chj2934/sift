"""Vault layer: markdown notes with YAML frontmatter and [[wikilinks]]."""

from sift.vault.notes import (
    IdConflict,
    Note,
    SaveResult,
    canonical_url,
    delete_note,
    iter_notes,
    legacy_slug,
    load_note,
    locate_note,
    note_path,
    note_slug,
    same_document,
    save_note,
    walk_note_files,
    write_lock,
    write_note,
)
from sift.vault.schema import NOTE_TYPES, Frontmatter

__all__ = [
    "Note",
    "Frontmatter",
    "NOTE_TYPES",
    "IdConflict",
    "SaveResult",
    "canonical_url",
    "delete_note",
    "iter_notes",
    "legacy_slug",
    "load_note",
    "locate_note",
    "note_path",
    "note_slug",
    "same_document",
    "save_note",
    "walk_note_files",
    "write_lock",
    "write_note",
]
