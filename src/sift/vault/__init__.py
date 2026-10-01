"""Vault layer: markdown notes with YAML frontmatter and [[wikilinks]]."""

from sift.vault.notes import Note, load_note, note_path, save_note, iter_notes
from sift.vault.schema import NOTE_TYPES, Frontmatter

__all__ = [
    "Note",
    "Frontmatter",
    "NOTE_TYPES",
    "load_note",
    "save_note",
    "note_path",
    "iter_notes",
]
