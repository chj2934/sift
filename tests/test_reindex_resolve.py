"""Resolving a note reference (MCP get_note / resolve_idea) through the vault catalog.

get_note_by_slug used to try `vault/<type>/<slug>.md`, then `rglob(f"{slug}.md")`, then
parse the whole vault (2-4 s per call) and return the first note whose slug OR id
matched - so a note whose filename slugified to another note's id won, and ids that
shared an 80-character prefix could never be told apart.
"""

from __future__ import annotations

import os
import time


def _save(vault, nid: str, title: str, body: str = "body", ntype: str = "technique"):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    return save_note(
        vault, Note(meta=Frontmatter(id=nid, type=ntype, title=title), body=body), stamp=False
    )


def test_an_exact_id_wins_over_a_filename_that_slugifies_to_it(vault_path):
    from sift.pipeline import get_note_by_slug

    _save(vault_path, "cookie-sandwich", "Quoted cookie values", body="A's body")
    _save(vault_path, "tech-other", "cookie-sandwich", body="B's body")  # file: cookie-sandwich.md

    note = get_note_by_slug("cookie-sandwich")
    assert note is not None and note.meta.id == "cookie-sandwich"
    assert get_note_by_slug("tech-other").body.strip() == "B's body"


def test_ids_sharing_an_80_char_prefix_resolve_apart_and_the_legacy_slug_is_ambiguous(
    vault_path,
):
    from slugify import slugify

    from sift.pipeline import get_note_by_slug, resolve_note

    stem = "writeup-" + "-".join(["request-smuggling-via-chunk-extensions"] * 3)
    a, b = f"{stem}-first-article", f"{stem}-second-article"
    _save(vault_path, a, "First article", ntype="writeup")
    _save(vault_path, b, "Second article", ntype="writeup")

    assert get_note_by_slug(b).meta.id == b  # by id
    assert get_note_by_slug(slugify(b)).meta.id == b  # by the uncapped slug

    legacy = slugify(b, max_length=80)
    assert legacy == slugify(a, max_length=80)
    hit = resolve_note(legacy)
    assert hit.ambiguous and {c["note_id"] for c in hit.candidate_info()} == {a, b}
    assert get_note_by_slug(legacy) is None, "an ambiguous slug must not pick a note"


def test_a_vault_path_resolves_but_a_file_outside_the_vault_does_not(vault_path, tmp_path):
    from sift.pipeline import resolve_note
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    inside = _save(vault_path, "tech-in", "Inside note")
    assert resolve_note(str(inside)).note.meta.id == "tech-in"
    assert resolve_note(inside.relative_to(vault_path).as_posix()).note.meta.id == "tech-in"

    outside = tmp_path / "outside.md"
    outside.write_text(
        Note(meta=Frontmatter(id="secret", type="technique", title="Out"), body="x").render(),
        encoding="utf-8",
    )
    assert resolve_note(str(outside)).note is None
    assert resolve_note("../outside.md").note is None


def test_an_id_edited_outside_sift_is_seen_at_once(vault_path):
    """The catalog is a cache: a row whose file now carries another id is re-read."""
    from sift.pipeline import get_note_by_slug, resolve_note

    path = _save(vault_path, "old-id", "Edited by hand")
    assert get_note_by_slug("old-id") is not None  # warms the catalog

    text = path.read_text(encoding="utf-8").replace("id: old-id", "id: new-id")
    path.write_text(text, encoding="utf-8")
    t = time.time() + 5
    os.utime(path, (t, t))

    assert resolve_note("old-id").note is None
    assert get_note_by_slug("new-id").meta.id == "new-id"


def test_a_missing_note_is_none_and_never_prints(vault_path, capfd):
    from sift.pipeline import get_note_by_slug

    _save(vault_path, "tech-real", "Real")
    (vault_path / "technique" / "empty.md").write_text("", encoding="utf-8")
    assert get_note_by_slug("does-not-exist") is None
    assert get_note_by_slug("") is None
    assert capfd.readouterr().out == ""


def test_the_returned_note_is_a_fresh_object(vault_path):
    """resolve_idea mutates what it gets back; a cached object would poison the cache."""
    from sift.pipeline import get_note_by_slug

    _save(vault_path, "idea-1", "An idea")
    first = get_note_by_slug("idea-1")
    first.meta.tags.append("mutated")
    first.body = "changed in memory only"
    again = get_note_by_slug("idea-1")
    assert again is not first
    assert "mutated" not in again.meta.tags and again.body.strip() == "body"
