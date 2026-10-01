"""Filenames are note titles, so the vault opens natively in Obsidian.

`meta.id` remains the stable identity used by the index; only the filename changed.
The point is that `[[Cookie sandwich - reading HttpOnly cookies]]` typed in Obsidian
resolves to the file, and the file list is readable instead of a wall of truncated
slugs like `tech-arbitrary-web-cache-deception-and-poisoning-via-url-parser-discrepanci.md`.
"""

from __future__ import annotations

import pytest


def _meta(title: str, note_id: str = "tech-x", note_type: str = "technique"):
    from sift.vault.schema import Frontmatter

    return Frontmatter(id=note_id, type=note_type, title=title)


def test_filename_is_the_title(vault_path):
    from sift.vault.notes import note_path

    p = note_path(vault_path, _meta("Cookie sandwich: reading HttpOnly cookies"))
    # The colon is illegal on Windows; everything else survives, spaces and case included.
    assert p.name == "Cookie sandwich- reading HttpOnly cookies.md"
    assert p.parent.name == "technique"


@pytest.mark.parametrize("bad", ['a<b', 'a>b', 'a:b', 'a"b', "a/b", "a\\b", "a|b", "a?b", "a*b"])
def test_windows_illegal_characters_are_replaced(bad):
    from sift.vault.notes import title_to_filename

    stem = title_to_filename(bad)
    assert not any(c in stem for c in '<>:"/\\|?*')
    assert stem


def test_unicode_titles_are_preserved():
    """Orange Tsai's posts are Chinese; slugify would have flattened them to nothing."""
    from sift.vault.notes import title_to_filename

    assert title_to_filename("101 年全國大專院校資安技能金盾獎") == "101 年全國大專院校資安技能金盾獎"


def test_windows_reserved_names_are_escaped():
    from sift.vault.notes import title_to_filename

    assert title_to_filename("CON") != "CON"
    assert title_to_filename("nul").lower() != "nul"


def test_long_titles_are_truncated_on_a_word_boundary():
    from sift.vault.notes import FILENAME_MAX, title_to_filename

    long = "Arbitrary web cache deception and poisoning via URL parser discrepancies " * 4
    stem = title_to_filename(long)
    assert len(stem) <= FILENAME_MAX
    assert not stem.endswith(" ")


def test_trailing_dots_and_spaces_are_stripped():
    """Windows silently drops them, which would desync path-on-disk from path-in-index."""
    from sift.vault.notes import title_to_filename

    assert title_to_filename("Some technique...  ") == "Some technique"


def test_empty_title_falls_back_to_the_id():
    from sift.vault.notes import title_to_filename

    assert title_to_filename("", "tech-fallback-id") == "tech-fallback-id"
    assert title_to_filename("///") != ""


def test_roundtrip_save_and_load(vault_path):
    from sift.vault.notes import Note, load_note, save_note

    note = Note(meta=_meta("SAML roulette: the hacker always wins"), body="body")
    path = save_note(vault_path, note)
    assert path.name == "SAML roulette- the hacker always wins.md"
    assert load_note(path).meta.title == "SAML roulette: the hacker always wins"


def test_obsidian_style_title_wikilink_resolves(vault_path):
    """The whole point: a link written the way Obsidian writes them must work in sift."""
    from sift.index.graph import build_link_index, clear_link_index_cache, expand
    from sift.vault.notes import Note, save_note

    clear_link_index_cache()
    save_note(
        vault_path,
        Note(meta=_meta("Cookie sandwich", "tech-cookie"), body="see [[Unicode overflow]]"),
    )
    save_note(vault_path, Note(meta=_meta("Unicode overflow", "tech-unicode"), body="target"))

    idx = build_link_index(vault_path)
    linked = expand(["tech-cookie"], idx)
    assert linked == ["tech-unicode"], f"title wikilink did not resolve: {linked}"
