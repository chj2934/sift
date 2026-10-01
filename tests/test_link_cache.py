"""Link-graph cache.

Building the graph parses every note - 3.3s over 9,020 notes, which made
`search_memory(expand_links=True)` 40x slower than a plain search. It is cached
against a (count, newest mtime) fingerprint that costs ~13ms.

A stale graph would be worse than a slow one, so these tests are about invalidation:
a note written mid-hunt must be visible to the very next query.
"""

from __future__ import annotations

import os
import time


def _write(vault, name: str, body: str = "hello", links: str = ""):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    note = Note(
        meta=Frontmatter(id=name, type="technique", title=name, links=[links] if links else []),
        body=body,
    )
    return save_note(vault, note)


def test_cache_returns_the_same_object_when_nothing_changed(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    first = build_link_index(vault_path)
    second = build_link_index(vault_path)
    assert first is second, "identical vault state should not be re-parsed"


def test_new_note_invalidates_the_cache(vault_path):
    """A capture_idea write mid-hunt must show up on the next query."""
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    assert "alpha" in build_link_index(vault_path)

    _write(vault_path, "beta")
    refreshed = build_link_index(vault_path)
    assert "beta" in refreshed, "cache served a stale graph after a new note"


def test_deleted_note_invalidates_the_cache(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    path = _write(vault_path, "doomed")
    assert "doomed" in build_link_index(vault_path)

    path.unlink()
    assert "doomed" not in build_link_index(vault_path)


def test_edited_note_invalidates_the_cache(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    _write(vault_path, "beta")
    assert build_link_index(vault_path)["beta"].all_links() == []

    # Bump mtime deterministically rather than relying on filesystem granularity.
    path = _write(vault_path, "beta", links="alpha")
    future = time.time() + 10
    os.utime(path, (future, future))

    assert "alpha" in build_link_index(vault_path)["beta"].all_links()


def test_fingerprint_ignores_underscore_files(vault_path):
    """_state.json and _rejects.jsonl churn constantly; they must not bust the cache."""
    from sift.index.graph import _fingerprint

    _write(vault_path, "alpha")
    before = _fingerprint(vault_path)
    (vault_path / "technique" / "_scratch.md").write_text("noise", encoding="utf-8")
    assert _fingerprint(vault_path) == before


def test_use_cache_false_always_rebuilds(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    _write(vault_path, "alpha")
    a = build_link_index(vault_path, use_cache=False)
    b = build_link_index(vault_path, use_cache=False)
    assert a is not b


# --- wikilink alias resolution ---------------------------------------------------
# Every cross-link in the first batch of technique notes was broken: they were written
# from the title ("[[cookie-sandwich...]]") while the note's slug carries an id prefix
# ("tech-cookie-sandwich..."). 0 of 5 resolved, so the link graph returned nothing.


def test_wikilink_without_the_id_prefix_resolves(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    _write(vault_path, "tech-cookie-sandwich", body="see [[unicode-overflow]]")
    _write(vault_path, "tech-unicode-overflow", body="the target")

    idx = build_link_index(vault_path)
    assert "unicode-overflow" in idx, "alias not registered"
    # ...and expansion must emit the CANONICAL slug, not the alias, so callers can
    # pass it straight to get_note.
    assert expand(["tech-cookie-sandwich"], idx) == ["tech-unicode-overflow"]


def test_a_real_slug_is_never_shadowed_by_an_alias(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache

    clear_link_index_cache()
    real = _write(vault_path, "unicode-overflow", body="the genuine article")
    _write(vault_path, "tech-unicode-overflow", body="would alias to the same key")

    idx = build_link_index(vault_path)
    assert idx["unicode-overflow"].path == real, "alias overwrote a real note"


def test_unknown_wikilinks_are_ignored(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    _write(vault_path, "tech-alpha", body="see [[does-not-exist]]")
    idx = build_link_index(vault_path)
    assert expand(["tech-alpha"], idx) == []
