"""Which note a link key resolves to.

Slugs used to be cut to 80 characters, so distinct notes shared one (16 real pairs)
and a link resolved to one note while get_note returned the other. The graph answers
to every form a link can take, strongest first:

    exact id > slug > legacy 80-char slug / uncapped id slug > filename > title
    > slug without its source prefix

A key several notes claim at the same strength is recorded in `ambiguous` and
followed to every claimant instead of silently resolving to one of them.
"""

from __future__ import annotations

import pytest


def _put(vault, rel: str, note_id: str, *, title=None, body="body", links=(), note_type=None):
    """Write a note by hand, so the test controls the exact filename."""
    import yaml

    meta = {"id": note_id, "type": note_type or rel.split("/")[0], "title": title or note_id}
    if links:
        meta["links"] = list(links)
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    path.write_text(f"---\n{fm}---\n\n{body}\n", encoding="utf-8")
    return path


@pytest.fixture
def uncapped_slugs(monkeypatch):
    """Note.slug as it is once slugs stop being cut at 80 characters.

    The vault package owns that change; patching it here keeps these tests about the
    graph, whichever form Note.slug currently has.
    """
    from slugify import slugify

    from sift.index.graph import clear_link_index_cache
    from sift.vault.notes import Note

    monkeypatch.setattr(
        Note, "slug", property(lambda self: slugify(self.meta.id) or slugify(self.meta.title))
    )
    clear_link_index_cache()
    yield
    clear_link_index_cache()


# Long ids/titles: their slugs run past 80 characters and share the first 80.
_BASE = "research-" + "a-very-long-shared-article-title-prefix-" * 3
_PREFIX = "Arbitrary web cache deception and poisoning via URL parser discrepancies in CDN edge"


@pytest.mark.parametrize(
    "text",
    [
        "x" * 200,
        _PREFIX + " normalisation",
        "a" * 79 + " bcd",  # a separator lands exactly on the cut
        "a" * 80 + "-b",
        "'Quoted' \"title\" with 1,000 numbers and 3.5 decimals " * 3,
        "Ünïcödé and 中文 and emoji 🚀 " * 5,
        "--leading and trailing--" * 6,
        _BASE + "first",
        "short",
        "",
    ],
)
def test_cap_matches_slugify_max_length(text):
    """The graph derives the legacy 80-char forms from the uncapped slug instead of
    slugifying twice; that is only sound while the two agree."""
    from slugify import slugify

    from sift.index.graph import _cap

    assert _cap(slugify(text)) == slugify(text, max_length=80)


def test_a_long_title_wikilink_resolves(vault_path):
    """Wikilinks are slugified uncapped, so a title alias cut at 80 never matched them."""
    from slugify import slugify

    from sift.index.graph import build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    title = _PREFIX + " normalisation"
    assert len(slugify(title)) > 80
    _put(vault_path, "technique/Long.md", "tech-long", title=title)
    _put(vault_path, "technique/Linker.md", "tech-linker", body=f"see [[{title}]]")

    assert expand(["tech-linker"], build_link_index(vault_path)) == ["tech-long"]


def test_long_titles_sharing_an_80_char_prefix_each_resolve_to_their_own_note(vault_path):
    from slugify import slugify

    from sift.index.graph import _cap, build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    one, two = f"{_PREFIX} part one", f"{_PREFIX} part two"
    shared = _cap(slugify(one))
    assert shared == _cap(slugify(two))
    _put(vault_path, f"technique/{one}.md", "tech-one", title=one)
    _put(vault_path, f"technique/{two}.md", "tech-two", title=two)
    _put(vault_path, "technique/Linker.md", "tech-linker", body=f"[[{two}]] then [[{one}]]")

    idx = build_link_index(vault_path)
    assert expand(["tech-linker"], idx) == ["tech-two", "tech-one"]
    # The truncated form is contested, and says so rather than picking one.
    assert {r.note_id for r in idx.ambiguous[shared]} == {"tech-one", "tech-two"}


def test_legacy_80_char_slug_still_resolves_once_slugs_are_uncapped(vault_path, uncapped_slugs):
    """Every index row keeps its 80-char slug until `reindex --force`, and links
    written from old remember() results carry it forever. Both must still resolve."""
    from slugify import slugify

    from sift.index.graph import build_link_index, expand

    long_id = _BASE + "first"
    legacy = slugify(long_id, max_length=80)
    _put(vault_path, "writeup/Target.md", long_id, title="Target", body="see [[find-old]]")
    _put(vault_path, "finding/Old.md", "find-old", links=[legacy])

    idx = build_link_index(vault_path)
    target = idx[long_id]
    assert target.slug == slugify(long_id) != legacy
    assert idx[legacy] is target
    assert expand(["find-old"], idx) == [target.slug]  # an old links: entry
    assert expand([legacy], idx) == ["find-old"]  # a seed from an old index row


def test_ids_sharing_an_80_char_prefix_stay_distinct(vault_path, uncapped_slugs):
    from slugify import slugify

    from sift.index.graph import build_link_index, expand

    id_a, id_b = _BASE + "first", _BASE + "second"
    legacy = slugify(id_a, max_length=80)
    assert legacy == slugify(id_b, max_length=80)
    _put(vault_path, "writeup/A.md", id_a, title="A", body="[[n-a]]")
    _put(vault_path, "writeup/B.md", id_b, title="B", body="[[n-b]]")
    _put(vault_path, "technique/n-a.md", "n-a")
    _put(vault_path, "technique/n-b.md", "n-b")
    _put(vault_path, "finding/Linker.md", "linker", links=[legacy])

    idx = build_link_index(vault_path)
    assert expand([slugify(id_b)], idx) == ["n-b"]
    assert expand([id_b], idx) == ["n-b"]
    assert {r.note_id for r in idx.ambiguous[legacy]} == {id_a, id_b}
    assert sorted(expand(["linker"], idx)) == sorted([slugify(id_a), slugify(id_b)])


def test_exact_id_resolves_even_while_slugs_collide(vault_path):
    """Holds with today's Note.slug too: an id is exact even when two ids share their
    80-char slug."""
    from sift.index.graph import build_link_index, clear_link_index_cache, expand, expand_records

    clear_link_index_cache()
    id_a, id_b = _BASE + "first", _BASE + "second"
    _put(vault_path, "writeup/A.md", id_a, title="A", body="[[n-a]]")
    _put(vault_path, "writeup/B.md", id_b, title="B", body="[[n-b]]")
    _put(vault_path, "technique/n-a.md", "n-a")
    _put(vault_path, "technique/n-b.md", "n-b")

    idx = build_link_index(vault_path)
    assert idx[id_a].path.name == "A.md" and idx[id_b].path.name == "B.md"
    assert expand([id_b], idx) == ["n-b"]
    assert [r.note_id for r in expand_records([id_a], idx)] == ["n-a"]


def test_a_title_link_prefers_the_file_named_by_it(vault_path):
    """Obsidian resolves [[Shared title]] to `Shared title.md`. The "(2)" sibling that a
    second note with the same title gets sorts first (' ' < '.'), so first-claim-wins
    handed it the link."""
    from sift.index.graph import build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    _put(vault_path, "technique/Shared title.md", "tech-first", title="Shared title")
    _put(vault_path, "technique/Shared title (2).md", "tech-second", title="Shared title")
    _put(
        vault_path,
        "technique/Linker.md",
        "tech-linker",
        body="[[Shared title]] and [[Shared title (2)]]",
    )

    assert expand(["tech-linker"], build_link_index(vault_path)) == ["tech-first", "tech-second"]


def test_a_contested_key_is_flagged_and_followed_to_every_claimant(vault_path):
    from sift.index.graph import (
        build_link_index,
        clear_link_index_cache,
        expand,
        expand_records,
        lookup,
    )

    clear_link_index_cache()
    _put(vault_path, "technique/One.md", "dup-id", title="One", body="[[n-one]]")
    _put(vault_path, "report/Two.md", "dup-id", title="Two", body="[[n-two]]")
    _put(vault_path, "technique/n-one.md", "n-one")
    _put(vault_path, "technique/n-two.md", "n-two")
    _put(vault_path, "finding/Linker.md", "linker", body="[[dup-id]]")

    idx = build_link_index(vault_path)
    claimants = idx.ambiguous["dup-id"]
    assert [r.path.name for r in claimants] == ["Two.md", "One.md"]  # path order
    assert lookup(idx, "dup-id") == claimants
    # A plain lookup gets the first in path order - the note get_note returns.
    assert idx["dup-id"] is claimants[0]
    assert {r.path.name for r in expand_records(["linker"], idx)} == {"One.md", "Two.md"}
    assert sorted(expand(["dup-id"], idx)) == ["n-one", "n-two"]
    assert lookup(idx, "no-such-key") == ()


def test_note_ids_work_as_seeds(vault_path):
    """A search hit's note_id is exact; its slug is lossy."""
    from sift.index.graph import build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    _put(vault_path, "writeup/Foo.md", "Research_Foo.Bar", title="Foo", body="[[n-x]]")
    _put(vault_path, "technique/n-x.md", "n-x")

    assert expand(["Research_Foo.Bar"], build_link_index(vault_path)) == ["n-x"]


def test_seed_notes_are_never_returned_as_neighbours(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    _put(vault_path, "technique/A.md", "Note_A", title="A", body="[[Note_B]]")
    _put(vault_path, "technique/B.md", "Note_B", title="B", body="[[Note_A]]")

    idx = build_link_index(vault_path)
    assert expand(["Note_A"], idx) == ["note-b"]
    assert expand(["Note_A", "Note_B"], idx) == []


def test_hops_and_limit(vault_path):
    from sift.index.graph import build_link_index, clear_link_index_cache, expand

    clear_link_index_cache()
    _put(vault_path, "technique/a.md", "a", body="[[b]]")
    _put(vault_path, "technique/b.md", "b", body="[[c]]")
    _put(vault_path, "technique/c.md", "c", body="[[a]]")

    idx = build_link_index(vault_path)
    assert expand(["a"], idx) == ["b"]
    assert expand(["a"], idx, hops=2) == ["b", "c"]
    assert expand(["a"], idx, hops=2, limit=1) == ["b"]
    assert expand(["a"], idx, limit=0) == []


def test_expand_still_accepts_a_plain_dict_of_notes(vault_path):
    """expand() is public; a caller holding {slug: Note} must keep working."""
    from sift.index.graph import expand
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    a = Note(meta=Frontmatter(id="a", type="technique", title="a"), body="[[b]]")
    b = Note(meta=Frontmatter(id="b", type="technique", title="b"), body="")
    assert expand(["a"], {"a": a, "b": b}) == ["b"]


def test_warm_link_index_builds_in_the_background(vault_path, monkeypatch):
    """An MCP server can start the ~5 s cold parse at startup; the first expanding
    query then shares that build instead of paying for its own."""
    import os
    import time

    from sift.index import graph

    graph.clear_link_index_cache()
    path = _put(vault_path, "technique/a.md", "a", body="[[b]]")
    t = time.time() - 3600
    os.utime(path, (t, t))

    thread = graph.warm_link_index(vault_path)
    thread.join(timeout=30)
    assert not thread.is_alive() and thread.daemon

    from sift.vault import catalog

    parsed: list = []
    real = catalog.load_note
    monkeypatch.setattr(catalog, "load_note", lambda p: parsed.append(p) or real(p))
    assert "a" in graph.build_link_index(vault_path)
    assert parsed == [], "the warmed graph was not reused"
