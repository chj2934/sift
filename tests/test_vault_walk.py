"""One enumeration rule for every vault walker (K4).

`.trash` (Obsidian's and sift's soft delete), `.obsidian`, `_templates`, Obsidian's
configured Templates folder and README.md are not notes. Trashed notes carry valid
frontmatter, so walking them brought deleted notes back as duplicate ids.
"""

from __future__ import annotations


def _put(path, note_id="x", note_type="technique"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: {note_id}\ntype: {note_type}\ntitle: {note_id}\n---\n\nb\n", encoding="utf-8"
    )
    return path


def test_structural_skips(vault_path):
    from sift.vault.notes import iter_notes, walk_note_files

    _put(vault_path / "technique" / "Real.md", "real")
    _put(vault_path / "technique" / "web" / "Nested.md", "nested")
    _put(vault_path / ".trash" / "technique" / "Real.md", "real")
    _put(vault_path / ".obsidian" / "x.md", "obs")
    _put(vault_path / "_templates" / "T.md", "tmpl")
    _put(vault_path / "technique" / "_draft.md", "draft")
    _put(vault_path / "technique" / ".hidden.md", "hidden")
    _put(vault_path / "README.md", "readme1")
    _put(vault_path / "technique" / "readme.md", "readme2")
    (vault_path / "technique" / "notes.txt").write_text("not markdown", encoding="utf-8")
    _put(vault_path / "technique" / "Upper.MD", "upper")

    assert [n.meta.id for n in iter_notes(vault_path)] == ["real", "upper", "nested"]
    rel = [p.relative_to(vault_path).as_posix() for p in walk_note_files(vault_path)]
    assert rel == ["technique/Real.md", "technique/Upper.MD", "technique/web/Nested.md"]


def test_obsidian_templates_folder_is_skipped(vault_path):
    from sift.vault.notes import iter_notes

    (vault_path / ".obsidian").mkdir()
    (vault_path / ".obsidian" / "templates.json").write_text(
        '{"folder": "Templates"}', encoding="utf-8"
    )
    _put(vault_path / "Templates" / "Finding template.md", "{{title}}")
    _put(vault_path / "finding" / "Real.md", "real", "finding")
    assert [n.meta.id for n in iter_notes(vault_path)] == ["real"]


def test_configured_ignore_dirs(vault_path, monkeypatch):
    """`vault_ignore_dirs` (read with getattr until the setting exists): a bare name
    matches at any depth, a path with a slash is vault-relative."""
    from types import SimpleNamespace

    from sift import config
    from sift.vault.notes import iter_notes

    real_get = config.get_settings
    real = real_get()
    fake = SimpleNamespace(
        vault_ignore_dirs="Archive, finding/private",
        resolved_db=real.resolved_db,
        resolved_vault=real.resolved_vault,
    )

    def fake_get_settings():
        return fake

    fake_get_settings.cache_clear = real_get.cache_clear  # conftest teardown calls it
    monkeypatch.setattr(config, "get_settings", fake_get_settings)
    _put(vault_path / "finding" / "Keep.md", "keep", "finding")
    _put(vault_path / "finding" / "private" / "Secret.md", "secret", "finding")
    _put(vault_path / "report" / "private" / "Kept.md", "kept-too", "report")
    _put(vault_path / "report" / "Archive" / "Old.md", "old", "report")
    assert sorted(n.meta.id for n in iter_notes(vault_path)) == ["keep", "kept-too"]


def test_walk_order_matches_sorted_rglob(vault_path):
    from sift.vault.notes import walk_note_files

    for rel in ("b/z.md", "a.md", "a/b.md", "a/a.md", "c.md", "b.md"):
        _put(vault_path / rel)
    expected = sorted(vault_path.rglob("*.md"))
    assert list(walk_note_files(vault_path)) == expected


def test_duplicate_ids_are_yielded_and_reported_once(vault_path):
    import logging

    from sift.vault.notes import iter_notes

    records = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    lg = logging.getLogger("sift.vault")
    lg.addHandler(handler)
    try:
        _put(vault_path / "cve" / "A.md", "CVE-1", "cve")
        _put(vault_path / "cve" / "B.md", "CVE-1", "cve")
        for _ in range(2):
            assert [n.path.name for n in iter_notes(vault_path)] == ["A.md", "B.md"]
    finally:
        lg.removeHandler(handler)
    dup = [r.getMessage() for r in records if "duplicate files for one id" in r.getMessage()]
    assert len(dup) == 1 and "B.md" in dup[0] and "cve/A.md" in dup[0]
