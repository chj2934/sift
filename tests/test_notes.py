from __future__ import annotations

from sift.vault.notes import Note, load_note, save_note
from sift.vault.schema import Frontmatter


def test_frontmatter_roundtrip(vault_path):
    meta = Frontmatter(
        id="CVE-2024-1234",
        type="cve",
        title="Test CVE",
        cwe=["79", "CWE-89"],  # normalization: bare number -> CWE-79
        severity="High",  # -> lowercased
        tags="single-tag",  # -> list
    )
    note = Note(meta=meta, body="## Description\n\nsomething\n\nrelated: [[other-note]]")
    path = save_note(vault_path, note)

    assert path.exists()
    assert path.name == "Test CVE.md"  # filename is the title, for Obsidian

    reloaded = load_note(path)
    assert reloaded.meta.cwe == ["CWE-79", "CWE-89"]
    assert reloaded.meta.severity == "high"
    assert reloaded.meta.tags == ["single-tag"]
    assert reloaded.meta.ingested is not None


def test_wikilink_extraction():
    note = Note(
        meta=Frontmatter(id="x", type="finding", title="x", links=["explicit-link"]),
        body="see [[Body Link]] and [[another|aliased]] and [[explicit-link]]",
    )
    links = note.all_links()
    assert "body-link" in links
    assert "another" in links
    assert "explicit-link" in links
    assert len(links) == len(set(links))  # deduped


def test_render_has_frontmatter_fence():
    note = Note(meta=Frontmatter(id="x", type="technique", title="T"), body="hi")
    out = note.render()
    assert out.startswith("---\n")
    assert "\n---\n" in out
    assert out.rstrip().endswith("hi")
