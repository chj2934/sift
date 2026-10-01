"""`sift ingest notes`: frontmatter backfill for hand-written notes.

Every case here destroyed a real user note before: the backfill rebuilt frontmatter
from five fields (dropping url/program/severity/created/links), split `tags: ssrf`
into letters, gave a UTF-8-BOM note a new id while demoting its YAML into the body,
and aborted the whole run on one bad file. The index is not under test here
(`pipeline.reindex` is stubbed in `backfill_and_index`).
"""

from __future__ import annotations

import pytest


def _write(path, text, *, bom=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + data)
    return path


def _only_note(folder):
    files = sorted(folder.glob("*.md"))
    assert len(files) == 1, [f.name for f in files]
    return files[0]


@pytest.fixture
def no_reindex(monkeypatch):
    """backfill_and_index's indexing pass, stubbed."""
    from sift import pipeline

    calls = []

    class _Stats:
        notes = 0

    def fake_reindex(vault=None, **kwargs):
        calls.append(vault)
        return _Stats()

    monkeypatch.setattr(pipeline, "reindex", fake_reindex)
    return calls


def test_partial_frontmatter_keeps_every_key(vault_path):
    from sift.ingest.local_notes import backfill
    from sift.vault.notes import load_note

    _write(
        vault_path / "inbox" / "ssrf-in-pdf.md",
        "---\n"
        "url: https://hackerone.com/reports/1\n"
        "program: Acme\n"
        "severity: High\n"
        "created: 2026-08-01\n"
        "cwe: 918\n"
        "links: [target-acme]\n"
        "tags: ssrf\n"
        "extra: {bounty_note: paid}\n"
        "aliases: [pdf ssrf]\n"
        "---\n"
        "# SSRF in the PDF renderer\n\nThe renderer fetched http://169.254.169.254/.\n",
    )

    fixed, problems = backfill(vault_path)

    assert (fixed, problems) == (1, [])
    note = load_note(_only_note(vault_path / "finding"))
    m = note.meta
    assert m.title == "SSRF in the PDF renderer" and m.type == "finding"
    assert m.id == "local-ssrf-in-pdf" and m.source == "manual"
    assert m.url == "https://hackerone.com/reports/1" and m.program == "Acme"
    assert m.severity == "high" and m.created.isoformat() == "2026-08-01"
    assert m.cwe == ["CWE-918"] and m.links == ["target-acme"]
    assert m.tags == ["ssrf"], "a scalar tag must not become letters"
    assert m.extra == {"bounty_note": "paid"}
    assert m.model_extra == {"aliases": ["pdf ssrf"]}, "unknown keys stay at the top level"
    assert "169.254.169.254" in note.body
    assert not (vault_path / "inbox" / "ssrf-in-pdf.md").exists(), "the loose original moved"


def test_no_frontmatter_at_all_is_adopted(vault_path):
    from sift.ingest.local_notes import backfill
    from sift.vault.notes import load_note

    _write(
        vault_path / "technique" / "cookie-tossing.md",
        "# Cookie tossing\n\nSet it on a parent domain.\n",
    )

    assert backfill(vault_path) == (1, [])
    note = load_note(_only_note(vault_path / "technique"))
    assert note.meta.type == "technique" and note.meta.title == "Cookie tossing"
    assert note.body.startswith("# Cookie tossing")


def test_a_bom_note_with_complete_frontmatter_is_left_byte_identical(vault_path):
    from sift.ingest.local_notes import backfill
    from sift.vault.notes import load_note

    path = _write(
        vault_path / "finding" / "Mine.md",
        "---\nid: my-own-id\ntype: finding\ntitle: Mine\n---\n\nbody\n",
        bom=True,
    )
    before = path.read_bytes()

    assert backfill(vault_path) == (0, [])
    assert path.read_bytes() == before
    assert load_note(path).meta.id == "my-own-id"


def test_a_bom_note_missing_a_key_keeps_its_id_and_its_yaml(vault_path):
    from sift.ingest.local_notes import backfill
    from sift.vault.notes import load_note

    _write(
        vault_path / "finding" / "Partial.md",
        "---\nid: keep-this-id\ntitle: Partial\nprogram: Acme\n---\n\nbody text\n",
        bom=True,
    )

    assert backfill(vault_path) == (1, [])
    note = load_note(_only_note(vault_path / "finding"))
    assert note.meta.id == "keep-this-id" and note.meta.program == "Acme"
    assert "program:" not in note.body and note.body.strip() == "body text"


def test_empty_id_and_title_are_filled(vault_path):
    from sift.ingest.local_notes import backfill
    from sift.vault.notes import load_note

    _write(vault_path / "finding" / "blank.md", "---\nid:\ntitle: ''\n---\n# Real title\n\ntext\n")

    assert backfill(vault_path) == (1, [])
    note = load_note(_only_note(vault_path / "finding"))
    assert note.meta.id == "local-blank" and note.meta.title == "Real title"


def test_an_unknown_type_is_coerced_and_remembered(vault_path):
    from sift.ingest.local_notes import backfill
    from sift.vault.notes import load_note

    _write(vault_path / "inbox" / "idea.md", "---\ntype: idea\n---\n# Try JWT kid traversal\n")

    assert backfill(vault_path) == (1, [])
    note = load_note(_only_note(vault_path / "finding"))
    assert note.meta.type == "finding" and note.meta.extra["original_type"] == "idea"


def test_a_complete_but_invalid_note_is_reported_not_coerced(vault_path):
    from sift.ingest.local_notes import backfill

    path = _write(
        vault_path / "finding" / "x.md", "---\nid: x-1\ntype: idea\ntitle: X\n---\nbody\n"
    )
    before = path.read_bytes()

    fixed, problems = backfill(vault_path)

    assert fixed == 0 and [p for p, _ in problems] == [path]
    assert path.read_bytes() == before


def test_one_bad_file_does_not_stop_the_others(vault_path):
    from sift.ingest.local_notes import backfill

    bad = _write(vault_path / "inbox" / "bad.md", "---\ncreated: soon\n---\n# Bad date\n")
    before = bad.read_bytes()
    _write(vault_path / "inbox" / "good.md", "# Good note\n\ntext\n")
    _write(vault_path / "inbox" / "cp1252.md", "")
    (vault_path / "inbox" / "cp1252.md").write_bytes(b"# caf\xe9\n")

    fixed, problems = backfill(vault_path)

    assert fixed == 1
    assert sorted(p.name for p, _ in problems) == ["bad.md", "cp1252.md"]
    assert bad.read_bytes() == before, "a file that cannot be backfilled is left untouched"


def test_trash_templates_and_empty_files_are_never_adopted(vault_path):
    from sift.ingest.local_notes import backfill

    trashed = _write(vault_path / ".trash" / "old.md", "# Deleted note\n")
    template = _write(vault_path / "_templates" / "finding.md", "# {{title}}\n")
    empty = _write(vault_path / "inbox" / "empty.md", "")

    fixed, problems = backfill(vault_path)

    assert fixed == 0
    assert [p.name for p, _ in problems] == ["empty.md"]
    assert trashed.exists() and template.exists() and empty.read_bytes() == b""
    assert not (vault_path / "finding").exists()


def test_a_file_already_at_its_canonical_name_is_rewritten_in_place(vault_path):
    """On Windows `finding/my idea.md` IS `finding/My Idea.md`; saving "elsewhere"
    used to see this very file as occupied and fork `My Idea (2).md`."""
    from sift.ingest.local_notes import backfill
    from sift.vault.notes import load_note

    path = _write(vault_path / "finding" / "My Idea.md", "# My Idea\n\ntext\n")

    assert backfill(vault_path) == (1, [])
    assert _only_note(vault_path / "finding") == path
    assert load_note(path).meta.title == "My Idea"


def test_two_loose_files_with_one_stem_do_not_overwrite_each_other(vault_path):
    from sift.ingest.local_notes import backfill
    from sift.vault.notes import load_note

    _write(vault_path / "inbox" / "idea.md", "# Idea\n\nfirst idea body\n")
    _write(vault_path / "drafts" / "idea.md", "# Idea\n\nsecond idea body\n")

    assert backfill(vault_path)[0] == 2
    notes = [load_note(p) for p in sorted((vault_path / "finding").glob("*.md"))]
    assert len(notes) == 2 and len({n.meta.id for n in notes}) == 2
    assert {n.body.strip().splitlines()[-1] for n in notes} == {
        "first idea body",
        "second idea body",
    }


def test_backfill_and_index_still_unpacks_and_reports(vault_path, no_reindex):
    from sift.ingest.local_notes import backfill_and_index

    _write(vault_path / "inbox" / "a.md", "# A\n")
    _write(vault_path / "inbox" / "b.md", "---\nid: b\ntype: idea\ntitle: B\n---\n")

    res = backfill_and_index()
    fixed, indexed = res

    assert (fixed, indexed) == (1, 0)
    assert [p.name for p, _ in res.problems] == ["b.md"]
    assert no_reindex == [vault_path], "indexing is the incremental reindex"


def test_backfill_prints_nothing(vault_path, capfd):
    from sift.ingest.local_notes import backfill

    _write(vault_path / "inbox" / "bad.md", "---\ncreated: soon\n---\n# Bad\n")
    _write(vault_path / "inbox" / "ok.md", "# Ok\n")
    backfill(vault_path)
    assert capfd.readouterr().out == ""
