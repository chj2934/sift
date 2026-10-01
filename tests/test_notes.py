from __future__ import annotations

import logging

import pytest


@pytest.fixture
def vault_log():
    """Warnings logged by the vault package, captured on its own logger.

    A handler on `sift.vault` itself (not caplog's root handler) still sees records
    when a CLI test elsewhere in the session has set `propagate = False` on `sift`.
    """

    class _Collect(logging.Handler):
        def __init__(self) -> None:
            super().__init__(logging.DEBUG)
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    lg = logging.getLogger("sift.vault")
    handler = _Collect()
    old_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)


def _warnings(records, needle: str) -> list[str]:
    return [
        r.getMessage() for r in records if r.levelno >= logging.WARNING and needle in r.getMessage()
    ]


def _write_raw(path, text: str, *, bom: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" + data) if bom else data)


# --- round trip -----------------------------------------------------------------


def test_frontmatter_roundtrip(vault_path):
    from sift.vault.notes import Note, load_note, save_note
    from sift.vault.schema import Frontmatter

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
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

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
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    note = Note(meta=Frontmatter(id="x", type="technique", title="T"), body="hi")
    out = note.render()
    assert out.startswith("---\n")
    assert "\n---\n" in out
    assert out.rstrip().endswith("hi")


# --- wikilinks: headings, blocks, folders, long titles ----------------------------


@pytest.mark.parametrize(
    "link, want",
    [
        ("[[Target#Heading]]", ["target"]),
        ("[[Target#Heading|alias]]", ["target"]),
        ("[[Target#^block-id]]", ["target"]),
        ("[[folder/Target]]", ["target"]),
        ("[[Target.md]]", ["target"]),
        ("[[Target|]]", ["target"]),
        ("[[#Heading in this note]]", []),
    ],
)
def test_section_and_path_wikilinks_resolve_to_the_note(link, want):
    """A heading or block part used to make the whole link fail to match."""
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    note = Note(meta=Frontmatter(id="x", type="finding", title="x"), body=f"see {link} here")
    assert note.wikilinks == want


def test_long_title_link_and_slug_are_not_cut_at_80():
    from slugify import slugify

    from sift.vault.notes import Note, legacy_slug
    from sift.vault.schema import Frontmatter

    long_title = (
        "Arbitrary web cache deception and poisoning via URL parser discrepancies in CDN edge"
    )
    body = f"see [[{long_title}]]"
    note = Note(meta=Frontmatter(id="x", type="finding", title="x"), body=body)
    assert note.wikilinks == [slugify(long_title)]
    assert len(note.wikilinks[0]) > 80

    base = "research-" + "a-very-long-shared-article-title-prefix-" * 3
    a = Note(meta=Frontmatter(id=base + "first", type="writeup", title="A"), body="")
    b = Note(meta=Frontmatter(id=base + "second", type="writeup", title="B"), body="")
    assert a.slug != b.slug  # used to be one shared 80-char slug
    assert a.legacy_slug == b.legacy_slug == slugify(base + "first", max_length=80)
    assert legacy_slug(base + "first") == a.legacy_slug


# --- hand-edited frontmatter: one bad field no longer drops the note ---------------


@pytest.mark.parametrize(
    "fm, check",
    [
        ("cwe: [79, 89]", lambda m: m.cwe == ["CWE-79", "CWE-89"]),
        ("cwe: 79", lambda m: m.cwe == ["CWE-79"]),
        ("tags: [2024, xss]", lambda m: m.tags == ["2024", "xss"]),
        ("tags: xss", lambda m: m.tags == ["xss"]),
        ("created: 2024-05-01T10:00:00", lambda m: str(m.created) == "2024-05-01"),
        ("extra: null", lambda m: m.extra == {}),
        ("bounty: '$1,000'", lambda m: m.bounty == 1000.0),
    ],
)
def test_hand_typed_values_are_coerced(tmp_path, fm, check):
    from sift.vault.notes import load_note

    p = tmp_path / "n.md"
    _write_raw(p, f"---\nid: n\ntype: technique\ntitle: T\n{fm}\n---\n\nbody\n")
    assert check(load_note(p).meta)


def test_type_case_and_numeric_title_are_accepted(tmp_path):
    from sift.vault.notes import load_note

    p = tmp_path / "n.md"
    _write_raw(p, "---\nid: 1234\ntype: Technique\ntitle: 404\n---\n\nbody\n")
    meta = load_note(p).meta
    assert (meta.id, meta.type, meta.title) == ("1234", "technique", "404")


def test_missing_required_key_still_fails(tmp_path):
    from pydantic import ValidationError

    from sift.vault.notes import load_note

    p = tmp_path / "n.md"
    _write_raw(p, "---\nid: n\ntype: technique\n---\n\nbody\n")
    with pytest.raises(ValidationError):
        load_note(p)


def test_unknown_keys_and_bad_values_survive_a_resave(vault_path):
    """Obsidian `aliases`, a user's own `status`, and a value sift cannot validate
    (`bounty: lots`) were deleted the next time sift re-saved the note."""
    import yaml

    from sift.vault.notes import load_note, save_note

    p = vault_path / "technique" / "T.md"
    _write_raw(
        p,
        "---\nid: tech-t\ntype: technique\ntitle: T\naliases: [Tee]\nstatus: draft\n"
        "reviewed: 2024-05-01\nbounty: lots\nextra: [1]\n---\n\nbody\n",
    )
    note = load_note(p)
    assert note.meta.invalid_fields == {"bounty": "lots", "extra": [1]}
    note.body = "edited"
    assert save_note(vault_path, note) == p

    raw = yaml.safe_load(p.read_text(encoding="utf-8").split("---")[1])
    assert raw["aliases"] == ["Tee"]
    assert raw["status"] == "draft"
    assert str(raw["reviewed"]) == "2024-05-01"
    assert raw["bounty"] == "lots"
    assert raw["extra"] == [1]
    assert load_note(p).body.strip() == "edited"


def test_code_built_frontmatter_has_no_stray_keys():
    from sift.vault.schema import Frontmatter

    meta = Frontmatter(id="x", type="cve", title="t", tags=["a"], extra={"k": 1})
    assert meta.model_extra == {}
    assert meta.invalid_fields == {}


def test_bom_note_loads_and_is_rewritten_in_place(vault_path):
    """PowerShell 5.1's Out-File writes a BOM, which hid the frontmatter fence."""
    from sift.vault.notes import Note, load_note, save_note
    from sift.vault.schema import Frontmatter

    p = vault_path / "technique" / "Bommed.md"
    _write_raw(p, "---\nid: tech-bom\ntype: technique\ntitle: Bommed\n---\n\nbody\n", bom=True)
    assert load_note(p).meta.id == "tech-bom"

    fresh = Note(meta=Frontmatter(id="tech-bom", type="technique", title="Bommed"), body="new")
    assert save_note(vault_path, fresh) == p
    assert sorted(x.name for x in p.parent.iterdir()) == ["Bommed.md"]  # no "(2)"


# --- atomic writes --------------------------------------------------------------


def test_failed_rewrite_leaves_the_old_file_intact(vault_path, monkeypatch):
    import os

    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    note = Note(meta=Frontmatter(id="tech-a", type="technique", title="Atomic"), body="v1")
    path = save_note(vault_path, note)
    before = path.read_bytes()

    def boom(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", boom)
    note.body = "v2"
    with pytest.raises(OSError):
        save_note(vault_path, note)
    monkeypatch.undo()

    assert path.read_bytes() == before  # not truncated, not half-written
    assert [p.name for p in path.parent.iterdir()] == ["Atomic.md"]  # no temp debris


def test_lone_surrogate_cannot_truncate_a_note(vault_path):
    """One bad character used to abort write_text after it had truncated the file."""
    from sift.vault.notes import Note, load_note, save_note
    from sift.vault.schema import Frontmatter

    note = Note(meta=Frontmatter(id="tech-s", type="technique", title="Surrogate"), body="ok")
    path = save_note(vault_path, note)
    note.body = "x \ud83d y"
    assert save_note(vault_path, note) == path
    assert path.stat().st_size > 0
    loaded = load_note(path)
    assert loaded.body.strip() == note.body.strip() == "x ? y"  # disk == what gets indexed


# --- non-notes: skipped once, logged, never printed -----------------------------


def test_non_notes_are_skipped_once_and_never_printed(vault_path, capfd, vault_log):
    from sift.vault.notes import Note, iter_notes, save_note
    from sift.vault.schema import Frontmatter

    save_note(vault_path, Note(meta=Frontmatter(id="good", type="report", title="Good"), body="b"))
    (vault_path / "report" / "empty.md").write_bytes(b"")  # what a crashed write left
    _write_raw(vault_path / "START HERE.md", "# Welcome\n\nno frontmatter here\n")
    _write_raw(vault_path / "report" / "broken.md", "---\nid: [unclosed\n---\n\nx\n")

    for _ in range(2):  # list_notes / stats rescan the vault on every call
        assert [n.meta.id for n in iter_notes(vault_path)] == ["good"]

    assert capfd.readouterr().out == ""  # stdout is the MCP JSON-RPC channel
    skipped = _warnings(vault_log, "skipping unreadable note")
    assert len(skipped) == 3, skipped  # one line per file, not per pass
    assert any("empty.md: empty file" in m for m in skipped)
    assert any("START HERE.md: no frontmatter" in m and "ingest local" in m for m in skipped)
    assert all("\n" not in m for m in skipped)
    assert not any("unclosed" in m for m in skipped)  # never echoes file content


def test_a_changed_bad_file_is_reported_again(vault_path, vault_log):
    import os

    from sift.vault.notes import iter_notes

    p = vault_path / "report" / "empty.md"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    list(iter_notes(vault_path))
    p.write_text("still no frontmatter\n", encoding="utf-8")
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 5))
    list(iter_notes(vault_path))
    assert len(_warnings(vault_log, "empty.md")) == 2


# --- mtime is taken before the read ----------------------------------------------


def test_load_note_records_the_mtime_from_before_the_read(vault_path, monkeypatch):
    """An edit landing while a note is read must leave the recorded mtime older than
    the file, so the next incremental reindex picks it up."""
    import os
    import time

    from sift.vault import notes as notes_mod
    from sift.vault.notes import Note, load_note, save_note
    from sift.vault.schema import Frontmatter

    note = Note(meta=Frontmatter(id="tech-m", type="technique", title="Mtime"), body="one")
    path = save_note(vault_path, note)
    assert note.mtime == path.stat().st_mtime  # set by the write too
    assert load_note(path).mtime == path.stat().st_mtime

    real_read = notes_mod.read_note_text

    def edit_during_read(p):
        text = real_read(p)
        later = time.time() + 20
        os.utime(p, (later, later))  # the user saves in Obsidian mid-read
        return text

    monkeypatch.setattr(notes_mod, "read_note_text", edit_during_read)
    loaded = load_note(path)
    assert loaded.mtime < path.stat().st_mtime
