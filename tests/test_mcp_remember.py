"""`remember` is an upsert, never a fork (K2).

Calling remember again with a title it already used used to write `Title (2).md`
under a new id, leave the old chunks indexed, and - because ids were cut to 80
characters with the timestamp at the end - give both notes one slug, so
get_note(slug) returned whichever file sorted first. Now: ids fit in 80 characters
(the id is its own slug), a repeated title appends to the note remember already
wrote (same type and program), `note_id` appends explicitly, and ingested notes are
never merged into or appended to.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

DIM = 32
TITLE = "Acme SSO state parameter is not bound to the session"


class FakeEmbedder:
    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts, *, kind="passage", batch_size=32):
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str):
        return self._vec(text)

    def embed_one(self, text: str):
        return self._vec(text)


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_DIM", str(DIM))
    from sift.config import get_settings

    get_settings.cache_clear()
    fake = FakeEmbedder()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: fake)
    yield


def _md_files(vault: Path) -> list[Path]:
    return sorted(p for p in vault.rglob("*.md") if ".trash" not in p.parts)


def _rows_for(note_id: str) -> int:
    from sift.index.store import Store

    tbl = Store().table()
    return tbl.count_rows(f"note_id = '{note_id}'")


def test_long_title_id_is_its_own_slug(vault_path):
    from sift.mcp_server import remember

    res = remember(title="Prototype pollution " * 12, body_md="gadget chain notes")
    assert len(res["note_id"]) <= 80
    assert res["slug"] == res["note_id"]
    assert re.fullmatch(r"find-[a-z0-9-]+-\d{17}", res["note_id"])


def test_long_titles_sharing_a_prefix_get_distinct_slugs(vault_path):
    from sift.mcp_server import get_note, remember

    stem = "Blind SSRF through the PDF renderer's font loader on the reporting service "
    a = remember(title=stem + "via @font-face", body_md="font face body")
    b = remember(title=stem + "via SVG image href", body_md="svg href body")
    assert a["note_id"] != b["note_id"] and a["slug"] != b["slug"]
    assert "font face body" in get_note(a["slug"])["body"]
    assert "svg href body" in get_note(b["slug"])["body"]


def test_same_title_twice_appends_to_one_note(vault_path):
    from sift.mcp_server import get_note, remember, search_memory

    first = remember(title=TITLE, body_md="Not exploitable: state is checked server-side.")
    second = remember(title=TITLE, body_md="Correction: exploitable when the IdP retries.")

    assert first["updated"] is False
    assert second["updated"] is True
    assert second["merged_into"] == first["note_id"]
    assert second["note_id"] == first["note_id"]
    assert second["path"] == first["path"]

    files = _md_files(vault_path)
    assert files == [Path(first["path"])], files  # no 'Title (2).md'
    body = get_note(first["note_id"])["body"]
    assert "Not exploitable" in body and "exploitable when the IdP retries" in body
    assert "## Update" in body

    # One note in the index, carrying the new text; the old rows were replaced.
    hits = search_memory(query="IdP retries exploitable state", k=5)["results"]
    ids = [h["note_id"] for h in hits]
    assert ids.count(first["note_id"]) == 1, ids
    assert _rows_for(first["note_id"]) == second["chunks_indexed"] > 0


def test_note_id_appends_explicitly_and_keeps_the_title(vault_path):
    from sift.mcp_server import get_note, remember

    first = remember(title=TITLE, body_md="B1 original analysis", tags=["sso"], cwe=["CWE-352"])
    out = remember(
        title="ignored for an append",
        body_md="B2 follow-up",
        note_id=first["note_id"],
        tags=["oauth"],
        cwe=["352", "CWE-601"],
    )
    assert out["updated"] is True and "merged_into" not in out
    note = get_note(first["note_id"])
    assert note["frontmatter"]["title"] == TITLE
    assert "B1 original analysis" in note["body"] and "B2 follow-up" in note["body"]
    assert note["frontmatter"]["tags"] == ["sso", "oauth"]
    assert note["frontmatter"]["cwe"] == ["CWE-352", "CWE-601"]  # normalised, not duplicated
    assert len(_md_files(vault_path)) == 1


def test_same_title_other_program_or_type_is_a_separate_note(vault_path):
    from sift.mcp_server import remember

    a = remember(title="IDOR in /api/users", body_md="acme body", program="acme")
    b = remember(title="IDOR in /api/users", body_md="globex body", program="globex")
    c = remember(title="IDOR in /api/users", body_md="technique body", type="technique")
    assert len({a["note_id"], b["note_id"], c["note_id"]}) == 3
    assert not any(r.get("merged_into") for r in (a, b, c))
    assert "acme body" in Path(a["path"]).read_text(encoding="utf-8")
    assert "globex body" not in Path(a["path"]).read_text(encoding="utf-8")

    # Positive control: same program again does merge.
    d = remember(title="idor in /API/users", body_md="more acme", program="ACME")
    assert d["merged_into"] == a["note_id"]


def _ingested(vault: Path, title: str = TITLE) -> Path:
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    meta = Frontmatter(
        id="h1-123456",
        type="finding",
        title=title,
        source="hackerone-public",
        url="https://hackerone.com/reports/123456",
    )
    return save_note(vault, Note(meta=meta, body="Disclosed report body."))


def test_never_merges_into_or_appends_to_an_ingested_note(vault_path):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import remember

    path = _ingested(vault_path)
    before = path.read_bytes()

    res = remember(title=TITLE, body_md="my own take")
    assert res["updated"] is False and res["note_id"] != "h1-123456"

    with pytest.raises(ToolError, match="re-ingest"):
        remember(title=TITLE, body_md="appended", note_id="h1-123456")
    assert path.read_bytes() == before


def test_marks_the_note_user_authored_even_with_a_caller_source(vault_path):
    from sift.mcp_server import remember
    from sift.pipeline import get_note_by_slug
    from sift.quality import AUTHORED_VIA, is_user_authored

    res = remember(
        title="My report summary", body_md="body", type="report", source="hackerone-public"
    )
    note = get_note_by_slug(res["note_id"])
    assert note.meta.source == "hackerone-public"
    assert note.meta.extra[AUTHORED_VIA] == "sift-remember"
    assert is_user_authored(note.meta)


def test_created_defaults_to_today_and_takes_an_iso_date(vault_path):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import remember
    from sift.pipeline import get_note_by_slug

    a = remember(title="Dated today", body_md="x")
    assert get_note_by_slug(a["note_id"]).meta.created == datetime.now(UTC).date()
    b = remember(title="Old disclosure", body_md="x", type="report", created="2019-03-04")
    assert get_note_by_slug(b["note_id"]).meta.created.isoformat() == "2019-03-04"

    n = len(_md_files(vault_path))
    with pytest.raises(ToolError, match="ISO date"):
        remember(title="Bad date", body_md="x", created="last tuesday")
    assert len(_md_files(vault_path)) == n


@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": "", "body_md": "x"},
        {"title": "t", "body_md": "   "},
        {"title": "t", "body_md": "x", "type": "findings"},
        {"title": "t", "body_md": "x", "note_id": "no-such-note"},
    ],
)
def test_bad_arguments_raise_tool_errors(vault_path, kwargs):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import remember

    with pytest.raises(ToolError):
        remember(**kwargs)
    assert _md_files(vault_path) == []


def test_appends_to_the_newest_of_old_duplicates_and_leaves_the_rest(vault_path):
    """Before this fix, repeats forked 'Title.md' and 'Title (2).md'. The newest one
    gets the update; the other is left alone (never deleted on a title match)."""
    from sift.mcp_server import remember
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    paths = []
    for i, ts in enumerate(("20260901120000000", "20260902120000000")):
        meta = Frontmatter(
            id=f"find-old-title-{ts}", type="finding", title=TITLE, source="sift-remember"
        )
        paths.append(save_note(vault_path, Note(meta=meta, body=f"old body {i}")))
    assert paths[1].name.endswith("(2).md")
    os.utime(paths[0], (2_000_000_000, 2_000_000_000))  # the '(1)' file is newest
    os.utime(paths[1], (1_900_000_000, 1_900_000_000))
    older = paths[1].read_bytes()

    out = remember(title=TITLE, body_md="the correction")
    assert out["merged_into"] == "find-old-title-20260901120000000"
    assert "the correction" in paths[0].read_text(encoding="utf-8")
    assert paths[1].read_bytes() == older
    assert len(_md_files(vault_path)) == 2


def test_appends_to_one_file_when_old_copies_share_an_id(vault_path):
    """Two files carrying one id (an Obsidian copy): the id alone is ambiguous, so the
    newest file is updated directly - no third file, the copy untouched."""
    from sift.mcp_server import remember
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    meta = Frontmatter(
        id="find-copied-20260901120000000", type="finding", title=TITLE, source="sift-remember"
    )
    first = save_note(vault_path, Note(meta=meta, body="original"))
    copy = first.with_name(f"{first.stem} (copy).md")
    copy.write_bytes(first.read_bytes())
    os.utime(first, (1_900_000_000, 1_900_000_000))
    os.utime(copy, (2_000_000_000, 2_000_000_000))  # the copy is newest
    before = first.read_bytes()

    out = remember(title=TITLE, body_md="the update")
    assert out["merged_into"] == "find-copied-20260901120000000"
    assert Path(out["path"]) == copy
    assert "the update" in copy.read_text(encoding="utf-8")
    assert first.read_bytes() == before
    assert len(_md_files(vault_path)) == 2


def test_concurrent_same_title_calls_end_in_one_note(vault_path):
    """Lookup-then-write runs under the vault lock: the second call sees the first's
    note and appends, instead of both creating (or one overwriting the other)."""
    from sift.mcp_server import remember

    errors: list[BaseException] = []
    results: list[dict] = []
    barrier = threading.Barrier(2)

    def run(text: str) -> None:
        try:
            barrier.wait(timeout=10)
            results.append(remember(title=TITLE, body_md=text, program="acme"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(t,)) for t in ("alpha text", "bravo text")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors

    files = _md_files(vault_path)
    assert len(files) == 1, files
    text = files[0].read_text(encoding="utf-8")
    assert "alpha text" in text and "bravo text" in text
    assert len({r["note_id"] for r in results}) == 1


def test_ids_minted_in_one_millisecond_never_collide(vault_path, monkeypatch):
    when = datetime(2026, 10, 1, 9, 30, 0, 42000, tzinfo=UTC)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return when if tz is None else when.astimezone(tz)

    monkeypatch.setattr("sift.mcp_server.datetime", Frozen)
    from sift.mcp_server import remember

    a = remember(title="Same instant", body_md="first", program="one")
    b = remember(title="Same instant", body_md="second", program="two")
    assert a["note_id"] != b["note_id"]
    assert a["note_id"].endswith("20261001093000042")
    assert b["note_id"].endswith("20261001093000043")
    assert "first" in Path(a["path"]).read_text(encoding="utf-8")
    assert "second" in Path(b["path"]).read_text(encoding="utf-8")
