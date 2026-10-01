"""capture_idea / resolve_idea round-trip.

The point of the status loop is that dead ends get recorded. These tests pin the
behaviour that makes that work: status lives in `extra` (filterable) and is mirrored
into a `status/` tag, and resolving replaces the old status rather than accumulating.
Resolving writes back into the file the idea lives in (even one renamed in
Obsidian), refuses a reference it would have to guess, and never touches a note
that is not a captured idea.
"""

from __future__ import annotations

import hashlib
import re
import threading
from datetime import UTC, datetime

import pytest

IDEA = "Try JWT alg confusion on the SSO callback - it accepts both RS256 and HS256"
WHY = "The callback echoes the kid header, suggesting a key lookup we may control."
DIM = 32


class FakeEmbedder:
    """Hashed bag-of-words -> unit vector; offline and instant."""

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


def _capture(**over):
    from sift.mcp_server import capture_idea

    kwargs = dict(idea=IDEA, reasoning=WHY, target="acme", tags=["jwt"], cwe=["CWE-347"])
    kwargs.update(over)
    return capture_idea(**kwargs)


def _files_with_id(vault, note_id: str) -> list:
    out = []
    for p in vault.rglob("*.md"):
        if f"id: {note_id}\n" in p.read_text(encoding="utf-8"):
            out.append(p)
    return out


def _frozen_clock(monkeypatch, when: datetime) -> None:
    """Every `datetime.now()` in mcp_server returns `when` (same millisecond)."""

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return when if tz is None else when.astimezone(tz)

    monkeypatch.setattr("sift.mcp_server.datetime", Frozen)


def test_capture_starts_as_hypothesis(vault_path):
    res = _capture()
    assert res["saved"] is True
    assert res["status"] == "hypothesis"

    from sift.pipeline import get_note_by_slug

    note = get_note_by_slug(res["slug"])
    assert note.meta.extra["status"] == "hypothesis"
    assert "status/hypothesis" in note.meta.tags
    assert note.meta.program == "acme"
    assert note.meta.cwe == ["CWE-347"]
    assert WHY in note.body
    assert res["note_id"] == note.meta.id


def test_capture_is_dated_and_marked_as_user_authored(vault_path):
    res = _capture()
    from sift.pipeline import get_note_by_slug
    from sift.quality import AUTHORED_VIA, is_user_authored

    note = get_note_by_slug(res["note_id"])
    assert note.meta.created == datetime.now(UTC).date()
    assert note.meta.extra[AUTHORED_VIA] == "sift-capture-idea"
    assert is_user_authored(note.meta)


def test_long_idea_id_is_its_own_slug(vault_path):
    res = _capture(idea="Probe the " + "very " * 60 + "long parameter list for mass assignment")
    assert len(res["note_id"]) <= 80
    assert res["slug"] == res["note_id"]
    assert re.fullmatch(r"idea-[a-z0-9-]+-\d{17}", res["note_id"])


def test_resolve_records_a_dead_end(vault_path):
    res = _capture()
    from sift.mcp_server import resolve_idea

    out = resolve_idea(
        slug=res["slug"],
        status="failed",
        notes="RS256 validated properly; kid is not attacker-controlled.",
    )
    assert out["updated"] is True
    assert out["note_id"] == res["note_id"]

    from sift.pipeline import get_note_by_slug

    note = get_note_by_slug(res["slug"])
    assert note.meta.extra["status"] == "failed"
    assert "RS256 validated properly" in note.body
    # The whole point: a failure is retrievable, not lost.
    assert "**Status:** failed" in note.body


def test_status_tag_is_replaced_not_accumulated(vault_path):
    res = _capture()
    from sift.mcp_server import resolve_idea

    resolve_idea(slug=res["slug"], status="partial", notes="worked only when logged out")
    resolve_idea(slug=res["slug"], status="worked", notes="chained with cookie tossing")

    from sift.pipeline import get_note_by_slug

    note = get_note_by_slug(res["slug"])
    status_tags = [t for t in note.meta.tags if t.startswith("status/")]
    assert status_tags == ["status/worked"], status_tags
    assert note.meta.extra["status"] == "worked"


@pytest.mark.parametrize("bad", ["hypothesis", "done", ""])
def test_rejects_bad_status(vault_path, bad):
    res = _capture()
    from pathlib import Path

    from fastmcp.exceptions import ToolError

    from sift.mcp_server import resolve_idea

    path = Path(res["path"])
    before = path.read_bytes()
    with pytest.raises(ToolError):
        resolve_idea(slug=res["slug"], status=bad, notes="x")
    assert path.read_bytes() == before


def test_resolve_unknown_slug_is_an_error(vault_path):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import resolve_idea

    with pytest.raises(ToolError):
        resolve_idea(slug="nope-does-not-exist", status="failed", notes="x")


def test_resolve_writes_back_to_a_renamed_file(vault_path):
    """The user renamed the idea in Obsidian: the outcome lands in that file, and no
    second file appears at the old title path."""
    res = _capture()
    from pathlib import Path

    old = Path(res["path"])
    renamed = old.with_name("JWT confusion idea.md")
    old.rename(renamed)

    from sift.mcp_server import list_notes, resolve_idea

    out = resolve_idea(slug=res["note_id"], status="failed", notes="RS256 enforced")
    assert Path(out["path"]) == renamed

    carriers = _files_with_id(vault_path, res["note_id"])
    assert carriers == [renamed], carriers
    assert "**Status:** failed" in renamed.read_text(encoding="utf-8")
    assert not old.exists()
    assert list_notes(status="hypothesis")["count"] == 0
    assert list_notes(status="failed")["count"] == 1  # positive control


def test_resolve_refuses_a_note_that_is_not_an_idea(vault_path):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import remember, resolve_idea

    res = remember(
        title="Cookie tossing primer", body_md="Subdomain sets a cookie.", type="technique"
    )
    from pathlib import Path

    before = Path(res["path"]).read_bytes()
    with pytest.raises(ToolError, match="not a captured idea"):
        resolve_idea(slug=res["note_id"], status="worked", notes="x")
    assert Path(res["path"]).read_bytes() == before


def test_two_long_ideas_in_one_millisecond_resolve_independently(vault_path, monkeypatch):
    """Ids used to cut the idea at 60 chars, so two long ideas captured together got
    one slug and resolve_idea edited whichever sorted first."""
    _frozen_clock(monkeypatch, datetime(2026, 10, 1, 12, 0, 0, 123000, tzinfo=UTC))
    stem = "Abuse the export endpoint's predictable job ids to download other tenants' "
    a = _capture(idea=stem + "invoices")
    b = _capture(idea=stem + "payroll files")
    assert a["note_id"] != b["note_id"]
    assert a["slug"] != b["slug"]

    from sift.mcp_server import resolve_idea
    from sift.pipeline import get_note_by_slug

    resolve_idea(slug=a["slug"], status="failed", notes="ids are random")
    assert get_note_by_slug(a["note_id"]).meta.extra["status"] == "failed"
    assert get_note_by_slug(b["note_id"]).meta.extra["status"] == "hypothesis"


def test_same_idea_twice_in_one_millisecond_never_overwrites(vault_path, monkeypatch):
    """Same text, same clock: the second capture used to mint the first one's id and
    overwrite it (a save is an upsert by id)."""
    _frozen_clock(monkeypatch, datetime(2026, 10, 1, 12, 0, 0, 5000, tzinfo=UTC))
    a = _capture()
    b = _capture()
    assert a["note_id"] != b["note_id"]

    from sift.mcp_server import resolve_idea
    from sift.pipeline import get_note_by_slug

    resolve_idea(slug=a["note_id"], status="failed", notes="dead end")
    assert get_note_by_slug(a["note_id"]).meta.extra["status"] == "failed"
    assert get_note_by_slug(b["note_id"]).meta.extra["status"] == "hypothesis"


def test_resolve_refuses_an_ambiguous_slug(vault_path):
    """Two legacy ideas whose 81+ char ids share the 80-char legacy slug: resolving
    by that slug must refuse rather than edit a guessed note."""
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import resolve_idea
    from sift.vault.notes import Note, legacy_slug, save_note
    from sift.vault.schema import Frontmatter

    prefix = "idea-" + "x" * 76  # 81 chars: the legacy slug cuts both ids here
    paths = []
    for suffix, title in (("-one", "First legacy idea"), ("-two", "Second legacy idea")):
        meta = Frontmatter(
            id=prefix + suffix,
            type="technique",
            title=title,
            source="sift-capture-idea",
            tags=["idea", "status/hypothesis"],
            extra={"status": "hypothesis"},
        )
        paths.append(save_note(vault_path, Note(meta=meta, body="**Status:** hypothesis\n\nidea")))
    shared = legacy_slug(prefix + "-one")
    assert shared == legacy_slug(prefix + "-two")

    before = [p.read_bytes() for p in paths]
    with pytest.raises(ToolError, match="ambiguous"):
        resolve_idea(slug=shared, status="failed", notes="x")
    assert [p.read_bytes() for p in paths] == before

    # Positive control: the exact id resolves the one note.
    out = resolve_idea(slug=prefix + "-two", status="failed", notes="x")
    assert out["note_id"] == prefix + "-two"
    assert paths[0].read_bytes() == before[0]


def test_resolve_never_rebuilds_the_fts_index(vault_path, monkeypatch):
    res = _capture()
    from sift.index.store import Store

    def boom(*a, **k):
        raise AssertionError("MCP tools must not rebuild or compact the index")

    monkeypatch.setattr(Store, "ensure_fts", boom)
    monkeypatch.setattr(Store, "optimize", boom)

    from sift.mcp_server import resolve_idea

    assert resolve_idea(slug=res["note_id"], status="worked", notes="ok")["updated"] is True


def test_concurrent_resolves_keep_both_outcomes(vault_path):
    """The lookup-mutate-save runs under the vault write lock, so two resolves of one
    idea serialise instead of one overwriting the other's outcome."""
    res = _capture()
    from sift.mcp_server import resolve_idea

    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def run(status: str, text: str) -> None:
        try:
            barrier.wait(timeout=10)
            resolve_idea(slug=res["note_id"], status=status, notes=text)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=("failed", "first outcome text")),
        threading.Thread(target=run, args=("partial", "second outcome text")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors

    from sift.pipeline import get_note_by_slug

    body = get_note_by_slug(res["note_id"]).body
    assert "first outcome text" in body
    assert "second outcome text" in body
    assert len(_files_with_id(vault_path, res["note_id"])) == 1
