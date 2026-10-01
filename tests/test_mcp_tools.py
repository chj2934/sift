"""MCP tools: lookups, listings, corrections, forgetting and single-URL capture.

Everything runs against the temp vault/db from conftest with a fake embedder, so it
is offline. Tools are called directly (FastMCP's decorator returns the function) and,
for the protocol-level behaviour (annotations, isError), through an in-memory client.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sys
import threading
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

DIM = 32


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


# --------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------


def _save(vault: Path, note_id: str, title: str, body: str = "body", **meta) -> Path:
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    meta.setdefault("type", "finding")
    fm = Frontmatter(id=note_id, title=title, **meta)
    return save_note(vault, Note(meta=fm, body=body), stamp=False)


def _age(path: Path, epoch: float) -> None:
    """Backdate a file. A same-size touch changes no folder mtime, so tell the catalog
    (Obsidian's edits are seen by the next stat walk, a second later)."""
    from sift.config import get_settings
    from sift.vault.catalog import get_catalog

    os.utime(path, (epoch, epoch))
    get_catalog(get_settings().resolved_vault()).invalidate([path])


def _indexed_paths(note_id: str) -> set[str]:
    from sift.index.store import Store, norm_path

    return {norm_path(p) for nid, p, _m in Store().indexed_files() if nid == note_id}


def _live_md(vault: Path) -> list[Path]:
    return sorted(p for p in vault.rglob("*.md") if ".trash" not in p.parts)


def _client_call(tool: str, args: dict):
    from fastmcp import Client

    from sift.mcp_server import mcp

    async def go():
        async with Client(mcp) as c:
            return await c.call_tool(tool, args, raise_on_error=False)

    return asyncio.run(go())


# --------------------------------------------------------------------------------
# get_note
# --------------------------------------------------------------------------------

LONG = "finding-" + "a" * 80  # 88 chars: two such ids share the 80-char legacy slug


def test_get_note_exact_id_beats_a_colliding_legacy_slug(vault_path):
    from sift.mcp_server import get_note
    from sift.vault.notes import legacy_slug

    _save(vault_path, LONG + "-one", "Colliding one", "one body")
    _save(vault_path, LONG + "-two", "Colliding two", "two body")

    assert "two body" in get_note(LONG + "-two")["body"]
    assert "one body" in get_note(LONG + "-one")["body"]
    assert get_note(LONG + "-two")["note_id"] == LONG + "-two"

    shared = legacy_slug(LONG + "-one")
    out = get_note(shared)
    assert out["error"] == "ambiguous slug"
    assert {c["note_id"] for c in out["candidates"]} == {LONG + "-one", LONG + "-two"}


def test_get_note_by_id_ignores_a_title_that_slugs_to_it(vault_path):
    """Note B's title slugifies to note A's id: get_note(A's id) must return A."""
    from sift.mcp_server import get_note

    _save(vault_path, "ssrf-in-pdf-export", "Real note A", "A body")
    _save(vault_path, "b-123", "ssrf in pdf export", "B body")
    assert "A body" in get_note("ssrf-in-pdf-export")["body"]


def test_get_note_accepts_vault_paths_only(vault_path, tmp_path):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import get_note

    p = _save(vault_path, "path-note", "Path note", "found by path")
    assert "found by path" in get_note(str(p))["body"]

    outside = tmp_path / "outside.md"
    outside.write_text("---\nid: path-note-x\ntype: finding\ntitle: x\n---\n\nsecret\n", "utf-8")
    with pytest.raises(ToolError):
        get_note(str(outside))
    with pytest.raises(ToolError):
        get_note("no-such-note")


def test_get_note_section_and_max_chars(vault_path):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import get_note

    body = (
        "# Overview\n\nintro text\n\n## Repro\n\nstep one\n\n```bash\n# not a heading\ncurl x\n```\n"
        "\n### Detail\n\ndeep\n\n## Impact\n\nimpact text " + "z" * 500
    )
    _save(vault_path, "sections", "Sections", body)
    part = get_note("sections", section="repro")
    assert part["body"].startswith("## Repro")
    assert "# not a heading" in part["body"] and "deep" in part["body"]
    assert "impact text" not in part["body"]
    with pytest.raises(ToolError, match="headings"):
        get_note("sections", section="not a heading")

    cut = get_note("sections", max_chars=200)
    assert cut["truncated"] is True and len(cut["body"]) == 200
    assert cut["body_chars"] == len(get_note("sections")["body"])


# --------------------------------------------------------------------------------
# list_notes / stats
# --------------------------------------------------------------------------------


def test_list_notes_puts_undated_agent_notes_by_when_they_were_written(vault_path):
    from sift.mcp_server import list_notes

    old = _save(
        vault_path, "CVE-2019-0001", "Old CVE", type="cve", source="nvd", created=date(2019, 1, 1)
    )
    _age(old, datetime(2026, 9, 1, tzinfo=UTC).timestamp())  # ingested recently
    mine = _save(vault_path, "find-mine-20260930120000000", "My finding", source="sift-remember")
    _age(mine, datetime(2026, 9, 30, tzinfo=UTC).timestamp())

    first = list_notes(limit=1)
    assert first["count"] == 2
    assert first["notes"][0]["note_id"] == "find-mine-20260930120000000"
    assert "updated" in first["notes"][0]


def test_list_notes_orders_undated_notes_by_write_time_and_sorts(vault_path):
    from sift.mcp_server import list_notes

    a = _save(vault_path, "a", "Undated A")
    b = _save(vault_path, "b", "Undated B")
    dated = _save(vault_path, "c", "Dated 2019", created=date(2019, 5, 5))
    _age(a, datetime(2018, 1, 1, 10, tzinfo=UTC).timestamp())
    _age(b, datetime(2018, 1, 1, 11, tzinfo=UTC).timestamp())
    _age(dated, datetime(2017, 1, 1, tzinfo=UTC).timestamp())

    ids = [n["note_id"] for n in list_notes()["notes"]]
    assert ids == ["c", "b", "a"]  # 2019 disclosure, then same-day undated by time
    recent = [n["note_id"] for n in list_notes(sort="recent")["notes"]]
    assert recent == ["b", "a", "c"]


def test_list_notes_filters_and_paging(vault_path):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import list_notes

    _save(
        vault_path,
        "i1",
        "Idea one",
        type="technique",
        source="sift-capture-idea",
        program="Acme",
        tags=["idea", "JWT"],
        extra={"status": "hypothesis"},
    )
    _save(
        vault_path,
        "i2",
        "Idea two",
        type="technique",
        source="sift-capture-idea",
        program="acme",
        tags=["idea"],
        extra={"status": "failed"},
    )
    _save(
        vault_path,
        "r1",
        "Report",
        type="report",
        source="hackerone-public",
        created=date(2026, 9, 1),
    )

    assert {n["note_id"] for n in list_notes(type="technique")["notes"]} == {"i1", "i2"}
    assert [n["note_id"] for n in list_notes(status="hypothesis")["notes"]] == ["i1"]
    assert list_notes(program="ACME")["count"] == 2
    assert [n["note_id"] for n in list_notes(tag="jwt")["notes"]] == ["i1"]
    assert list_notes(source="sift-capture-idea")["count"] == 2
    assert list_notes(since="2026-09-01", sort="created", type="report")["count"] == 1
    assert list_notes(since="2026-09-02", type="report")["count"] == 0

    page = list_notes(limit=1, offset=1)
    assert page["count"] == 3 and len(page["notes"]) == 1 and page["offset"] == 1

    for bad in (
        {"type": "findings"},
        {"limit": -1},
        {"limit": 0},
        {"status": "done"},
        {"since": "yesterday"},
        {"sort": "oldest"},
        {"offset": -2},
    ):
        with pytest.raises(ToolError):
            list_notes(**bad)


def test_list_notes_does_not_reparse_an_unchanged_vault(vault_path, monkeypatch):
    from sift.mcp_server import list_notes
    from sift.vault import catalog

    for i in range(5):
        p = _save(vault_path, f"n{i}", f"Note {i}")
        _age(p, 1_700_000_000 + i)  # not "racy": written long before the read
    assert list_notes()["count"] == 5

    calls = []
    real = catalog.load_note
    monkeypatch.setattr(catalog, "load_note", lambda p: calls.append(p) or real(p))
    assert list_notes()["count"] == 5
    assert calls == []

    # Positive control: an edit is picked up (one re-parse, not five).
    p = _save(vault_path, "n0", "Note zero renamed")
    _age(p, 1_700_000_100)
    titles = {n["title"] for n in list_notes()["notes"]}
    assert "Note zero renamed" in titles
    assert len(calls) == 1


def test_stats_counts_from_the_catalog_and_reports_skipped_files(vault_path, capfd):
    from sift.mcp_server import get_note, list_notes, remember, stats

    remember(title="Indexed finding", body_md="some text")
    _save(vault_path, "CVE-2024-1", "A cve", type="cve", source="nvd")
    (vault_path / "report").mkdir(exist_ok=True)
    (vault_path / "report" / "empty.md").write_bytes(b"")
    (vault_path / "START HERE.md").write_text("Welcome, no frontmatter.\n", encoding="utf-8")

    out = stats()
    assert out["notes_by_type"] == {"finding": 1, "cve": 1}
    assert out["total_notes"] == 2
    assert out["index_chunks"] > 0
    assert out["skipped_notes"] == 2
    assert sorted(out["skipped_paths"]) == ["START HERE.md", "report/empty.md"]
    assert out["duplicate_id_groups"] == 0
    list_notes()
    with pytest.raises(Exception):  # noqa: B017 - ToolError; only stdout matters here
        get_note("nothing-here")
    stats()

    # K1/K4: nothing reaches stdout (the JSON-RPC wire), whatever was skipped.
    assert capfd.readouterr().out == ""


# --------------------------------------------------------------------------------
# search_memory
# --------------------------------------------------------------------------------


def test_search_memory_returns_ids_dates_and_warnings(vault_path):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import remember, search_memory

    res = remember(title="GraphQL batching bypasses rate limit", body_md="alias batching on login")
    out = search_memory(query="graphql alias batching", k=3)
    hit = next(h for h in out["results"] if h["note_id"] == res["note_id"])
    assert hit["slug"] == res["note_id"]
    assert hit["created"] == datetime.now(UTC).date().isoformat()
    assert isinstance(out["warnings"], list)
    assert "note_id" in out["hint"]

    with pytest.raises(ToolError):
        search_memory(query="x", type="findings")
    with pytest.raises(ToolError):
        search_memory(query="x", k=0)
    with pytest.raises(ToolError):
        search_memory(query="x", min_quality=101)
    with pytest.raises(ToolError, match="cwe"):
        search_memory(query="x", cwe="79 OR 1=1")


# --------------------------------------------------------------------------------
# update_note
# --------------------------------------------------------------------------------


def test_update_note_rewrites_in_place_and_replaces_rows(vault_path):
    from sift.mcp_server import get_note, remember, search_memory, update_note

    res = remember(title="Host header poisoning", body_md="wrong host: staging.acme.test")
    out = update_note(res["note_id"], body_md="correct host: login.acme.test", tags_add=["fixed"])
    assert out["note_id"] == res["note_id"] and out["path"] == res["path"]
    assert set(out["changed"]) >= {"body", "tags"}
    assert _live_md(vault_path) == [Path(res["path"])]
    note = get_note(res["note_id"])
    assert "login.acme.test" in note["body"] and "staging" not in note["body"]
    assert "fixed" in note["frontmatter"]["tags"]

    hits = search_memory(query="login acme host", k=5)["results"]
    assert [h["note_id"] for h in hits].count(res["note_id"]) == 1
    assert all("staging" not in h["excerpt"] for h in hits)


def test_update_note_title_change_leaves_one_file(vault_path):
    from sift.mcp_server import get_note, remember, update_note

    res = remember(title="Old title for the bug", body_md="text")
    out = update_note(res["note_id"], title="New title for the bug")
    files = _live_md(vault_path)
    assert len(files) == 1 and files[0].name == "New title for the bug.md"
    assert out["renamed_from"] == res["path"]
    assert get_note(res["note_id"])["frontmatter"]["title"] == "New title for the bug"
    from sift.index.store import norm_path

    assert _indexed_paths(res["note_id"]) == {norm_path(files[0])}


def test_update_note_case_only_title_change_keeps_the_file(vault_path):
    from sift.mcp_server import get_note, remember, update_note

    res = remember(title="cookie tossing", body_md="text")
    update_note(res["note_id"], title="Cookie Tossing")
    files = _live_md(vault_path)
    assert len(files) == 1
    assert get_note(res["note_id"])["frontmatter"]["title"] == "Cookie Tossing"
    assert "text" in files[0].read_text(encoding="utf-8")


def test_update_note_emptied_body_clears_its_rows(vault_path):
    from sift.index.store import Store
    from sift.mcp_server import remember, update_note

    res = remember(title="To be emptied", body_md="content that will go")
    assert Store().table().count_rows(f"note_id = '{res['note_id']}'") > 0
    out = update_note(res["note_id"], body_md="")
    assert out["chunks_indexed"] == 0
    assert Store().table().count_rows(f"note_id = '{res['note_id']}'") == 0


def test_update_note_guards(vault_path):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import get_note, update_note

    p = _save(vault_path, "CVE-2024-2", "Ingested CVE", "nvd text", type="cve", source="nvd")
    before = p.read_bytes()
    with pytest.raises(ToolError, match="force"):
        update_note("CVE-2024-2", append_md="my note")
    assert p.read_bytes() == before
    with pytest.raises(ToolError, match="nothing to change"):
        update_note("CVE-2024-2")
    with pytest.raises(ToolError):
        update_note("missing-id", body_md="x")

    out = update_note("CVE-2024-2", tags_add=["triaged", "x"], tags_remove=["x"], force=True)
    assert out["updated"] is True
    assert get_note("CVE-2024-2")["frontmatter"]["tags"] == ["triaged"]

    _save(vault_path, LONG + "-one", "One", source="manual")
    _save(vault_path, LONG + "-two", "Two", source="manual")
    from sift.vault.notes import legacy_slug

    with pytest.raises(ToolError, match="ambiguous"):
        update_note(legacy_slug(LONG + "-one"), body_md="x")


# --------------------------------------------------------------------------------
# forget_note
# --------------------------------------------------------------------------------


def test_forget_note_trashes_unindexes_and_tombstones(vault_path):
    from fastmcp.exceptions import ToolError

    from sift.index.store import Store
    from sift.mcp_server import forget_note, list_notes, remember
    from sift.tombstones import load_tombstones
    from sift.vault.notes import iter_notes

    res = remember(title="Dead lead", body_md="nothing here", url="https://example.com/post")
    keep = remember(title="Keeper", body_md="still useful")
    assert Store().table().count_rows(f"note_id = '{res['note_id']}'") > 0  # positive control

    out = forget_note(res["note_id"], reason="wrong target")
    assert out["forgotten"] is True and "errors" not in out
    assert out["index_rows_deleted"] > 0
    trashed = Path(out["trashed"][0])
    assert ".trash" in trashed.parts and trashed.exists()
    assert "wrong target" in trashed.read_text(encoding="utf-8")
    assert not Path(res["path"]).exists()

    assert Store().table().count_rows(f"note_id = '{res['note_id']}'") == 0
    assert Store().table().count_rows(f"note_id = '{keep['note_id']}'") > 0
    assert [n["note_id"] for n in list_notes()["notes"]] == [keep["note_id"]]
    assert res["note_id"] not in {n.meta.id for n in iter_notes(vault_path)}
    from sift.mcp_server import get_note

    with pytest.raises(ToolError):
        get_note(res["note_id"])

    tomb = load_tombstones()
    assert tomb.has_id(res["note_id"], source="sift-remember")
    # A user note's url is provenance, not the article: it stays ingestible.
    assert not tomb.has_url("https://example.com/post")


def test_forget_note_tombstones_an_ingested_articles_url(vault_path):
    from sift.mcp_server import forget_note
    from sift.tombstones import load_tombstones

    _save(
        vault_path,
        "writeup-junk",
        "Junk",
        type="writeup",
        source="pentesterland",
        url="https://blog.example.org/junk/",
    )
    forget_note("writeup-junk", reason="spam")
    tomb = load_tombstones()
    assert tomb.has_id("writeup-junk", source="pentesterland")
    assert tomb.has_url("https://blog.example.org/junk", source="research")


def test_forget_note_needs_the_exact_id(vault_path):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import forget_note

    p = _save(vault_path, "finding-exact-id", "Some title")
    with pytest.raises(ToolError, match="exact note_id"):
        forget_note("Some title", reason="x")
    with pytest.raises(ToolError, match="no note"):
        forget_note("nope", reason="x")
    with pytest.raises(ToolError, match="reason"):
        forget_note("finding-exact-id", reason=" ")
    assert p.exists()


# --------------------------------------------------------------------------------
# capture_url
# --------------------------------------------------------------------------------

ARTICLE = "A long post-cutoff writeup about request smuggling. " * 40


@pytest.fixture
def fake_fetch(monkeypatch):
    """Stand-in for the network half of sift.ingest.single_url: `fetch_url_note` is
    replaced, the module's other rules (`is_pre_cutoff`) stay real."""
    calls: list[str] = []
    state = {"created": date(2026, 9, 15), "raise": None, "title": "Smuggling via chunk ext"}

    def fetch_url_note(url: str):
        from slugify import slugify

        from sift.vault.notes import Note
        from sift.vault.schema import Frontmatter

        calls.append(url)
        if state["raise"]:
            raise state["raise"]
        meta = Frontmatter(
            id="writeup-" + slugify(state["title"], max_length=90),
            type="writeup",
            title=state["title"],
            source="blog.example.com",
            url=url,
            created=state["created"],
            tags=["writeup"],
        )
        return Note(meta=meta, body=ARTICLE)

    from sift.ingest import single_url

    monkeypatch.setattr(single_url, "fetch_url_note", fetch_url_note)
    return calls, state


def test_capture_url_saves_indexes_and_dedupes(vault_path, fake_fetch):
    from sift.index.store import Store
    from sift.mcp_server import capture_url, get_note

    calls, _ = fake_fetch
    out = capture_url(
        "https://blog.example.com/smuggling?utm_source=x", program="acme", tags=["http"]
    )
    assert out["saved"] is True and out["new"] is True
    assert out["chunks_indexed"] > 0
    note = get_note(out["note_id"])
    assert note["frontmatter"]["program"] == "acme"
    assert {"writeup", "http"} <= set(note["frontmatter"]["tags"])
    assert note["frontmatter"]["extra"]["captured_via"] == "mcp-capture-url"
    assert Store().table().count_rows(f"note_id = '{out['note_id']}'") == out["chunks_indexed"]

    again = capture_url("http://www.blog.example.com/smuggling/")
    assert again["saved"] is False and again["existing"] is True
    assert again["note_id"] == out["note_id"]
    assert len(calls) == 1  # no second fetch
    assert len(_live_md(vault_path)) == 1


def test_capture_url_refuses_pre_cutoff_unless_forced(vault_path, fake_fetch):
    from sift.mcp_server import capture_url

    _, state = fake_fetch
    state["created"] = date(2020, 1, 1)
    out = capture_url("https://blog.example.com/old")
    assert out["saved"] is False and out["reason"] == "pre-cutoff"
    assert out["published"] == "2020-01-01"
    assert _live_md(vault_path) == []
    forced = capture_url("https://blog.example.com/old", force=True)
    assert forced["saved"] is True and forced["pre_cutoff"] is True


def test_capture_url_reports_a_rejected_page_and_writes_nothing(vault_path, fake_fetch, capfd):
    from sift.mcp_server import capture_url

    _, state = fake_fetch
    state["raise"] = ValueError("js-rendered or boilerplate")
    out = capture_url("https://spa.example.com/post")
    assert out == {
        "saved": False,
        "url": "https://spa.example.com/post",
        "reason": "js-rendered or boilerplate",
    }
    assert _live_md(vault_path) == []
    assert capfd.readouterr().out == ""


def test_capture_url_rejects_non_http_urls_without_fetching(vault_path, fake_fetch):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import capture_url

    calls, _ = fake_fetch
    for bad in ("file:///etc/passwd", "ftp://example.com/x", "not a url", ""):
        with pytest.raises(ToolError):
            capture_url(bad)
    assert calls == []


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://[::1]/",
        "http://2130706433/",  # 127.0.0.1 as one number
        "http://localhost/",
        "https://router.local/setup",
        "http://0.0.0.0/",
    ],
)
def test_capture_url_refuses_local_targets_without_fetching(vault_path, fake_fetch, url):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import capture_url

    calls, _ = fake_fetch
    with pytest.raises(ToolError, match="refusing"):
        capture_url(url)
    assert calls == []


def test_capture_url_allows_public_hosts(vault_path, fake_fetch):
    """Positive control for the guard: public names and addresses go to the fetcher."""
    from sift.mcp_server import capture_url

    calls, _ = fake_fetch
    assert capture_url("https://93.184.215.14/post")["saved"] is True
    assert calls == ["https://93.184.215.14/post"]


def test_capture_url_respects_tombstones(vault_path, fake_fetch):
    from sift.mcp_server import capture_url
    from sift.tombstones import record_tombstones

    calls, _ = fake_fetch
    record_tombstones(urls=["https://blog.example.com/forgotten"], reason="test")
    out = capture_url("https://blog.example.com/forgotten")
    assert out["saved"] is False and "tombstoned" in out["reason"]
    assert calls == []
    assert capture_url("https://blog.example.com/forgotten", force=True)["saved"] is True


def test_capture_url_keys_a_title_clash_by_url(vault_path, fake_fetch):
    """Another article already holds the title-derived id: the capture gets a
    url-hashed id and the existing note is untouched."""
    from sift.mcp_server import capture_url

    p = _save(
        vault_path,
        "writeup-smuggling-via-chunk-ext",
        "Smuggling via chunk ext",
        "other article",
        type="writeup",
        source="blog.example.com",
        url="https://other.example.net/post",
    )
    before = p.read_bytes()
    out = capture_url("https://blog.example.com/smuggling")
    assert out["saved"] is True
    assert out["note_id"].startswith("writeup-smuggling-via-chunk-ext-")
    assert p.read_bytes() == before


def test_capture_url_without_the_ingest_module_is_a_tool_error(vault_path, monkeypatch):
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import capture_url

    monkeypatch.setitem(sys.modules, "sift.ingest.single_url", None)
    with pytest.raises(ToolError, match="unavailable"):
        capture_url("https://example.com/a")


PROSE = (
    "The front end forwards chunk extensions verbatim while the back end strips them, "
    "so a request that both sides frame differently lets an attacker queue a second "
    "request on the shared connection. We confirmed it against a staging host, measured "
    "the timing difference, and reported the desync with a minimal reproduction. "
)


def test_capture_url_through_the_real_fetcher(vault_path, monkeypatch):
    """Contract with sift.ingest.single_url: an article is saved and indexed, a
    refused page comes back as a reason, and a redirect to an internal address is
    stopped on the hop. Offline: a mock transport and a patched resolver."""
    import functools

    import httpx

    single_url = pytest.importorskip("sift.ingest.single_url")
    from sift.index.store import Store
    from sift.mcp_server import capture_url

    page = (
        "<html><head><title>Request smuggling via chunk extensions</title>"
        '<meta property="article:published_time" content="2026-09-20T10:00:00Z"></head>'
        "<body><nav>home about</nav><article><h1>Request smuggling via chunk extensions</h1>"
        + "".join(f"<p>{PROSE}</p>" for _ in range(8))
        + "</article></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/forbidden":
            return httpx.Response(403, text="no")
        if request.url.path == "/bounce":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})
        return httpx.Response(200, text=page, headers={"content-type": "text/html"})

    def resolve(host: str, port: int) -> list[str]:
        return [host] if host[0].isdigit() else ["93.184.215.14"]

    monkeypatch.setattr(single_url, "_resolve", resolve)
    monkeypatch.setattr(
        single_url,
        "fetch_url_note",
        functools.partial(single_url.fetch_url_note, transport=httpx.MockTransport(handler)),
    )

    out = capture_url("https://blog.example.com/smuggling")
    assert out["saved"] is True, out
    assert out["chunks_indexed"] > 0
    assert Store().table().count_rows(f"note_id = '{out['note_id']}'") == out["chunks_indexed"]
    assert "chunk extensions verbatim" in Path(out["path"]).read_text(encoding="utf-8")

    refused = capture_url("https://blog.example.com/forbidden")
    assert refused["saved"] is False and "403" in refused["reason"]
    hop = capture_url("https://blog.example.com/bounce")
    assert hop["saved"] is False and "127.0.0.1" in hop["reason"]
    assert len(_live_md(vault_path)) == 1


# --------------------------------------------------------------------------------
# protocol level: annotations, isError, no background work under tests
# --------------------------------------------------------------------------------


def test_tool_annotations():
    from fastmcp import Client

    from sift.mcp_server import mcp

    async def go():
        async with Client(mcp) as c:
            return {t.name: t.annotations for t in await c.list_tools()}

    ann = asyncio.run(go())
    for name in ("search_memory", "get_note", "list_notes", "stats"):
        assert ann[name].model_dump(by_alias=True)["readOnlyHint"] is True, name
    for name in ("remember", "capture_idea", "resolve_idea", "capture_url"):
        dumped = ann[name].model_dump(by_alias=True)
        assert dumped["readOnlyHint"] is False and dumped["destructiveHint"] is False, name
    for name in ("update_note", "forget_note"):
        assert ann[name].model_dump(by_alias=True)["destructiveHint"] is True, name
    assert ann["capture_url"].model_dump(by_alias=True)["openWorldHint"] is True


def test_bad_arguments_are_is_error_on_the_wire(vault_path):
    res = _client_call("list_notes", {"type": "findings"})
    assert res.is_error is True
    res = _client_call("get_note", {"slug": "nothing-here"})
    assert res.is_error is True
    ok = _client_call("list_notes", {})  # positive control
    assert ok.is_error is False


def test_in_memory_sessions_start_no_background_work(vault_path, monkeypatch):
    from sift import mcp_server

    started: list[str] = []
    monkeypatch.setattr(mcp_server, "_warm_vault", lambda: started.append("vault"))
    monkeypatch.setattr(mcp_server, "_warm_models", lambda: started.append("models"))
    monkeypatch.setattr(mcp_server, "_background_started", False)
    monkeypatch.setattr(mcp_server, "_background_enabled", False)
    _client_call("list_notes", {})
    assert started == []

    # Positive control: in a served process (main() enabled it) initialize starts both, once.
    monkeypatch.setattr(mcp_server, "_background_enabled", True)
    _client_call("list_notes", {})
    _client_call("list_notes", {})
    for t in threading.enumerate():
        if t.name.startswith("sift-warm"):
            t.join(timeout=10)
    assert sorted(started) == ["models", "vault"]


def test_search_triggers_the_index_sync_only_when_serving(vault_path, monkeypatch):
    from sift import mcp_server, pipeline

    calls: list[int] = []
    monkeypatch.setattr(pipeline, "sync_in_background", lambda *a, **k: calls.append(1))
    mcp_server.remember(title="sync probe", body_md="text")

    monkeypatch.setattr(mcp_server, "_background_enabled", False)
    mcp_server.search_memory(query="sync probe")
    assert calls == []

    from sift.config import get_settings

    monkeypatch.setattr(mcp_server, "_background_enabled", True)
    monkeypatch.setenv("SIFT_MCP_AUTO_SYNC", "false")
    get_settings.cache_clear()
    mcp_server.search_memory(query="sync probe")
    assert calls == []

    monkeypatch.setenv("SIFT_MCP_AUTO_SYNC", "true")
    get_settings.cache_clear()
    mcp_server.search_memory(query="sync probe")
    assert calls == [1]
