"""End-to-end index + search smoke test with a deterministic fake embedder
(so it runs offline with no model download)."""

from __future__ import annotations

import hashlib
import re

import pytest

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words -> unit vector. Similar text -> similar vector."""

    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0
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


def _make_note(nid, ntype, title, body, **meta):
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    return Note(meta=Frontmatter(id=nid, type=ntype, title=title, **meta), body=body)


def test_index_and_search(vault_path):
    from sift.index.store import Store
    from sift.pipeline import reindex, search
    from sift.vault.notes import save_note

    save_note(
        vault_path,
        _make_note(
            "r1",
            "report",
            "Stored XSS via SVG upload",
            "## Summary\nAn uploaded SVG file executes javascript in the app origin. "
            "Stored cross site scripting through the avatar upload endpoint.",
            cwe=["CWE-79"],
            program="Acme",
        ),
    )
    save_note(
        vault_path,
        _make_note(
            "r2",
            "report",
            "SQL injection in reporting export",
            "## Summary\nThe date parameter of the CSV export is concatenated into a SQL query "
            "allowing union based sql injection and database exfiltration.",
            cwe=["CWE-89"],
            program="Acme",
        ),
    )
    save_note(
        vault_path,
        _make_note(
            "t1",
            "technique",
            "SVG upload XSS bypass",
            "Bypass image upload filters by using an SVG with an onload handler running javascript.",
            cwe=["CWE-79"],
        ),
    )

    stats = reindex(force=True)
    assert stats.notes == 3
    assert Store().count() >= 3

    res = search("svg file upload cross site scripting javascript", k=3)
    assert res.hits
    assert res.hits[0].note_id in {"r1", "t1"}

    # type filter
    res2 = search("svg upload javascript", k=5, filters={"type": "technique"})
    assert all(h.type == "technique" for h in res2.hits)
    assert res2.hits[0].note_id == "t1"

    # cwe filter
    res3 = search("injection database", k=5, filters={"cwe": "CWE-89"})
    assert res3.hits and res3.hits[0].note_id == "r2"


def test_search_link_expansion(vault_path):
    from sift.pipeline import reindex, search
    from sift.vault.notes import save_note

    save_note(
        vault_path,
        _make_note(
            "r1",
            "report",
            "Stored XSS via SVG upload",
            "## Summary\nStored cross site scripting through SVG avatar upload. See "
            "[[svg-upload-xss-bypass]].\n\n## Steps to reproduce\n1. Upload a crafted SVG "
            "avatar containing an onload handler.\n2. View another user's profile; the "
            "script executes in the app origin.\n\n```\nGET /avatars/evil.svg HTTP/1.1\n```\n\n"
            "## Impact\nAccount takeover via session theft.",
            severity="high",
            extra={"has_bounty": True, "vote_count": 12},
        ),
    )
    save_note(
        vault_path,
        _make_note(
            "svg-upload-xss-bypass",
            "technique",
            "SVG upload XSS bypass",
            "Bypass upload filters with an SVG onload handler.",
        ),
    )
    reindex(force=True)

    res = search("svg avatar upload scripting", k=1, expand_links=True)
    assert res.hits[0].note_id == "r1"
    assert any(nb["slug"] == "svg-upload-xss-bypass" for nb in res.linked)


def test_quality_reweights_ranking(vault_path):
    from sift.pipeline import reindex, search
    from sift.vault.notes import save_note

    body = (
        "## Summary\nCache poisoning via the X-Forwarded-Host header on the login page.\n\n"
        "## Steps to reproduce\n1. Send a request with `X-Forwarded-Host: evil.com`.\n"
        "2. The response caches an absolute redirect to evil.com.\n\n"
        "```\nGET / HTTP/1.1\nX-Forwarded-Host: evil.com\n```\n\n## Impact\nStored redirect / XSS.\n"
    )
    # Same text; one is a marked dupe with no bounty, the other bountied + well-voted.
    save_note(
        vault_path,
        _make_note("weak", "report", "Cache poisoning via XFH", body, extra={"is_dupe": True}),
    )
    save_note(
        vault_path,
        _make_note(
            "strong",
            "report",
            "Cache poisoning via XFH",
            body,
            severity="high",
            extra={"has_bounty": True, "vote_count": 25},
        ),
    )
    reindex(force=True)

    res = search("cache poisoning x-forwarded-host redirect", k=2)
    assert [h.note_id for h in res.hits][0] == "strong"
    assert res.hits[0].quality > res.hits[1].quality

    # min_quality filter drops the weak one entirely
    res2 = search("cache poisoning x-forwarded-host redirect", k=5, min_quality=55)
    assert [h.note_id for h in res2.hits] == ["strong"]


# ---- filters, recall and error surfacing, at the Store level ------------------
#
# A where clause that fails used to be retried WITHOUT the filter, so every filter
# test below asserts that excluded notes are absent - presence alone would pass on
# an unfiltered result.


def _chunk(
    nid, text, *, title=None, ntype="report", cwe=(), program="", quality=50, idx=0, path=None
):
    from sift.index.store import ChunkRow

    title = title or nid
    return ChunkRow(
        note_id=nid,
        slug=nid,
        type=ntype,
        title=title,
        heading="",
        text=text,
        chunk_index=idx,
        vector=FakeEmbedder()._vec(f"{title}\n{text}"),
        cwe=list(cwe),
        program=program,
        quality=quality,
        path=f"/vault/{ntype}/{nid}.md" if path is None else path,
    )


def _index(rows):
    from sift.index.store import Store

    store = Store()
    store.add_chunks(rows)
    store.ensure_fts()
    return store


def _ids(query, **kw):
    from sift.index.store import Store

    return {h.note_id for h in Store().search(query, k=10, **kw)}


def test_cwe_filter_matches_the_exact_token_only():
    from sift.index.store import Store

    body = "command injection input validation flaw"
    _index(
        [
            _chunk("n78", body, cwe=["CWE-78"]),
            _chunk("n787", body, cwe=["CWE-787"]),
            _chunk("n20", body, cwe=["CWE-20"]),
            _chunk("n200", body, cwe=["CWE-200"]),
            _chunk("n209-79", body, cwe=["CWE-209", "CWE-79"]),
        ]
    )

    assert _ids(body, filters={"cwe": "CWE-78"}) == {"n78"}, "CWE-78 matched CWE-787"
    assert _ids(body, filters={"cwe": "78"}) == {"n78"}
    assert _ids(body, filters={"cwe": "cwe-20"}) == {"n20"}, "CWE-20 matched CWE-200/209"
    assert _ids(body, filters={"cwe": "79"}) == {"n209-79"}, "second token of a list"
    assert _ids(body, filters={"cwe": "CWE_78"}) == {"n78"}, "_ must not act as a wildcard"
    with pytest.raises(ValueError, match="cwe filter"):
        Store().search(body, filters={"cwe": "CWE-7%"})


def test_program_filter_ignores_case_and_keeps_apostrophes():
    body = "ssrf through the webhook url"
    _index(
        [
            _chunk("a1", body, program="Acme"),
            _chunk("a2", body, program="acme"),
            _chunk("b1", body, program="Other"),
            _chunk("o1", body, program="O'Brien"),
        ]
    )

    for program in ("ACME", "acme", "Acme"):
        assert _ids(body, filters={"program": program}) == {"a1", "a2"}
    assert _ids(body, filters={"program": "o'brien"}) == {"o1"}
    assert _ids(body, filters={"program": "Acme", "type": "report"}) == {"a1", "a2"}
    assert _ids(body, filters={"program": "Acme", "type": "cve"}) == set()


def test_min_quality_prefilters_before_the_candidate_pool():
    """Thin stubs filled the 40-chunk pool, so min_quality=70 used to return nothing."""
    from sift.index import embed
    from sift.index.store import Store
    from sift.pipeline import search

    q = "jwt algorithm confusion"
    # More stubs than even the widened pool (40 -> 120), so pool widening alone cannot
    # rescue the good note: only a quality prefilter in the query can.
    rows = [
        _chunk(
            f"stub-{i}",
            f"jwt algorithm confusion jwt algorithm confusion {i}",
            title=f"CVE stub {i}",
            ntype="cve",
            quality=30,
        )
        for i in range(150)
    ]
    rows.append(
        _chunk(
            "writeup-good",
            "How a jwt library accepted an HMAC signature keyed with the RSA public key, "
            "letting us forge admin tokens: key handling, the patch, and three related "
            "endpoints that trusted the same verification helper.",
            title="Forging admin tokens",
            ntype="writeup",
            quality=90,
        )
    )
    store = _index(rows)

    # Precondition: the good note is outside both candidate lists, widened ones included.
    vec = embed.get_embedder().embed_query(q)
    for pool in (40, 120):
        vhits, fhits = store._retrieve(store.table(), vec, q, pool, None)
        assert "writeup-good" not in {r["note_id"] for r in vhits + fhits}, pool

    assert [h.slug for h in Store().search(q, k=8, pool=40, min_quality=70)] == ["writeup-good"]
    assert [h.slug for h in search(q, k=8, min_quality=70).hits] == ["writeup-good"]
    assert Store().search(q, k=8, pool=40, min_quality=70, filters={"type": "cve"}) == []


def test_dimension_mismatch_is_raised_not_answered_with_nothing(monkeypatch):
    from sift.index.store import IndexDimMismatch, Store
    from sift.pipeline import search

    _index([_chunk("a", "some text about oauth")])
    monkeypatch.setenv("SIFT_EMBED_DIM", str(DIM * 2))
    from sift.config import get_settings

    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="reindex --force"):
        Store().search("oauth")
    with pytest.raises(IndexDimMismatch):
        search("oauth")


def test_a_failing_filter_never_returns_unfiltered_rows(monkeypatch):
    from sift.index.store import Store

    _index([_chunk("r1", "xss payload"), _chunk("t1", "xss payload", ntype="technique")])
    monkeypatch.setattr(Store, "_where", staticmethod(lambda filters: "nonexistent_col = 'x'"))

    with pytest.raises(RuntimeError, match="vector search failed"):
        Store().search("xss payload", filters={"type": "technique"})


def test_a_failed_keyword_search_degrades_with_a_warning(monkeypatch, capsys):
    store = _index([_chunk("a", "stored xss in the avatar upload")])
    assert [h.note_id for h in store.search("stored xss avatar", k=3)] == ["a"]
    assert store.warnings == []

    tbl = store.table()
    real = tbl.search

    def search(query=None, *args, **kwargs):
        if kwargs.get("query_type") == "fts":
            raise RuntimeError("fts index is corrupt")
        return real(query, *args, **kwargs)

    monkeypatch.setattr(tbl, "search", search)
    hits = store.search("stored xss avatar", k=3)

    assert [h.note_id for h in hits] == ["a"], "vector results must survive an FTS failure"
    assert len(store.warnings) == 1 and "fts" in store.warnings[0]
    assert "corrupt" in store.warnings[0]
    assert capsys.readouterr().out == "", "the MCP server's stdout is the JSON-RPC wire"


def test_a_repeated_chunk_id_is_credited_once():
    """Two files sharing an id used to share chunk ids too, and RRF summed both rows."""
    from sift.index.store import Store

    text = "prototype pollution gadget chain in the template engine"
    title = "Prototype pollution"
    _index(
        [
            _chunk("dup", text, title=title, path=""),
            _chunk("dup", text, title=title, path=""),  # same chunk id, written twice
            _chunk("solo", text, title=title, path=""),
        ]
    )

    hits = {h.note_id: h for h in Store().search(text, k=5)}
    assert hits["dup"].score / hits["solo"].score < 1.5, "a duplicated row doubled the score"


def test_a_crowded_pool_is_widened_once():
    """One note's chunks filling the whole pool left fewer than k notes."""
    from sift.index import embed
    from sift.index.store import Store

    text = "graphql batching rate limit bypass"
    rows = [_chunk("big", text, title="graphql batching", idx=i) for i in range(50)]
    rows += [
        _chunk(
            f"other-{j}",
            f"{text} lorem ipsum dolor sit amet consectetur adipiscing elit",
            title=f"Unrelated heading {j}",
        )
        for j in range(5)
    ]
    store = _index(rows)

    vec = embed.get_embedder().embed_query(text)
    vhits, fhits = store._retrieve(store.table(), vec, text, 40, None)
    assert {r["note_id"] for r in vhits + fhits} == {"big"}, "precondition: a crowded pool"

    hits = Store().search(text, k=4, pool=40)
    assert len(hits) == 4
    assert "big" in {h.note_id for h in hits}


def test_a_long_lived_store_sees_edits_at_once_and_survives_a_drop():
    """The MCP server may hold one Store; remember-then-search must see the write."""
    from sift.index.store import Store

    reader, writer = Store(), Store()
    writer.replace_notes([_chunk("n1", "original wording about web cache deception")])
    writer.ensure_fts()
    assert reader.search("web cache deception", k=1)[0].excerpt.startswith("original")

    writer.upsert_note([_chunk("n1", "corrected wording about web cache deception")])
    assert reader.search("web cache deception", k=1)[0].excerpt.startswith("corrected")

    rebuilder = Store()
    rebuilder.drop()
    assert reader.search("web cache deception", k=1) == []
    rebuilder.add_chunks([_chunk("n2", "rebuilt index about web cache deception")])
    assert [h.note_id for h in reader.search("web cache deception", k=1)] == ["n2"]


def test_index_note_then_search_returns_the_new_text(vault_path):
    from sift.pipeline import index_note, search

    note = _make_note(
        "idea-1", "finding", "Cache key confusion", "First guess about the cache key."
    )
    index_note(note)
    assert search("cache key", k=1).hits[0].excerpt.startswith("First guess")

    note.body = "Confirmed: the cache key ignores the port."
    index_note(note)
    hits = search("cache key", k=3).hits
    assert [h.note_id for h in hits] == ["idea-1"]
    assert hits[0].excerpt.startswith("Confirmed")


def test_search_without_an_index_returns_nothing_cleanly():
    from sift.index.store import TABLE, Store

    store = Store()
    assert store.search("anything") == [] and store.warnings == []
    assert TABLE not in store._table_names(), "a search must not create the table"

    store.table()  # an empty table with no FTS index yet
    assert store.search("anything") == [] and store.warnings == []
