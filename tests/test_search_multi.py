"""Multi-query search: several phrasings searched together and fused by rank.

The point is vocabulary mismatch. A note that says `View::Teardown` is found by the
identifier query, a note that only describes the bug in prose by the descriptive one,
and one call returns both - ranked by RRF across every phrasing's vector and keyword
lists, so a note matched by several phrasings rises.
"""

from __future__ import annotations

import hashlib
import re

import pytest

DIM = 32


class FakeEmbedder:
    """Hashed bag-of-words -> unit vector; counts batched query embeddings."""

    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def __init__(self):
        self.query_batches: list[list[str]] = []

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts, *, kind="passage", batch_size=32):
        if kind == "query":
            self.query_batches.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str):
        return self.embed([text], kind="query")[0]

    def embed_one(self, text: str):
        return self._vec(text)


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_DIM", str(DIM))
    from sift.config import get_settings

    get_settings.cache_clear()
    emb = FakeEmbedder()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: emb)
    return emb


def _chunk(nid, text, *, ntype="finding", program=""):
    from sift.index.store import ChunkRow

    return ChunkRow(
        note_id=nid,
        slug=nid,
        type=ntype,
        title=nid,
        heading="",
        text=text,
        chunk_index=0,
        vector=FakeEmbedder()._vec(f"{nid}\n{text}"),
        program=program,
        quality=50,
        path=f"/vault/{ntype}/{nid}.md",
    )


def _index(rows):
    from sift.index.store import Store

    store = Store()
    store.add_chunks(rows)
    store.ensure_fts()
    return store


CORPUS = [
    ("ident", "crash in teardownchildren when the parent iterates its list"),
    ("prose", "use after free while a widget is destroyed during iteration"),
    ("both", "teardownchildren use after free while destroyed during iteration"),
    ("noise1", "stored cross site scripting in the avatar upload form"),
    ("noise2", "server side request forgery against the cloud metadata service"),
    ("noise3", "jwt algorithm confusion accepted an hs256 token"),
]


def _corpus():
    return _index([_chunk(nid, text) for nid, text in CORPUS])


def test_each_phrasing_contributes_hits_the_other_misses():
    from sift.index.store import Store

    _corpus()
    q_ident, q_prose = "teardownchildren", "use after free widget destroyed"
    alone_ident = {h.note_id for h in Store().search(q_ident, k=2)}
    alone_prose = {h.note_id for h in Store().search(q_prose, k=2)}
    fused = {h.note_id for h in Store().search([q_ident, q_prose], k=3)}

    assert "ident" in fused and "prose" in fused and "both" in fused
    # The fused call recovers what each single phrasing found.
    assert alone_ident <= fused | {"both"} and alone_prose <= fused | {"both"}


def test_a_note_matched_by_several_phrasings_ranks_first_and_says_so():
    from sift.index.store import Store

    _corpus()
    hits = Store().search(["teardownchildren", "use after free widget destroyed"], k=3)

    assert hits[0].note_id == "both"
    assert hits[0].matched_queries == [0, 1]
    by_id = {h.note_id: h for h in hits}
    assert 0 in by_id["ident"].matched_queries
    assert 1 in by_id["prose"].matched_queries


def _row(nid):
    return {"id": f"{nid}#0", "note_id": nid, "quality": 50}


def test_attribution_is_each_phrasings_own_top_k_not_pool_membership():
    """`c` sits in both phrasings' candidate pools but in neither one's own top 2:
    it is attributed to no phrasing. Pool membership is true of nearly every hit."""
    from sift.index.store import Store

    q0 = [_row(n) for n in ("a", "b", "c", "d")]
    q1 = [_row(n) for n in ("e", "f", "c", "g")]
    lists = [(0, q0), (0, q0), (1, q1), (1, q1)]
    hits = {h.note_id: h for h in Store._fuse_many(lists, 0)}

    Store._attribute(list(hits.values()), lists, n=2, k=2, min_quality=0)

    assert hits["a"].matched_queries == [0] and hits["b"].matched_queries == [0]
    assert hits["e"].matched_queries == [1] and hits["f"].matched_queries == [1]
    assert hits["c"].matched_queries == []


def test_a_single_phrasing_attributes_every_hit_to_itself():
    from sift.index.store import Store

    _corpus()
    hits = Store().search("use after free", k=5)
    assert hits and all(h.matched_queries == [0] for h in hits)


def test_one_string_and_a_one_item_list_are_the_same_search():
    from sift.index.store import Store

    _corpus()
    one = [(h.note_id, round(h.score, 9)) for h in Store().search("jwt hs256 token", k=4)]
    lst = [(h.note_id, round(h.score, 9)) for h in Store().search(["jwt hs256 token"], k=4)]
    assert one == lst
    assert all(h.matched_queries == [0] for h in Store().search("jwt hs256 token", k=4))


def test_queries_are_embedded_in_one_batch(fake):
    from sift.index.store import Store

    _corpus()
    fake.query_batches.clear()
    Store().search(["teardownchildren", "use after free", "cloud metadata"], k=3)
    assert fake.query_batches == [["teardownchildren", "use after free", "cloud metadata"]]


def test_blank_and_repeated_phrasings_are_dropped():
    from sift.pipeline import normalize_queries

    assert normalize_queries("  xss ", ["XSS", "", "  ", "ssrf", "xss"]) == ["xss", "ssrf"]
    assert normalize_queries(None, None) == []


def test_filters_apply_to_every_phrasing():
    from sift.index.store import Store

    _index(
        [
            _chunk("in-scope", "teardownchildren use after free", program="Acme"),
            _chunk("out-of-scope", "teardownchildren use after free", program="Other"),
        ]
    )
    hits = Store().search(["teardownchildren", "use after free"], k=5, filters={"program": "acme"})
    assert [h.note_id for h in hits] == ["in-scope"]


# ---------------------------------------------------------------- MCP tool


def test_mcp_search_memory_takes_queries_and_reports_matches():
    from sift.mcp_server import search_memory

    _corpus()
    out = search_memory(queries=["teardownchildren", "use after free widget destroyed"], k=3)

    assert out["queries"] == ["teardownchildren", "use after free widget destroyed"]
    first = out["results"][0]
    assert first["note_id"] == "both" and first["matched_queries"] == [0, 1]


def test_mcp_search_memory_single_query_keeps_its_old_shape():
    from sift.mcp_server import search_memory

    _corpus()
    out = search_memory(query="jwt hs256 token", k=2)
    assert out["query"] == "jwt hs256 token"
    assert out["results"] and "matched_queries" not in out["results"][0]


def test_mcp_search_memory_merges_query_into_queries():
    from sift.mcp_server import search_memory

    _corpus()
    out = search_memory(query="teardownchildren", queries=["use after free", "TEARDOWNCHILDREN"])
    assert out["queries"] == ["teardownchildren", "use after free"]


def test_mcp_search_memory_rejects_no_query_and_too_many():
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import search_memory

    with pytest.raises(ToolError, match="query"):
        search_memory()
    with pytest.raises(ToolError, match="query"):
        search_memory(query="  ", queries=["", " "])
    with pytest.raises(ToolError, match="at most"):
        search_memory(queries=[f"phrasing {i}" for i in range(7)])
