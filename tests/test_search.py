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
    save_note(vault_path, _make_note("weak", "report", "Cache poisoning via XFH", body,
                                     extra={"is_dupe": True}))
    save_note(vault_path, _make_note("strong", "report", "Cache poisoning via XFH", body,
                                     severity="high",
                                     extra={"has_bounty": True, "vote_count": 25}))
    reindex(force=True)

    res = search("cache poisoning x-forwarded-host redirect", k=2)
    assert [h.note_id for h in res.hits][0] == "strong"
    assert res.hits[0].quality > res.hits[1].quality

    # min_quality filter drops the weak one entirely
    res2 = search("cache poisoning x-forwarded-host redirect", k=5, min_quality=55)
    assert [h.note_id for h in res2.hits] == ["strong"]
