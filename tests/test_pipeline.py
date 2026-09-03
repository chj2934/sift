"""Ingest -> vault -> index -> search, with the fake embedder and a stubbed feed."""

from __future__ import annotations

import hashlib
import re

import pytest

DIM = 64


class FakeEmbedder:
    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def _vec(self, text: str):
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed(self, texts, *, kind="passage", batch_size=32):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_DIM", str(DIM))
    from sift.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: FakeEmbedder())
    yield


_KEV_SAMPLE = [
    {
        "cveID": "CVE-2021-44228",
        "vendorProject": "Apache",
        "product": "Log4j2",
        "vulnerabilityName": "Apache Log4j2 Remote Code Execution",
        "dateAdded": "2021-12-10",
        "shortDescription": "JNDI features do not protect against attacker-controlled LDAP lookups, allowing remote code execution.",
        "requiredAction": "Apply updates.",
        "cwes": ["CWE-502", "CWE-917"],
    },
    {
        "cveID": "CVE-2022-22965",
        "vendorProject": "VMware",
        "product": "Spring Framework",
        "vulnerabilityName": "Spring4Shell",
        "dateAdded": "2022-04-04",
        "shortDescription": "Spring MVC data binding on JDK 9+ allows remote code execution via class loader manipulation.",
        "requiredAction": "Apply updates.",
        "cwes": ["CWE-94"],
    },
]


def test_kev_ingest_and_search(monkeypatch):
    from sift.ingest import kev
    from sift.ingest.base import load_state, run_source
    from sift.pipeline import search

    monkeypatch.setattr(kev, "fetch", lambda: _KEV_SAMPLE)

    res = run_source("kev", kev.source())
    assert res.written == 2
    assert res.errors == 0
    assert res.indexed_chunks >= 2

    hits = search("log4j jndi ldap remote code execution", k=2).hits
    assert hits and hits[0].note_id == "CVE-2021-44228"

    filtered = search("remote code execution", k=5, filters={"cwe": "CWE-94"}).hits
    assert [h.note_id for h in filtered] == ["CVE-2022-22965"]

    assert "kev" in load_state()


def test_reindex_is_idempotent(monkeypatch, vault_path):
    from sift.index.store import Store
    from sift.ingest import kev
    from sift.ingest.base import run_source
    from sift.pipeline import reindex

    monkeypatch.setattr(kev, "fetch", lambda: _KEV_SAMPLE)
    run_source("kev", kev.source())
    n1 = Store().count()

    reindex(force=False)
    reindex(force=False)
    assert Store().count() == n1  # no duplicate chunks
