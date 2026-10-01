"""Incremental reindex.

Adding one note used to re-embed all 9,000+ notes — minutes of GPU work to index a
single file. Chunk rows already carry the source file's mtime, so an incremental pass
skips everything untouched. `--force` still rebuilds everything.
"""

from __future__ import annotations

import os
import time

import pytest

DIM = 64


class FakeEmbedder:
    """Counts how many passages were embedded — the thing under test."""

    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def __init__(self):
        self.embedded = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for i, tok in enumerate(text.split()):
            v[hash(tok) % DIM] += 1.0
            if i > 200:
                break
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts, *, kind="passage", batch_size=32):
        self.embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_DIM", str(DIM))
    from sift import config

    config.get_settings.cache_clear()
    embedder = FakeEmbedder()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: embedder)
    return embedder


def _write(vault, name: str, body: str = "some technique body text here"):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    return save_note(
        vault,
        Note(meta=Frontmatter(id=name, type="technique", title=name), body=body),
        stamp=False,
    )


def test_second_reindex_embeds_nothing(vault_path, fake):
    from sift.pipeline import reindex

    for i in range(4):
        _write(vault_path, f"note-{i}")

    first = reindex(force=True)
    assert first.notes == 4
    assert fake.embedded > 0

    fake.embedded = 0
    second = reindex()

    assert second.unchanged == 4
    assert second.notes == 0
    assert fake.embedded == 0, "unchanged notes were re-embedded"


def test_only_the_changed_note_is_reembedded(vault_path, fake):
    from sift.pipeline import reindex

    for i in range(4):
        _write(vault_path, f"note-{i}")
    reindex(force=True)

    path = _write(vault_path, "note-2", body="rewritten body with different words")
    future = time.time() + 10
    os.utime(path, (future, future))

    fake.embedded = 0
    stats = reindex()

    assert stats.notes == 1, "expected exactly the edited note to be reindexed"
    assert stats.unchanged == 3
    assert fake.embedded > 0


def test_a_new_note_is_picked_up(vault_path, fake):
    """capture_idea writing a note mid-hunt must land in the index on the next pass."""
    from sift.pipeline import reindex

    _write(vault_path, "existing")
    reindex(force=True)

    _write(vault_path, "brand-new")
    stats = reindex()

    assert stats.notes == 1
    assert stats.unchanged == 1


def test_force_ignores_mtimes_entirely(vault_path, fake):
    from sift.pipeline import reindex

    for i in range(3):
        _write(vault_path, f"note-{i}")
    reindex(force=True)

    fake.embedded = 0
    stats = reindex(force=True)

    assert stats.unchanged == 0, "--force must rebuild everything"
    assert stats.notes == 3
    assert fake.embedded > 0


def test_search_still_finds_an_unchanged_note_after_incremental(vault_path, fake):
    """The skip must not drop rows — the whole point is the index stays complete."""
    from sift.pipeline import reindex, search

    _write(vault_path, "cookie-sandwich", body="quoted cookie value parsing httponly bypass")
    _write(vault_path, "other", body="completely unrelated content about pdfs")
    reindex(force=True)
    reindex()  # incremental pass skips both

    hits = search("quoted cookie value parsing", k=2).hits
    assert any(h.note_id == "cookie-sandwich" for h in hits)
