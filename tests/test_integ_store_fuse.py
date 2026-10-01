"""Store result fusion and the targeted path read the pipeline relies on.

* The per-note bonus for extra matched chunks is capped. Chunks now fit the 512-token
  embedder window, so a long note is cut into more pieces than before; an uncapped
  +25% per extra chunk would have ranked notes by length after the forced reindex.
* `Store.stored_paths` is the one place that builds the ``note_id IN (...)`` read the
  pipeline used to assemble from the store's private quoting helper.
"""

from __future__ import annotations

import pytest


def _row(nid: str, i: int) -> dict:
    return {"id": f"{nid}::{i}", "note_id": nid, "title": nid, "text": f"chunk {i}"}


def _fused(rows_v: list[dict], rows_f: list[dict]):
    from sift.index.store import Store

    return {h.note_id: h for h in Store._fuse(rows_v, rows_f, 0)}


def test_extra_chunk_credit_stops_after_the_strongest_few():
    from sift.index.store import _MAX_EXTRA_CHUNKS

    top = 1 + _MAX_EXTRA_CHUNKS
    many = [_row("long", i) for i in range(20)]
    few = many[:top]

    long_note = _fused(many, many)["long"]
    trimmed = _fused(few, few)["long"]

    assert long_note.matched_chunks == 20, "every matched chunk is still counted"
    assert long_note.score == pytest.approx(trimmed.score), "chunks past the cap add nothing"


def test_extra_chunks_up_to_the_cap_still_earn_credit():
    """Control: the bonus itself is kept - a second matching chunk still helps."""
    one = _fused([_row("n", 0)], [_row("n", 0)])["n"]
    two = _fused([_row("n", 0), _row("n", 1)], [_row("n", 0), _row("n", 1)])["n"]
    assert two.score > one.score and two.matched_chunks == 2


def test_a_long_note_no_longer_outranks_a_sharper_one_by_length_alone():
    """A note whose single chunk tops both lists against one with twenty weaker
    matches: the gap the long note gets from its extra chunks is bounded."""
    from sift.index.store import _EXTRA_CHUNK_CREDIT, _MAX_EXTRA_CHUNKS, RRF_K

    sharp = [_row("sharp", 0)]
    long_rows = [_row("long", i) for i in range(20)]
    hits = _fused(sharp + long_rows, sharp + long_rows)

    # Both get the same quality/recency multiplier, so compare the ratio: the long
    # note's best chunk ranks second in both lists, the sharp note's first.
    best_ratio = (2.0 / (RRF_K + 2)) / (2.0 / (RRF_K + 1))
    bound = best_ratio * (1 + _EXTRA_CHUNK_CREDIT * _MAX_EXTRA_CHUNKS)
    assert hits["long"].score / hits["sharp"].score <= bound + 1e-9
    uncapped = best_ratio + _EXTRA_CHUNK_CREDIT * sum(
        (2.0 / (RRF_K + 2 + i)) / (2.0 / (RRF_K + 1)) for i in range(1, 20)
    )
    assert bound < uncapped / 2, "the cap matters at this size"


def test_stored_paths_reads_only_the_named_notes(vault_path, fake_embedder):
    # fake_embedder: tests/conftest.py (no model, no network).
    from sift.index.store import Store
    from sift.pipeline import index_notes
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    store = Store()
    assert store.stored_paths(["x"]) == [], "no table: empty, and none is created"
    assert not store.index_info()["exists"]

    saved = []
    for nid, title in (("a-1", "Alpha"), ("b-o'brien", "O'Brien's note")):
        note = Note(meta=Frontmatter(id=nid, type="finding", title=title), body=f"{title} body")
        save_note(vault_path, note)
        saved.append(note)
    index_notes(saved, store)

    assert store.stored_paths(["a-1"]) == [str(saved[0].path)]
    assert store.stored_paths(["b-o'brien", "missing"]) == [str(saved[1].path)]
    assert sorted(store.stored_paths(["a-1", "b-o'brien"])) == sorted(str(n.path) for n in saved)
    assert store.stored_paths([]) == []


def test_pipeline_no_longer_imports_the_private_quoting_helper():
    import inspect

    from sift import pipeline

    assert "_in_predicate" not in inspect.getsource(pipeline)
