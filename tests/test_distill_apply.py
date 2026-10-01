"""`distill export` / `distill apply`: the in-session gate's write path.

Each test here pins a way the round-trip used to lose or duplicate material: one bad
verdict line aborting a whole file, drops logged twice on a re-apply, two keeps sharing
a technique title overwriting each other, and an export that `apply` could not match
without repeating `--type`.
"""

from __future__ import annotations

import json

import pytest

DIM = 64


class FakeEmbedder:
    """Deterministic bag-of-words vectors - no model download, no GPU."""

    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in text.split()[:200]:
            v[hash(tok) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts, *, kind="passage", batch_size=32):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embedder(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_DIM", str(DIM))
    from sift import config

    config.get_settings.cache_clear()
    embedder = FakeEmbedder()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: embedder)
    return embedder


def _writeup(vault, nid, title, url, *, note_type="writeup", body=None):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    note = Note(
        meta=Frontmatter(id=nid, type=note_type, title=title, url=url, source="example"),
        body=body or f"Article body for {title}. " * 40,
    )
    save_note(vault, note)
    return note


def _drop(url, **over):
    row = {
        "url": url,
        "decision": "drop",
        "already_known": "Known methodology.",
        "reason": "already-known",
        "justification": "Nothing new.",
    }
    row.update(over)
    return row


def _keep(url, title, **over):
    row = {
        "url": url,
        "decision": "keep",
        "already_known": "Only the general class.",
        "reason": "obscure-variant",
        "justification": "Specific bypass I would fumble.",
        "technique_title": title,
        "when_to_try": "When the target parses cookies twice.",
        "body_md": f"Distilled body for {url}.",
    }
    row.update(over)
    return row


def _verdicts(tmp_path, rows=(), *, lines=None, encoding="utf-8", name="verdicts.jsonl"):
    path = tmp_path / name
    lines = lines if lines is not None else [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding=encoding)
    return path


def _apply(path, note_type="writeup"):
    from sift.distill.manual import apply_verdicts, collect

    return apply_verdicts(path, collect(note_type, skip_gated=False, prefilter=False))


def _technique_files(vault):
    return sorted(p.name for p in (vault / "technique").glob("*.md"))


def _technique_ids(vault):
    from sift.vault.notes import iter_notes

    return sorted(n.meta.id for n in iter_notes(vault, note_type="technique"))


def _indexed_ids():
    from sift.index.store import Store

    return sorted(set(Store().table().to_arrow().column("note_id").to_pylist()))


# --- reading the verdicts file --------------------------------------------------------


def test_a_bom_prefixed_file_applies(tmp_path, vault_path):
    """PowerShell writes a UTF-8 BOM; json.loads used to reject the whole file."""
    _writeup(vault_path, "w1", "Post one", "https://a.tld/p")
    res = _apply(_verdicts(tmp_path, [_drop("https://a.tld/p")], encoding="utf-8-sig"))
    assert (res["dropped"], res["skipped"]) == (1, 0)


def test_a_utf16_file_applies(tmp_path, vault_path):
    _writeup(vault_path, "w1", "Post one", "https://a.tld/p")
    res = _apply(_verdicts(tmp_path, [_drop("https://a.tld/p")], encoding="utf-16"))
    assert res["dropped"] == 1


def test_raw_unicode_line_separators_inside_a_row_apply(tmp_path, vault_path):
    """U+2028/U+0085 are legal inside a JSON string; splitlines() cut the row there."""
    from sift.distill.rejects import load_rejects

    _writeup(vault_path, "w1", "Post one", "https://a.tld/p")
    row = _drop("https://a.tld/p", justification="copied\u2028from\x85the article")
    res = _apply(_verdicts(tmp_path, [row]))
    assert res["dropped"] == 1
    assert load_rejects()[0]["justification"] == "copied\u2028from\x85the article"


def test_one_bad_line_does_not_abort_the_file(tmp_path, vault_path):
    _writeup(vault_path, "w1", "Post one", "https://a.tld/p")
    _writeup(vault_path, "w2", "Post two", "https://b.tld/p")
    lines = [
        json.dumps(_drop("https://a.tld/p")),
        '{"url": "https://c.tld/p", "body_md": "an unescaped',
        json.dumps(_drop("https://b.tld/p")),
    ]
    res = _apply(_verdicts(tmp_path, lines=lines))
    assert res["dropped"] == 2
    assert res["skipped"] == 1
    assert any(p.startswith("verdicts.jsonl:2:") for p in res["problems"])


def test_a_non_utf8_file_is_reported_not_guessed(tmp_path, vault_path):
    from sift.distill.manual import apply_verdicts

    path = tmp_path / "verdicts.jsonl"
    row = json.dumps(_drop("https://a.tld/p", justification="café"), ensure_ascii=False)
    path.write_bytes(row.encode("cp1252"))  # an editor saving in the Windows codepage
    res = apply_verdicts(path, [])
    assert res["dropped"] == 0
    assert "not UTF-8" in res["problems"][0]


def test_problems_are_returned_and_never_printed(tmp_path, vault_path, capsys):
    """apply is library code; stdout is the MCP wire if it is ever exposed there."""
    _writeup(vault_path, "w1", "Post one", "https://a.tld/p")
    lines = ["[1, 2]", json.dumps(_drop("https://nowhere.tld/x")), "{broken"]
    res = _apply(_verdicts(tmp_path, lines=lines))
    assert res["skipped"] == 3
    assert len(res["problems"]) == 3
    assert "row is list" in res["problems"][0]
    assert "no candidate matches" in res["problems"][1]
    assert capsys.readouterr().out == ""


# --- validating rows ------------------------------------------------------------------


def test_a_keep_with_an_unknown_reason_is_skipped(tmp_path, vault_path):
    """The reason becomes a tag; 'novel: I did not know' used to land in the tag list."""
    _writeup(vault_path, "w1", "Post one", "https://a.tld/p")
    res = _apply(_verdicts(tmp_path, [_keep("https://a.tld/p", "Some technique", reason="novel")]))
    assert (res["kept"], res["skipped"]) == (0, 1)
    assert "bad keep reason" in res["problems"][0]
    assert not (vault_path / "technique").exists()


@pytest.mark.parametrize(
    "over, why",
    [
        ({"technique_title": 5}, "technique_title must be a string"),
        ({"tags": ["xss", 1]}, "tags must be a list of strings"),
        ({"cwe": {"id": 79}}, "cwe must be a list of strings"),
    ],
)
def test_badly_typed_fields_are_skipped_before_anything_is_written(tmp_path, vault_path, over, why):
    _writeup(vault_path, "w1", "Post one", "https://a.tld/p")
    res = _apply(_verdicts(tmp_path, [_keep("https://a.tld/p", "Some technique", **over)]))
    assert res["kept"] == 0
    assert why in res["problems"][0]


def test_a_bare_string_tag_is_one_tag():
    from sift.distill.candidates import Candidate
    from sift.distill.manual import _validate
    from sift.distill.technique import build_note

    cand = Candidate(title="t", url="https://a.tld/p", text="x", source="s")
    note = build_note(
        cand, title="T", when_to_try="w", body_md="b", keep_reason="post-cutoff", tags="xss"
    )
    assert note.meta.tags == ["post-cutoff", "technique", "xss"]

    row = _keep("https://a.tld/p", "T", tags="xss", cwe="CWE-79")
    assert _validate(row) is None
    assert (row["tags"], row["cwe"]) == (["xss"], ["CWE-79"])


# --- idempotency ----------------------------------------------------------------------


def test_reapplying_a_drop_logs_it_once(tmp_path, vault_path):
    from sift.distill.rejects import load_rejects

    _writeup(vault_path, "w1", "Post one", "https://a.tld/p/")
    path = _verdicts(tmp_path, [_drop("https://a.tld/p/")])
    first, second = _apply(path), _apply(path)
    assert (first["dropped"], second["dropped"], second["duplicates"]) == (1, 0, 1)
    assert len(load_rejects()) == 1


def test_a_url_repeated_in_one_file_is_logged_once(tmp_path, vault_path):
    from sift.distill.rejects import load_rejects

    _writeup(vault_path, "w1", "Post one", "https://a.tld/p")
    res = _apply(
        _verdicts(tmp_path, [_drop("https://a.tld/p"), _drop("https://a.tld/p?source=rss")])
    )
    assert (res["dropped"], res["duplicates"]) == (1, 1)
    assert len(load_rejects()) == 1


def test_every_drop_is_logged(tmp_path, vault_path):
    from sift.distill.rejects import load_rejects

    for i in range(3):
        _writeup(vault_path, f"w{i}", f"Post {i}", f"https://blog{i}.tld/p")
    res = _apply(_verdicts(tmp_path, [_drop(f"https://blog{i}.tld/p") for i in range(3)]))
    assert res["dropped"] == 3
    rows = load_rejects()
    assert [r["title"] for r in rows] == ["Post 0", "Post 1", "Post 2"]
    assert all(r["already_known"] and r["gated_by"] == "in-session" for r in rows)


# --- technique ids --------------------------------------------------------------------


def test_same_title_for_two_articles_gives_two_notes(tmp_path, vault_path, fake_embedder):
    """Both keeps used to share `tech-cookie-sandwich-technique`: one file, one set of
    index rows, and the first article silently un-gated."""
    from sift.distill.candidates import url_key
    from sift.distill.manual import gated_urls

    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    _writeup(vault_path, "w2", "Article two", "https://two.tld/p")
    title = "Cookie sandwich technique"
    res = _apply(
        _verdicts(tmp_path, [_keep("https://one.tld/p", title), _keep("https://two.tld/p", title)])
    )

    assert (res["kept"], res["collisions"]) == (2, 1)
    ids = _technique_ids(vault_path)
    assert len(set(ids)) == 2 and "tech-cookie-sandwich-technique" in ids
    assert len(_technique_files(vault_path)) == 2
    assert set(ids) <= set(_indexed_ids())
    assert {url_key("https://one.tld/p"), url_key("https://two.tld/p")} <= gated_urls()


def test_reapplying_a_keep_rewrites_the_same_note(tmp_path, vault_path, fake_embedder):
    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    path = _verdicts(tmp_path, [_keep("https://one.tld/p", "Cookie sandwich technique")])
    first = _apply(path)
    files, indexed = _technique_files(vault_path), _indexed_ids()
    second = _apply(path)

    assert (first["kept"], second["kept"], second["collisions"]) == (1, 1, 0)
    assert _technique_files(vault_path) == files == ["Cookie sandwich technique.md"]
    assert _indexed_ids() == indexed


def test_reapplying_a_collision_file_changes_nothing(tmp_path, vault_path, fake_embedder):
    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    _writeup(vault_path, "w2", "Article two", "https://two.tld/p")
    title = "Cookie sandwich technique"
    path = _verdicts(
        tmp_path, [_keep("https://one.tld/p", title), _keep("https://two.tld/p", title)]
    )
    _apply(path)
    files, ids, indexed = _technique_files(vault_path), _technique_ids(vault_path), _indexed_ids()

    again = _apply(path)
    assert again["collisions"] == 0
    assert (_technique_files(vault_path), _technique_ids(vault_path), _indexed_ids()) == (
        files,
        ids,
        indexed,
    )


def test_titles_sharing_a_70_char_slug_stay_distinct_through_reindex(
    tmp_path, vault_path, fake_embedder
):
    """Two different titles cut to the same 70-character id used to flip which one was
    indexed on every incremental reindex."""
    from sift.pipeline import reindex

    stem = "Request smuggling via chunk extensions in a reverse proxy that normalises"
    t1, t2 = f"{stem} headers first", f"{stem} bodies later"
    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    _writeup(vault_path, "w2", "Article two", "https://two.tld/p")
    res = _apply(
        _verdicts(tmp_path, [_keep("https://one.tld/p", t1), _keep("https://two.tld/p", t2)])
    )

    ids = _technique_ids(vault_path)
    assert res["collisions"] == 1 and len(set(ids)) == 2
    reindex()
    assert set(ids) <= set(_indexed_ids())


def test_a_handwritten_note_on_the_plain_id_is_not_overwritten(tmp_path, vault_path, fake_embedder):
    from sift.vault.notes import Note, iter_notes, save_note
    from sift.vault.schema import Frontmatter

    mine = Note(
        meta=Frontmatter(id="tech-cookie-sandwich", type="technique", title="Cookie sandwich"),
        body="My own notes from a live target.",
    )
    save_note(vault_path, mine)
    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    res = _apply(_verdicts(tmp_path, [_keep("https://one.tld/p", "Cookie sandwich")]))

    assert res["collisions"] == 1
    bodies = {n.meta.id: n.body for n in iter_notes(vault_path, note_type="technique")}
    assert bodies["tech-cookie-sandwich"] == "My own notes from a live target."
    assert len(bodies) == 2


def test_rejudging_an_article_under_a_new_title_does_not_duplicate_it(
    tmp_path, vault_path, fake_embedder
):
    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    _apply(_verdicts(tmp_path, [_keep("https://one.tld/p", "First title")], name="a.jsonl"))
    res = _apply(_verdicts(tmp_path, [_keep("https://one.tld/p/", "Second title")], name="b.jsonl"))

    assert (res["kept"], res["skipped"]) == (0, 1)
    assert "already distilled as tech-first-title" in res["problems"][0]
    assert _technique_ids(vault_path) == ["tech-first-title"]


def test_disambiguated_ids_fit_the_slug_cut():
    from sift.distill.technique import technique_id
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    title = "A very long technique title " * 6
    hashed = technique_id(title, "https://one.tld/p")
    assert len(hashed) <= 75
    assert hashed == technique_id(title, "https://one.tld/p")  # deterministic
    assert hashed != technique_id(title, "https://two.tld/p")
    note = Note(meta=Frontmatter(id=hashed, type="technique", title=title), body="b")
    assert note.slug == hashed  # Note.slug cuts at 80 characters


# --- indexing -------------------------------------------------------------------------


class _RecordingStore:
    created = 0

    def __init__(self):
        type(self).created += 1
        self.calls: list[str] = []
        _RecordingStore.last = self

    def ensure_fts(self):
        self.calls.append("ensure_fts")

    def optimize(self):
        self.calls.append("optimize")
        return {"ran": False, "error": None}


def test_keeps_are_indexed_in_one_batch_and_maintained_once(tmp_path, vault_path, monkeypatch):
    """Per-note index_note cost two LanceDB commits per keep, each rewriting a manifest
    that lists every fragment."""
    batches = []

    def fake_index_notes(notes, store=None, **kw):
        batches.append([n.meta.id for n in notes])
        return len(notes)

    monkeypatch.setattr("sift.pipeline.index_notes", fake_index_notes)
    monkeypatch.setattr("sift.index.store.Store", _RecordingStore)
    for i in range(3):
        _writeup(vault_path, f"w{i}", f"Article {i}", f"https://blog{i}.tld/p")
    res = _apply(
        _verdicts(tmp_path, [_keep(f"https://blog{i}.tld/p", f"Technique {i}") for i in range(3)])
    )

    assert res["kept"] == 3
    assert len(batches) == 1 and len(batches[0]) == 3
    # One maintenance pass: Store.optimize also builds the FTS index when it is missing.
    assert _RecordingStore.last.calls == ["optimize"]


def test_a_new_index_gets_its_fts_index_from_the_one_maintenance_pass(
    tmp_path, vault_path, fake_embedder
):
    """`optimize` alone must leave the keeps keyword-searchable on a table that had no
    FTS index yet (the old code called ensure_fts first for that)."""
    from sift.index.store import Store

    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    res = _apply(_verdicts(tmp_path, [_keep("https://one.tld/p", "Technique")]))

    assert res["kept"] == 1 and res["problems"] == [] and res["chunks_indexed"] > 0
    assert Store().has_fts()


def test_a_maintenance_error_in_the_optimize_report_is_surfaced(tmp_path, vault_path, monkeypatch):
    """Store.optimize never raises; it reports a failure in its return value."""

    class ReportingStore:
        def ensure_fts(self):
            return True

        def optimize(self):
            return {"ran": False, "error": "commit conflict"}

    monkeypatch.setattr("sift.pipeline.index_notes", lambda notes, store=None, **kw: len(notes))
    monkeypatch.setattr("sift.index.store.Store", ReportingStore)
    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    res = _apply(_verdicts(tmp_path, [_keep("https://one.tld/p", "Technique")]))
    assert res["kept"] == 1
    assert res["problems"] == [
        "verdicts.jsonl: index maintenance failed (commit conflict); the notes are saved and indexed"
    ]


def test_a_note_the_vault_refuses_is_reported_and_the_batch_carries_on(
    tmp_path, vault_path, monkeypatch
):
    """An exception from save_note (e.g. the vault's IdConflict) used to abort apply,
    leaving every keep saved before it unindexed."""
    import sift.distill.manual as manual

    real_save = manual.save_note
    indexed: list[str] = []

    def picky_save(vault, note, **kw):
        if note.meta.title == "Refused":
            raise ValueError("id 'tech-refused' is already on disk as a different document")
        return real_save(vault, note, **kw)

    def fake_index_notes(notes, store=None, **kw):
        indexed.extend(n.meta.id for n in notes)
        return len(notes)

    monkeypatch.setattr(manual, "save_note", picky_save)
    monkeypatch.setattr("sift.pipeline.index_notes", fake_index_notes)
    monkeypatch.setattr("sift.index.store.Store", _RecordingStore)
    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    _writeup(vault_path, "w2", "Article two", "https://two.tld/p")
    rows = [_keep("https://one.tld/p", "Accepted first"), _keep("https://two.tld/p", "Refused")]
    rows.append(_keep("https://one.tld/p", "Accepted first"))  # a repeat still lands once
    res = _apply(_verdicts(tmp_path, rows))

    assert (res["kept"], res["skipped"]) == (2, 1)
    assert "could not save technique note 'Refused'" in res["problems"][0]
    # Indexed once: both copies in one batch would have added its chunks twice.
    assert indexed == ["tech-accepted-first"]
    assert _technique_files(vault_path) == ["Accepted first.md"]


def test_a_drop_only_apply_never_opens_the_index(tmp_path, vault_path, monkeypatch):
    class NoStore:
        def __init__(self):
            raise AssertionError("the index was opened for a drop-only apply")

    monkeypatch.setattr("sift.index.store.Store", NoStore)
    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    assert _apply(_verdicts(tmp_path, [_drop("https://one.tld/p")]))["dropped"] == 1


def test_an_indexing_failure_is_reported_and_the_notes_stay_saved(
    tmp_path, vault_path, monkeypatch
):
    def broken(notes, store=None, **kw):
        raise RuntimeError("embedder exploded")

    monkeypatch.setattr("sift.pipeline.index_notes", broken)
    monkeypatch.setattr("sift.index.store.Store", _RecordingStore)
    _writeup(vault_path, "w1", "Article one", "https://one.tld/p")
    res = _apply(_verdicts(tmp_path, [_keep("https://one.tld/p", "Technique")]))

    assert (res["kept"], res["chunks_indexed"]) == (1, 0)
    assert "sift reindex" in res["problems"][0]
    assert _technique_files(vault_path) == ["Technique.md"]


# --- export -> apply round-trip -------------------------------------------------------


def test_export_rows_point_at_the_full_note(tmp_path, vault_path):
    from sift.distill.candidates import GATE_TEXT_CHARS
    from sift.distill.manual import collect, write_candidates

    _writeup(
        vault_path, "long", "Long article", "https://a.tld/p", body="x" * (GATE_TEXT_CHARS + 10)
    )
    _writeup(vault_path, "short", "Short article", "https://b.tld/p", body="short body")
    out = tmp_path / "candidates.jsonl"
    write_candidates(collect("writeup", prefilter=False), out)

    rows = {r["note_id"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    assert rows["long"]["truncated"] is True
    assert rows["long"]["chars_total"] == GATE_TEXT_CHARS + 10
    assert len(rows["long"]["text"]) == GATE_TEXT_CHARS
    assert rows["short"]["truncated"] is False
    assert rows["long"]["slug"] == "long"
    assert rows["long"]["path"].endswith("Long article.md")


def test_export_keeps_every_row_on_one_line(tmp_path, vault_path):
    from sift.distill.manual import collect, load_candidates, write_candidates

    _writeup(vault_path, "w1", "Title with\u2028a line separator", "https://a.tld/p")
    out = tmp_path / "candidates.jsonl"
    write_candidates(collect("writeup", prefilter=False), out)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
    assert load_candidates(out)[0].title == "Title with\u2028a line separator"


def test_apply_against_the_export_needs_no_type(tmp_path, vault_path):
    """A `--type report` export applied without repeating `--type` used to skip every
    row as 'no matching candidate'; matching against the export itself removes that."""
    from sift.distill.manual import apply_verdicts, collect, load_candidates, write_candidates

    _writeup(vault_path, "r1", "A disclosed report", "https://h1.tld/reports/1", note_type="report")
    export = tmp_path / "candidates.jsonl"
    write_candidates(collect("report", prefilter=False), export)
    path = _verdicts(tmp_path, [_drop("https://h1.tld/reports/1")])

    stale = apply_verdicts(path, collect("writeup", skip_gated=False))  # the old --type default
    assert (stale["dropped"], stale["skipped"]) == (0, 1)

    res = apply_verdicts(path, load_candidates(export))
    assert (res["dropped"], res["skipped"]) == (1, 0)


def test_a_technique_note_links_back_to_its_source(tmp_path, vault_path, fake_embedder):
    """The note is distilled from an 8,000-char excerpt; the link is how search reaches
    the full article."""
    from sift.index.graph import build_link_index, expand
    from sift.vault.notes import iter_notes

    _writeup(vault_path, "writeup-cookie", "Cookie sandwich writeup", "https://one.tld/p")
    _apply(_verdicts(tmp_path, [_keep("https://one.tld/p", "Cookie sandwich")]))

    (tech,) = list(iter_notes(vault_path, note_type="technique"))
    assert tech.meta.links == ["writeup-cookie"]
    assert "writeup-cookie" in expand([tech.slug], build_link_index(vault_path))


def test_existing_candidate_constructors_still_work():
    from datetime import date

    from sift.distill.candidates import Candidate

    c = Candidate("t", "https://a.tld/p", "text", "src", date(2026, 1, 1))
    assert (c.source_id, c.source_slug, c.path) == ("", "", "")
