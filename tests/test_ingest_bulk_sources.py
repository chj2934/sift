"""Bulk sources on a re-run: nothing re-yielded that is already stored and cannot
have changed, nothing the user added overwritten, and the Chrome ledger never
rebuilt from a truncated window.

Before: h1-public, hacktivity and chromium-fixes re-yielded every record on every
run (rewrite + re-embed), h1-mine overwrote the user's annotations, and
`ingest chrome-releases --limit 1` replaced the rolling ledger with one release's rows
while it still claimed the full window.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest


@pytest.fixture
def index(monkeypatch):
    from sift.ingest import base

    state: dict = {"batches": []}

    class _Store:
        def indexed_ids(self):
            return {i for b in state["batches"] for i in b}

        def optimize(self):
            return {"error": None}

    store = _Store()

    def fake_index(notes, _store):
        state["batches"].append([n.meta.id for n in notes])
        return len(notes)

    monkeypatch.setattr(base, "index_notes", fake_index)
    monkeypatch.setattr(base, "Store", lambda: store)
    return state


@pytest.fixture(autouse=True)
def _cutoff(monkeypatch):
    monkeypatch.setenv("SIFT_MODEL_CUTOFF", "2026-04-01")
    from sift.config import get_settings

    get_settings.cache_clear()


# --- h1-public --------------------------------------------------------------------------


class _Table:
    def __init__(self, rows):
        self.rows = rows

    def to_pylist(self):
        return list(self.rows)


def _h1_rows(*ids):
    return [
        {"id": i, "title": f"Report {i}", "vulnerability_information": f"report {i} details " * 5}
        for i in ids
    ]


def test_h1_public_skips_stored_reports(vault_path, index, monkeypatch):
    from sift.ingest import h1_public
    from sift.ingest.base import run_source

    rows = _h1_rows(1, 2)
    monkeypatch.setattr(h1_public, "_download_table", lambda client, fname: _Table(rows))
    monkeypatch.setattr(h1_public, "FILES", ["train.parquet"])
    run_source("h1-public", h1_public.source())

    rows.append(_h1_rows(3)[0])
    ids = [n.meta.id for n in h1_public.source()]
    assert ids == ["h1-3"]
    assert [n.meta.id for n in h1_public.source(refresh=True)] == ["h1-1", "h1-2", "h1-3"]


def test_h1_public_limit_counts_stored_reports(vault_path, index, monkeypatch):
    """`--limit N` means the first N reports, so a deeper second run adds only the rest."""
    from sift.ingest import h1_public
    from sift.ingest.base import run_source

    rows = _h1_rows(1, 2, 3, 4)
    monkeypatch.setattr(h1_public, "_download_table", lambda client, fname: _Table(rows))
    monkeypatch.setattr(h1_public, "FILES", ["train.parquet"])
    run_source("h1-public", h1_public.source(limit=2))

    assert [n.meta.id for n in h1_public.source(limit=3)] == ["h1-3"]


# --- h1-mine ----------------------------------------------------------------------------


def _my_report(state="triaged", body="Original writeup body."):
    return {
        "id": "777",
        "attributes": {
            "state": state,
            "title": "IDOR in billing API",
            "vulnerability_information": body,
            "created_at": "2026-08-01T00:00:00Z",
            "triaged_at": "2026-08-02T00:00:00Z",
            "closed_at": "2026-08-09T00:00:00Z" if state == "resolved" else None,
        },
        "relationships": {"program": {"data": {"attributes": {"name": "Acme"}}}},
    }


def test_h1_mine_rerun_keeps_annotations_and_updates_state(vault_path, index, monkeypatch):
    from sift.ingest import h1_api
    from sift.ingest.base import run_source
    from sift.vault.notes import load_note, save_note

    items = [_my_report()]
    monkeypatch.setattr(h1_api, "_client", lambda: _NullClient())
    monkeypatch.setattr(h1_api, "_paginate", lambda client, path, params: iter(items))
    run_source("h1-mine", h1_api.my_reports())

    path = next((vault_path / "report").glob("*.md"))
    note = load_note(path)
    note.body += "\n\n## Retest\nStill reproducible on v2."
    note.meta.tags.append("follow-up")
    note.meta.links.append("target-acme")
    save_note(vault_path, note)

    items[:] = [_my_report(state="resolved", body="API body changed")]
    res = run_source("h1-mine", h1_api.my_reports())

    after = load_note(path)
    assert res.updated == 1 and len(list((vault_path / "report").glob("*.md"))) == 1
    assert "## Retest\nStill reproducible on v2." in after.body
    assert "Original writeup body." in after.body
    assert {"follow-up", "resolved"} <= set(after.meta.tags) and "triaged" not in after.meta.tags
    assert after.meta.links == ["target-acme"]
    assert after.meta.extra["state"] == "resolved" and after.meta.extra["closed_at"]

    again = run_source("h1-mine", h1_api.my_reports())
    assert again.unchanged == 1


class _NullClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_hacktivity_skips_stored_reports_within_the_same_window(vault_path, index, monkeypatch):
    from sift.ingest import h1_api
    from sift.ingest.base import run_source

    def item(n):
        return {
            "id": str(n),
            "attributes": {"title": f"Report {n}", "vulnerability_information": "x" * 50},
        }

    feed = [item(3), item(2), item(1)]  # newest first
    monkeypatch.setattr(h1_api, "_client", lambda: _NullClient())
    monkeypatch.setattr(h1_api, "_paginate", lambda client, path, params: iter(feed))
    run_source("h1-hacktivity", h1_api.hacktivity(limit=2))

    feed.insert(0, item(4))
    ids = [n.meta.id for n in h1_api.hacktivity(limit=2)]
    assert ids == ["h1act-4"], "the newest window, not ever-deeper paging"


# --- chromium-fixes ---------------------------------------------------------------------


def test_chromium_fixes_skips_commits_already_stored(vault_path, index, monkeypatch):
    from sift.ingest import chromium_fixes as cf
    from sift.ingest.base import run_source

    commits = [
        cf.Commit(
            "a" * 40, "2026-09-18", "Dev", "Fix UAF in CredentialManager task", "Bug: 1", ["x.cc"]
        ),
    ]
    monkeypatch.setattr(cf, "_resolved_src", lambda: Path("."))
    monkeypatch.setattr(cf, "fetch_log", lambda src, since, paths: "log")
    monkeypatch.setattr(cf, "parse_log", lambda raw: iter(list(commits)))
    run_source("chromium-fixes", cf.source())

    commits.append(cf.Commit("b" * 40, "2026-09-19", "Dev", "Fix UAF in Dawn", "Bug: 2", ["y.cc"]))
    assert [n.meta.id for n in cf.source()] == ["chromium-fix-bbbbbbbbbbbb"]
    assert len(list(cf.source(refresh=True))) == 2


# --- chromium-docs ----------------------------------------------------------------------


def test_a_doc_whose_revision_url_and_title_both_changed_is_still_updated(vault_path, index):
    """The id is the doc's path. Without pinning the note to its file, a sync that
    changed the revision in the URL and the H1 would be refused as a different
    document (IdConflict) and the doc would never update again."""
    from sift.ingest.base import run_source
    from sift.ingest.existing import attach_existing
    from sift.vault.notes import Note, load_note
    from sift.vault.schema import Frontmatter

    def doc(rev, title, body):
        meta = Frontmatter(
            id="chromium-doc-docs-security-faq-md",
            type="reference",
            title=title,
            source="chromium-src",
            url=f"https://source.chromium.org/chromium/chromium/src/+/{rev}:docs/security/faq.md",
        )
        return Note(meta=meta, body=body)

    run_source("chromium-docs", [doc("aaa", "Security FAQ", "old text")])
    plain = run_source("chromium-docs", [doc("bbb", "Chrome Security FAQ", "new text")])
    assert plain.id_conflicts == 1, "the hazard this guards against"

    res = run_source(
        "chromium-docs",
        [attach_existing(vault_path, doc("bbb", "Chrome Security FAQ", "new text"))],
    )

    files = list((vault_path / "reference").glob("*.md"))
    assert res.updated == 1 and len(files) == 1
    assert load_note(files[0]).body.strip() == "new text"


# --- chrome-releases --------------------------------------------------------------------

ROW = (
    "[$3,000][<a href='x'> {bug} </a>] High CVE-2026-{n}: Use after free in Dawn. "
    "Reported by Someone on 2026-08-0{d}<br>"
)


def _entry(n: int, day: int, *, reward="$3,000"):
    body = ROW.format(bug=1000 + n, n=9000 + n, d=day).replace("$3,000", reward)
    return {
        "title": {"$t": "Stable Channel Update for Desktop"},
        "published": {"$t": f"2026-09-{10 + n:02d}T09:00:00.000-07:00"},
        "content": {"$t": f"Stable updated to 140.0.{n}.1. {body}"},
        "link": [{"rel": "alternate", "href": f"https://chromereleases.test/{n}"}],
    }


@pytest.fixture
def blogger(monkeypatch):
    """Serve pages of release entries; `fail_after` makes later pages raise."""
    from sift.ingest import chrome_releases as cr

    state = {"entries": [_entry(3, 3), _entry(2, 2), _entry(1, 1)], "fail_after": None}

    def fetch_page(client, label, start):
        if state["fail_after"] is not None and start > state["fail_after"]:
            import httpx

            raise httpx.ConnectError("timed out")
        page = state["entries"][start - 1 : start - 1 + 2]  # pages of two
        return page

    monkeypatch.setattr(cr, "_fetch_page", fetch_page)
    monkeypatch.setattr(cr, "PAGE_SIZE", 2)
    return state


def test_limit_caps_notes_not_the_ledger(vault_path, blogger):
    from sift.ingest import chrome_releases as cr

    notes = list(cr.source(limit=1))

    releases = [n for n in notes if n.meta.id != "chrome-vrp-ledger"]
    ledger = [n for n in notes if n.meta.id == "chrome-vrp-ledger"]
    assert len(releases) == 1 and len(ledger) == 1
    assert ledger[0].meta.extra["rows"] == 3, "the ledger saw every release in the window"
    assert "security fixes" in releases[0].meta.title and "1 security" not in releases[0].meta.title


def test_an_incomplete_walk_does_not_rewrite_the_ledger(vault_path, blogger):
    from sift.ingest import chrome_releases as cr

    blogger["fail_after"] = 1  # the second page times out

    notes = list(cr.source())

    assert "chrome-vrp-ledger" not in [n.meta.id for n in notes]
    assert len(notes) == 2, "the releases that were fetched are still written"


def test_a_corrected_release_post_is_written_again(vault_path, index, blogger):
    from sift.ingest import chrome_releases as cr
    from sift.ingest.base import run_source
    from sift.vault.notes import load_note

    run_source("chrome-releases", cr.source(ledger=False))
    assert list(cr.source(ledger=False)) == [], "unchanged posts are not re-yielded"

    blogger["entries"][0] = _entry(3, 3, reward="$7,000")  # Google corrected the reward
    res = run_source("chrome-releases", cr.source(ledger=False))

    assert res.updated == 1 and res.written == 0
    files = list((vault_path / "reference").glob("*.md"))
    assert len(files) == 3
    assert any("$7,000" in load_note(p).body for p in files)


def test_release_created_date(vault_path, blogger):
    from sift.ingest import chrome_releases as cr

    notes = [n for n in cr.source(ledger=False)]
    assert notes[0].meta.created == date(2026, 9, 13)
