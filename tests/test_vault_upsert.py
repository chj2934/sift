"""save_note is an upsert by id (contract C5).

Before: a retitled, moved, renamed-in-Obsidian or re-ingested note got a *second*
file carrying the same id (76 ids in 152 files in the real vault), and the index
flipped between the copies on every reindex. Two different documents that share an
id (KEV and NVD on one CVE, truncated title ids) must still never overwrite each
other: that write is refused with IdConflict.
"""

from __future__ import annotations

import pytest


def _note(note_id, title, *, body="body", note_type="technique", source=None, url=None):
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    meta = Frontmatter(id=note_id, type=note_type, title=title, source=source, url=url)
    return Note(meta=meta, body=body)


def _files(vault):
    """Every note file, vault-relative, excluding the trash."""
    return sorted(
        p.relative_to(vault).as_posix()
        for p in vault.rglob("*.md")
        if ".trash" not in p.relative_to(vault).parts
    )


def _ids_on_disk(vault):
    from sift.vault.notes import iter_notes

    return sorted(n.meta.id for n in iter_notes(vault))


# --- one file per id -------------------------------------------------------------


def test_retitle_rewrites_and_renames_the_one_file(vault_path):
    from sift.vault.notes import load_note, write_note

    note = _note("tech-a", "Old title", source="manual")
    first = write_note(vault_path, note)
    assert first.created and not first.title_clash

    fresh = _note("tech-a", "New title", body="v2", source="manual")  # no path: by id
    res = write_note(vault_path, fresh)
    assert not res.created and res.renamed
    assert _files(vault_path) == ["technique/New title.md"]
    assert load_note(res.path).body.strip() == "v2"


def test_renamed_in_obsidian_then_saved_keeps_the_users_name(vault_path):
    """resolve_idea / epss re-save a loaded note; the user had renamed its file."""
    import os

    from sift.vault.notes import load_note, save_note

    path = save_note(vault_path, _note("idea-1", "Long hypothesis title", source="sift"))
    mine = path.with_name("Short.md")
    os.replace(path, mine)

    loaded = load_note(mine)
    loaded.meta.extra["status"] = "failed"
    assert save_note(vault_path, loaded) == mine
    assert _files(vault_path) == ["technique/Short.md"]

    # Even a fresh object (no path) finds it by id and leaves the user's name alone.
    again = _note("idea-1", "Long hypothesis title v2", source="sift")
    assert save_note(vault_path, again) == mine
    assert _files(vault_path) == ["technique/Short.md"]


def test_moved_into_a_subfolder_stays_there(vault_path):
    import os

    from sift.vault.notes import save_note

    path = save_note(vault_path, _note("tech-m", "Moved", source="manual"))
    sub = vault_path / "technique" / "web"
    sub.mkdir()
    os.replace(path, sub / path.name)

    assert (
        save_note(vault_path, _note("tech-m", "Moved", body="v2", source="manual"))
        == sub / "Moved.md"
    )
    assert _files(vault_path) == ["technique/web/Moved.md"]


def test_a_gap_in_the_numbered_chain_never_forks_a_copy(vault_path):
    from sift.vault.notes import save_note

    save_note(vault_path, _note("t-1", "Same"))
    p2 = save_note(vault_path, _note("t-2", "Same"))
    p3 = save_note(vault_path, _note("t-3", "Same"))
    assert (p2.name, p3.name) == ("Same (2).md", "Same (3).md")
    p2.unlink()  # e.g. `sift prune --yes`

    for locate in (True, False):  # by id, and through the filename probe alone
        assert save_note(vault_path, _note("t-3", "Same", body="again"), locate_by_id=locate) == p3
    assert _files(vault_path) == ["technique/Same (3).md", "technique/Same.md"]


def test_explicit_existing_path_is_used(vault_path):
    from sift.vault.notes import save_note

    path = save_note(vault_path, _note("tech-e", "Elsewhere"))
    custom = path.with_name("Custom name.md")
    path.rename(custom)
    assert save_note(vault_path, _note("tech-e", "Elsewhere"), existing=custom) == custom
    assert _files(vault_path) == ["technique/Custom name.md"]


def test_a_path_now_holding_another_id_is_left_untouched(vault_path):
    from sift.vault.notes import load_note, save_note

    path = save_note(vault_path, _note("tech-x", "Shared name"))
    loaded = load_note(path)
    path.unlink()
    other = save_note(vault_path, _note("tech-y", "Shared name", body="someone else"))
    assert other == path  # the name was reused by a different note

    saved = save_note(vault_path, loaded)
    assert saved != path
    assert load_note(path).meta.id == "tech-y"
    assert load_note(path).body.strip() == "someone else"


def test_user_chosen_filename_is_not_renamed_on_retitle(vault_path):
    from sift.vault.notes import write_note

    p = vault_path / "technique" / "My own name.md"
    p.parent.mkdir(parents=True)
    p.write_text("---\nid: tech-u\ntype: technique\ntitle: Alpha\n---\n\nb\n", encoding="utf-8")
    res = write_note(vault_path, _note("tech-u", "Beta"))
    assert res.path == p and not res.renamed


def test_rename_never_lands_on_another_note(vault_path):
    from sift.vault.notes import load_note, write_note

    write_note(vault_path, _note("tech-b", "Beta"))
    write_note(vault_path, _note("tech-a", "Alpha"))
    res = write_note(vault_path, _note("tech-a", "Beta"))
    assert res.path.name == "Beta (2).md"
    assert load_note(vault_path / "technique" / "Beta.md").meta.id == "tech-b"
    assert _files(vault_path) == ["technique/Beta (2).md", "technique/Beta.md"]


def test_rename_false_keeps_the_filename(vault_path):
    from sift.vault.notes import write_note

    write_note(vault_path, _note("tech-r", "Before"))
    res = write_note(vault_path, _note("tech-r", "After"), rename=False)
    assert res.path.name == "Before.md"


# --- same id, different document: refused, never overwritten -----------------------


def test_kev_after_nvd_is_refused_not_clobbered(vault_path):
    from sift.vault.notes import IdConflict, save_note

    nvd = save_note(
        vault_path,
        _note("CVE-2024-0001", "CVE-2024-0001: heap overflow in libfoo", note_type="cve",
              source="nvd", url="https://nvd.nist.gov/vuln/detail/CVE-2024-0001",
              body="## Description\n\nNVD text"),
    )  # fmt: skip
    before = nvd.read_bytes()
    kev = _note("CVE-2024-0001", "CVE-2024-0001: Foo heap overflow", note_type="cve",
                source="cisa-kev", url="https://nvd.nist.gov/vuln/detail/CVE-2024-0001",
                body="## Required action\n\npatch")  # fmt: skip
    with pytest.raises(IdConflict) as err:
        save_note(vault_path, kev)
    assert err.value.reason == "source" and err.value.path == nvd
    assert err.value.existing_source == "nvd"
    assert nvd.read_bytes() == before
    assert _files(vault_path) == ["cve/CVE-2024-0001- heap overflow in libfoo.md"]


def test_colliding_title_ids_with_different_urls_are_refused(vault_path):
    """Two different articles whose truncated title slugs give one id."""
    from sift.vault.notes import IdConflict, save_note

    a = save_note(vault_path, _note("writeup-x", "Account takeover via OAuth", note_type="writeup",
                                    source="writeups", url="https://a.tld/post"))  # fmt: skip
    with pytest.raises(IdConflict) as err:
        save_note(vault_path, _note("writeup-x", "Account takeover via OAuth state", note_type="writeup",
                                    source="writeups", url="https://b.tld/other"))  # fmt: skip
    assert err.value.reason == "url"
    assert _files(vault_path) == [a.relative_to(vault_path).as_posix()]


@pytest.mark.parametrize(
    "url",
    [
        "http://www.a.tld/post/",
        "https://a.tld/post?utm_source=rss#comments",
        "https://A.TLD/post?ref=feed",
    ],
)
def test_same_article_seen_through_another_url_form_is_an_update(vault_path, url):
    from sift.vault.notes import load_note, save_note

    first = save_note(vault_path, _note("writeup-y", "Same article", note_type="writeup",
                                        source="writeups", url="https://a.tld/post"))  # fmt: skip
    again = save_note(vault_path, _note("writeup-y", "Same article, retitled", note_type="writeup",
                                        source="writeups", url=url, body="v2"))  # fmt: skip
    assert _files(vault_path) == [again.relative_to(vault_path).as_posix()]
    assert load_note(again).body.strip() == "v2"
    assert first.parent == again.parent


def test_verify_identity_false_is_an_explicit_merge(vault_path):
    from sift.vault.notes import save_note

    path = save_note(vault_path, _note("CVE-2024-0002", "t", note_type="cve", source="nvd"))
    merged = _note("CVE-2024-0002", "t", note_type="cve", source="cisa-kev", body="merged")
    assert save_note(vault_path, merged, verify_identity=False) == path
    assert _ids_on_disk(vault_path) == ["CVE-2024-0002"]


def test_same_document_and_canonical_url_rules():
    from sift.vault.notes import canonical_url, same_document
    from sift.vault.schema import Frontmatter

    assert canonical_url("https://www.Example.com/a/b/?utm_medium=x&p=12&fbclid=z#frag") == (
        "example.com/a/b?p=12"
    )
    assert canonical_url("http://example.com/?p=1") != canonical_url("http://example.com/?p=2")
    assert canonical_url(None) == "" and canonical_url("  ") == ""

    meta = Frontmatter(id="w", type="writeup", title="T", source="s", url="https://a.tld/x")
    base = {"id": "w", "source": "s", "title": "T", "url": "https://a.tld/x"}
    assert same_document(base, meta) == (True, "")
    assert same_document({**base, "id": "v"}, meta) == (False, "id")
    assert same_document({**base, "source": "nvd"}, meta) == (False, "source")
    assert same_document({**base, "title": "Other"}, meta) == (True, "")  # retitle, same url
    assert same_document({**base, "url": "https://a.tld/y"}, meta) == (True, "")  # same title
    assert same_document({**base, "url": "https://b.tld/z", "title": "Other"}, meta) == (
        False,
        "url",
    )
    no_url = Frontmatter(id="w", type="writeup", title="T", source="s")
    assert same_document({**base, "url": None, "title": "Other"}, no_url) == (True, "")
    assert same_document({**base, "title": "Other"}, no_url) == (True, "")  # one side has none


def test_locate_note_by_id_and_by_document(vault_path):
    from sift.vault.notes import locate_note, save_note
    from sift.vault.schema import Frontmatter

    path = save_note(vault_path, _note("writeup-z", "Z", note_type="writeup", source="w",
                                       url="https://a.tld/z"))  # fmt: skip
    assert locate_note(vault_path, "writeup-z") == path
    same = Frontmatter(id="writeup-z", type="writeup", title="Z", source="w", url="http://a.tld/z/")
    other = Frontmatter(
        id="writeup-z", type="writeup", title="Q", source="w", url="https://b.tld/q"
    )
    assert locate_note(vault_path, "writeup-z", meta=same) == path
    assert locate_note(vault_path, "writeup-z", meta=other) is None
    assert locate_note(vault_path, "nope") is None


def test_title_clash_is_reported_only_for_a_new_file(vault_path):
    """run_source counts collisions; an existing note found by id under another name
    is not one (it used to be counted as `actual != note_path()`)."""
    from sift.vault.notes import write_note

    assert not write_note(vault_path, _note("a", "Same")).title_clash
    clash = write_note(vault_path, _note("b", "Same"))
    assert clash.created and clash.title_clash and clash.path.name == "Same (2).md"
    again = write_note(vault_path, _note("b", "Same"))
    assert not again.created and not again.title_clash and again.path == clash.path


def test_existing_duplicates_are_reported_and_left_alone(vault_path):
    from sift.vault.notes import load_note, write_note

    d = vault_path / "technique"
    d.mkdir()
    for name, body in (("Dup.md", "one"), ("Dup elsewhere.md", "two")):
        (d / name).write_text(f"---\nid: tech-d\ntype: technique\ntitle: Dup\n---\n\n{body}\n",
                              encoding="utf-8")  # fmt: skip
    res = write_note(vault_path, _note("tech-d", "Dup", body="three"))
    assert res.path.name == "Dup.md"  # the copy at its title path wins
    assert [p.name for p in res.duplicates] == ["Dup elsewhere.md"]
    assert load_note(d / "Dup elsewhere.md").body.strip() == "two"


# --- soft delete -----------------------------------------------------------------


def test_delete_moves_every_copy_to_the_trash(vault_path):
    from sift.vault.catalog import fresh_catalog
    from sift.vault.notes import delete_note, iter_notes, load_note, locate_note, save_note

    save_note(vault_path, _note("keep", "Keep"))
    path = save_note(vault_path, _note("gone", "Gone", note_type="finding"))
    sub = vault_path / "finding" / "old"
    sub.mkdir()
    (sub / "Gone copy.md").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    moved = delete_note("gone", vault=vault_path, reason="wrong host")
    rel = sorted(p.relative_to(vault_path).as_posix() for p in moved)
    assert rel == [".trash/finding/Gone.md", ".trash/finding/old/Gone copy.md"]
    assert load_note(moved[0]).meta.extra["deleted_reason"] == "wrong host"
    assert [n.meta.id for n in iter_notes(vault_path)] == ["keep"]
    assert locate_note(vault_path, "gone") is None
    assert fresh_catalog(vault_path).by_id("gone") == ()

    # Deleting the same name again never overwrites what the trash holds.
    save_note(vault_path, _note("gone", "Gone", note_type="finding"))
    again = delete_note("gone", vault=vault_path)
    assert [p.name for p in again] == ["Gone (2).md"]
    assert delete_note("never-existed", vault=vault_path) == []


# --- concurrency and cost ----------------------------------------------------------


def test_parallel_saves_of_one_title_never_lose_a_note(vault_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from sift.vault.notes import load_note, save_note

    barrier = threading.Barrier(8)

    def save(i):
        barrier.wait()
        return save_note(vault_path, _note(f"race-{i}", "Race", body=f"body {i}"))

    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(save, range(8)))
    assert len(set(paths)) == 8
    assert sorted(load_note(p).meta.id for p in paths) == sorted(f"race-{i}" for i in range(8))
    assert len(_files(vault_path)) == 8


def test_parallel_saves_of_one_note_leave_one_whole_file(vault_path):
    from concurrent.futures import ThreadPoolExecutor

    from sift.vault.notes import load_note, save_note

    save_note(vault_path, _note("same", "Same note"))
    with ThreadPoolExecutor(8) as pool:
        list(
            pool.map(
                lambda i: save_note(vault_path, _note("same", "Same note", body=f"v{i}")), range(16)
            )
        )
    assert _files(vault_path) == ["technique/Same note.md"]
    assert load_note(vault_path / "technique" / "Same note.md").body.strip().startswith("v")


def test_bulk_saves_do_not_rewalk_the_vault_per_note(vault_path, monkeypatch):
    from sift.vault import catalog as catalog_mod
    from sift.vault import notes as notes_mod
    from sift.vault.notes import save_note

    for i in range(20):  # an existing vault to search
        save_note(vault_path, _note(f"old-{i}", f"Old {i}", note_type="report"))

    walks = 0
    real_walk = catalog_mod.walk_note_entries

    def counting_walk(*a, **k):
        nonlocal walks
        walks += 1
        return real_walk(*a, **k)

    monkeypatch.setattr(catalog_mod, "walk_note_entries", counting_walk)
    monkeypatch.setattr(notes_mod, "_CARRIER_MAX_AGE_S", 3600.0)  # time-independent
    for i in range(300):
        save_note(vault_path, _note(f"new-{i}", f"New {i}", note_type="report"))
    assert walks <= 3, walks
    assert len(_files(vault_path)) == 320


def test_a_file_created_outside_sift_is_seen_by_the_next_save(vault_path, monkeypatch):
    """Folder mtimes catch an Obsidian create/rename even inside the staleness window."""
    from sift.vault import notes as notes_mod
    from sift.vault.notes import save_note

    monkeypatch.setattr(notes_mod, "_CARRIER_MAX_AGE_S", 3600.0)
    save_note(vault_path, _note("seed", "Seed"))
    hand = vault_path / "technique" / "Typed by hand.md"
    hand.write_text("---\nid: tech-h\ntype: technique\ntitle: Hand\n---\n\nb\n", encoding="utf-8")
    assert save_note(vault_path, _note("tech-h", "Hand", body="v2")) == hand
    assert _files(vault_path) == ["technique/Seed.md", "technique/Typed by hand.md"]


def test_the_write_lock_is_reentrant_and_lives_outside_the_vault(vault_path):
    from sift.config import get_settings
    from sift.vault.notes import save_note, write_lock

    with write_lock(vault_path), write_lock(vault_path):
        save_note(vault_path, _note("locked", "Locked"))
    assert not any(p.suffix == ".lock" for p in vault_path.rglob("*"))
    assert (get_settings().resolved_db() / "_sift").is_dir()
