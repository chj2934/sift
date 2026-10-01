"""Helpers `ingest.base` offers the sources: `safe_get`, the identity helpers and
`KnownNotes`. Offline: httpx runs against a MockTransport."""

from __future__ import annotations

import logging

import pytest

BAD_URLS = [
    "http://[::1/x",  # httpx.InvalidURL: neither an HTTPError nor a ValueError
    "https://xn--ls8h.la/p",  # IDNA-invalid (an emoji label)
    "https://\U0001f4a9.la/p",  # the same, unencoded
    "https://a.tld/\x00x",  # a control character
]


def _client(seen: list[str]):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/bad":
            return httpx.Response(500, text="oops")
        return httpx.Response(200, text="<html>ok</html>")

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("url", BAD_URLS)
def test_safe_get_survives_links_httpx_cannot_parse(url, capfd, caplog):
    from sift.ingest.base import safe_get

    seen: list[str] = []
    with _client(seen) as client:
        assert safe_get(client, url, what="writeups") is None
        # Positive control in the same run: a good link still fetches.
        assert safe_get(client, "https://a.tld/good").text == "<html>ok</html>"
    assert capfd.readouterr().out == ""
    assert "writeups: fetch failed" in caplog.text


def test_safe_get_status_handling(caplog):
    from sift.ingest.base import safe_get

    caplog.set_level(logging.WARNING)
    seen: list[str] = []
    with _client(seen) as client:
        assert safe_get(client, "https://a.tld/bad") is None
        r = safe_get(client, "https://a.tld/bad", raise_for_status=False)
        assert r is not None and r.status_code == 500
    assert len(seen) == 2


def test_fetch_errors_cover_what_a_per_item_fetch_raises():
    import httpx

    from sift.ingest.base import FETCH_ERRORS

    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client:
        for url in BAD_URLS:
            try:
                client.get(url)
            except FETCH_ERRORS:
                continue
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"{url!r} raised {type(exc).__name__}, not in FETCH_ERRORS")
            # Accepted by httpx: fine, nothing to catch.


@pytest.mark.parametrize(
    ("note_id", "title", "derived"),
    [
        ("research-security-advisory", "Security Advisory", True),
        ("writeup-account-takeover", "Account Takeover", True),
        ("top10-desync-endgame", "Desync endgame", True),
        ("CVE-2021-44228", "Apache Log4j2 RCE", False),
        ("h1-12345", "XSS in search", False),
        ("chromium-doc-docs-security-faq-md", "Chromium Security FAQ", False),
        ("research-security-advisory-1a2b3c4d", "Security Advisory", False),
    ],
)
def test_is_title_derived(note_id, title, derived):
    from sift.ingest.base import is_title_derived
    from sift.vault.schema import Frontmatter

    meta = Frontmatter(id=note_id, type="writeup", title=title)
    assert is_title_derived(meta) is derived


def test_url_note_id_needs_a_url():
    from sift.ingest.base import url_note_id

    with pytest.raises(ValueError):
        url_note_id("research-x", "")
    assert url_note_id("research-x-", "https://a.tld/p").startswith("research-x-")
    assert "--" not in url_note_id("research-x-", "https://a.tld/p")


def _save(vault, note_id, title, url, source="blog.tld"):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    meta = Frontmatter(id=note_id, type="writeup", title=title, url=url, source=source)
    return save_note(vault, Note(meta=meta, body="text"))


def test_known_urls_are_scoped_to_the_id_family(vault_path):
    """A Top-10 nomination of an article a research feed carried is still its own note
    (tags, nomination year): the top10 skip must only see top10 notes."""
    from sift.ingest.base import KnownNotes

    _save(vault_path, "research-desync", "Desync", "https://blog.tld/desync")
    known = KnownNotes(vault_path)
    assert known.has_url("https://blog.tld/desync/?utm_source=x")
    assert known.has_url("https://blog.tld/desync", family="research")
    assert not known.has_url("https://blog.tld/desync", family="top10")
    assert known.ids_for_url("http://www.blog.tld/desync") == ("research-desync",)
    assert not known.has_url(None) and not known.has_url("")


def test_known_notes_sees_what_a_run_adds(vault_path):
    """A source that yields an entry should `add` it, so the same post from a second
    feed in one run is skipped."""
    from sift.ingest.base import KnownNotes
    from sift.vault.schema import Frontmatter

    known = KnownNotes(vault_path)
    meta = Frontmatter(id="research-new", type="writeup", title="New", url="https://b.tld/n")
    assert not known.has(meta)
    known.add(meta)
    assert known.has(meta)
    assert known.resolve_id(meta) == "research-new"


def test_resolve_id_prefers_the_article_already_on_disk(vault_path):
    """An upstream retitle (same URL, new title-derived id) updates the existing note
    instead of forking a second one; a different URL under a taken id is re-keyed."""
    from sift.ingest.base import KnownNotes, url_note_id
    from sift.vault.schema import Frontmatter

    _save(vault_path, "research-old-title", "Old title", "https://blog.tld/p/1")
    known = KnownNotes(vault_path)

    retitled = Frontmatter(id="research-new-title", type="writeup", title="New title",
                           url="https://blog.tld/p/1")  # fmt: skip
    assert known.resolve_id(retitled) == "research-old-title"

    other = Frontmatter(id="research-old-title", type="writeup", title="Old title",
                        url="https://blog.tld/p/2")  # fmt: skip
    assert known.resolve_id(other) == url_note_id("research-old-title", "https://blog.tld/p/2")

    no_url = Frontmatter(id="research-old-title", type="writeup", title="Old title")
    assert known.resolve_id(no_url) == "research-old-title"
    assert len(list((vault_path / "writeup").glob("*.md"))) == 1, "resolve_id never writes"


def test_have_note_finds_a_renamed_or_moved_note(vault_path):
    """The old have_note probed only the title path; a note the user renamed or moved
    in Obsidian (or saved as 'Title (2).md') read as missing and was fetched again."""
    from sift.ingest.base import have_note
    from sift.vault.schema import Frontmatter

    path = _save(vault_path, "chrome-release-120", "Chrome 120 - 12 fixes", "https://cr.tld/120",
                 source="chrome-releases")  # fmt: skip
    moved = vault_path / "writeup" / "sub" / "my own name.md"
    moved.parent.mkdir()
    path.rename(moved)

    meta = Frontmatter(id="chrome-release-120", type="writeup", title="Chrome 120 - 13 fixes",
                       url="https://cr.tld/120", source="chrome-releases")  # fmt: skip
    assert have_note(vault_path, meta)
    other = meta.model_copy(update={"source": "someone-else"})
    assert not have_note(vault_path, other), "same id, different document"
