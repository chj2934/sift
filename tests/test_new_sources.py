"""Offline tests for the PentesterLand and Top-10 sources.

Both keep parsing in pure helpers so these run against fixture strings with no
network, matching test_ingest_research.py.
"""

from __future__ import annotations

PL_ENTRY = {
    "Links": [
        {
            "Title": "Vulnerabilities in Open Source C2 Frameworks",
            "Link": "https://blog.includesecurity.com/2024/09/vulns-in-c2/",
        }
    ],
    "Authors": ["Laurence Tennant"],
    "Programs": ["Bishop Fox (Sliver)", "Havoc"],
    "Bugs": ["RCE", "OS command injection"],
    "Bounty": "$1,500",
    "PublicationDate": "2024-09-18",
    "AddedDate": "2024-09-24",
}

TOP10_HTML = """
<p>Nominations:</p>
<a href="https://twitter.com/albinowax">@albinowax</a>
<a href="https://portswigger.net/research/something">our own research</a>
<a href="https://adragos.ro/fontleak/">Fontleak: stealing text with CSS fonts</a>
<a href="https://adragos.ro/fontleak/">fontleak</a>
<a href="https://api.whatsapp.com/send?text=share">share</a>
<a href="https://github.com/someone/poc-repo">PoC</a>
<a href="https://gist.github.com/x/abc123">gist writeup</a>
<a href="https://0x44.xyz/blog/cve-2023-4369/">CVE-2023-4369 <b>deep dive</b></a>
"""


def test_pentesterland_entry_maps_onto_frontmatter():
    from sift.ingest.writeups import _to_note

    note = _to_note(PL_ENTRY)
    assert note is not None
    assert note.meta.title == "Vulnerabilities in Open Source C2 Frameworks"
    assert note.meta.url == "https://blog.includesecurity.com/2024/09/vulns-in-c2/"
    assert note.meta.source == "blog.includesecurity.com"
    assert note.meta.created.year == 2024
    assert note.meta.program == "Bishop Fox (Sliver)"
    assert note.meta.bounty == 1500.0
    assert note.meta.extra["bug_classes"] == ["RCE", "OS command injection"]


def test_pentesterland_handles_no_bounty_and_bad_date():
    from sift.ingest.writeups import _to_note

    entry = {**PL_ENTRY, "Bounty": "-", "PublicationDate": "not-a-date"}
    note = _to_note(entry)
    assert note.meta.bounty is None
    assert note.meta.created is None


def test_pentesterland_skips_entries_with_no_link():
    from sift.ingest.writeups import _to_note

    assert _to_note({**PL_ENTRY, "Links": []}) is None
    assert _to_note({**PL_ENTRY, "Links": [{"Title": "x", "Link": ""}]}) is None


def test_top10_extracts_research_links_only():
    from sift.ingest.top10 import _extract_links

    links = _extract_links(TOP10_HTML)
    hosts = {u.split("/")[2] for u in links}

    assert "adragos.ro" in hosts
    assert "0x44.xyz" in hosts
    assert "gist.github.com" in hosts, "gists are usually the writeup itself"
    # Social, sharing, bare repos and PortSwigger's own posts are excluded.
    assert "twitter.com" not in hosts
    assert "api.whatsapp.com" not in hosts
    assert "portswigger.net" not in hosts
    assert "github.com" not in hosts, "bare repos are PoCs, not writeups"


def test_top10_prefers_the_longest_anchor_text():
    """The same URL often appears twice - once titled, once as a bare link."""
    from sift.ingest.top10 import _extract_links

    links = _extract_links(TOP10_HTML)
    assert links["https://adragos.ro/fontleak/"] == "Fontleak: stealing text with CSS fonts"


def test_top10_strips_markup_inside_anchor_text():
    from sift.ingest.top10 import _extract_links

    links = _extract_links(TOP10_HTML)
    assert links["https://0x44.xyz/blog/cve-2023-4369/"] == "CVE-2023-4369 deep dive"


def test_top10_prefers_the_page_title_over_the_anchor():
    """Anchors in nomination lists are prose as often as titles, so <title> wins."""
    from sift.ingest.top10 import _article_title

    html = "<html><head><title>Fontleak - stealing text with CSS</title></head></html>"
    assert _article_title(html, "here", "https://x.tld") == "Fontleak - stealing text with CSS"
    assert _article_title(html, "an earlier post", "https://x.tld") == (
        "Fontleak - stealing text with CSS"
    )


def test_top10_falls_back_to_anchor_when_page_has_no_title():
    from sift.ingest.top10 import _article_title

    assert _article_title("<html></html>", "Cookie tossing attacks explained", "https://x.tld") == (
        "Cookie tossing attacks explained"
    )
    # A prose anchor is not a title - better to end up with the host than a fragment.
    assert _article_title("<html></html>", "an earlier post", "https://x.tld") == "an earlier post"


def test_top10_denies_subdomains_of_denied_hosts():
    """uk.linkedin.com slipped an exact-match denylist and produced a note."""
    from sift.ingest.top10 import _is_research_link

    assert not _is_research_link("https://uk.linkedin.com/in/sdalili")
    assert not _is_research_link("https://mobile.twitter.com/albinowax")
    assert _is_research_link("https://blog.doyensec.com/2024/01/thing.html")


def test_top10_skips_author_bio_and_index_pages():
    from sift.ingest.top10 import _is_research_link

    assert not _is_research_link("https://blog.flomb.net/about/")
    assert not _is_research_link("https://example.com/tags")
    assert _is_research_link("https://example.com/about-ssrf-in-pdf-renderers")
