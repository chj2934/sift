from __future__ import annotations

import feedparser

from sift.ingest.research import _entry_date, _html_to_text, _to_note

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>PortSwigger Research</title>
<item>
  <title><![CDATA[CRLF-Powered Desync Attacks]]></title>
  <link>https://portswigger.net/research/crlf-powered-desync-attacks</link>
  <pubDate>Wed, 05 Aug 2026 00:00:00 GMT</pubDate>
  <description><![CDATA[<p>A <b>new</b> smuggling class via CRLF injection.</p>]]></description>
  <guid isPermaLink="false">abc123</guid>
</item></channel></rss>"""


def test_entry_to_writeup_note():
    entry = feedparser.parse(RSS).entries[0]
    assert _entry_date(entry).isoformat() == "2026-08-05"

    note = _to_note(entry, "https://portswigger.net/research/rss")
    assert note is not None
    assert note.meta.type == "writeup"
    assert note.meta.id == "research-crlf-powered-desync-attacks"
    assert note.meta.url == "https://portswigger.net/research/crlf-powered-desync-attacks"
    assert note.meta.created.year == 2026
    assert "research" in note.meta.tags and "portswigger.net" in note.meta.tags
    assert "smuggling" in note.body.lower()
    assert note.meta.url in note.body  # link kept in the body


def test_html_to_text_strips_markup_and_scripts():
    out = _html_to_text("<p>hello <script>evil()</script><b>world</b></p>")
    assert "evil" not in out
    assert "hello" in out and "world" in out


def test_entry_without_link_is_skipped():
    entry = feedparser.parse(RSS).entries[0]
    entry["link"] = ""
    assert _to_note(entry, "https://portswigger.net/research/rss") is None
