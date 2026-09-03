from __future__ import annotations

from sift.vault.chunk import chunk_markdown


def test_splits_on_headings():
    body = "# Title\n\nintro para\n\n## Impact\n\nbad things happen\n\n## Remediation\n\nfix it"
    chunks = chunk_markdown(body, max_tokens=1000)
    headings = {c.heading for c in chunks}
    assert "Title > Impact" in headings or "Impact" in " ".join(headings)
    assert any("bad things" in c.text for c in chunks)


def test_large_section_is_packed_with_overlap():
    para = "lorem ipsum dolor sit amet " * 40  # ~200 tokens
    body = "## Big\n\n" + "\n\n".join([para] * 10)
    chunks = chunk_markdown(body, max_tokens=300, overlap_tokens=50)
    assert len(chunks) > 3
    # consecutive chunks should share some trailing/leading text (overlap)
    assert any(
        chunks[i].text.split("\n\n")[-1] in chunks[i + 1].text for i in range(len(chunks) - 1)
    )


def test_empty_body():
    assert chunk_markdown("") == []


def test_plain_text_no_headings():
    chunks = chunk_markdown("just a sentence with no markdown structure at all")
    assert len(chunks) == 1
    assert chunks[0].heading == ""
