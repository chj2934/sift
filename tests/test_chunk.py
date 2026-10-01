from __future__ import annotations

import random
import re

import pytest

TITLE = "Stored XSS in the admin invoice export leads to account takeover"

POC_BODY = (
    "## Reproduction\n\nRun the following:\n\n```bash\n# install deps\npip install foo\n\n"
    "# exploit\npython exploit.py --target x\n```\n\nShell transcript:\n\n```\n$ nc -lvp 4444\n"
    "# id\nuid=0(root)\n```\n\n## Impact\n\nRoot RCE."
)


def _ws_count(text: str) -> int:
    """Fake tokenizer: one token per whitespace-separated word, whitespace free."""
    return len(text.split())


def _context(heading: str) -> str:  # what the pipeline embeds before a chunk
    return f"{TITLE}\n{heading}\n" if heading else f"{TITLE}\n"


def _assert_fits(chunks, count, window: int = 512) -> None:
    for c in chunks:
        used = 2 + count(_context(c.heading)) + count(c.text)
        assert used <= window, (c.index, used, c.text[:80])


def _assert_fences_balanced(text: str) -> None:
    from sift.vault.chunk import _FENCE_RE, _fenced_lines

    lines = text.split("\n")
    fenced = _fenced_lines(lines)
    stray = [ln for i, ln in enumerate(lines) if _FENCE_RE.match(ln) and i not in fenced]
    assert not stray, (stray, text[:120])


# --------------------------------------------------------------------------- basics


def test_splits_on_headings():
    from sift.vault.chunk import chunk_markdown

    body = "# Title\n\nintro para\n\n## Impact\n\nbad things happen\n\n## Remediation\n\nfix it"
    chunks = chunk_markdown(body, max_tokens=1000)
    headings = {c.heading for c in chunks}
    assert "Title > Impact" in headings or "Impact" in " ".join(headings)
    assert any("bad things" in c.text for c in chunks)


def test_large_section_is_packed_with_overlap():
    from sift.vault.chunk import chunk_markdown

    para = "lorem ipsum dolor sit amet " * 40  # ~200 tokens
    body = "## Big\n\n" + "\n\n".join([para] * 10)
    chunks = chunk_markdown(body, max_tokens=300, overlap_tokens=50)
    assert len(chunks) > 3
    # consecutive chunks should share some trailing/leading text (overlap)
    assert any(
        chunks[i].text.split("\n\n")[-1] in chunks[i + 1].text for i in range(len(chunks) - 1)
    )


def test_empty_body():
    from sift.vault.chunk import chunk_markdown

    assert chunk_markdown("") == []


def test_plain_text_no_headings():
    from sift.vault.chunk import chunk_markdown

    chunks = chunk_markdown("just a sentence with no markdown structure at all")
    assert len(chunks) == 1
    assert chunks[0].heading == ""


def test_chunking_is_deterministic():
    """The pipeline zips chunks with their vectors; chunking twice must agree exactly."""
    from sift.vault.chunk import chunk_markdown

    body = (
        POC_BODY + "\n\n" + "\n".join(f"curl -sk https://t.io/{i}?a={i * 7919}" for i in range(400))
    )
    first = [(c.heading, c.text, c.index) for c in chunk_markdown(body, count_tokens=_ws_count)]
    again = [(c.heading, c.text, c.index) for c in chunk_markdown(body, count_tokens=_ws_count)]
    assert first == again
    assert [i for _, _, i in first] == list(range(len(first)))


# --------------------------------------------------------------------------- fences vs headings


def test_hash_lines_inside_fences_are_not_headings():
    from sift.vault.chunk import chunk_markdown

    chunks = chunk_markdown(POC_BODY)
    assert {c.heading for c in chunks} == {"Reproduction", "Impact"}
    assert any("# install deps" in c.text and "python exploit.py" in c.text for c in chunks)
    assert any("# id" in c.text and "uid=0(root)" in c.text for c in chunks)
    impact = [c for c in chunks if c.heading == "Impact"]
    assert [c.text for c in impact] == ["Root RCE."]
    for c in chunks:
        _assert_fences_balanced(c.text)


def test_tilde_fence_is_not_closed_by_backticks():
    from sift.vault.chunk import chunk_markdown

    body = "## A\n\n~~~\n```\n# not a heading\n```\n~~~\n\n## B\n\ntext"
    chunks = chunk_markdown(body)
    assert {c.heading for c in chunks} == {"A", "B"}
    assert "# not a heading" in next(c.text for c in chunks if c.heading == "A")


def test_unclosed_fence_does_not_swallow_later_headings():
    from sift.vault.chunk import chunk_markdown

    body = "## A\n\n```\nstray opener left by extraction\n\n## B\n\nbody of b"
    chunks = chunk_markdown(body)
    assert any(c.heading.endswith("B") for c in chunks)
    assert next(c for c in chunks if "body of b" in c.text).heading.endswith("B")


def test_info_string_line_is_never_a_closer():
    from sift.vault.chunk import chunk_markdown

    body = "## A\n\n```bash\necho 1\n```bash\n# comment\n```\n\n## B\n\ntext"
    chunks = chunk_markdown(body)
    assert {c.heading for c in chunks} == {"A", "B"}
    assert "# comment" in next(c.text for c in chunks if c.heading == "A")


def test_closer_must_be_as_long_as_opener():
    from sift.vault.chunk import chunk_markdown

    body = "## A\n\n````md\n```\n# inner\n```\n````\n\n## B\n\ntext"
    assert {c.heading for c in chunk_markdown(body)} == {"A", "B"}


def test_backtick_in_info_string_is_not_a_fence():
    from sift.vault.chunk import chunk_markdown

    body = "## A\n\n``` a`b\n\n# Real heading\n\ntext\n\n```\n"
    assert "Real heading" in {c.heading for c in chunk_markdown(body)}


def test_paragraph_keeps_first_line_indentation():
    """An indented ``` line is indented code, not a fence; the chunk must not make it one."""
    from sift.vault.chunk import chunk_markdown

    body = "## I\n\n    ```python\n    x = 1\n\n## J\n\ntext"
    chunks = chunk_markdown(body)
    assert chunks[0].heading == "I"
    assert chunks[0].text == "    ```python\n    x = 1"


# --------------------------------------------------------------------------- window budget


def test_long_curl_fence_fits_window_and_every_piece_is_a_closed_fence():
    from sift.vault.chunk import chunk_markdown

    rnd = random.Random(1)
    curl = "\n".join(
        f"# step {i}\ncurl -sk 'https://api.t.io/v1/u/{i}?t={rnd.randbytes(6).hex()}' "
        f"-H 'Authorization: Bearer {rnd.randbytes(9).hex()}' -d 'a=1&b=2&c=3'"
        for i in range(200)
    )
    body = "## PoC\n\nSend these:\n\n```bash\n" + curl + "\n```\n\n## Impact\n\nAccount takeover."
    chunks = chunk_markdown(body, count_tokens=_ws_count, context=_context)
    _assert_fits(chunks, _ws_count)
    assert {c.heading for c in chunks} == {"PoC", "Impact"}
    fence_chunks = [c for c in chunks if "curl -sk" in c.text]
    assert len(fence_chunks) > 3
    for c in fence_chunks:
        _assert_fences_balanced(c.text)
        assert "```bash\n" in c.text  # re-opened with its info string
        assert c.text.rstrip().endswith("```")
    for i in range(200):  # no command lost
        assert any(f"/v1/u/{i}?" in c.text for c in fence_chunks)


def test_table_without_blank_lines_fits_window_and_repeats_header():
    from sift.vault.chunk import chunk_markdown

    head = "| Method | Endpoint | Auth | Status |\n|---|---|---|---|"
    rows = [f"| GET | /api/v2/items/{i}/export?fmt=csv | bearer | 200 |" for i in range(600)]
    body = "## Endpoints\n\nAll routes:\n" + head + "\n" + "\n".join(rows)
    chunks = chunk_markdown(body, count_tokens=_ws_count, context=_context)
    _assert_fits(chunks, _ws_count)
    assert len(chunks) > 10
    assert chunks[0].text.startswith("All routes:\n" + head + "\n")
    for c in chunks[1:]:
        assert c.text.startswith(head + "\n")
    for row in rows:
        assert any(row in c.text for c in chunks)


def test_rows_after_the_table_are_not_filed_under_its_header():
    from sift.vault.chunk import chunk_markdown

    head = "| A | B |\n|---|---|"
    rows = "\n".join(f"| a{i} | b{i} |" for i in range(300))
    body = "## T\n\n" + head + "\n" + rows + "\n" + " ".join(["prose"] * 900)
    chunks = chunk_markdown(body, count_tokens=_ws_count, context=_context)
    _assert_fits(chunks, _ws_count)
    prose = [c for c in chunks if "prose" in c.text]
    assert len(prose) >= 2
    for c in prose:
        assert head + "\nprose" not in c.text  # never a fake row under a repeated header
        assert re.search(r"(\A|\n\n)prose", c.text)  # starts a chunk or its own paragraph
    assert sum(c.text.count("prose") for c in prose) >= 900


def test_giant_single_paragraph_fits_window_without_losing_or_merging_words():
    from sift.vault.chunk import chunk_markdown

    para = " ".join(f"word{i} lorem ipsum dolor" for i in range(7000))  # ~28k tokens
    body = "## Notes\n\n" + para
    chunks = chunk_markdown(body, count_tokens=_ws_count, context=_context)
    _assert_fits(chunks, _ws_count)
    assert len(chunks) > 50
    for c in chunks:
        assert c.text in body  # a contiguous slice of the note: nothing merged or dropped
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(para.split())


def test_long_line_keeps_its_whitespace_and_sentences():
    from sift.vault.chunk import chunk_markdown

    line = " ".join(
        f"Sentence {i} explains the  bypass of filter{i} via header injection; it works."
        for i in range(400)
    )
    chunks = chunk_markdown("## S\n\n" + line)
    assert len(chunks) > 5
    for c in chunks:
        assert c.text in line


def test_split_sentence_tail_keeps_its_space_before_the_next_sentence():
    """A sentence too long for the window is split on words; its tail is then re-packed
    with the following sentences and must not fuse with them ("end.Next")."""
    from sift.vault.chunk import chunk_markdown

    giant = " ".join(f"w{i}" for i in range(1500)) + "."
    line = "Short opener here. " + giant + " Next short sentence. Another one follows."
    chunks = chunk_markdown("## S\n\n" + line, count_tokens=_ws_count, context=_context)
    _assert_fits(chunks, _ws_count)
    assert len(chunks) >= 3
    for c in chunks:
        assert c.text in line, c.text[-60:]
    assert any("w1499. Next short sentence." in c.text for c in chunks)


def test_whitespace_free_blob_is_split_without_gluing_neighbours():
    """A run longer than the window is hard-cut; its tail must not fuse with the next word."""
    import base64

    from sift.vault.chunk import chunk_markdown, estimate_tokens

    blob = base64.b64encode(random.Random(3).randbytes(2600)).decode()
    body = f"## B\n\nlead words here {blob} trailing words after the blob."
    chunks = chunk_markdown(body)
    for c in chunks:
        assert 2 + 64 + estimate_tokens("B") + estimate_tokens(c.text) <= 512
        for word in c.text.split():
            assert word in body.split() or word in blob, word[:40]
    fragments = "".join(w for c in chunks for w in c.text.split() if w in blob)
    assert blob in fragments
    assert any("trailing words after the blob." in c.text for c in chunks)


def test_default_estimate_reserves_room_for_heading_and_title():
    from sift.vault.chunk import DEFAULT_CONTEXT_RESERVE, chunk_markdown, estimate_tokens

    rnd = random.Random(5)
    hexdump = "\n".join(rnd.randbytes(16).hex(" ") for _ in range(400))
    body = "## Memory > Heap dump\n\n" + hexdump + "\n\n" + "plain words " * 2000
    chunks = chunk_markdown(body)
    assert len(chunks) > 5
    for c in chunks:
        overhead = 2 + DEFAULT_CONTEXT_RESERVE + estimate_tokens(c.heading)
        assert overhead + estimate_tokens(c.text) <= 512


def test_long_context_leaves_the_text_a_floor():
    from sift.vault.chunk import MIN_TEXT_TOKENS, chunk_markdown

    body = "## H\n\n" + " ".join(f"w{i}" for i in range(3000))
    chunks = chunk_markdown(
        body, count_tokens=_ws_count, context=lambda h: "ctx " * 1000, overlap_tokens=0
    )
    assert all(0 < _ws_count(c.text) <= MIN_TEXT_TOKENS for c in chunks)
    assert len(chunks) >= 3000 // MIN_TEXT_TOKENS


def test_overlap_never_pushes_a_chunk_over_budget():
    from sift.vault.chunk import chunk_markdown

    paras = [" ".join(["x"] * n) for n in (100, 30, 30, 30, 200, 5, 5, 300, 40)]
    body = "## O\n\n" + "\n\n".join(paras * 4)
    chunks = chunk_markdown(body, count_tokens=_ws_count, context=_context, overlap_tokens=80)
    _assert_fits(chunks, _ws_count)
    assert len(chunks) > 4


# --------------------------------------------------------------------------- estimator


def test_estimate_is_subadditive_across_whitespace():
    from sift.vault.chunk import estimate_tokens

    rnd = random.Random(9)
    words = ["curl", "-sk", "https://a.io/x?y=1", "deadbeef" * 3, "Bearer", "eyJhbGciOi", "ὕβρις"]
    for _ in range(300):
        a = " ".join(rnd.choice(words) for _ in range(rnd.randint(1, 8)))
        b = " ".join(rnd.choice(words) for _ in range(rnd.randint(1, 8)))
        assert estimate_tokens(a + " " + b) <= estimate_tokens(a) + estimate_tokens(b)
        assert estimate_tokens(a + "\n" + b) <= estimate_tokens(a) + estimate_tokens(b)
    assert estimate_tokens("") == 0


def test_estimate_counts_code_far_above_chars_over_four():
    from sift.vault.chunk import estimate_tokens

    rnd = random.Random(2)
    hexline = rnd.randbytes(64).hex()
    curl = "curl -sk 'https://api.t.io/v1/users/42?token=" + hexline + "' -H 'X-Id: 1'"
    for sample in (hexline, curl):
        assert estimate_tokens(sample) > 2 * (len(sample) // 4)


def _cached_bge_tokenizer():
    """The bge-large tokenizer from the local HF cache only (never the network)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from tokenizers import Tokenizer
    except ImportError:
        return None
    for repo in ("BAAI/bge-large-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5"):
        path = try_to_load_from_cache(repo, "tokenizer.json")
        if isinstance(path, str):
            tok = Tokenizer.from_file(path)
            tok.no_truncation()
            return tok
    return None


def test_estimate_never_undercounts_the_real_bge_tokenizer():
    """Control: the estimate is the only thing between a default chunk and truncation."""
    import base64

    from sift.vault.chunk import chunk_markdown, estimate_tokens

    tok = _cached_bge_tokenizer()
    if tok is None:
        pytest.skip("bge tokenizer not in the local HF cache")

    def real(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    rnd = random.Random(11)
    samples = [
        "\n".join(
            f"curl -sk 'https://api.t.io/v1/u/{i}?token={rnd.randbytes(12).hex()}' "
            f"-H 'Authorization: Bearer {base64.urlsafe_b64encode(rnd.randbytes(24)).decode()}'"
            for i in range(20)
        ),
        "\n".join(rnd.randbytes(16).hex(" ") for _ in range(40)),
        base64.b64encode(rnd.randbytes(1200)).decode(),
        "| Method | Endpoint | Status |\n|---|---|---|\n"
        + "\n".join(f"| POST | /api/v1/orders/{i}/refund | 403 |" for i in range(40)),
        "def handler(request):\n    user_id = request.GET['id']\n    return render(user_id)\n" * 15,
        "The attacker controls the redirect_uri parameter, so the OAuth code leaks. " * 20,
        "認証バイパスの脆弱性を発見しました。管理者権限で任意のファイルを読み取れます。" * 10,
    ]
    for s in samples:
        assert estimate_tokens(s) >= real(s), s[:60]
    for s in samples:
        for c in chunk_markdown("## PoC\n\n" + s * 3):
            assert real(f"{TITLE}\n{c.heading}\n{c.text}") + 2 <= 512


def test_exact_counter_fills_the_real_window():
    from sift.vault.chunk import chunk_markdown

    tok = _cached_bge_tokenizer()
    if tok is None:
        pytest.skip("bge tokenizer not in the local HF cache")

    def real(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    body = "## Notes\n\n" + " ".join(f"param{i}=value{i * 31}&" for i in range(3000))
    chunks = chunk_markdown(body, count_tokens=real, context=_context)
    sizes = [real(_context(c.heading) + c.text) + 2 for c in chunks]
    assert max(sizes) <= 512
    assert sorted(sizes)[len(sizes) // 2] > 400  # exact counting packs close to the window


def test_heading_regex_unchanged_for_plain_markdown():
    from sift.vault.chunk import chunk_markdown

    body = "# A\n\none\n\n## B\n\ntwo\n\n### C\n\nthree\n\n## D\n\nfour"
    got = [(c.heading, c.text) for c in chunk_markdown(body)]
    assert got == [("A", "one"), ("A > B", "two"), ("A > B > C", "three"), ("A > D", "four")]
    assert not any(re.match(r"^#", t) for _, t in got)
