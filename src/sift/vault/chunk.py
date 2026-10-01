"""Header-aware chunking of markdown note bodies.

Chunk sizing is measured in *approximate* tokens (chars / 4) to avoid a
tokenizer dependency. BGE-M3 handles 8k tokens, so ~800-token chunks with
~100-token overlap leave plenty of headroom and keep retrieval granular.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 100


@dataclass
class Chunk:
    text: str
    heading: str  # nearest heading path, e.g. "Impact" or "Steps > Setup"
    index: int


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split body into (heading_path, text) sections on markdown headings."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append((" > ".join(p for p in stack if p), text))
        buf.clear()

    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
        else:
            buf.append(line)
    flush()
    return sections or [("", body.strip())]


def _pack(text: str, heading: str, max_tokens: int, overlap_tokens: int) -> list[tuple[str, str]]:
    """Greedily pack paragraphs into <= max_tokens windows with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []
    out: list[tuple[str, str]] = []
    cur: list[str] = []
    cur_tok = 0
    for para in paras:
        ptok = _approx_tokens(para)
        if cur and cur_tok + ptok > max_tokens:
            out.append((heading, "\n\n".join(cur)))
            # carry overlap: keep trailing paras up to overlap budget
            carry: list[str] = []
            budget = overlap_tokens
            for prev in reversed(cur):
                t = _approx_tokens(prev)
                if t > budget:
                    break
                carry.insert(0, prev)
                budget -= t
            cur = carry
            cur_tok = sum(_approx_tokens(p) for p in cur)
        cur.append(para)
        cur_tok += ptok
    if cur:
        out.append((heading, "\n\n".join(cur)))
    return out


def chunk_markdown(
    body: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, text in _split_sections(body):
        for h, packed in _pack(text, heading, max_tokens, overlap_tokens):
            chunks.append(Chunk(text=packed, heading=h, index=len(chunks)))
    if not chunks and body.strip():
        chunks.append(Chunk(text=body.strip(), heading="", index=0))
    return chunks
