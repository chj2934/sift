"""Header- and fence-aware chunking of markdown note bodies, sized to the embedder window.

The pipeline embeds every chunk as ``context + text``, where the context is whatever it
puts in front (passage prefix, note title, heading path). Embedders truncate silently at
their window - bge-*-v1.5 stops at 512 tokens *including* [CLS] and [SEP] - so text past
that point never reaches the vector, only BM25. ``chunk_markdown`` therefore sizes every
chunk so that ``special_tokens + count(context(heading)) + count(text) <= max_tokens``.

Counting. Pass the embedder's own tokenizer as ``count_tokens`` (no special tokens, no
truncation) and the budget is exact. Without one, :func:`estimate_tokens` is used: a
cheap model of BERT WordPiece pre-tokenisation that over-counts on purpose. Calibrated
against the bge-large tokenizer on ~10k windows of prose, Python/JS code, curl and HTTP
transcripts, hex dumps, base64/JWT, markdown tables, JSON, tracebacks and CJK text, it
never came in under the real count, so the defaults stay inside a 512 window. The old
``chars / 4`` estimate under-counted code by up to 3x and curl/hex/base64 by 2-3x.

Structure:

* Headings split sections. A ``#`` line inside a closed code fence is a shell comment
  or a root prompt, not a heading. A fence opener that is never closed counts as text,
  so a stray fence cannot swallow the rest of the note's headings.
* A section is packed from blank-line paragraphs, except that a closed fence is a single
  unit: blank lines inside code are not paragraph breaks.
* A unit over budget is split on lines, then sentences, then words, then characters.
  A split fence is closed and re-opened, with the same info string, in every piece. A
  split table repeats its header row. Each piece still reads correctly on its own.
* Consecutive chunks share up to ``overlap_tokens`` of trailing text.

Changing anything here moves chunk boundaries for notes that are already indexed, and an
incremental reindex skips unchanged files. Bump ``CHUNKER_VERSION`` and run
``sift reindex --force``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

# Recorded by the indexer so a chunker change can be detected; bump on any change that
# moves chunk boundaries.
CHUNKER_VERSION = "fence-window-1"

# bge-*-v1.5 max_seq_length (sentence_bert_config.json). Counts [CLS]/[SEP] too.
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_SPECIAL_TOKENS = 2
# Allowance for the note title the pipeline prepends when the caller passes no `context`.
DEFAULT_CONTEXT_RESERVE = 64
# A chunk's text never gets less than this, however long the context (progress guarantee).
MIN_TEXT_TOKENS = 128
# Multiplier on the WordPiece cost model. 1.15 was the smallest margin with no
# under-count in calibration; 1.2 leaves headroom for text unlike the calibration set.
ESTIMATE_MARGIN = 1.2
# Text longer than this many chars per budgeted token is split without being counted
# first (an optimisation only: anything that turns out to fit is re-packed whole).
_OBVIOUSLY_OVER_CHARS_PER_TOKEN = 16

TokenCounter = Callable[[str], int]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?;])(\s+)")
_WORD_RE = re.compile(r"\s*\S+\s*")
_EDGE_BLANK_LINES_RE = re.compile(r"\A(?:[ \t]*\n)+|(?:\n[ \t]*)+\Z")

_LETTERS_RE = re.compile(r"[A-Za-z]+")
_DIGITS_RE = re.compile(r"[0-9]+")
_OTHER_RE = re.compile(r"[^\sA-Za-z0-9]")
_CAMEL_PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")
_VOWELS = frozenset("aeiouyAEIOUY")


@dataclass
class Chunk:
    text: str
    heading: str  # nearest heading path, e.g. "Impact" or "Steps > Setup"
    index: int


# --------------------------------------------------------------------------- estimate


def _word_cost(n: int) -> float:
    """Mean WordPiece pieces for a dictionary-like letter run of length ``n``."""
    if n == 1:
        return 1.0
    if n < 15:
        return 1.14 + 0.08 * n
    return 0.5 * n  # long single-case runs are mostly concatenations and ids


def _upper_cost(n: int) -> float:
    return 1.0 if n == 1 else max(1.5, 0.35 * n + 0.5)


@lru_cache(maxsize=1 << 16)
def _letters_cost(run: str) -> float:
    n = len(run)
    if run.islower() or (run[0].isupper() and run[1:].islower()):
        if n >= 8 and 4 * sum(c in _VOWELS for c in run) < n:
            return 0.7 * n  # vowel-poor: a hash or random id, about one piece per letter
        return _word_cost(n)
    if run.isupper():
        return _upper_cost(n)
    cost = 0.0
    for i, part in enumerate(_CAMEL_PART_RE.findall(run)):
        if part.isupper():
            if i or len(part) > 5:
                break
            cost += _upper_cost(len(part))
        elif len(part) >= 3 and any(c in _VOWELS for c in part):
            cost += _word_cost(len(part))
        else:
            break
    else:
        return cost  # camelCase / PascalCase built from word-like parts
    return 0.6 * n + 0.5  # random mixed case: base64, JWT segments, API keys


def estimate_tokens(text: str) -> int:
    """Conservative token count for a BERT-WordPiece embedder, without a tokenizer.

    Whitespace is free; every punctuation mark, symbol and non-ASCII character is one
    token (WordPiece splits each into its own piece); digit and letter runs are costed by
    length and case shape. The sum is scaled by ``ESTIMATE_MARGIN``. The estimate is
    sub-additive across whitespace (``estimate(a + " " + b) <= estimate(a) +
    estimate(b)``), so packing by summed estimates never overshoots.
    """
    if not text:
        return 0
    total = float(len(_OTHER_RE.findall(text)))
    for run in _LETTERS_RE.findall(text):
        total += _letters_cost(run)
    for run in _DIGITS_RE.findall(text):
        total += 0.6 * len(run) + 0.4
    return math.ceil(total * ESTIMATE_MARGIN)


# --------------------------------------------------------------------------- fences


def _fence_spans(lines: list[str]) -> list[tuple[int, int]]:
    """(opener, closer) line indexes, inclusive, of every *closed* code fence.

    CommonMark rules: a closer uses the opener's character, is at least as long, and
    has no info string; a backtick opener whose info string holds a backtick is not a
    fence. An opener with no closer is left as text, so later headings survive.
    """
    marks: list[tuple[str, int, bool, bool] | None] = []
    for line in lines:
        m = _FENCE_RE.match(line)
        if m is None:
            marks.append(None)
            continue
        fence, info = m.group(1), m.group(2)
        # (char, width, can close, cannot open)
        marks.append((fence[0], len(fence), not info.strip(), fence[0] == "`" and "`" in info))
    spans: list[tuple[int, int]] = []
    i, n = 0, len(lines)
    while i < n:
        mark = marks[i]
        if mark is not None and not mark[3]:
            ch, width = mark[0], mark[1]
            for j in range(i + 1, n):
                close = marks[j]
                if close is not None and close[0] == ch and close[1] >= width and close[2]:
                    spans.append((i, j))
                    i = j
                    break
        i += 1
    return spans


def _fenced_lines(lines: list[str]) -> set[int]:
    inside: set[int] = set()
    for start, end in _fence_spans(lines):
        inside.update(range(start, end + 1))
    return inside


# --------------------------------------------------------------------------- sections


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split body into (heading_path, text) sections on markdown headings."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        # Drop blank edge lines but keep the first line's indentation: a "    ```" line is
        # indented code, not a fence, and must not turn into one when re-scanned.
        text = _EDGE_BLANK_LINES_RE.sub("", "\n".join(buf)).rstrip()
        if text.strip():
            sections.append((" > ".join(p for p in stack if p), text))
        buf.clear()

    lines = body.splitlines()
    fenced = _fenced_lines(lines)
    for i, line in enumerate(lines):
        m = None if i in fenced else _HEADING_RE.match(line)
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


def _units(text: str) -> list[tuple[bool, str]]:
    """(is_fence, text) units: blank-line paragraphs, each closed fence kept whole.

    A paragraph keeps its first line's indentation: stripping it would turn an indented
    ``    ```python`` line (indented code, or a docstring) into a fence opener.
    """
    lines = text.split("\n")
    spans = dict(_fence_spans(lines))
    units: list[tuple[bool, str]] = []
    para: list[str] = []

    def flush() -> None:
        joined = "\n".join(para).rstrip()
        if joined:
            units.append((False, joined))
        para.clear()

    i = 0
    while i < len(lines):
        end = spans.get(i)
        if end is not None:
            flush()
            units.append((True, "\n".join(lines[i : end + 1])))
            i = end + 1
            continue
        if lines[i].strip():
            para.append(lines[i])
        else:
            flush()
        i += 1
    flush()
    return units


# --------------------------------------------------------------------------- packing

Atom = tuple[str, int]  # (text, tokens)


def _pack(atoms: list[Atom], sep: str, join: int, budget: int, overlap: int) -> list[Atom]:
    """Greedily pack atoms (each within budget) into windows of at most ``budget`` tokens.

    ``join`` is the token cost of ``sep`` (0 for WordPiece). After each full window the
    trailing atoms worth up to ``overlap`` tokens are repeated at the start of the next
    one - never the whole window (it would be a subset of the next) and never so much
    that the next atom no longer fits.
    """
    out: list[Atom] = []
    cur: list[Atom] = []
    cur_tok = 0
    for text, tok in atoms:
        if cur and cur_tok + join + tok > budget:
            out.append((sep.join(t for t, _ in cur), cur_tok))
            carry: list[Atom] = []
            carry_tok = 0
            for prev in reversed(cur[1:]):
                add = prev[1] + (join if carry else 0)
                if carry_tok + add > overlap:
                    break
                carry.insert(0, prev)
                carry_tok += add
            while carry and carry_tok + join + tok > budget:
                dropped = carry.pop(0)
                carry_tok -= dropped[1] + (join if carry else 0)
            cur, cur_tok = carry, carry_tok
        if cur:
            cur_tok += join
        cur.append((text, tok))
        cur_tok += tok
    if cur:
        out.append((sep.join(t for t, _ in cur), cur_tok))
    return out


def _lines(text: str) -> list[str]:
    return text.split("\n")


def _sentences(text: str) -> list[str]:
    """Sentences, each keeping the whitespace that follows it."""
    bits = _SENTENCE_END_RE.split(text)  # [sentence, gap, sentence, gap, ..., sentence]
    parts = [bits[i] + (bits[i + 1] if i + 1 < len(bits) else "") for i in range(0, len(bits), 2)]
    return [p for p in parts if p]


def _words(text: str) -> list[str]:
    """Words, each keeping the whitespace that follows it (the first also what precedes)."""
    return _WORD_RE.findall(text)


# Coarsest boundary first. "\n" keeps blank lines (they matter inside code). The other
# levels keep every whitespace character inside their parts, so joining with "" restores
# the original text exactly and two parts are always separated by whitespace: a
# WordPiece tokenizer never merges tokens across such a join.
_LEVELS: tuple[tuple[str, Callable[[str], list[str]]], ...] = (
    ("\n", _lines),
    ("", _sentences),
    ("", _words),
)


def _count_if_fits(text: str, budget: int, count: TokenCounter) -> int | None:
    """Token count of ``text`` if it fits the budget, else None.

    Text far longer than any budget is not counted at all. Calling it over budget is
    always safe - splitting and re-packing reassembles anything that turns out to fit -
    and it saves tokenising every big unit twice.
    """
    if len(text) > _OBVIOUSLY_OVER_CHARS_PER_TOKEN * budget:
        return None
    n = count(text)
    return n if n <= budget else None


def _split_text(
    text: str,
    budget: int,
    overlap: int,
    count: TokenCounter,
    joins: dict[str, int],
    level: int = 0,
) -> list[Atom]:
    """Split text into pieces of at most ``budget`` tokens, coarsest boundaries first."""
    sep, parts = "", []
    while level < len(_LEVELS):
        sep, atomize = _LEVELS[level]
        level += 1
        parts = atomize(text)
        if len(parts) > 1:
            break
    else:
        return _split_chars(text, budget, count)
    atoms: list[Atom] = []
    for part in parts:
        n = _count_if_fits(part, budget, count)
        if n is not None:
            atoms.append((part, n))
        else:
            atoms.extend(_split_text(part, budget, overlap, count, joins, level))
    pieces: list[Atom] = []
    for piece, n in _pack(atoms, sep, joins.get(sep, 0), budget, overlap):
        if sep:
            piece = _EDGE_BLANK_LINES_RE.sub("", piece)
        # Below the line level whitespace is kept: the caller may join these with "".
        if piece.strip():
            pieces.append((piece, n))
    return pieces


def _max_prefix(s: str, budget: int, count: TokenCounter, guess: int) -> int:
    """Length (within ~2%) of the longest prefix of ``s`` that fits; at least 1.

    Brackets the answer by galloping out from ``guess``, then bisects. Prefix counts
    are not strictly monotonic (cutting a word can re-tokenise it), so the result is
    always a length that was actually counted and fitted, or 1 to guarantee progress.
    """
    n = len(s)
    g = max(1, min(n, guess))
    step = max(1, g // 16)
    if count(s[:g]) <= budget:
        lo, hi = g, n + 1
        while lo < n:
            probe = min(n, lo + step)
            if count(s[:probe]) > budget:
                hi = probe
                break
            lo, step = probe, step * 2
        if lo >= n:
            return n
    else:
        lo, hi = 0, g
        while hi > 1:
            probe = max(1, hi - step)
            if count(s[:probe]) <= budget:
                lo = probe
                break
            hi, step = probe, step * 2
    while hi - lo > max(1, lo // 64):
        mid = (lo + hi) // 2
        if count(s[:mid]) <= budget:
            lo = mid
        else:
            hi = mid
    return max(lo, 1)


def _soft_cut(s: str, cut: int) -> int:
    """Move a hard cut back to just after punctuation, if any is within the last 20%."""
    for i in range(cut - 1, cut - cut // 5, -1):
        if not s[i].isalnum():
            return i + 1
    return cut


def _split_chars(text: str, budget: int, count: TokenCounter) -> list[Atom]:
    """Last resort for a run with no whitespace at all (minified JSON, base64, hex).

    Every piece but the last is reported as ``budget`` tokens, i.e. full, so the packer
    never puts two fragments of the run back into one window: a mid-run join can
    tokenise to more than the sum of its parts (``abc`` + ``def`` is not ``abcdef``).
    The run's own leading and trailing whitespace stays on the first and last piece, so
    the pieces still join with "" to the text around them.
    """
    core = text.strip()
    if not core:
        return []
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    pieces: list[Atom] = []
    rest = core
    probe = rest[: 8 * budget]
    chars_per_token = len(probe) / max(1, count(probe))
    while rest:
        cut = _max_prefix(rest, budget, count, int(budget * chars_per_token))
        if cut >= len(rest):
            pieces.append((rest, count(rest)))
            break
        head = rest[: _soft_cut(rest, cut)]
        n = count(head)
        if n > budget:  # counts are not strictly monotonic; fall back to the hard cut
            head = rest[:cut]
            n = count(head)
        pieces.append((head, max(n, budget)))
        chars_per_token = len(head) / max(1, n)
        rest = rest[len(head) :]
    if lead or trail:
        first, n = pieces[0]
        pieces[0] = (lead + first, n)
        last = pieces[-1][0] + trail
        pieces[-1] = (last, count(last))
    return pieces


def _split_fence(
    fence: str, budget: int, overlap: int, count: TokenCounter, joins: dict[str, int]
) -> list[Atom]:
    """Split an over-budget code fence; every piece is closed and re-opened."""
    lines = fence.split("\n")
    opener, closer = lines[0], lines[-1]
    frame = count(opener) + count(closer) + 2 * joins["\n"]
    inner = budget - frame
    if len(lines) < 3 or inner < max(1, budget // 4):
        return _split_text(fence, budget, overlap, count, joins)
    body = "\n".join(lines[1:-1])
    return [
        (f"{opener}\n{piece}\n{closer}", n + frame)
        for piece, n in _split_text(body, inner, overlap, count, joins)
    ]


def _table_sep_index(lines: list[str]) -> int:
    for k in range(1, len(lines)):
        if "|" in lines[k] and "|" in lines[k - 1] and _TABLE_SEP_RE.match(lines[k]):
            return k
    return -1


def _split_table(
    lines: list[str],
    k: int,
    budget: int,
    overlap: int,
    count: TokenCounter,
    joins: dict[str, int],
) -> list[Atom] | None:
    """Split an over-budget table by rows, repeating the header row in every piece.

    Lines before the header are a lead-in, kept with the first piece when short. Lines
    after the last row (no ``|``) are not rows: they are chunked on their own rather than
    filed under a repeated header that does not describe them.
    """
    end = k + 1
    while end < len(lines) and "|" in lines[end]:
        end += 1
    lead, head, rows, tail = lines[: k - 1], lines[k - 1 : k + 1], lines[k + 1 : end], lines[end:]
    if not rows:
        return None
    nl = joins["\n"]
    out: list[Atom] = []
    lead_text = "\n".join(lead).rstrip()
    lead_tok = count(lead_text) + nl if lead_text else 0
    if lead_tok > budget // 4:  # a long lead-in is chunked on its own, not repeated
        out.extend(_split_block(lead_text, budget, overlap, count, joins))
        lead_text, lead_tok = "", 0
    head_text = "\n".join(head)
    head_tok = count(head_text) + nl
    inner = budget - head_tok - lead_tok
    if inner < max(1, budget // 4):
        return None
    for i, (piece, n) in enumerate(_split_text("\n".join(rows), inner, overlap, count, joins)):
        if i == 0 and lead_text:
            out.append((f"{lead_text}\n{head_text}\n{piece}", lead_tok + head_tok + n))
        else:
            out.append((f"{head_text}\n{piece}", head_tok + n))
    tail_text = "\n".join(tail).rstrip()
    if tail_text:
        out.extend(_split_block(tail_text, budget, overlap, count, joins))
    return out


def _split_block(
    text: str, budget: int, overlap: int, count: TokenCounter, joins: dict[str, int]
) -> list[Atom]:
    """A paragraph as atoms within budget: whole if it fits, else by table rows or text."""
    n = _count_if_fits(text, budget, count)
    if n is not None:
        return [(text, n)]
    lines = text.split("\n")
    k = _table_sep_index(lines)
    if k > 0:
        pieces = _split_table(lines, k, budget, overlap, count, joins)
        if pieces:
            return pieces
    return _split_text(text, budget, overlap, count, joins)


def _chunk_section(
    text: str, budget: int, overlap: int, count: TokenCounter, joins: dict[str, int]
) -> list[str]:
    atoms: list[Atom] = []
    for is_fence, unit in _units(text):
        if not is_fence:
            atoms.extend(_split_block(unit, budget, overlap, count, joins))
            continue
        n = _count_if_fits(unit, budget, count)
        if n is not None:
            atoms.append((unit, n))
        else:
            atoms.extend(_split_fence(unit, budget, overlap, count, joins))
    return [t.rstrip() for t, _ in _pack(atoms, "\n\n", joins["\n\n"], budget, overlap)]


def _join_cost(sep: str, count: TokenCounter) -> int:
    """Extra tokens a separator adds between two pieces (0 for WordPiece)."""
    return max(0, count(f"x{sep}x") - 2 * count("x"))


def _text_budget(
    heading: str,
    window: int,
    special_tokens: int,
    count: TokenCounter,
    context: Callable[[str], str] | None,
) -> int:
    if context is not None:
        overhead = count(context(heading))
    else:
        overhead = DEFAULT_CONTEXT_RESERVE + (count(heading) if heading else 0)
    floor = max(1, min(MIN_TEXT_TOKENS, window // 4))
    return max(floor, window - special_tokens - overhead)


def chunk_markdown(
    body: str,
    *,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count_tokens: TokenCounter | None = None,
    context: Callable[[str], str] | None = None,
    special_tokens: int = DEFAULT_SPECIAL_TOKENS,
) -> list[Chunk]:
    """Split a note body into heading-labelled chunks that fit the embedder window.

    Args:
        max_tokens: the embedder's window (sentence-transformers ``max_seq_length``),
            special tokens included. ``None`` means ``DEFAULT_MAX_TOKENS``.
        overlap_tokens: trailing text repeated at the start of the next chunk.
        count_tokens: the embedder's tokenizer as ``text -> token count``, without
            special tokens and without truncation. ``None`` uses the conservative
            :func:`estimate_tokens`.
        context: ``heading -> exact text the caller embeds before the chunk``,
            separator included (e.g. ``f"{passage_prefix}{title}\\n{heading}\\n"``).
            ``None`` reserves the heading plus ``DEFAULT_CONTEXT_RESERVE`` tokens for a
            title the chunker cannot see.
        special_tokens: tokens the embedder adds to every passage ([CLS] + [SEP]).

    Every chunk satisfies ``special_tokens + count(context(heading)) + count(text) <=
    max_tokens``, unless the context alone leaves less than ``MIN_TEXT_TOKENS``
    (capped at a quarter of the window), in which case the text gets that floor.
    The result is a pure function of the arguments, so chunking the same note twice
    gives the same chunks.
    """
    count = count_tokens or estimate_tokens
    window = max_tokens or DEFAULT_MAX_TOKENS
    joins = {sep: _join_cost(sep, count) for sep in ("\n\n", "\n")}
    chunks: list[Chunk] = []
    for heading, text in _split_sections(body):
        budget = _text_budget(heading, window, special_tokens, count, context)
        overlap = max(0, min(overlap_tokens, budget // 2))
        for piece in _chunk_section(text, budget, overlap, count, joins):
            chunks.append(Chunk(text=piece, heading=heading, index=len(chunks)))
    return chunks
