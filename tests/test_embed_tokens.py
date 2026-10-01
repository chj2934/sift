"""Token budget API: counts and windows come from tokenizer/config files, never the weights.

The tokenizer is a tiny word-level one built in the test, saved as a local model
directory, so nothing touches the network or the Hugging Face cache.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pytest

WORDS = ["a", "b", "c", "curl", "x", "y", "z", "search", "document", "title", "impact", "poc"]


def _toy_tokenizer(*, truncate_at: int | None = None, pad_to: int | None = None):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3}
    for w in WORDS:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    if truncate_at:
        tok.enable_truncation(max_length=truncate_at)
    if pad_to:
        tok.enable_padding(pad_id=3, pad_token="[PAD]", length=pad_to)
    return tok


def _model_dir(
    root: Path,
    *,
    tokenizer=None,
    sbert: dict | None = None,
    tok_cfg: dict | None = None,
) -> Path:
    d = root / "toy-model"
    d.mkdir()
    if tokenizer is not None:
        tokenizer.save(str(d / "tokenizer.json"))
    if sbert is not None:
        (d / "sentence_bert_config.json").write_text(json.dumps(sbert), encoding="utf-8")
    if tok_cfg is not None:
        (d / "tokenizer_config.json").write_text(json.dumps(tok_cfg), encoding="utf-8")
    return d


@pytest.fixture(autouse=True)
def _clean():
    from sift.index import embed

    embed.get_embedder.cache_clear()
    yield
    embed.get_embedder.cache_clear()


@pytest.fixture
def records():
    got: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            got.append(record)

    handler = _Keep(level=logging.WARNING)
    logger = logging.getLogger("sift.index")
    logger.addHandler(handler)
    yield got
    logger.removeHandler(handler)


@pytest.fixture
def no_model_loads(monkeypatch):
    """Fail loudly if anything tries to load embedding weights."""
    from sift.index import embed

    calls: list[tuple[str, str]] = []

    def refuse(model_name: str, device: str):
        calls.append((model_name, device))
        raise AssertionError("token counting must not load the model")

    monkeypatch.setattr(embed, "_construct_backend", refuse)
    return calls


def test_counts_ignore_truncation_and_padding_baked_into_tokenizer_json(tmp_path):
    """fastembed's tokenizer truncates at the window, so it can never see an overflow."""
    from tokenizers import Tokenizer

    from sift.index.embed import Embedder

    d = _model_dir(tmp_path, tokenizer=_toy_tokenizer(truncate_at=8, pad_to=16))
    text = " ".join(["a"] * 20)

    # Control: the file as saved really does truncate/pad, so the trap is live.
    raw = Tokenizer.from_file(str(d / "tokenizer.json"))
    assert len(raw.encode(text, add_special_tokens=False).ids) == 16

    assert Embedder(str(d), "cpu").count_tokens(text) == 20


def test_counts_exclude_special_tokens_and_report_them_separately(tmp_path, no_model_loads):
    from sift.index.embed import Embedder

    d = _model_dir(tmp_path, tokenizer=_toy_tokenizer(), sbert={"max_seq_length": 256})
    emb = Embedder(str(d), "cpu")
    assert emb.count_tokens("a b, c!") == 5  # a b , c !
    assert emb.count_tokens("") == 0
    assert emb.num_special_tokens == 2  # [CLS] ... [SEP]
    assert emb.max_seq_length == 256
    assert no_model_loads == []
    assert emb._backend is None


def test_passage_budget_subtracts_prefix_header_specials_and_margin(tmp_path):
    from sift.index.embed import Embedder

    d = _model_dir(tmp_path, tokenizer=_toy_tokenizer(), sbert={"max_seq_length": 256})
    emb = Embedder(str(d), "cpu")
    header = "title\nimpact poc\n"  # the pipeline's title + heading lines: 3 tokens
    assert emb.passage_budget(header) == 256 - 2 - 3 - 2
    assert emb.passage_budget(header, margin=0) == 256 - 2 - 3

    emb.passage_prefix = "search_document: "  # nomic/e5 models prefix every passage
    assert emb.count_tokens("search_document: ") == 4  # search _ document :
    assert emb.passage_budget(header) == 256 - 2 - (4 + 3) - 2

    # A chunk sized to the budget fits the window once the passage is assembled.
    budget = emb.passage_budget(header)
    passage = emb.passage_prefix + header + " ".join(["curl"] * budget)
    assert emb.count_tokens(passage) + emb.num_special_tokens <= emb.max_seq_length


def test_window_is_the_smaller_configured_limit(tmp_path):
    from sift.index.embed import Embedder

    d = _model_dir(
        tmp_path,
        tokenizer=_toy_tokenizer(),
        sbert={"max_seq_length": 384},
        tok_cfg={"model_max_length": 512},
    )
    assert Embedder(str(d), "cpu").max_seq_length == 384


def test_window_from_configs_ignores_sentinels_and_junk():
    from sift.index.embed import _window_from_configs

    assert _window_from_configs([{"max_seq_length": 512}]) == 512
    # tokenizer_config.json uses ~1e30 for "unlimited".
    assert _window_from_configs([{"model_max_length": 1000000000000000019884624838656}]) is None
    assert _window_from_configs([{"max_seq_length": 512}, {"model_max_length": 1e30}]) == 512
    assert _window_from_configs([None, {"max_length": 128, "model_max_length": 512}]) == 128
    assert _window_from_configs([{"max_seq_length": True}, {"max_seq_length": "512"}]) is None
    assert _window_from_configs([{"max_seq_length": 0}, {}]) is None
    assert _window_from_configs([]) is None


def test_window_falls_back_to_known_models_then_512(monkeypatch):
    from sift.index import embed

    monkeypatch.setattr(embed, "_find_model_file", lambda *a, **k: None)
    assert embed.Embedder("nomic-ai/nomic-embed-text-v1.5", "cpu").max_seq_length == 8192
    assert embed.Embedder("BAAI/bge-large-en-v1.5", "cpu").max_seq_length == 512
    assert embed.Embedder("someone/unknown-model", "cpu").max_seq_length == 512


def test_missing_tokenizer_degrades_to_a_conservative_estimate(
    monkeypatch, tmp_path, capsys, records, no_model_loads
):
    from sift.index import embed

    # A hub model id with nothing cached: one download attempt, which fails (offline).
    attempts: list[tuple[str, str]] = []
    monkeypatch.setattr(embed, "_hf_cached", lambda *a, **k: None)
    monkeypatch.setattr(embed, "_fastembed_cached", lambda *a, **k: None)
    monkeypatch.setattr(embed, "_hf_download", lambda repo, name: attempts.append((repo, name)))

    emb = embed.Embedder("someone/unknown-model", "cpu")
    poc = "curl -sk 'https://a.test/x?id=1'"
    assert emb.count_tokens(poc) == embed._estimate_tokens(poc)
    assert emb.count_tokens(poc * 2) == embed._estimate_tokens(poc * 2)
    assert emb.num_special_tokens == 2
    assert attempts == [("someone/unknown-model", "tokenizer.json")]  # tried once, cached
    assert sum("conservative estimates" in r.getMessage() for r in records) == 1
    assert capsys.readouterr().out == ""
    assert no_model_loads == []

    # Control: the same lookup chain finds a real tokenizer when one exists.
    d = _model_dir(tmp_path, tokenizer=_toy_tokenizer())
    assert embed.Embedder(str(d), "cpu").count_tokens(poc) != embed._estimate_tokens(poc)


def test_fallback_estimate_is_the_chunkers_calibrated_one(monkeypatch):
    """One cost model: a chunk sized by the chunker's estimate counts the same here."""
    import sys

    from sift.index import embed
    from sift.vault.chunk import estimate_tokens

    samples = ["curl -sk 'https://a.test/x?id=1'", "| 1 | /api | GET |\n" * 40, "plain prose."]
    for s in samples:
        assert embed._estimate_tokens(s) == estimate_tokens(s)

    # If the chunker cannot be imported, the self-contained estimate still answers.
    monkeypatch.setitem(sys.modules, "sift.vault.chunk", None)
    for s in samples:
        assert embed._estimate_tokens(s) == embed._wordpiece_estimate(s)


def test_estimate_errs_high_where_chars_over_4_errs_low():
    from sift.index.embed import _wordpiece_estimate as _estimate_tokens

    poc = "curl -sk 'https://a.test/x?id=1'"
    # curl(2) - sk ' https(2) : / / a . test(2) / x ? id = 1 '  -> 21
    assert _estimate_tokens(poc) == 21
    assert _estimate_tokens(poc) > 2 * (len(poc) // 4)
    assert _estimate_tokens("user_id") == 4  # user(2) _ id
    sha = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    assert _estimate_tokens(sha) == math.ceil(len(sha) * 0.75)  # hashes shred
    cjk = "".join(chr(c) for c in (0x653B, 0x6483, 0x8005, 0x306F))
    assert _estimate_tokens(cjk) == 4  # one per character
    assert _estimate_tokens("2026 1234567") == 1 + 4
    assert _estimate_tokens("") == 0


def test_tokenizer_and_window_are_looked_up_once(monkeypatch, tmp_path):
    from sift.index import embed

    d = _model_dir(tmp_path, tokenizer=_toy_tokenizer(), sbert={"max_seq_length": 300})
    lookups: list[str] = []
    real = embed._find_model_file

    def counting(model_name, filename, **kw):
        lookups.append(filename)
        return real(model_name, filename, **kw)

    monkeypatch.setattr(embed, "_find_model_file", counting)
    emb = embed.Embedder(str(d), "cpu")
    for _ in range(3):
        emb.count_tokens("a b c")
        assert emb.max_seq_length == 300
    assert lookups.count("tokenizer.json") == 1
    assert lookups.count("sentence_bert_config.json") == 1


def test_chunks_built_with_the_embedder_budget_fit_its_window(tmp_path, no_model_loads):
    """The embedder's API, wired into chunk_markdown the way the pipeline does it."""
    from sift.index.embed import Embedder
    from sift.vault.chunk import chunk_markdown

    d = _model_dir(tmp_path, tokenizer=_toy_tokenizer(), sbert={"max_seq_length": 96})
    emb = Embedder(str(d), "cpu")
    emb.passage_prefix = "search_document: "  # a prefixed model costs window too
    title = "title impact poc"

    def context(heading: str) -> str:
        return emb.passage_prefix + title + "\n" + heading + "\n"

    curl = "\n".join(f"# step {i}\ncurl -sk 'https://a.test/x?id={i}'" for i in range(200))
    table = "| a | b | c |\n|---|---|---|\n" + "\n".join(f"| x{i} | y | z |" for i in range(600))
    body = f"# PoC\n\n```bash\n{curl}\n```\n\n# Endpoints\n\n{table}\n\n# Prose\n\n" + (
        "a b c " * 3000
    )
    chunks = chunk_markdown(
        body,
        max_tokens=emb.max_seq_length,
        count_tokens=emb.count_tokens,
        special_tokens=emb.num_special_tokens,
        context=context,
    )

    assert len(chunks) > 30  # the oversized blocks really were split
    for c in chunks:
        used = emb.num_special_tokens + emb.count_tokens(context(c.heading) + c.text)
        assert used <= emb.max_seq_length, (used, c.heading, c.text[:80])
    assert {c.heading for c in chunks} >= {"PoC", "Endpoints", "Prose"}  # '# step' is code
    assert no_model_loads == []
