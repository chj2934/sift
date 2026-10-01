"""Offline tests for the novelty gate.

The gate's *judgement* can only be checked against the live API — that lives in
`test_gate_calibration.py` and is opt-in. What's tested here is everything around
it: prompt construction, response parsing, failure handling, and the reject log.
"""

from __future__ import annotations

import json
from datetime import date

import pytest


class _FakeBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, payload: dict, stop_reason: str = "end_turn"):
        self.content = [_FakeBlock(json.dumps(payload))]
        self.stop_reason = stop_reason


class _FakeClient:
    """Records the request and replays a canned verdict."""

    def __init__(self, payload: dict, stop_reason: str = "end_turn"):
        self._payload = payload
        self._stop_reason = stop_reason
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._payload, self._stop_reason)


KEEP = {
    "decision": "keep",
    "already_known": "Nothing — this is unfamiliar.",
    "reason": "post-cutoff",
    "justification": "Describes a parser differential I don't recognise.",
}
DROP = {
    "decision": "drop",
    "already_known": "Standard reflected XSS methodology.",
    "reason": "already-known",
    "justification": "I can already explain this without the article.",
}


def _candidate(**over):
    from sift.distill.candidates import Candidate

    base = dict(
        title="Some technique",
        url="https://example.com/post",
        text="body text here",
        source="example.com",
        created=date(2026, 5, 1),
    )
    base.update(over)
    return Candidate(**base)


def test_prompt_carries_the_fields_the_gate_reasons_over():
    from sift.distill.gate import _build_user_prompt

    prompt = _build_user_prompt(_candidate())
    assert "Some technique" in prompt
    assert "example.com" in prompt
    assert "2026-05-01" in prompt
    assert "body text here" in prompt


def test_prompt_handles_unknown_publish_date():
    from sift.distill.gate import _build_user_prompt

    assert "Published: unknown" in _build_user_prompt(_candidate(created=None))


def test_gate_text_is_capped_but_note_body_is_not():
    from sift.distill.candidates import GATE_TEXT_CHARS

    cand = _candidate(text="x" * (GATE_TEXT_CHARS + 5000))
    assert len(cand.gate_text()) == GATE_TEXT_CHARS
    assert len(cand.text) == GATE_TEXT_CHARS + 5000


def test_keep_verdict_parses():
    from sift.distill.gate import judge

    v = judge(_candidate(), client=_FakeClient(KEEP))
    assert v.keep is True
    assert v.reason == "post-cutoff"


def test_drop_verdict_parses():
    from sift.distill.gate import judge

    v = judge(_candidate(), client=_FakeClient(DROP))
    assert v.keep is False
    assert v.reason == "already-known"


def test_request_pins_model_and_constrains_output():
    from sift.distill.gate import judge

    client = _FakeClient(KEEP)
    judge(_candidate(), client=client)
    sent = client.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    # A free-form response would break parsing, so the schema must be enforced.
    assert sent["output_config"]["format"]["schema"]["required"]


def test_refusal_raises_rather_than_silently_keeping():
    from sift.distill.gate import GateError, judge

    with pytest.raises(GateError):
        judge(_candidate(), client=_FakeClient(KEEP, stop_reason="refusal"))


def test_unparseable_response_raises():
    from sift.distill.gate import GateError, judge

    client = _FakeClient({"decision": "keep"})  # missing required fields
    with pytest.raises(GateError):
        judge(_candidate(), client=client)


def test_judge_many_logs_every_reject():
    from sift.distill.gate import judge_many
    from sift.distill.rejects import load_rejects

    cands = [_candidate(title=f"t{i}") for i in range(3)]
    out = list(judge_many(cands, client=_FakeClient(DROP)))

    assert len(out) == 3
    rejects = load_rejects()
    assert [r["title"] for r in rejects] == ["t0", "t1", "t2"]
    assert all(r["already_known"] for r in rejects)


def test_one_transient_failure_does_not_abort_the_batch():
    from sift.distill.gate import judge_many

    class Flaky(_FakeClient):
        def __init__(self):
            super().__init__(DROP)
            self.n = 0

        def create(self, **kwargs):
            self.n += 1
            if self.n == 2:
                raise ConnectionError("transient")
            return _FakeResponse(DROP)

    cands = [_candidate(title=f"t{i}") for i in range(3)]
    out = list(judge_many(cands, client=Flaky()))

    # The middle candidate is skipped; the run continues to the third.
    assert [c.title for c, _ in out] == ["t0", "t2"]


def test_reject_log_survives_a_torn_line(vault_path):
    from sift.distill.rejects import REJECTS_FILE, load_rejects

    (vault_path / REJECTS_FILE).write_text(
        '{"title": "good"}\n{"title": "tor\n', encoding="utf-8"
    )
    assert [r["title"] for r in load_rejects()] == ["good"]


def test_missing_credentials_give_an_actionable_error(monkeypatch):
    from sift import config
    from sift.distill import gate

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    config.get_settings.cache_clear()
    with pytest.raises(gate.GateError, match="ANTHROPIC_API_KEY"):
        gate.judge(_candidate())


def test_collect_keeps_one_candidate_per_url(vault_path):
    """The same article arrives from the research feed, a Top-10 nomination and
    PentesterLand. Judging it three times costs money and duplicates the output."""
    from sift.distill.manual import collect
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    url = "https://blog.doyensec.com/2024/07/02/cspt2csrf.html"
    for i, (src, body) in enumerate(
        [("research", "short body " * 60), ("top10", "much longer body " * 200)]
    ):
        save_note(
            vault_path,
            Note(
                meta=Frontmatter(id=f"{src}-cspt", type="writeup", title=f"CSPT2CSRF {i}", url=url),
                body=body,
            ),
        )

    cands = collect("writeup", skip_gated=False)
    assert len(cands) == 1, f"expected one candidate per url, got {len(cands)}"
    # The fuller copy wins - the gate should judge the best available text.
    assert len(cands[0].text) > 1000


def test_collect_ignores_query_strings_when_deduping(vault_path):
    from sift.distill.manual import collect
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    for i, u in enumerate(
        ["https://x.tld/post", "https://x.tld/post/", "https://x.tld/post?utm_source=rss"]
    ):
        save_note(
            vault_path,
            Note(meta=Frontmatter(id=f"n{i}", type="writeup", title=f"T{i}", url=u), body="b " * 300),
        )
    assert len(collect("writeup", skip_gated=False)) == 1
