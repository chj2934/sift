"""Offline tests for the behavioural gate and the evaluation harness.

The experiment only means anything if stage 1 genuinely cannot see the technique, so
the leak check below is the load-bearing test in this file.
"""

from __future__ import annotations

import json
from datetime import date

import pytest


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, payload, stop_reason="end_turn"):
        self.content = [_Block(json.dumps(payload))]
        self.stop_reason = stop_reason


class _Stream:
    """Stands in for the SDK's MessageStreamManager. `reply` is a response, or a
    callable producing one (raising there is how the SDK surfaces an HTTP error)."""

    def __init__(self, reply):
        self._reply = reply

    def __enter__(self):
        if callable(self._reply):
            self._reply = self._reply()
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._reply


class _TwoStageClient:
    """Replays a prediction then a score, recording what each stage was shown."""

    def __init__(self, prediction, scored):
        self._queue = [prediction, scored]
        self.seen: list[str] = []
        self.messages = self

    def stream(self, **kwargs):
        self.seen.append(kwargs["messages"][0]["content"])
        return _Stream(self.reply(**kwargs))

    def reply(self, **kwargs):
        return _Resp(self._queue.pop(0))


KNEW = {
    "recognised": True,
    "mechanism": "Reflected XSS via unencoded echo.",
    "how_to_test": "Inject a marker.",
}
BLANK = {"recognised": False, "mechanism": "Not recognised.", "how_to_test": "Unknown."}
SCORE_KNEW = {
    "has_technique": True,
    "match": "knew-it",
    "what_it_missed": "nothing",
    "justification": "prediction matched",
}
SCORE_MISSED = {
    "has_technique": True,
    "match": "missed",
    "what_it_missed": "the whole mechanism",
    "justification": "no idea",
}
SCORE_PARTIAL = {
    "has_technique": True,
    "match": "partial",
    "what_it_missed": "exact payloads",
    "justification": "class only",
}
# Unfamiliar AND worthless - a tool release. The failure case that motivated
# has_technique: an earlier gate kept a Go fuzzing tool purely because it was unknown.
SCORE_NO_TECHNIQUE = {
    "has_technique": False,
    "match": "missed",
    "what_it_missed": "n/a",
    "justification": "tool announcement",
}

SECRET = "THE-SECRET-MECHANISM-IS-A-BARE-CR-IN-A-CHUNK-EXTENSION"


def _candidate():
    from sift.distill.candidates import Candidate

    return Candidate(
        title="Chunk-extension desync",
        url="https://example.com/p",
        text="Intro paragraph that sets the scene. " * 5 + SECRET + " and much more detail.",
        source="example.com",
        created=date(2026, 4, 2),
    )


def test_stage_one_never_sees_the_article():
    """The load-bearing test: if stage 1 sees any article text, the memory test
    becomes a reading test and the whole experiment is void."""
    from sift.distill.behavioral import judge_behavioral

    client = _TwoStageClient(BLANK, SCORE_MISSED)
    judge_behavioral(_candidate(), client=client)

    predict_prompt, score_prompt = client.seen
    assert SECRET not in predict_prompt, "stage 1 leaked the article's mechanism"
    assert "Intro paragraph" not in predict_prompt, "stage 1 must see no body text at all"
    assert SECRET in score_prompt, "stage 2 must see the full article"
    # Title, source, date and instruction only.
    assert len(predict_prompt) < 300


def test_knew_it_drops():
    from sift.distill.behavioral import judge_behavioral

    v = judge_behavioral(_candidate(), client=_TwoStageClient(KNEW, SCORE_KNEW))
    assert v.keep is False
    assert v.reason == "already-known"
    # The prediction is the evidence of what was known - keep it in the log.
    assert "Reflected XSS" in v.already_known


def test_missed_keeps_and_records_not_recognised():
    from sift.distill.behavioral import judge_behavioral

    v = judge_behavioral(_candidate(), client=_TwoStageClient(BLANK, SCORE_MISSED))
    assert v.keep is True
    assert v.reason == "post-cutoff"
    assert "Not recognised" in v.already_known


def test_partial_keeps_as_obscure_variant():
    from sift.distill.behavioral import judge_behavioral

    v = judge_behavioral(_candidate(), client=_TwoStageClient(KNEW, SCORE_PARTIAL))
    assert v.keep is True
    assert v.reason == "obscure-variant"
    assert v.justification.startswith("[partial]")


def test_refusal_surfaces_as_gate_error():
    from sift.distill.behavioral import judge_behavioral
    from sift.distill.gate import GateError

    class Refusing(_TwoStageClient):
        def reply(self, **kwargs):
            return _Resp(BLANK, stop_reason="refusal")

    with pytest.raises(GateError):
        judge_behavioral(_candidate(), client=Refusing(BLANK, SCORE_MISSED))


# --- evaluation harness ---


def test_labels_fixture_is_intact():
    from sift.distill.evaluate import load_labels

    labels = load_labels()
    assert len(labels) == 40
    assert sum(1 for d in labels.values() if d == "keep") == 21
    assert sum(1 for d in labels.values() if d == "drop") == 19


def test_score_separates_false_drops_from_false_keeps():
    from sift.distill.evaluate import score_gate
    from sift.distill.gate import Verdict

    cands = [(_candidate(), "keep"), (_candidate(), "drop"), (_candidate(), "keep")]
    calls = {"n": 0}

    def always_drop(cand, *, client=None):
        calls["n"] += 1
        return Verdict(
            decision="drop", already_known="x", reason="already-known", justification="j"
        )

    s = score_gate("always-drop", always_drop, cands)
    assert s.total == 3
    assert s.agree == 1  # only the genuine drop
    assert len(s.false_drops) == 2
    assert len(s.false_keeps) == 0
    assert s.accuracy == pytest.approx(1 / 3)


def test_score_survives_a_failing_candidate():
    from sift.distill.evaluate import score_gate

    def explodes(cand, *, client=None):
        raise RuntimeError("boom")

    s = score_gate("broken", explodes, [(_candidate(), "keep")])
    assert s.total == 0
    assert len(s.errors) == 1
    assert s.accuracy == 0.0


def test_unfamiliar_but_worthless_is_still_dropped():
    """The gosentry case: genuinely unknown to the model, but a tool release with no
    reusable technique. Novelty alone must not be enough to keep something."""
    from sift.distill.behavioral import judge_behavioral

    v = judge_behavioral(_candidate(), client=_TwoStageClient(BLANK, SCORE_NO_TECHNIQUE))
    assert v.keep is False
    assert v.reason == "already-known"
    assert "no reusable technique" in v.justification


def test_technique_the_model_missed_is_kept():
    from sift.distill.behavioral import judge_behavioral

    v = judge_behavioral(_candidate(), client=_TwoStageClient(BLANK, SCORE_MISSED))
    assert v.keep is True


# --- malformed stages surface as GateError ---


def test_a_prediction_missing_fields_is_a_gate_error():
    """A pydantic ValidationError used to escape as a non-GateError."""
    from sift.distill.behavioral import judge_behavioral
    from sift.distill.gate import GateError

    with pytest.raises(GateError, match="unusable prediction"):
        judge_behavioral(_candidate(), client=_TwoStageClient({"recognised": True}, SCORE_MISSED))


@pytest.mark.parametrize(
    "scored",
    [
        {"has_technique": True, "what_it_missed": "x", "justification": "j"},  # no match
        {"has_technique": "yes", "match": "missed", "what_it_missed": "x", "justification": "j"},
    ],
)
def test_a_malformed_score_is_a_gate_error(scored):
    """A missing `match` used to escape as a bare KeyError."""
    from sift.distill.behavioral import judge_behavioral
    from sift.distill.gate import GateError

    with pytest.raises(GateError, match="unusable score"):
        judge_behavioral(_candidate(), client=_TwoStageClient(BLANK, scored))


def test_a_truncated_stage_says_so():
    from sift.distill.behavioral import judge_behavioral
    from sift.distill.gate import GateError

    class Truncating(_TwoStageClient):
        def reply(self, **kwargs):
            return _Resp(BLANK, stop_reason="max_tokens")

    with pytest.raises(GateError, match="truncated"):
        judge_behavioral(_candidate(), client=Truncating(BLANK, SCORE_MISSED))


# --- run-fatal errors stop the evaluation ---


def _api_error(status: int):
    """A real SDK status error, built offline the way the SDK builds it."""
    import anthropic
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    cls = {
        400: anthropic.BadRequestError,
        401: anthropic.AuthenticationError,
        403: anthropic.PermissionDeniedError,
        404: anthropic.NotFoundError,
    }.get(status, anthropic.APIStatusError)
    return cls(f"HTTP {status}", response=httpx2.Response(status, request=req), body=None)


class _Failing:
    """A client whose calls fail with the given statuses in turn (None = succeed)."""

    def __init__(self, statuses):
        self._statuses = list(statuses)
        self.calls = 0
        self.messages = self

    def stream(self, **kwargs):
        return _Stream(self._reply)

    def _reply(self):
        self.calls += 1
        status = self._statuses.pop(0) if self._statuses else None
        if status is not None:
            raise _api_error(status)
        return _Resp(
            {
                "decision": "drop",
                "already_known": "k",
                "reason": "already-known",
                "justification": "j",
            }
        )


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_a_run_fatal_api_error_stops_the_run_after_one_call(status):
    """A revoked key, billing problem or bad model id fails every call alike; the run
    used to make all 120 calls and report 0/0."""
    from sift.distill.evaluate import score_gate
    from sift.distill.gate import GateError, judge

    client = _Failing([status] * 5)
    with pytest.raises(GateError, match=f"HTTP {status}"):
        score_gate("self-report", judge, [(_candidate(), "keep")] * 5, client=client)
    assert client.calls == 1


def test_three_bad_requests_in_a_row_stop_the_run():
    from sift.distill.evaluate import score_gate
    from sift.distill.gate import GateError, judge

    client = _Failing([400] * 5)
    with pytest.raises(GateError, match="consecutive 400"):
        score_gate("self-report", judge, [(_candidate(), "keep")] * 5, client=client)
    assert client.calls == 3


def test_scattered_bad_requests_do_not_stop_the_run():
    """A 400 can be specific to one candidate's content."""
    from sift.distill.evaluate import score_gate
    from sift.distill.gate import judge

    client = _Failing([400, 400, None, 400, None])
    s = score_gate("self-report", judge, [(_candidate(), "drop")] * 5, client=client)
    assert (s.total, len(s.errors), s.agree) == (2, 3, 2)


def test_a_bad_effort_setting_stops_the_run_before_any_call(monkeypatch):
    from sift import config
    from sift.distill.evaluate import score_gate
    from sift.distill.gate import GateError, judge

    monkeypatch.setenv("SIFT_GATE_EFFORT", "extreme")
    config.get_settings.cache_clear()
    client = _Failing([])
    with pytest.raises(GateError, match="SIFT_GATE_EFFORT"):
        score_gate("self-report", judge, [(_candidate(), "keep")] * 3, client=client)
    assert client.calls == 0


# --- eval judges the copy production would gate ---


def _writeup(vault, nid, title, url, body):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    save_note(
        vault, Note(meta=Frontmatter(id=nid, type="writeup", title=title, url=url), body=body)
    )


def test_eval_scores_the_copy_production_would_gate(vault_path):
    """With several copies of one article, eval used to take whichever loaded last -
    for 5 of the 40 labelled articles a shorter aggregator copy than the gate sees."""
    from sift.distill.evaluate import labelled_candidates
    from sift.distill.manual import collect

    url = "https://example.com/research/chunk-desync"
    _writeup(vault_path, "full", "A full article", url, "full article body " * 300)
    _writeup(vault_path, "teaser", "B aggregator teaser", url, "teaser text " * 10)

    (pair,) = labelled_candidates({url: "keep"}, vault_path)
    assert pair[0].text.startswith("full article body")
    (prod,) = collect("writeup", skip_gated=False, prefilter=False)
    assert prod.text == pair[0].text


def test_eval_finds_labels_whatever_their_url_form(vault_path):
    from sift.distill.evaluate import labelled_candidates

    _writeup(vault_path, "a", "Some article", "https://example.com/p/", "body " * 50)
    pairs = labelled_candidates({"https://example.com/p?utm_source=x": "drop"}, vault_path)
    assert [(c.title, d) for c, d in pairs] == [("Some article", "drop")]


def test_eval_ignores_the_gate_log(vault_path):
    """The labelled articles are already judged; filtering gated urls would empty the
    eval set without a word."""
    from sift.distill.candidates import Candidate
    from sift.distill.evaluate import labelled_candidates
    from sift.distill.gate import Verdict
    from sift.distill.rejects import record_reject

    url = "https://example.com/p"
    _writeup(vault_path, "a", "Some article", url, "body " * 50)
    record_reject(
        Candidate(title="Some article", url=url, text="", source="s"),
        Verdict(decision="drop", already_known="k", reason="already-known", justification="j"),
    )
    assert len(labelled_candidates({url: "drop"}, vault_path)) == 1
