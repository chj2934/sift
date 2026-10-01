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


class _TwoStageClient:
    """Replays a prediction then a score, recording what each stage was shown."""

    def __init__(self, prediction, scored):
        self._queue = [prediction, scored]
        self.seen: list[str] = []
        self.messages = self

    def create(self, **kwargs):
        self.seen.append(kwargs["messages"][0]["content"])
        return _Resp(self._queue.pop(0))


KNEW = {"recognised": True, "mechanism": "Reflected XSS via unencoded echo.", "how_to_test": "Inject a marker."}
BLANK = {"recognised": False, "mechanism": "Not recognised.", "how_to_test": "Unknown."}
SCORE_KNEW = {"has_technique": True, "match": "knew-it", "what_it_missed": "nothing", "justification": "prediction matched"}
SCORE_MISSED = {"has_technique": True, "match": "missed", "what_it_missed": "the whole mechanism", "justification": "no idea"}
SCORE_PARTIAL = {"has_technique": True, "match": "partial", "what_it_missed": "exact payloads", "justification": "class only"}
# Unfamiliar AND worthless - a tool release. The failure case that motivated
# has_technique: an earlier gate kept a Go fuzzing tool purely because it was unknown.
SCORE_NO_TECHNIQUE = {"has_technique": False, "match": "missed", "what_it_missed": "n/a", "justification": "tool announcement"}

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
        def create(self, **kwargs):
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
        return Verdict(decision="drop", already_known="x", reason="already-known", justification="j")

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
