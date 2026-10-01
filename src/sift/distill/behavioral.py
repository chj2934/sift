"""Behavioural novelty gate: test what the model knows instead of asking it.

`gate.py` asks the model to self-report ("do you already know this?"). Self-reported
knowledge is poorly calibrated in every model - a model can be confidently wrong in
both directions about its own recall.

This variant observes instead. Two stages:

  1. PREDICT - the model sees only the title and a one-line framing, and must explain
     the technique and how to test for it, from memory alone.
  2. SCORE   - the model then sees the full article next to its own prediction, and
     judges whether the prediction actually captured the real mechanism.

Knew it -> drop. Missed the mechanism -> keep.

Cost is close to the self-report gate despite the extra call, because stage 1's input
is a title (tens of tokens) rather than an article.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from sift.config import get_settings
from sift.distill.candidates import Candidate
from sift.distill.gate import GateError, Verdict, _client

# Stage 1 sees NO article text. An earlier version passed the opening 300 characters
# for context, but research posts routinely state their technique in the first
# sentence ("In this post I'll introduce the cookie sandwich technique, which lets you
# bypass HttpOnly") - which turns the memory test into a reading test and voids the
# experiment. Title, source and date only: exactly what a hunter sees in a feed.
#
# The cost is opaque titles ("SAML roulette", "Fickle PDFs") the model cannot place
# even when it knows the underlying work, which would show up as false keeps. That is
# a measurable failure mode, and `sift distill eval` is what measures it.
PREDICT_CONTEXT_CHARS = 0


class Prediction(BaseModel):
    recognised: bool = Field(description="whether the technique is recognised at all")
    mechanism: str = Field(description="how the technique works, from memory")
    how_to_test: str = Field(description="how you would test for it")


PREDICT_SYSTEM = """\
You are being tested on what you already know. You will be given only the title of a \
piece of security research, and at most a couple of opening lines.

From memory alone, explain the technique: the underlying mechanism, and how you would \
test a target for it. Be specific and concrete - name the exact headers, payloads, \
parameters or parser behaviours involved if you know them.

Do not hedge or describe the general vulnerability class when you actually know the \
specific technique. Equally, do not guess: if you do not recognise this specific \
technique, set recognised to false and say plainly what you do and do not know. \
Guessing plausibly is the failure mode that ruins this test.\
"""

SCORE_SYSTEM = """\
You are scoring a knowledge test in order to decide whether an article belongs in a \
knowledge base that supplements you.

You will see: a prediction written from the title alone, and then the actual article.

Answer TWO independent questions.

**1. has_technique** - does this article actually teach a reusable technique a tester \
could apply to a different target?

Set it false for: tool and product announcements ("we built X", "introducing X", \
release notes), conference and event posts, interviews and profiles, indexes and \
link roundups, competition reports, personal experience pieces, and news about the \
industry. These can be entirely unfamiliar to you and still be worthless here - being \
unknown is not the same as being worth knowing. This question is about the article, \
not about you.

**2. match** - did the prediction capture the article's ACTUAL core technique, the \
specific mechanism that makes it work?

- knew-it: the prediction described the real mechanism correctly. The article teaches \
you nothing you did not already have.
- partial: the prediction had the general class right but missed the specific \
mechanism, the exact payloads, or the conditions that make it work.
- missed: the prediction did not describe this technique at all, or was wrong.

Be strict about `knew-it`: a vague gesture at the right vulnerability class is NOT \
knowing the technique. But do not reward a prediction for being verbose - a long \
answer that never names the actual mechanism is `missed`.

Ignore whether the article is well written, recent, or from a respected source.\
"""

_PREDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "recognised": {"type": "boolean"},
        "mechanism": {"type": "string"},
        "how_to_test": {"type": "string"},
    },
    "required": ["recognised", "mechanism", "how_to_test"],
    "additionalProperties": False,
}

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_technique": {"type": "boolean"},
        "match": {"type": "string", "enum": ["knew-it", "partial", "missed"]},
        "what_it_missed": {"type": "string"},
        "justification": {"type": "string"},
    },
    "required": ["has_technique", "match", "what_it_missed", "justification"],
    "additionalProperties": False,
}

# Which keep-reason to record for each behavioural outcome.
_REASON = {"partial": "obscure-variant", "missed": "post-cutoff"}


def _call(client, model: str, effort: str, system: str, user: str, schema: dict) -> dict:
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    )
    if resp.stop_reason == "refusal":
        raise GateError("model refused")
    try:
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)
    except (StopIteration, json.JSONDecodeError) as exc:
        raise GateError(f"unusable response: {exc}") from exc


def predict(candidate: Candidate, *, client=None) -> Prediction:
    """Stage 1 - explain the technique from the title alone."""
    s = get_settings()
    client = client or _client()
    published = candidate.created.isoformat() if candidate.created else "unknown"
    user = (
        f"Title: {candidate.title}\n"
        f"Source: {candidate.source}\n"
        f"Published: {published}\n\n"
        "Explain this technique from memory. You have not been shown the article."
    )
    return Prediction(**_call(client, s.gate_model, s.gate_effort, PREDICT_SYSTEM, user, _PREDICT_SCHEMA))


def judge_behavioral(candidate: Candidate, *, client=None) -> Verdict:
    """Two-stage behavioural gate. Returns the same Verdict shape as `gate.judge`."""
    s = get_settings()
    client = client or _client()

    p = predict(candidate, client=client)
    user = (
        f"# Prediction written from the title alone\n"
        f"Recognised: {p.recognised}\n"
        f"Mechanism: {p.mechanism}\n"
        f"How to test: {p.how_to_test}\n\n"
        f"# The actual article\n"
        f"Title: {candidate.title}\n\n{candidate.gate_text()}"
    )
    scored = _call(client, s.gate_model, s.gate_effort, SCORE_SYSTEM, user, _SCORE_SCHEMA)

    match = scored["match"]
    has_technique = bool(scored["has_technique"])
    # Both conditions must hold. Novelty alone is not enough: a tool announcement is
    # genuinely unfamiliar and still worthless, which is exactly how an earlier version
    # of this gate kept a Go fuzzing release.
    keep = has_technique and match != "knew-it"

    if not has_technique:
        reason, note = "already-known", "no reusable technique"
    else:
        reason, note = _REASON.get(match, "already-known"), match

    return Verdict(
        decision="keep" if keep else "drop",
        # The prediction *is* the evidence of what was already known - far more
        # informative in the reject log than a self-assessment would be.
        already_known=(p.mechanism if p.recognised else "Not recognised from the title."),
        reason=reason,
        justification=f"[{note}] {scored['justification']}",
    )
