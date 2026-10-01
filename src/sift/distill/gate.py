"""The novelty gate: does the reasoning model already know this?

This is the whole point of the pipeline. A technique note explaining something Opus
can already explain is worse than useless — it costs tokens to retrieve and it
dilutes ranking, pushing genuinely rare hits down the list. So the default is
*drop*, and material has to earn its place.

Only Opus can judge what Opus knows; a local model would be guessing about weights
it has no visibility into. That's why the gate is an API call rather than the local
Qwen used elsewhere for grunt work.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from pydantic import BaseModel, Field

from sift.config import get_settings
from sift.distill.candidates import Candidate

# Why a candidate was worth keeping. `already-known` is the drop reason.
KEEP_REASONS = ("post-cutoff", "obscure-variant", "operational-detail")
_ALL_REASONS = (*KEEP_REASONS, "already-known")


class GateError(RuntimeError):
    """Gate could not run (missing credentials, unusable response)."""


class Verdict(BaseModel):
    decision: str = Field(description="keep or drop")
    already_known: str = Field(description="what the model already knows about this topic")
    reason: str = Field(description="one of: " + ", ".join(_ALL_REASONS))
    justification: str = Field(description="one sentence")

    @property
    def keep(self) -> bool:
        return self.decision == "keep"


GATE_SYSTEM = """\
You are triaging security research for a personal knowledge base that supplements \
YOU — the same model that will later read it. The base is only worth having if it \
holds what you do NOT already know. Anything you can already explain accurately is \
noise: it costs retrieval tokens and pushes genuinely rare material down the ranking.

For each candidate, first write `already_known`: honestly summarise what you already \
know about this specific technique, from your own training. Do this BEFORE deciding. \
If you can already describe the mechanism and how to test for it, that is a drop, no \
matter how well-written the article is.

Keep ONLY if one of these clearly applies:
- post-cutoff: published after your training data ends AND describes a technique or \
vulnerability class you do not recognise.
- obscure-variant: you know the general class, but this is a specific edge case or \
bypass whose details you would fumble or get wrong if asked from memory.
- operational-detail: exact payloads, tool flags, or bypass strings you could not \
reproduce accurately without the source.

Drop everything else with reason `already-known`. In particular, drop:
- General methodology for well-known bug classes (IDOR, reflected XSS, SQLi, SSRF, \
open redirect, basic JWT attacks, standard CSRF).
- Reference material and cheat-sheets covering established techniques.
- Single-target findings with no reusable technique ("I found an IDOR on example.com").
- Tool announcements, conference recaps, and commentary.

Be strict. A small base of things you genuinely lack beats a large one you mostly \
already know. When in doubt, drop.\
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["keep", "drop"]},
        "already_known": {"type": "string"},
        "reason": {"type": "string", "enum": list(_ALL_REASONS)},
        "justification": {"type": "string"},
    },
    "required": ["decision", "already_known", "reason", "justification"],
    "additionalProperties": False,
}


def _build_user_prompt(candidate: Candidate) -> str:
    """Pure — kept separate from the API call so tests can assert on it offline."""
    published = candidate.created.isoformat() if candidate.created else "unknown"
    return (
        f"Title: {candidate.title}\n"
        f"Source: {candidate.source}\n"
        f"Published: {published}\n"
        f"URL: {candidate.url}\n\n"
        f"--- content ---\n{candidate.gate_text()}"
    )


def _client():
    import os

    import anthropic  # deferred: keeps the dep off the CLI hot import path

    key = get_settings().anthropic_api_key
    if key:
        return anthropic.Anthropic(api_key=key)
    # No key in .env is not the same as no credentials — the SDK also resolves
    # ANTHROPIC_AUTH_TOKEN and `ant auth login` profiles. Only bail if nothing is
    # available, since the SDK itself fails late (at request time) with a TypeError.
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise GateError(
            "no Anthropic credentials for the novelty gate. "
            "Set ANTHROPIC_API_KEY in .env, or run `ant auth login`."
        )
    return anthropic.Anthropic()


def judge(candidate: Candidate, *, client=None) -> Verdict:
    """Ask the model whether it already knows this. Raises GateError on failure."""
    import json

    s = get_settings()
    client = client or _client()
    resp = client.messages.create(
        model=s.gate_model,
        max_tokens=4000,
        system=GATE_SYSTEM,
        messages=[{"role": "user", "content": _build_user_prompt(candidate)}],
        output_config={
            "effort": s.gate_effort,
            "format": {"type": "json_schema", "schema": _SCHEMA},
        },
    )
    if resp.stop_reason == "refusal":
        raise GateError(f"gate refused: {candidate.title!r}")
    try:
        text = next(b.text for b in resp.content if b.type == "text")
        return Verdict(**json.loads(text))
    except (StopIteration, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise GateError(f"unusable gate response for {candidate.title!r}: {exc}") from exc


def judge_many(
    candidates: Iterable[Candidate], *, client=None
) -> Iterator[tuple[Candidate, Verdict]]:
    """Judge a stream, logging rejects. One bad candidate never aborts the run."""
    from sift.distill.rejects import record_reject

    client = client or _client()
    for cand in candidates:
        # Broad by design, matching ingest.base.run_source: one transient API error
        # must not abort a 500-candidate batch.
        try:
            verdict = judge(cand, client=client)
        except Exception as exc:
            print(f"  ! gate: {cand.title!r}: {exc}")
            continue
        if not verdict.keep:
            record_reject(cand, verdict)
        yield cand, verdict
