"""The novelty gate: does the reasoning model already know this?

This is the whole point of the pipeline. A technique note explaining something Opus
can already explain is worse than useless — it costs tokens to retrieve and it
dilutes ranking, pushing genuinely rare hits down the list. So the default is
*drop*, and material has to earn its place.

Only Opus can judge what Opus knows; a local model would be guessing about weights
it has no visibility into. That's why the gate is an API call rather than the local
Qwen used elsewhere for grunt work. For the same reason the gate never falls back to
another model on a refusal: a different model's verdict says nothing about this one's
knowledge, so a refusal is an error, not a verdict.

Day-to-day gating is in-session (`distill.manual`); the API path here is what
`distill eval` calibrates.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from sift.config import get_settings
from sift.distill.candidates import Candidate

# Why a candidate was worth keeping. `already-known` is the drop reason.
KEEP_REASONS = ("post-cutoff", "obscure-variant", "operational-detail")
_ALL_REASONS = (*KEEP_REASONS, "already-known")

# Accepted values of `output_config.effort`. Checked where the gate runs rather than in
# Settings: the MCP server loads Settings for every tool call, and a typo in a
# gate-only variable must not take search down with it.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Thinking tokens count against max_tokens, and at 4,000 a long think cut the JSON
# verdict off mid-string, which surfaced as a misleading parse error.
MAX_TOKENS = 16_000

# Gate calls stream, so the read timeout bounds the silence between two stream events,
# not the whole generation. A non-streaming call can only be given one timeout for
# both, and the SDK's 10 minutes (retried twice) let a stalled connection hold a run
# for about half an hour per candidate. Now a stall fails in two minutes, while a long
# high-effort think still completes because events keep arriving.
STREAM_READ_TIMEOUT_S = 120.0
CONNECT_TIMEOUT_S = 10.0

# HTTP statuses that fail every remaining call in a run the same way. One of these on
# the first candidate means the next 119 would repeat it, paid for nothing.
_FATAL_STATUS = {
    401: "authentication failed - check ANTHROPIC_API_KEY or run `ant auth login`",
    402: "billing problem on the Anthropic account",
    403: "permission denied for this credential",
    404: "model not found - check SIFT_GATE_MODEL",
}


class GateError(RuntimeError):
    """Gate could not run (missing credentials, unusable response)."""


class GateConfigError(GateError):
    """The gate cannot run at all - credentials, model, effort or billing.

    Unlike a refused or unparseable reply, retrying the next candidate only repeats it,
    so batch callers stop on this instead of logging it per candidate.
    """


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


def gate_settings() -> tuple[str, str]:
    """(model, effort) for gate calls. Raises GateConfigError on an unknown effort."""
    s = get_settings()
    if s.gate_effort not in EFFORT_LEVELS:
        raise GateConfigError(
            f"SIFT_GATE_EFFORT={s.gate_effort!r} is not one of {'|'.join(EFFORT_LEVELS)}"
        )
    return s.gate_model, s.gate_effort


def _resolve_client():
    """An Anthropic client from whatever credentials the SDK can find.

    A key from Settings (.env or the environment) is passed explicitly; without one the
    SDK walks its own chain - ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, an
    `ANTHROPIC_PROFILE` or `ant auth login` profile, workload identity. This used to
    check only the two env vars, so the `ant auth login` the error message recommends
    was refused.
    """
    import anthropic  # deferred: keeps the dep off the CLI hot import path

    key = get_settings().anthropic_api_key or None
    try:
        client = anthropic.Anthropic(api_key=key)
    except anthropic.AnthropicError as exc:  # an explicitly selected profile is broken
        raise GateConfigError(
            f"Anthropic credentials are configured but unusable: {exc}. Fix the profile, "
            "set ANTHROPIC_API_KEY in .env, or run `ant auth login`."
        ) from exc
    # The SDK itself only fails late, at request time, when it found nothing.
    if client.api_key is None and client.auth_token is None and client.credentials is None:
        raise GateConfigError(
            "no Anthropic credentials for the novelty gate. "
            "Set ANTHROPIC_API_KEY in .env, or run `ant auth login`."
        )
    return client


def _client():
    """A client for gate calls, after checking the gate settings it will be used with."""
    gate_settings()  # fail before any spend, not on the first candidate
    return _resolve_client()


def credentials_available() -> bool:
    """Whether the SDK can resolve Anthropic credentials (env, .env or an ant profile).

    Never raises, so it is safe in a test skip condition.
    """
    try:
        _resolve_client()
    except Exception:  # noqa: BLE001 - a probe: any failure means "not available"
        return False
    return True


def run_fatal(exc: BaseException) -> str | None:
    """Why `exc` would fail every remaining candidate too, or None if it is specific
    to this one (a refusal, a malformed reply, a transient error)."""
    if isinstance(exc, GateConfigError):
        return str(exc)
    status = getattr(exc, "status_code", None)
    if status not in _FATAL_STATUS:
        return None
    import anthropic

    if not isinstance(exc, anthropic.APIStatusError):
        return None
    return f"{_FATAL_STATUS[status]} (HTTP {status}: {exc})"


def _create_json(client, *, system: str, user: str, schema: dict, label: str) -> dict:
    """One structured-output call, shared by every gate design. Raises GateError.

    Streams and returns the final message (see STREAM_READ_TIMEOUT_S); API errors
    propagate as the SDK's own exceptions, for `run_fatal` to classify.
    """
    import anthropic  # deferred: keeps the dep off the CLI hot import path

    model, effort = gate_settings()
    with client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        timeout=anthropic.Timeout(STREAM_READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
    ) as stream:
        resp = stream.get_final_message()
    if resp.stop_reason == "refusal":
        raise GateError(f"{label}: the model refused")
    if resp.stop_reason == "max_tokens":
        raise GateError(f"{label}: reply truncated at max_tokens={MAX_TOKENS} (effort={effort})")
    try:
        data = json.loads(next(b.text for b in resp.content if b.type == "text"))
    except (StopIteration, json.JSONDecodeError) as exc:
        raise GateError(f"{label}: unusable response: {exc}") from exc
    if not isinstance(data, dict):
        raise GateError(f"{label}: expected a JSON object, got {type(data).__name__}")
    return data


def judge(candidate: Candidate, *, client=None) -> Verdict:
    """Ask the model whether it already knows this. Raises GateError on failure."""
    client = client or _client()
    data = _create_json(
        client,
        system=GATE_SYSTEM,
        user=_build_user_prompt(candidate),
        schema=_SCHEMA,
        label=f"gate {candidate.title!r}",
    )
    try:
        return Verdict(**data)
    except (TypeError, ValueError) as exc:
        raise GateError(f"unusable gate response for {candidate.title!r}: {exc}") from exc
