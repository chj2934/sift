"""Offline tests for the novelty gate.

The gate's *judgement* can only be checked against the live API — that lives in
`test_gate_calibration.py` and is opt-in. What's tested here is everything around
it: prompt construction, response parsing, failure handling, credentials, the reject
log, and which candidates reach the gate at all.
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
    def __init__(self, payload, stop_reason: str = "end_turn", *, raw: str | None = None):
        self.content = [_FakeBlock(raw if raw is not None else json.dumps(payload))]
        self.stop_reason = stop_reason


class _FakeStream:
    """Stands in for the SDK's MessageStreamManager: a context manager whose stream
    hands back the final message."""

    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._resp


class _FakeClient:
    """Records the request and replays a canned verdict."""

    def __init__(self, payload, stop_reason: str = "end_turn", *, raw: str | None = None):
        self._payload = payload
        self._stop_reason = stop_reason
        self._raw = raw
        self.calls: list[dict] = []
        self.messages = self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStream(_FakeResponse(self._payload, self._stop_reason, raw=self._raw))


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

# Every environment variable the SDK resolves credentials from.
_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
)


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


@pytest.fixture
def no_credentials(tmp_path, monkeypatch):
    """Hide every credential source: env vars, the project .env and on-disk profiles.

    ANTHROPIC_CONFIG_DIR is deliberately not used: the SDK treats setting it as an
    explicit profile choice and raises instead of reporting no credentials. The
    platform default config dir is moved instead (APPDATA on Windows, HOME elsewhere).
    """
    from sift import config

    for var in _CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    config.get_settings.cache_clear()
    return tmp_path


def _write_profile(root, token: str = "tok-test") -> None:
    """An `ant auth login`-style user_oauth profile, at both platform default dirs."""
    for base in (root / "appdata" / "Anthropic", root / "home" / ".config" / "anthropic"):
        (base / "configs").mkdir(parents=True)
        (base / "credentials").mkdir(parents=True)
        (base / "configs" / "default.json").write_text(
            json.dumps({"authentication": {"type": "user_oauth"}}), encoding="utf-8"
        )
        (base / "credentials" / "default.json").write_text(
            json.dumps({"access_token": token, "expires_at": 4102444800}), encoding="utf-8"
        )


# --- prompt and response handling ---------------------------------------------------


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


def test_request_leaves_room_for_thinking_before_the_verdict():
    """Thinking tokens count against max_tokens; 4,000 cut verdicts off mid-string."""
    from sift.distill.gate import MAX_TOKENS, judge

    client = _FakeClient(KEEP)
    judge(_candidate(), client=client)
    assert client.calls[0]["max_tokens"] == MAX_TOKENS >= 16_000


# --- the real SDK, offline: a mock HTTP transport serving an event stream -------------


def _sse(payload: dict, stop_reason: str = "end_turn") -> bytes:
    """A Messages API event stream: an empty thinking block, a ping, then the JSON
    verdict as text split over two deltas."""
    text = json.dumps(payload)
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "ping"},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": text[:15]},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": text[15:]},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 40},
        },
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _sdk_client(handler):
    import anthropic
    import httpx2

    return anthropic.Anthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def test_the_real_sdk_streams_the_verdict_with_a_stall_timeout():
    """The fakes above cannot show that the SDK accepts these arguments. A stream's read
    timeout bounds the silence between events; a non-streaming call's bounded the whole
    generation, so a stall held the run for 10 minutes per attempt."""
    import httpx2

    from sift.distill.gate import CONNECT_TIMEOUT_S, STREAM_READ_TIMEOUT_S, judge

    sent = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent["body"] = json.loads(request.content)
        sent["timeout"] = request.extensions.get("timeout")
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(DROP)
        )

    v = judge(_candidate(), client=_sdk_client(handler))
    assert (v.keep, v.reason) == (False, "already-known")
    assert sent["body"]["stream"] is True
    assert sent["body"]["output_config"]["format"]["type"] == "json_schema"
    assert sent["timeout"]["read"] == STREAM_READ_TIMEOUT_S
    assert sent["timeout"]["connect"] == CONNECT_TIMEOUT_S


def test_the_real_sdk_reports_a_truncated_stream():
    import httpx2

    from sift.distill.gate import GateError, judge

    def handler(request):
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(KEEP, "max_tokens")
        )

    with pytest.raises(GateError, match="truncated at max_tokens"):
        judge(_candidate(), client=_sdk_client(handler))


def test_a_revoked_key_stops_the_eval_after_one_call():
    """A real 401 from the SDK, end to end through score_gate."""
    import httpx2

    from sift.distill.evaluate import score_gate
    from sift.distill.gate import GateConfigError, judge

    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(
            401,
            json={
                "type": "error",
                "error": {"type": "authentication_error", "message": "invalid x-api-key"},
            },
        )

    with pytest.raises(GateConfigError, match="HTTP 401"):
        score_gate("self-report", judge, [(_candidate(), "drop")] * 4, client=_sdk_client(handler))
    assert len(calls) == 1


def test_refusal_raises_rather_than_silently_keeping():
    from sift.distill.gate import GateError, judge

    with pytest.raises(GateError):
        judge(_candidate(), client=_FakeClient(KEEP, stop_reason="refusal"))


def test_unparseable_response_raises():
    from sift.distill.gate import GateError, judge

    client = _FakeClient({"decision": "keep"})  # missing required fields
    with pytest.raises(GateError):
        judge(_candidate(), client=client)


def test_truncated_reply_is_reported_as_truncation_not_bad_json():
    from sift.distill.gate import GateError, judge

    client = _FakeClient(None, stop_reason="max_tokens", raw='{"decision": "keep", "already_kno')
    with pytest.raises(GateError, match="truncated at max_tokens"):
        judge(_candidate(), client=client)


def test_non_object_reply_raises_gate_error():
    from sift.distill.gate import GateError, judge

    with pytest.raises(GateError, match="JSON object"):
        judge(_candidate(), client=_FakeClient(["keep"]))


def test_invalid_effort_is_a_gate_error_and_costs_nothing(monkeypatch):
    """A typo in SIFT_GATE_EFFORT must fail at the gate, before any request - and must
    not fail Settings, which the MCP server loads for every tool call."""
    from sift import config
    from sift.distill.gate import GateError, judge

    monkeypatch.setenv("SIFT_GATE_EFFORT", "lwo")
    config.get_settings.cache_clear()
    assert config.get_settings().gate_effort == "lwo"

    client = _FakeClient(KEEP)
    with pytest.raises(GateError, match="SIFT_GATE_EFFORT"):
        judge(_candidate(), client=client)
    assert client.calls == []


# --- credentials ----------------------------------------------------------------------


def test_missing_credentials_give_an_actionable_error(no_credentials):
    from sift.distill import gate

    with pytest.raises(gate.GateError, match="ANTHROPIC_API_KEY"):
        gate.judge(_candidate())
    assert gate.credentials_available() is False


def test_a_key_in_dotenv_is_used(no_credentials, monkeypatch):
    """Positive control for the isolation above: the same setup plus a .env key works."""
    from sift import config
    from sift.distill import gate

    env = no_credentials / "test.env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-test-dotenv\n", encoding="utf-8")
    monkeypatch.setitem(config.Settings.model_config, "env_file", env)
    config.get_settings.cache_clear()

    assert gate._client().api_key == "sk-ant-test-dotenv"
    assert gate.credentials_available() is True


def test_an_ant_auth_login_profile_is_enough(no_credentials):
    """The error message says `ant auth login`; the gate used to refuse exactly that."""
    from sift.distill import gate

    _write_profile(no_credentials)
    client = gate._client()
    assert client.api_key is None
    assert client.credentials is not None
    assert gate.credentials_available() is True


def test_a_broken_explicit_profile_is_a_gate_error(no_credentials, monkeypatch):
    from sift.distill import gate

    monkeypatch.setenv("ANTHROPIC_PROFILE", "no-such-profile")
    with pytest.raises(gate.GateConfigError, match="unusable"):
        gate._client()
    assert gate.credentials_available() is False


def test_auth_token_env_var_is_enough(no_credentials, monkeypatch):
    from sift.distill import gate

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-env")
    assert gate.credentials_available() is True


# --- what reaches the gate ------------------------------------------------------------


def _save_note(vault, nid, title, url, *, note_type="writeup", body="body text " * 60):
    from sift.vault.notes import Note, save_note
    from sift.vault.schema import Frontmatter

    return save_note(
        vault,
        Note(
            meta=Frontmatter(id=nid, type=note_type, title=title, url=url, source="example"),
            body=body,
        ),
    )


def _drop(cand):
    from sift.distill.gate import Verdict
    from sift.distill.rejects import record_reject

    assert record_reject(
        cand, Verdict(decision="drop", already_known="k", reason="already-known", justification="j")
    )


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
            Note(
                meta=Frontmatter(id=f"n{i}", type="writeup", title=f"T{i}", url=u), body="b " * 300
            ),
        )
    assert len(collect("writeup", skip_gated=False)) == 1


def test_url_key_rules():
    from sift.distill.candidates import url_key

    assert url_key("https://X.tld/Post/") == "https://x.tld/Post"  # path case is kept
    assert url_key(" https://x.tld/p?source=rss&a=1#frag ") == "https://x.tld/p"
    assert url_key("HTTPS://x.tld/p#section") == "https://x.tld/p"
    assert url_key("https://x.tld/") == "https://x.tld"
    assert url_key("x.tld/p/") == "x.tld/p"
    assert url_key("") == url_key(None) == url_key(42) == ""


def test_judged_urls_with_slash_or_query_are_not_exported_again(vault_path):
    """628 of 1,209 pending urls end in '/' and 32 carry a query: compared raw against
    collect()'s normalised key, none of them ever counted as judged."""
    from sift.distill.manual import collect, gated_urls

    _save_note(vault_path, "a", "Trailing slash post", "https://x.tld/p/")
    _save_note(vault_path, "b", "Feed query post", "https://y.tld/p?source=rss----abc")
    _save_note(vault_path, "c", "Plain control post", "https://z.tld/p")

    first = collect("writeup", prefilter=False)
    assert len(first) == 3
    for cand in first:
        _drop(cand)

    assert collect("writeup", prefilter=False) == []
    assert {"https://x.tld/p", "https://y.tld/p", "https://z.tld/p"} <= gated_urls()


def test_a_technique_note_gates_its_article_in_any_url_form(vault_path):
    from sift.distill.manual import collect

    _save_note(vault_path, "w", "The article", "https://x.tld/research/p")
    _save_note(
        vault_path,
        "tech-x",
        "Distilled technique",
        "https://x.tld/research/p/",
        note_type="technique",
    )
    assert collect("writeup", prefilter=False) == []


def test_limited_batches_advance_once_judged(vault_path):
    """Under --limit the batch used to refill with the same already-judged items."""
    from sift.distill.manual import collect

    for i in range(6):
        _save_note(vault_path, f"n{i}", f"Post {i}", f"https://blog{i}.tld/post/")

    first = collect("writeup", limit=3, prefilter=False)
    assert len(first) == 3
    for cand in first:
        _drop(cand)
    second = collect("writeup", limit=3, prefilter=False)
    assert len(second) == 3
    assert {c.url for c in first}.isdisjoint({c.url for c in second})


def test_url_less_notes_are_not_exported(vault_path, caplog):
    """A verdict is matched by url, so a url-less candidate could never be applied - and
    it took a slot in every --limit batch for good."""
    from sift.distill.manual import collect

    _save_note(vault_path, "nourl", "No url here", None)
    _save_note(vault_path, "withurl", "Has a url", "https://x.tld/p")

    with caplog.at_level("WARNING", logger="sift.distill.manual"):
        cands = collect("writeup", limit=1, prefilter=False)
    assert [c.title for c in cands] == ["Has a url"]
    assert "no url" in caplog.text


# --- the reject log -------------------------------------------------------------------


def test_reject_log_survives_a_torn_line(vault_path):
    from sift.distill.rejects import REJECTS_FILE, load_rejects

    (vault_path / REJECTS_FILE).write_text('{"title": "good"}\n{"title": "tor\n', encoding="utf-8")
    assert [r["title"] for r in load_rejects()] == ["good"]


def test_a_torn_last_line_does_not_swallow_the_next_reject(vault_path):
    """A crash mid-write leaves no trailing newline; the next row used to be glued on."""
    from sift.distill.rejects import REJECTS_FILE, load_rejects

    (vault_path / REJECTS_FILE).write_bytes(b'{"title": "good"}\n{"title": "tor')
    _drop(_candidate(title="new-row", url="https://x.tld/new"))
    assert [r["title"] for r in load_rejects()] == ["good", "new-row"]


def test_unicode_line_separators_do_not_lose_a_reject(vault_path):
    """str.splitlines() splits at U+2028/U+2029/U+0085, which un-gated the url."""
    from sift.distill.manual import gated_urls
    from sift.distill.rejects import REJECTS_FILE, load_rejects

    _drop(_candidate(title="plain", url="https://a.tld/p"))
    _drop(_candidate(title="line\u2028separator", url="https://b.tld/p"))
    _drop(_candidate(title="next\x85line", url="https://c.tld/p"))
    assert [r["title"] for r in load_rejects()] == ["plain", "line\u2028separator", "next\x85line"]
    assert {"https://a.tld/p", "https://b.tld/p", "https://c.tld/p"} <= gated_urls()
    # Rows are written ASCII-escaped, so no reader can split them.
    (vault_path / REJECTS_FILE).read_bytes().decode("ascii")


def test_old_rows_with_raw_separators_still_load(vault_path):
    from sift.distill.rejects import REJECTS_FILE, load_rejects

    old = json.dumps({"title": "old\u2028row", "url": "https://x.tld/p"}, ensure_ascii=False)
    (vault_path / REJECTS_FILE).write_text(old + "\n", encoding="utf-8")
    assert [r["title"] for r in load_rejects()] == ["old\u2028row"]


def test_reject_rows_record_reason_and_gate(vault_path):
    from sift.distill.gate import Verdict
    from sift.distill.rejects import load_rejects, record_reject

    v = Verdict(decision="drop", already_known="k", reason="already-known", justification="j")
    record_reject(_candidate(), v, gated_by="in-session")
    (row,) = load_rejects()
    assert row["reason"] == "already-known"
    assert row["gated_by"] == "in-session"


def test_a_failed_reject_write_is_logged_not_printed(vault_path, capsys, caplog):
    from sift.distill.gate import Verdict
    from sift.distill.rejects import REJECTS_FILE, record_reject

    (vault_path / REJECTS_FILE).mkdir()  # unwritable as a file
    v = Verdict(decision="drop", already_known="k", reason="already-known", justification="j")
    with caplog.at_level("WARNING", logger="sift.distill.rejects"):
        assert record_reject(_candidate(), v) is False
    assert capsys.readouterr().out == ""
    assert "could not log" in caplog.text
