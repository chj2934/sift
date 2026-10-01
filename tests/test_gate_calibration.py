"""Live calibration of the novelty gate. Opt-in — costs a few cents.

    uv run --no-sync python -m pytest tests/test_gate_calibration.py -m calibration

The gate is the product: if it drifts, the vault silently fills with material the
model already knows and retrieval quality rots. Re-run this whenever GATE_SYSTEM
changes, and read the justifications rather than just the pass/fail.
"""

from __future__ import annotations

from datetime import date

import pytest

pytestmark = pytest.mark.calibration


@pytest.fixture(autouse=True)
def _needs_credentials():
    """Skip, with the reason on screen, when no credentials resolve.

    Checked per test with the SDK's own resolution - env, .env, or an `ant auth login`
    profile. The old check read only the ANTHROPIC_API_KEY env var, so a key kept in
    .env (as the gate's own error message advises) skipped every test: the check that
    guards GATE_SYSTEM changes silently never ran.
    """
    from sift.distill import gate

    if not gate.credentials_available():
        pytest.skip("live gate calibration needs Anthropic credentials (env, .env or ant profile)")


# --- fixtures: material whose verdict we already know the right answer to ---

WELL_KNOWN = """\
Reflected XSS happens when user input is echoed straight back into the response \
without encoding. To test for it, inject a marker such as xss1234 into each \
parameter and look for it in the response body. If it reflects, try breaking out \
of the surrounding context: <script>alert(1)</script> in HTML body, "> to close an \
attribute, or '; alert(1); // inside a JavaScript string. Check whether the \
application encodes < and > and whether a Content-Security-Policy is present. \
Remember to test headers and path segments as well as query parameters.\
"""

CHEATSHEET = """\
IDOR (Insecure Direct Object Reference) — Testing Checklist

Find endpoints that accept an object identifier: /api/user/1234, ?invoice_id=99.
Change the identifier to another user's value and see whether the object is returned.
Try: sequential IDs, UUIDs leaked elsewhere, IDs from a second account, wrapping the
value in an array, changing the HTTP method, and adding the parameter twice.
Escalate by chaining to account takeover where the object contains a reset token.
Always create two accounts so you have a legitimate second identifier to substitute.\
"""

OBSCURE_VARIANT = """\
When a front-end proxy normalises Transfer-Encoding but the back-end applies a \
lenient chunk-size parser, a chunk extension containing a bare CR (\\r without \\n) \
is treated as terminating by one parser and as part of the extension by the other. \
Specifically, sending "5;a=b\\rZZZZ\\r\\n" causes Akamai's parser to read a 5-byte \
chunk while the origin skips to the next \\r\\n and resynchronises four bytes later. \
The desync is only reachable when the request also carries a duplicated \
Content-Length whose second value is smaller than the first, because the front-end \
selects the last header instance while the origin selects the first.\
"""

NOVEL = """\
We found that the WebTransport datagram negotiation in the draft-11 implementation \
shipped by several CDNs derives its session key from the ALPN token concatenated \
with the SNI, without a length prefix. A hostname ending in the literal bytes of a \
valid ALPN token therefore produces a key collision with a different session. By \
registering a subdomain named h3-datagram.example.com and opening a session, an \
attacker can decrypt datagrams belonging to any co-tenant session on the same edge \
node whose ALPN is h3-datagram, because the concatenation is ambiguous.\
"""


def _judge(title, text, published):
    from sift.distill.candidates import Candidate
    from sift.distill.gate import judge

    return judge(
        Candidate(
            title=title,
            url="https://example.com/post",
            text=text,
            source="example.com",
            created=published,
        )
    )


@pytest.mark.parametrize(
    "title,text,published",
    [
        ("How to test for reflected XSS", WELL_KNOWN, date(2024, 3, 1)),
        ("IDOR testing checklist", CHEATSHEET, date(2023, 9, 12)),
    ],
)
def test_drops_what_the_model_already_knows(title, text, published):
    v = _judge(title, text, published)
    assert not v.keep, f"expected drop, got keep: {v.justification}"
    assert v.reason == "already-known"


@pytest.mark.parametrize(
    "title,text,published",
    [
        ("Chunk-extension CR desync against Akamai", OBSCURE_VARIANT, date(2026, 4, 2)),
        ("WebTransport datagram key collision via ALPN/SNI", NOVEL, date(2026, 7, 18)),
    ],
)
def test_keeps_what_the_model_lacks(title, text, published):
    v = _judge(title, text, published)
    assert v.keep, f"expected keep, got drop: {v.justification}"
    assert v.reason in ("post-cutoff", "obscure-variant", "operational-detail")


def test_verdict_states_what_it_already_knew():
    """The self-check is what makes the judgement honest — it must not come back empty."""
    v = _judge("How to test for reflected XSS", WELL_KNOWN, date(2024, 3, 1))
    assert len(v.already_known.strip()) > 20
