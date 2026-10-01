"""The prefilter is graded against real hand-judged titles.

A false drop is invisible and permanent - the candidate never reaches the gate and
never enters the vault. So the rules are held to zero false drops on the 21 titles
that were hand-judged KEEP, and are only expected to catch a subset of the 19 DROPs.
"""

from __future__ import annotations

# Hand-judged KEEP in the first PortSwigger batch. None of these may be prefiltered.
KEPT_TITLES = [
    "Bypassing character blocklists with unicode overflows",
    "Bypassing WAFs with the phantom $Version cookie",
    "Can AI do novel security research? Meet the HTTP Terminator",
    "Concealing payloads in URL credentials",
    "Cookie Chaos: How to bypass __Host and __Secure cookie prefixes",
    "CRLF-Powered Desync Attacks: Beheading HTTP Streams",
    "CSS:the bomb inside your inbox",
    "Drag and Pwnd: Leverage ASCII characters to exploit VS Code",
    "Fickle PDFs: exploiting browser rendering discrepancies",
    "Gotta cache 'em all: bending the rules of web cache exploitation",
    "HTTP/1.1 must die: the desync endgame",
    "Inline Style Exfiltration: leaking data with chained CSS conditionals",
    "Listen to the whispers: web timing attacks that actually work",
    "Making desync attacks easy with TRACE",
    "New crazy payloads in the URL Validation Bypass Cheat Sheet",
    "onwebkitplaybacktargetavailabilitychanged?! New exotic events in the XSS cheat sheet",
    "SAML roulette: the hacker always wins",
    "Splitting the email atom: exploiting parsers to bypass access controls",
    "Stealing HttpOnly cookies with the cookie sandwich technique",
    "The Fragile Lock: Novel Bypasses For SAML Authentication",
    "What's in a tag name? JavaScript, apparently",
]

# Hand-judged DROP. The prefilter should catch the structural ones for free.
DROPPED_TITLES = [
    "A hacking hat-trick: previewing three PortSwigger Research publications coming to DEF CON & Black Hat USA",
    "Beware the false false-positive: how to distinguish HTTP pipelining from request smuggling",
    "Document My Pentest: you hack, the AI writes it up!",
    "Finding that one weird endpoint, with Bambdas",
    "Hiding payloads in Java source code strings",
    "Introducing HTTP Anomaly Rank",
    "Introducing SignSaboteur: forge signed web tokens with ease",
    "Introducing the URL validation bypass cheat sheet",
    "Refining your HTTP perspective, with bambdas",
    "Repeater Strike: manual testing, amplified",
    "Shadow Repeater:AI-enhanced manual testing",
    "Top 10 web hacking techniques of 2023 - nominations open",
    "Top 10 web hacking techniques of 2023",
    "Top 10 web hacking techniques of 2024: nominations open",
    "Top 10 web hacking techniques of 2024",
    "Top 10 web hacking techniques of 2025: call for nominations",
    "Top 10 web hacking techniques of 2025",
    "Using form hijacking to bypass CSP",
    "WebSocket Turbo Intruder: Unearthing the WebSocket Goldmine",
]


def test_never_drops_hand_judged_keeps():
    """The rule that matters. A false drop here is silent data loss."""
    from sift.distill.prefilter import prefilter_reason

    wrongly_dropped = [t for t in KEPT_TITLES if prefilter_reason(t) is not None]
    assert wrongly_dropped == [], f"prefilter would discard real techniques: {wrongly_dropped}"


def test_catches_the_structural_drops():
    from sift.distill.prefilter import prefilter_reason

    caught = [t for t in DROPPED_TITLES if prefilter_reason(t) is not None]
    # 6 index posts + 3 "Introducing ..." + 1 conference preview
    assert len(caught) >= 10, f"only caught {len(caught)}: {caught}"


def test_meet_the_http_terminator_is_not_a_tool_announcement():
    """'Meet the HTTP Terminator' is a Black Hat whitepaper, not a product post -
    the tool-announcement rule anchors at the start of the title for this reason."""
    from sift.distill.prefilter import prefilter_reason

    assert prefilter_reason("Can AI do novel security research? Meet the HTTP Terminator") is None
    assert (
        prefilter_reason("Introducing SignSaboteur: forge signed web tokens") == "tool-announcement"
    )


def test_cheat_sheet_updates_still_reach_the_gate():
    """'New crazy payloads in the ... Cheat Sheet' was a KEEP - it carried genuinely
    new payloads. So cheat-sheet titles must not be a blanket prefilter rule."""
    from sift.distill.prefilter import prefilter_reason

    assert prefilter_reason("New crazy payloads in the URL Validation Bypass Cheat Sheet") is None
