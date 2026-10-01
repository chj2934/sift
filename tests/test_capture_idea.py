"""capture_idea / resolve_idea round-trip.

The point of the status loop is that dead ends get recorded. These tests pin the
behaviour that makes that work: status lives in `extra` (filterable) and is mirrored
into a `status/` tag, and resolving replaces the old status rather than accumulating.
"""

from __future__ import annotations

import pytest

IDEA = "Try JWT alg confusion on the SSO callback - it accepts both RS256 and HS256"
WHY = "The callback echoes the kid header, suggesting a key lookup we may control."


def _capture(**over):
    from sift.mcp_server import capture_idea

    kwargs = dict(idea=IDEA, reasoning=WHY, target="acme", tags=["jwt"], cwe=["CWE-347"])
    kwargs.update(over)
    return capture_idea(**kwargs)


def test_capture_starts_as_hypothesis(vault_path):
    res = _capture()
    assert res["saved"] is True
    assert res["status"] == "hypothesis"

    from sift.pipeline import get_note_by_slug

    note = get_note_by_slug(res["slug"])
    assert note.meta.extra["status"] == "hypothesis"
    assert "status/hypothesis" in note.meta.tags
    assert note.meta.program == "acme"
    assert note.meta.cwe == ["CWE-347"]
    assert WHY in note.body


def test_resolve_records_a_dead_end(vault_path):
    res = _capture()
    from sift.mcp_server import resolve_idea

    out = resolve_idea(
        slug=res["slug"],
        status="failed",
        notes="RS256 validated properly; kid is not attacker-controlled.",
    )
    assert out["updated"] is True

    from sift.pipeline import get_note_by_slug

    note = get_note_by_slug(res["slug"])
    assert note.meta.extra["status"] == "failed"
    assert "RS256 validated properly" in note.body
    # The whole point: a failure is retrievable, not lost.
    assert "**Status:** failed" in note.body


def test_status_tag_is_replaced_not_accumulated(vault_path):
    res = _capture()
    from sift.mcp_server import resolve_idea

    resolve_idea(slug=res["slug"], status="partial", notes="worked only when logged out")
    resolve_idea(slug=res["slug"], status="worked", notes="chained with cookie tossing")

    from sift.pipeline import get_note_by_slug

    note = get_note_by_slug(res["slug"])
    status_tags = [t for t in note.meta.tags if t.startswith("status/")]
    assert status_tags == ["status/worked"], status_tags
    assert note.meta.extra["status"] == "worked"


@pytest.mark.parametrize("bad", ["hypothesis", "done", ""])
def test_rejects_bad_status(vault_path, bad):
    res = _capture()
    from sift.mcp_server import resolve_idea

    assert "error" in resolve_idea(slug=res["slug"], status=bad, notes="x")


def test_resolve_unknown_slug_is_an_error(vault_path):
    from sift.mcp_server import resolve_idea

    assert "error" in resolve_idea(slug="nope-does-not-exist", status="failed", notes="x")
