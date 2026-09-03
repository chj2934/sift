from __future__ import annotations

from sift.ingest.base import clean_text, extract_cwes
from sift.ingest.h1_public import to_note as h1_to_note
from sift.ingest.kev import to_note as kev_to_note


def test_clean_text():
    assert clean_text("a\r\n\r\n\r\n\r\nb") == "a\n\nb"
    assert clean_text(None) == ""


def test_extract_cwes():
    assert extract_cwes("blah CWE-79 blah", "also cwe-89") == ["CWE-79", "CWE-89"]


def test_kev_record_to_note():
    rec = {
        "cveID": "CVE-2023-9999",
        "vendorProject": "Acme",
        "product": "Widget",
        "vulnerabilityName": "Acme Widget RCE",
        "dateAdded": "2023-06-01",
        "shortDescription": "Remote code execution in Widget.",
        "requiredAction": "Patch.",
        "cwes": ["CWE-94"],
    }
    note = kev_to_note(rec)
    assert note.meta.id == "CVE-2023-9999"
    assert note.meta.type == "cve"
    assert "CWE-94" in note.meta.cwe
    assert "kev" in note.meta.tags
    assert "Remote code execution" in note.body


def test_h1_public_record_to_note():
    row = {
        "id": 12345,
        "title": "Reflected XSS on search",
        "disclosed_at": "2024-02-02T00:00:00Z",
        "vulnerability_information": "The `q` parameter is reflected without encoding. "
        "Payload: <script>alert(1)</script>. This is a full description with detail.",
        "weakness": {"id": 1, "name": "Cross-site Scripting (XSS)"},
        "structured_scope": {"asset_identifier": "www.example.com", "max_severity": "medium"},
        "team": {"handle": "example", "profile": {"name": "Example Inc"}},
        "has_bounty?": True,
        "original_report_id": None,
    }
    note = h1_to_note(row)
    assert note is not None
    assert note.meta.id == "h1-12345"
    assert note.meta.cwe == ["CWE-79"]
    assert note.meta.severity == "medium"
    assert note.meta.program == "Example Inc"
    assert note.meta.url == "https://hackerone.com/reports/12345"


def test_h1_public_skips_thin_body():
    assert h1_to_note({"id": 1, "vulnerability_information": "too short"}) is None
