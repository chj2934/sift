"""NVD date windows: contiguous, inclusive, inside the API's 120-day limit.

The old windows ended at midnight at the *start* of their last day and the next one
began a day later, so one whole day per window - 8 days a year, every run, every CWE -
was never fetched.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest


@pytest.mark.parametrize(
    "today",
    [
        date(2024, 1, 1),  # the first day only
        date(2024, 4, 29),  # the last day of the first window
        date(2024, 4, 30),  # the first day of the second
        date(2026, 10, 1),
    ],
)
def test_windows_cover_every_day_exactly_once(today):
    from sift.ingest.nvd import WINDOW_DAYS, _iso, _windows

    windows = list(_windows(2024, today=today))

    assert windows[0][0] == date(2024, 1, 1)
    assert windows[-1][1] == today
    covered: list[date] = []
    for i, (first, last) in enumerate(windows):
        assert first <= last
        if i:
            assert first == windows[i - 1][1] + timedelta(days=1), "gap or overlap"
        start = datetime.fromisoformat(_iso(first))
        end = datetime.fromisoformat(_iso(last, end=True))
        assert end - start < timedelta(days=WINDOW_DAYS), "over the API's range limit"
        covered += [first + timedelta(days=n) for n in range((last - first).days + 1)]
    expected = [
        date(2024, 1, 1) + timedelta(days=n) for n in range((today - date(2024, 1, 1)).days + 1)
    ]
    assert covered == expected


def test_timestamps_are_inclusive_to_the_millisecond():
    from sift.ingest.nvd import _iso

    assert _iso(date(2024, 4, 30)) == "2024-04-30T00:00:00.000"
    assert _iso(date(2024, 4, 30), end=True) == "2024-04-30T23:59:59.999"


def test_a_future_since_year_yields_no_window():
    from sift.ingest.nvd import _windows

    assert list(_windows(2030, today=date(2026, 10, 1))) == []


def _vuln(cid, *cwes):
    return {
        "cve": {
            "id": cid,
            "published": "2026-08-01T00:00:00.000",
            "descriptions": [{"lang": "en", "value": f"{cid} lets an attacker do things."}],
            "weaknesses": [{"description": [{"value": c} for c in cwes]}],
        }
    }


def test_a_cve_in_two_cwes_is_yielded_once_and_bad_json_is_retried(monkeypatch):
    import httpx

    from sift.ingest import nvd

    calls: list[dict] = []
    bad_once = {"left": 1}

    def handler(request):
        params = dict(request.url.params)
        calls.append(params)
        if params["cweId"] == "CWE-78" and bad_once["left"]:
            bad_once["left"] -= 1
            return httpx.Response(200, text="<html>maintenance</html>")  # 200, not JSON
        vulns = [_vuln("CVE-2026-0001", "CWE-77", "CWE-78")]
        if params["cweId"] == "CWE-78":
            vulns.append(_vuln("CVE-2026-0002", "CWE-78"))
        return httpx.Response(200, json={"vulnerabilities": vulns, "totalResults": len(vulns)})

    real = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )
    monkeypatch.setattr(nvd.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        nvd,
        "_windows",
        lambda since_year, today=None: iter([(date(2026, 8, 1), date(2026, 8, 31))]),
    )

    notes = list(nvd.source(since_year=2026, cwes=["CWE-77", "CWE-78"]))

    assert [n.meta.id for n in notes] == ["CVE-2026-0001", "CVE-2026-0002"]
    assert calls[0]["pubEndDate"] == "2026-08-31T23:59:59.999"
    assert len(calls) == 3, "the non-JSON 200 was retried, not fatal"
