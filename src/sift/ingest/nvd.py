"""NVD 2.0 API -> `cve` notes, filtered to web-app-relevant CWEs and recent years.

Without an API key NVD allows 5 requests / 30s; with one, 50. We sleep
conservatively and page through 120-day windows (the API's max range) per CWE.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import httpx

from sift.config import get_settings
from sift.ingest.base import clean_text
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
WINDOW_DAYS = 120
PAGE = 2000

# Web / app security relevant CWEs — the classes a bug bounty hunter actually chases.
DEFAULT_CWES = [
    "CWE-79",  # XSS
    "CWE-89",  # SQLi
    "CWE-352",  # CSRF
    "CWE-918",  # SSRF
    "CWE-22",  # Path traversal
    "CWE-94",  # Code injection
    "CWE-77",  # Command injection
    "CWE-78",  # OS command injection
    "CWE-611",  # XXE
    "CWE-502",  # Insecure deserialization
    "CWE-287",  # Improper authentication
    "CWE-639",  # IDOR / authz
    "CWE-863",  # Incorrect authorization
    "CWE-1336",  # SSTI
    "CWE-434",  # Unrestricted file upload
    "CWE-384",  # Session fixation
]


def _iso(d: date) -> str:
    return datetime(d.year, d.month, d.day, tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%S.000")


def _windows(since_year: int) -> Iterator[tuple[date, date]]:
    start = date(since_year, 1, 1)
    today = date.today()
    cur = start
    while cur < today:
        end = min(cur + timedelta(days=WINDOW_DAYS), today)
        yield cur, end
        cur = end + timedelta(days=1)


def _headers() -> dict:
    key = get_settings().nvd_api_key
    return {"apiKey": key} if key else {}


def _severity(metrics: dict) -> tuple[str | None, float | None, str | None]:
    for k in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        arr = metrics.get(k)
        if arr:
            data = arr[0].get("cvssData", {})
            sev = data.get("baseSeverity") or arr[0].get("baseSeverity")
            return (
                sev.lower() if sev else None,
                data.get("baseScore"),
                data.get("vectorString"),
            )
    return None, None, None


def to_note(item: dict) -> Note | None:
    cve = item.get("cve", {})
    cid = cve.get("id")
    if not cid:
        return None
    descs = [d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"]
    desc = clean_text(descs[0] if descs else "")
    if desc.startswith("** REJECT **") or desc.startswith("** DISPUTED **"):
        return None

    cwes: list[str] = []
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            v = d.get("value", "")
            if v.startswith("CWE-") and v not in cwes:
                cwes.append(v)

    sev, score, vector = _severity(cve.get("metrics", {}))
    refs = [r["url"] for r in cve.get("references", []) if r.get("url")][:10]
    pub = cve.get("published", "")[:10] or None

    body = "\n\n".join(
        p
        for p in [
            "## Description",
            desc,
            f"## CVSS\n{sev or 'n/a'} ({score}) `{vector or ''}`" if score else "",
            "## References\n" + "\n".join(f"- {u}" for u in refs) if refs else "",
        ]
        if p
    )

    meta = Frontmatter(
        id=cid,
        type="cve",
        title=f"{cid} — {(desc[:80] + '…') if len(desc) > 80 else desc}",
        source="nvd",
        url=f"https://nvd.nist.gov/vuln/detail/{cid}",
        created=date.fromisoformat(pub) if pub else None,
        cwe=cwes,
        severity=sev,
        tags=["cve"],
        extra={"cvss_score": score, "cvss_vector": vector},
    )
    return Note(meta=meta, body=body)


def source(
    *, since_year: int | None = None, cwes: list[str] | None = None, max_notes: int | None = None
) -> Iterator[Note]:
    since_year = since_year or (date.today().year - 2)
    cwes = cwes or DEFAULT_CWES
    yielded = 0
    with httpx.Client(timeout=120, follow_redirects=True, headers=_headers()) as c:
        for cwe in cwes:
            for w_start, w_end in _windows(since_year):
                start_index = 0
                retries = 0
                while True:
                    params = {
                        "cweId": cwe,
                        "pubStartDate": _iso(w_start),
                        "pubEndDate": _iso(w_end),
                        "resultsPerPage": PAGE,
                        "startIndex": start_index,
                    }
                    try:
                        r = c.get(API, params=params)
                        r.raise_for_status()
                        retries = 0
                    except httpx.HTTPError as exc:
                        retries += 1
                        if retries > 5:
                            print(f"  ! nvd {cwe} {w_start}: giving up after 5 retries ({exc})")
                            break
                        print(f"  ! nvd {cwe} {w_start}: {exc}; retry {retries}/5 in 20s")
                        time.sleep(20)
                        continue
                    data = r.json()
                    vulns = data.get("vulnerabilities", [])
                    for item in vulns:
                        note = to_note(item)
                        if note:
                            yield note
                            yielded += 1
                            if max_notes and yielded >= max_notes:
                                return
                    total = data.get("totalResults", 0)
                    start_index += PAGE
                    time.sleep(0.7 if get_settings().nvd_api_key else 6.5)
                    if start_index >= total or not vulns:
                        break
