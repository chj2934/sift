"""NVD 2.0 API -> `cve` notes, filtered to web-app-relevant CWEs and recent years.

Without an API key NVD allows 5 requests / 30s; with one, 50. We sleep
conservatively and page through 120-day windows (the API's max range) per CWE.

Windows are contiguous and inclusive to the millisecond (`_windows`, `_iso`): the old
end bound was midnight at the *start* of the last day and the next window began a day
later, so one full day per window - 8 days a year, on every run, for every CWE - was
never fetched. A CVE carrying two of the CWEs is yielded once per run.

A CVE KEV already wrote is merged into rather than refused, and a re-ingest of an
edited description updates the note without wiping EPSS scores or the user's own
sections (`cve_merge`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from sift.config import get_settings
from sift.ingest.base import clean_text
from sift.ingest.cve_merge import merge_cve
from sift.ingest.existing import merge_into_existing
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

log = logging.getLogger(__name__)

API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
WINDOW_DAYS = 120
PAGE = 2000
RETRIES = 5
RETRY_SLEEP_S = 20

# For `run_source(..., merge=nvd.merge)`. Optional: `source()` already merges.
merge = merge_cve

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


def _iso(d: date, *, end: bool = False) -> str:
    """NVD reads an offset-less timestamp as UTC. The end bound is inclusive to the
    millisecond, so a window covers its last day in full."""
    return f"{d.isoformat()}T{'23:59:59.999' if end else '00:00:00.000'}"


def _windows(since_year: int, today: date | None = None) -> Iterator[tuple[date, date]]:
    """Contiguous, inclusive ``[first, last]`` day windows of at most `WINDOW_DAYS`
    days, from Jan 1 of `since_year` through `today` (default: today in UTC).

    Each window spans ``first 00:00:00.000`` to ``last 23:59:59.999`` - 119 days and
    change, inside NVD's 120-day limit - and the next starts the day after ``last``.
    """
    today = today or datetime.now(UTC).date()
    cur = date(since_year, 1, 1)
    while cur <= today:
        last = min(cur + timedelta(days=WINDOW_DAYS - 1), today)
        yield cur, last
        cur = last + timedelta(days=1)


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
    *,
    since_year: int | None = None,
    cwes: list[str] | None = None,
    max_notes: int | None = None,
    vault: Path | None = None,
) -> Iterator[Note]:
    since_year = since_year or (date.today().year - 2)
    cwes = cwes or DEFAULT_CWES
    vault = Path(vault) if vault is not None else get_settings().resolved_vault()
    seen: set[str] = set()  # a CVE tagged CWE-77 and CWE-78 comes back for both
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
                        "pubEndDate": _iso(w_end, end=True),
                        "resultsPerPage": PAGE,
                        "startIndex": start_index,
                    }
                    try:
                        r = c.get(API, params=params)
                        r.raise_for_status()
                        data = r.json()  # a 200 with an HTML error body is retryable too
                        if not isinstance(data, dict):
                            raise ValueError("response is not a JSON object")
                    except (httpx.HTTPError, ValueError) as exc:
                        retries += 1
                        if retries > RETRIES:
                            log.warning(
                                "nvd %s %s: giving up on this window after %d retries (%s)",
                                cwe, w_start, RETRIES, exc,
                            )  # fmt: skip
                            break
                        log.warning(
                            "nvd %s %s: %s; retry %d/%d in %ds",
                            cwe, w_start, exc, retries, RETRIES, RETRY_SLEEP_S,
                        )  # fmt: skip
                        time.sleep(RETRY_SLEEP_S)
                        continue
                    retries = 0
                    vulns = data.get("vulnerabilities", []) or []
                    for item in vulns:
                        cid = (item.get("cve") or {}).get("id") if isinstance(item, dict) else None
                        if not cid or cid in seen:
                            continue
                        seen.add(cid)
                        note = to_note(item)
                        if note is None:
                            continue
                        yield merge_into_existing(vault, note, merge_cve)
                        yielded += 1
                        if max_notes and yielded >= max_notes:
                            return
                    total = data.get("totalResults", 0)
                    start_index += PAGE
                    time.sleep(0.7 if get_settings().nvd_api_key else 6.5)
                    if start_index >= total or not vulns:
                        break
