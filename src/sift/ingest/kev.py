"""CISA Known Exploited Vulnerabilities catalog -> `cve` notes.

KEV is the highest-signal CVE subset: every entry has confirmed in-the-wild
exploitation. ~1,300 entries, one small JSON file.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import httpx

from sift.ingest.base import clean_text
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

FEED = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def fetch() -> list[dict]:
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        r = c.get(FEED)
        r.raise_for_status()
        return r.json().get("vulnerabilities", [])


def to_note(v: dict) -> Note:
    cve = v["cveID"]
    name = v.get("vulnerabilityName") or cve
    vendor = v.get("vendorProject", "")
    product = v.get("product", "")
    cwes = [c.upper() for c in v.get("cwes", []) if c]

    body = "\n\n".join(
        p
        for p in [
            f"**{vendor} {product}** — {name}".strip(" —"),
            "## Summary",
            clean_text(v.get("shortDescription")),
            "## Required action",
            clean_text(v.get("requiredAction")),
            f"## Ransomware use\n{v.get('knownRansomwareCampaignUse', 'Unknown')}",
            f"## Notes\n{clean_text(v.get('notes'))}" if v.get("notes") else "",
        ]
        if p
    )

    meta = Frontmatter(
        id=cve,
        type="cve",
        title=f"{cve} — {name}",
        source="cisa-kev",
        url=f"https://nvd.nist.gov/vuln/detail/{cve}",
        created=_parse_date(v.get("dateAdded")),
        cwe=cwes,
        tags=["kev", "known-exploited"]
        + ([vendor.lower()] if vendor else [])
        + ([product.lower()] if product else []),
        program=vendor or None,
        extra={
            "kev_date_added": v.get("dateAdded"),
            "kev_due_date": v.get("dueDate"),
        },
    )
    return Note(meta=meta, body=body)


def source() -> Iterator[Note]:
    for v in fetch():
        try:
            yield to_note(v)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! kev: bad record {v.get('cveID')}: {exc}")
