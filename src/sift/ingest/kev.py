"""CISA Known Exploited Vulnerabilities catalog -> `cve` notes.

KEV is the highest-signal CVE subset: every entry has confirmed in-the-wild
exploitation. ~1,300 entries, one small JSON file.

A CVE NVD already wrote is merged into, not duplicated or refused (`cve_merge`): the
note keeps NVD's description and CVSS and gains KEV's tags, required action and
ransomware use. Re-running on an unchanged catalog rewrites and re-embeds nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import httpx

from sift.config import get_settings
from sift.ingest.base import clean_text
from sift.ingest.cve_merge import merge_cve
from sift.ingest.existing import merge_into_existing
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

log = logging.getLogger(__name__)

FEED = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# For `run_source(..., merge=kev.merge)`. Optional: `source()` already merges.
merge = merge_cve


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


def source(*, vault: Path | None = None) -> Iterator[Note]:
    """Every KEV record, merged into the CVE note already in the vault if there is one."""
    vault = Path(vault) if vault is not None else get_settings().resolved_vault()
    for v in fetch():
        try:
            note = to_note(v)
        except Exception as exc:  # noqa: BLE001 - one malformed record never stops the feed
            log.warning("kev: bad record %s: %s", v.get("cveID") if isinstance(v, dict) else v, exc)
            continue
        yield merge_into_existing(vault, note, merge_cve)
