"""Public HackerOne disclosed reports (Hugging Face: Hacker0x01/hackerone_disclosed_reports).

~12.6k disclosed reports across train/test/validation parquet files. We pull the
parquet bytes directly with httpx and read them with pyarrow — no `datasets` or
`huggingface_hub` dependency.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import date

import httpx
import pyarrow.parquet as pq

from sift.ingest.base import clean_text, extract_cwes
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

BASE = "https://huggingface.co/datasets/Hacker0x01/hackerone_disclosed_reports/resolve/main/data"
FILES = [
    "train-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
    "validation-00000-of-00001.parquet",
]

# Common weakness-name -> CWE mapping (the dataset only gives free-text names).
_WEAKNESS_CWE = {
    "cross-site scripting": "CWE-79",
    "xss": "CWE-79",
    "sql injection": "CWE-89",
    "cross-site request forgery": "CWE-352",
    "csrf": "CWE-352",
    "server-side request forgery": "CWE-918",
    "ssrf": "CWE-918",
    "path traversal": "CWE-22",
    "directory traversal": "CWE-22",
    "insecure direct object reference": "CWE-639",
    "idor": "CWE-639",
    "improper authentication": "CWE-287",
    "improper authorization": "CWE-285",
    "privilege escalation": "CWE-269",
    "information disclosure": "CWE-200",
    "open redirect": "CWE-601",
    "xml external entities": "CWE-611",
    "xxe": "CWE-611",
    "deserialization": "CWE-502",
    "command injection": "CWE-77",
    "code injection": "CWE-94",
    "race condition": "CWE-362",
    "business logic": "CWE-840",
    "denial of service": "CWE-400",
    "server-side template injection": "CWE-1336",
    "ssti": "CWE-1336",
    "subdomain takeover": "CWE-350",
}


def _weakness_to_cwe(name: str | None) -> list[str]:
    if not name:
        return []
    low = name.lower()
    for key, cwe in _WEAKNESS_CWE.items():
        if key in low:
            return [cwe]
    return []


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _download_table(client: httpx.Client, fname: str):
    r = client.get(f"{BASE}/{fname}")
    r.raise_for_status()
    return pq.read_table(io.BytesIO(r.content))


def to_note(row: dict) -> Note | None:
    rid = row.get("id")
    if not rid:
        return None
    body = clean_text(row.get("vulnerability_information"))
    if not body or len(body) < 40:
        return None

    weakness = (row.get("weakness") or {}).get("name")
    scope = row.get("structured_scope") or {}
    team = row.get("team") or {}
    program = (team.get("profile") or {}).get("name") or team.get("handle")
    asset = scope.get("asset_identifier")
    sev = scope.get("max_severity")

    cwes = _weakness_to_cwe(weakness) or extract_cwes(body, weakness)
    tags = ["hackerone", "disclosed"]
    if weakness:
        tags.append(weakness.lower())

    meta = Frontmatter(
        id=f"h1-{rid}",
        type="report",
        title=clean_text(row.get("title")) or f"HackerOne report {rid}",
        source="hackerone-public",
        url=f"https://hackerone.com/reports/{rid}",
        created=_parse_date(row.get("disclosed_at") or row.get("created_at")),
        cwe=cwes,
        severity=sev if sev and sev != "none" else None,
        program=program,
        assets=[asset] if asset else [],
        bounty=None,
        tags=tags,
        extra={
            "weakness": weakness,
            "has_bounty": bool(row.get("has_bounty?")),
            "vote_count": row.get("vote_count"),
            "is_dupe": bool(row.get("original_report_id")),
        },
    )
    return Note(meta=meta, body=body)


def source(*, limit: int | None = None) -> Iterator[Note]:
    seen = 0
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        for fname in FILES:
            try:
                table = _download_table(client, fname)
            except httpx.HTTPError as exc:
                print(f"  ! h1-public: could not fetch {fname}: {exc}")
                continue
            for row in table.to_pylist():
                note = to_note(row)
                if not note:
                    continue
                yield note
                seen += 1
                if limit and seen >= limit:
                    return
