"""In-session gating: Claude Code is the gate, no API key needed.

`export` writes candidates to JSONL; the model reads them, judges each against the
rubric in `gate.GATE_SYSTEM`, and writes verdicts back; `apply` turns keeps into
technique notes and logs the drops.

The advantage over the unattended API path is that a verdict can be argued with
while it's being made. The cost is that it doesn't run on a schedule.
"""

from __future__ import annotations

import json
from pathlib import Path

from sift.config import get_settings
from sift.distill.candidates import Candidate
from sift.distill.technique import build_note
from sift.vault.notes import iter_notes, save_note

# Verdict rows must carry these; keeps need the distillation fields as well.
_REQUIRED = ("url", "decision", "already_known", "reason", "justification")
_KEEP_REQUIRED = ("technique_title", "when_to_try", "body_md")


def gated_urls() -> set[str]:
    """URLs already judged — derived from the reject log plus existing technique
    notes, so there's no separate state file to drift out of sync."""
    from sift.distill.rejects import load_rejects

    seen = {r.get("url") for r in load_rejects() if r.get("url")}
    vault = get_settings().resolved_vault()
    for note in iter_notes(vault, note_type="technique"):
        if note.meta.url:
            seen.add(note.meta.url)
    return seen


def collect(
    note_type: str = "writeup",
    *,
    limit: int | None = None,
    skip_gated: bool = True,
    prefilter: bool = True,
) -> list[Candidate]:
    """Gather ungated candidates from notes already in the vault.

    With `prefilter`, structurally-obvious drops never reach the gate at all - see
    `distill.prefilter`, which is graded against the hand-judged set.
    """
    from sift.distill.prefilter import prefilter_reason

    vault = get_settings().resolved_vault()
    seen = gated_urls() if skip_gated else set()
    # The same article legitimately arrives from several sources - the research feed,
    # a Top-10 nomination, and PentesterLand all carry Doyensec's CSPT2CSRF post.
    # Measured: 68 duplicate copies across 63 URLs. Judging each twice costs real
    # money and yields duplicate technique notes, so keep the longest body per URL.
    best: dict[str, Candidate] = {}
    out: list[Candidate] = []
    for note in iter_notes(vault, note_type=note_type):
        url = (note.meta.url or "").split("?")[0].rstrip("/")
        if url and url in seen:
            continue
        if prefilter and prefilter_reason(note.meta.title, note.meta.url or ""):
            continue
        cand = Candidate.from_note(note)
        if not url:
            out.append(cand)
        elif url not in best or len(cand.text) > len(best[url].text):
            best[url] = cand

    out.extend(best.values())
    return out[:limit] if limit else out


def write_candidates(candidates: list[Candidate], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(
                json.dumps(
                    {
                        "title": c.title,
                        "url": c.url,
                        "source": c.source,
                        "created": c.created.isoformat() if c.created else None,
                        "text": c.gate_text(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(candidates)


def _validate(row: dict) -> str | None:
    missing = [f for f in _REQUIRED if not row.get(f)]
    if row.get("decision") == "keep":
        missing += [f for f in _KEEP_REQUIRED if not row.get(f)]
    if missing:
        return f"missing {', '.join(missing)}"
    if row["decision"] not in ("keep", "drop"):
        return f"bad decision {row['decision']!r}"
    return None


def apply_verdicts(verdicts_path: Path, candidates: list[Candidate]) -> dict:
    """Write technique notes for keeps, log drops. Returns a summary."""
    from sift.distill.gate import Verdict
    from sift.distill.rejects import record_reject
    from sift.index.store import Store
    from sift.pipeline import index_note

    by_url = {c.url: c for c in candidates if c.url}
    kept = dropped = skipped = chunks = 0

    rows = [
        json.loads(line)
        for line in verdicts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    vault = get_settings().resolved_vault()
    store = Store()
    for row in rows:
        problem = _validate(row)
        if problem:
            print(f"  ! verdict skipped ({problem}): {row.get('url') or row.get('title')}")
            skipped += 1
            continue
        cand = by_url.get(row["url"])
        if cand is None:
            print(f"  ! verdict has no matching candidate: {row['url']}")
            skipped += 1
            continue

        if row["decision"] == "drop":
            record_reject(
                cand,
                Verdict(
                    decision="drop",
                    already_known=row["already_known"],
                    reason=row["reason"],
                    justification=row["justification"],
                ),
            )
            dropped += 1
            continue

        note = build_note(
            cand,
            title=row["technique_title"],
            when_to_try=row["when_to_try"],
            body_md=row["body_md"],
            keep_reason=row["reason"],
            already_known=row["already_known"],
            tags=row.get("tags"),
            cwe=row.get("cwe"),
        )
        save_note(vault, note, stamp=True)
        chunks += index_note(note, store)
        kept += 1

    if kept:
        store.ensure_fts()
    return {"kept": kept, "dropped": dropped, "skipped": skipped, "chunks_indexed": chunks}
