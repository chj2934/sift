"""In-session gating: Claude Code is the gate, no API key needed.

`export` writes candidates to JSONL; the model reads them, judges each against the
rubric in `gate.GATE_SYSTEM`, and writes verdicts back; `apply` turns keeps into
technique notes and logs the drops.

An exported row's `text` is only the gate excerpt - the first `GATE_TEXT_CHARS` of the
article, enough to judge it. Each row also carries the source note's `note_id`, `slug`,
`path`, `chars_total` and `truncated`. When `truncated` is true, read the full note
(`path`, or `get_note(slug)`) before writing `body_md` and `when_to_try`: the payloads
and conditions further down the article are exactly what a keep is for.

A verdict is one JSON object per line:

    url, decision ("keep" | "drop"), already_known, reason, justification   always
    technique_title, when_to_try, body_md                                   keeps
    tags, cwe (lists of strings)                                            optional

A keep's `reason` must be one of `gate.KEEP_REASONS`. `apply` reports a malformed row
and carries on, and re-applying a file is safe: a drop already in the reject log is not
logged twice, and a keep rewrites its own technique note rather than adding another.

There is no unattended path: `gate.judge` is used only by `distill eval` for calibration.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import date
from pathlib import Path

from sift.config import get_settings
from sift.distill.candidates import GATE_TEXT_CHARS, Candidate, best_copies, url_key
from sift.distill.gate import KEEP_REASONS, Verdict
from sift.distill.technique import build_note, technique_id
from sift.vault.notes import Note, iter_notes, save_note

log = logging.getLogger(__name__)

# Verdict rows must carry these; keeps need the distillation fields as well.
_REQUIRED = ("url", "decision", "already_known", "reason", "justification")
_KEEP_REQUIRED = ("technique_title", "when_to_try", "body_md")

# Keeps are embedded and indexed this many notes at a time: one delete and one add per
# batch instead of per note, since every LanceDB commit rewrites a manifest listing
# every fragment.
_INDEX_BATCH = 200

# JSON allows these raw inside a string, but some readers break lines at them. Escaped
# on export so every reader sees exactly one row per line.
_LINE_BREAKERS = {"\u2028": "\\u2028", "\u2029": "\\u2029", "\u0085": "\\u0085"}


def gated_urls() -> set[str]:
    """`url_key`s of everything already judged - derived from the reject log plus
    existing technique notes, so there's no separate state file to drift out of sync.

    Both sides are keyed: rows and notes store the raw url, and comparing those against
    `collect()`'s normalised key let judged articles back into every export.
    """
    from sift.distill.rejects import load_rejects
    from sift.vault.catalog import fresh_catalog

    seen = {url_key(r.get("url")) for r in load_rejects()}
    vault = get_settings().resolved_vault()
    # The catalog's rows carry each note's url: no technique body is parsed for this.
    for row in fresh_catalog(vault).rows("technique"):
        seen.add(url_key(row.url))
    seen.discard("")
    return seen


def collect(
    note_type: str = "writeup",
    *,
    limit: int | None = None,
    skip_gated: bool = True,
    prefilter: bool = True,
) -> list[Candidate]:
    """Gather ungated candidates from notes already in the vault, one per article.

    With `prefilter`, structurally-obvious drops never reach the gate at all - see
    `distill.prefilter`, which is graded against the hand-judged set.

    Notes without a url are left out: verdicts and the gate state are keyed by url, so
    such a candidate could never be applied or marked judged, and it took a slot in
    every `--limit` batch for good.
    """
    vault = get_settings().resolved_vault()
    seen = gated_urls() if skip_gated else set()
    best, no_url = best_copies(
        iter_notes(vault, note_type=note_type), seen=seen, prefilter=prefilter
    )
    if no_url:
        log.warning(
            "distill: skipped %d %s note(s) with no url - verdicts are matched by url: %s",
            len(no_url),
            note_type,
            ", ".join(repr(c.title[:60]) for c in no_url[:3]) + (" ..." if len(no_url) > 3 else ""),
        )
    out = list(best.values())
    return out[:limit] if limit else out


def _one_line_json(obj: dict) -> str:
    text = json.dumps(obj, ensure_ascii=False)
    for raw, escaped in _LINE_BREAKERS.items():
        text = text.replace(raw, escaped)
    return text


def write_candidates(candidates: list[Candidate], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in candidates:
            row = {
                "title": c.title,
                "url": c.url,
                "source": c.source,
                "created": c.created.isoformat() if c.created else None,
                "note_id": c.source_id,
                "slug": c.source_slug,
                "path": c.path,
                "chars_total": len(c.text),
                "truncated": len(c.text) > GATE_TEXT_CHARS,
                "text": c.gate_text(),
            }
            fh.write(_one_line_json(row) + "\n")
    return len(candidates)


def _read_jsonl(path: Path) -> tuple[list[tuple[int, object, str]], str]:
    """`(line number, value, error)` per non-blank line, and a whole-file error.

    One bad line costs that line, not the file: it comes back with an `error` and the
    rest still parse. Decodes UTF-8 with or without a BOM (PowerShell writes one) and
    BOM-marked UTF-16. Lines end only at \\n, \\r and \\r\\n: a raw U+2028 or U+0085
    copied from an article is legal inside a JSON string, and str.splitlines() used to
    cut the row there.
    """
    raw = path.read_bytes()
    try:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16")
        else:
            text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return [], f"not UTF-8 ({exc}) - save it as UTF-8 and apply again"
    entries: list[tuple[int, object, str]] = []
    for lineno, line in enumerate(io.StringIO(text, newline=None), 1):
        if not line.strip():
            continue
        try:
            entries.append((lineno, json.loads(line), ""))
        except json.JSONDecodeError as exc:
            entries.append(
                (
                    lineno,
                    None,
                    f"bad JSON ({exc.msg} at column {exc.colno}) - an unescaped newline "
                    "inside a string splits a row in two",
                )
            )
    return entries, ""


def load_candidates(path: Path) -> list[Candidate]:
    """Rebuild the candidates an export wrote, so `apply` matches verdicts against
    exactly what was judged.

    Apply used to rescan the vault for `--type writeup`, so a `--type report` export
    applied without repeating `--type` skipped every row as 'no matching candidate'.
    `text` comes back as the gate excerpt only; apply never needs more.
    """
    entries, fatal = _read_jsonl(path)
    if fatal:
        log.warning("distill: cannot read candidates from %s: %s", path.name, fatal)
    out: list[Candidate] = []
    for lineno, row, error in entries:
        if not error and not isinstance(row, dict):
            error = f"row is {type(row).__name__}, not an object"
        if not error:
            try:
                created = row.get("created")
                out.append(
                    Candidate(
                        title=str(row.get("title") or ""),
                        url=str(row.get("url") or ""),
                        text=str(row.get("text") or ""),
                        source=str(row.get("source") or "unknown"),
                        created=date.fromisoformat(created) if created else None,
                        source_id=str(row.get("note_id") or ""),
                        source_slug=str(row.get("slug") or ""),
                        path=str(row.get("path") or ""),
                    )
                )
                continue
            except (TypeError, ValueError) as exc:  # e.g. a malformed `created` date
                error = str(exc)
        log.warning(
            "distill: unreadable candidate row skipped: %s:%d: %s", path.name, lineno, error
        )
    return out


def _validate(row: object) -> str | None:
    """Why a verdict row can't be applied, or None. Normalises a bare-string tags/cwe.

    Types are checked here, before anything is written: a row with `"tags": "xss"` used
    to tag the note 's' and 'x', and a non-string field failed half-way through a batch.
    """
    if not isinstance(row, dict):
        return f"row is {type(row).__name__}, not an object"
    missing = [f for f in _REQUIRED if not row.get(f)]
    if row.get("decision") == "keep":
        missing += [f for f in _KEEP_REQUIRED if not row.get(f)]
    if missing:
        return f"missing {', '.join(missing)}"
    decision = row["decision"]
    if decision not in ("keep", "drop"):
        return f"bad decision {decision!r}"
    for f in (*_REQUIRED, *(_KEEP_REQUIRED if decision == "keep" else ())):
        if not isinstance(row[f], str):
            return f"{f} must be a string"
    if decision == "keep" and row["reason"] not in KEEP_REASONS:
        return f"bad keep reason {row['reason']!r} (want one of {', '.join(KEEP_REASONS)})"
    for f in ("tags", "cwe"):
        value = row.get(f)
        if isinstance(value, str):
            row[f] = [value]
        elif value is not None and not (
            isinstance(value, list) and all(isinstance(x, str) for x in value)
        ):
            return f"{f} must be a list of strings"
    return None


def _pick_technique_id(
    title: str,
    key: str,
    owners: dict[str, set[str]],
    by_url: dict[str, set[str]],
) -> tuple[str, bool, str]:
    """Choose the id a keep is saved under: `(note_id, collided, conflict)`.

    `owners` maps technique note id -> url keys of the notes carrying it, `by_url` the
    reverse; both cover the vault plus everything saved earlier in this apply.

    - This article already has a note under the plain or disambiguated id: reuse it, so
      re-applying a verdict rewrites that note instead of adding another.
    - This article already has a note under a different title: conflict, `note_id` is
      empty - a second technique note for one article is the noise the gate exists to
      prevent.
    - The plain id belongs to a note about another article, or to a hand-written note
      without a url: the url-hashed id, `collided` True. Saving under the plain id
      silently replaced that note and its index rows.
    """
    plain = technique_id(title)
    hashed = technique_id(title, key)
    mine = by_url.get(key, set())
    if plain in mine:
        return plain, False, ""
    if hashed in mine:
        return hashed, False, ""
    if mine:
        return (
            "",
            False,
            f"already distilled as {', '.join(sorted(mine))}; delete that note to re-distill it",
        )
    if not owners.get(plain):
        return plain, False, ""
    if owners.get(hashed):  # a sha1-prefix clash too: refuse rather than overwrite
        return "", False, f"ids {plain} and {hashed} both belong to other notes"
    return hashed, True, ""


def _finish_index(store) -> str | None:
    """Make the new rows keyword-searchable and keep the table compact - once per apply.

    `Store.optimize` compacts fragments, folds new rows into the FTS index and builds
    that index when the table has none. Returns a problem on failure - raised, or
    reported in optimize's `error` field: the notes are committed either way.
    """
    optimize = getattr(store, "optimize", None)  # a stubbed store may have none
    if not callable(optimize):
        return None
    try:
        result = optimize()
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])
    except Exception as exc:  # noqa: BLE001 - maintenance must not fail a finished apply
        return f"index maintenance failed ({exc}); the notes are saved and indexed"
    return None


def apply_verdicts(verdicts_path: Path, candidates: list[Candidate]) -> dict:
    """Write technique notes for keeps, log drops. Returns a summary.

    `candidates` are what the verdicts were judged against: `load_candidates(export)`,
    or `collect(note_type, skip_gated=False)` for a verdicts file whose export is gone.
    Rows are matched to candidates by `url_key`.

    Summary: `kept`, `dropped`, `skipped` (rows not applied) and `chunks_indexed`, plus
    `duplicates` (drops already in the reject log - a re-applied file), `collisions`
    (keeps saved under a url-hashed id because the plain one belongs to another
    article) and `problems` (one line per row not applied, `file:line: ...`; each is
    also logged as a warning on the `sift.distill.manual` logger).
    """
    from sift.distill.rejects import load_rejects, record_reject

    name = verdicts_path.name
    problems: list[str] = []
    skipped = 0

    def report(lineno: int | None, message: str) -> None:
        text = f"{name}:{lineno}: {message}" if lineno else f"{name}: {message}"
        problems.append(text)
        log.warning("distill apply: %s", text)

    entries, fatal = _read_jsonl(verdicts_path)
    if fatal:
        report(None, fatal)
        skipped += 1

    by_key = {k: c for c in candidates if (k := url_key(c.url))}
    vault = get_settings().resolved_vault()

    # Gate state before this apply. Updated as rows land, so a url repeated within one
    # file is treated exactly like one re-applied later.
    dropped_keys = {url_key(r.get("url")) for r in load_rejects()}
    dropped_keys.discard("")
    owners: dict[str, set[str]] = {}
    by_url: dict[str, set[str]] = {}
    from sift.vault.catalog import fresh_catalog

    for row in fresh_catalog(vault).rows("technique"):  # id and url, no body parsed
        key = url_key(row.url)
        owners.setdefault(row.id, set()).add(key)
        if key:
            by_url.setdefault(key, set()).add(row.id)
    shared = sorted(i for i, keys in owners.items() if len(keys) > 1)
    if shared:
        log.warning(
            "distill apply: %d technique id(s) are each carried by notes about different "
            "articles, so only one of each is searchable (not repaired here): %s",
            len(shared),
            ", ".join(shared[:5]),
        )

    kept = dropped = duplicates = collisions = chunks = 0
    # Keyed by note id: a keep repeated within one file rewrites its note, and indexing
    # both copies in one batch would add that note's chunks twice.
    pending: dict[str, Note] = {}
    store = None

    def flush() -> int:
        nonlocal pending, store
        batch, pending = list(pending.values()), {}
        if not batch:
            return 0
        from sift.index.store import Store
        from sift.pipeline import index_notes

        try:
            if store is None:
                store = Store()
            return index_notes(batch, store)
        except Exception as exc:  # noqa: BLE001 - the notes are saved; reindex finishes the job
            report(
                None,
                f"indexing {len(batch)} technique note(s) failed ({exc}); they are saved - "
                "run `sift reindex` to index them",
            )
            return 0

    for lineno, row, error in entries:
        if error:
            report(lineno, error)
            skipped += 1
            continue
        why = _validate(row)
        if why:
            label = (row.get("url") or row.get("title")) if isinstance(row, dict) else None
            report(lineno, f"skipped ({why}): {label or str(row)[:80]}")
            skipped += 1
            continue
        key = url_key(row["url"])
        cand = by_key.get(key)
        if cand is None:
            report(
                lineno,
                f"no candidate matches {row['url']} (apply against the export it was judged from)",
            )
            skipped += 1
            continue

        if row["decision"] == "drop":
            if key in dropped_keys:
                duplicates += 1  # a re-applied drop is already in the log
                continue
            verdict = Verdict(
                decision="drop",
                already_known=row["already_known"],
                reason=row["reason"],
                justification=row["justification"],
            )
            if not record_reject(cand, verdict, gated_by="in-session"):
                report(lineno, f"could not write the reject log for {cand.url}")
                skipped += 1
                continue
            dropped_keys.add(key)
            dropped += 1
            if by_url.get(key):
                report(
                    lineno,
                    f"dropped {cand.url}, which is already distilled as "
                    f"{', '.join(sorted(by_url[key]))} - delete that note if the drop stands",
                )
            continue

        title = row["technique_title"]
        note_id, collided, conflict = _pick_technique_id(title, key, owners, by_url)
        if not note_id:
            report(lineno, f"keep for {cand.url} not applied: {conflict}")
            skipped += 1
            continue
        if collided:
            collisions += 1
            log.warning(
                "distill apply: technique id %s belongs to another article; saved as %s",
                technique_id(title),
                note_id,
            )
        note = build_note(
            cand,
            title=title,
            when_to_try=row["when_to_try"],
            body_md=row["body_md"],
            keep_reason=row["reason"],
            already_known=row["already_known"],
            tags=row.get("tags"),
            cwe=row.get("cwe"),
            note_id=note_id,
        )
        try:
            save_note(vault, note, stamp=True)
        except Exception as exc:  # noqa: BLE001 - one row's failure must not strand the batch
            # Disk errors, or the vault refusing to overwrite a different document that
            # carries this id outside `technique/` (IdConflict). Raising here would also
            # leave every keep saved so far unindexed until the next `sift reindex`.
            report(lineno, f"could not save technique note {title!r}: {exc}")
            skipped += 1
            continue
        owners.setdefault(note_id, set()).add(key)
        by_url.setdefault(key, set()).add(note_id)
        kept += 1
        pending.pop(note_id, None)
        pending[note_id] = note  # the latest version of a repeated keep wins
        if len(pending) >= _INDEX_BATCH:
            chunks += flush()

    chunks += flush()
    if store is not None:
        trouble = _finish_index(store)
        if trouble:
            report(None, trouble)
    return {
        "kept": kept,
        "dropped": dropped,
        "skipped": skipped,
        "chunks_indexed": chunks,
        "duplicates": duplicates,
        "collisions": collisions,
        "problems": problems,
    }
