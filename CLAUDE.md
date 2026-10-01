# sift

Obsidian-like vector memory for bug bounty hunting: a markdown vault + LanceDB hybrid
search, exposed to Claude Code as an MCP server.

## Running things

`sift` is not on PATH. Use `uv run --no-sync sift ...`, or activate `.venv\Scripts\Activate.ps1`.

```
uv run --no-sync pytest              # 110 tests, offline
uv run --no-sync ruff check src tests
```

Live novelty-gate calibration costs a few cents and is opt-in:

```
uv run --no-sync pytest tests/test_gate_calibration.py -m calibration
```

## Working a bug bounty target

**Search sift before answering questions about a target, a technique, or a past
finding.** `search_memory` covers disclosed reports, CVEs, distilled techniques, and
the user's own notes. `search_memory(min_quality=70)` filters to high-quality hits.

**When you form a novel testing idea, call `capture_idea` before testing it.** Not for
routine methodology — for the specialized, non-obvious hypotheses that are worth
remembering. Then call `resolve_idea` with the outcome.

**Record failures as carefully as successes.** "Tried JWT alg confusion, RS256
validated properly" is exactly the note that stops a future session re-testing a dead
end. A `failed` outcome is a result, not a non-result.

`list_notes(status="hypothesis")` surfaces ideas that were never followed up.

**Each program gets a `target` note and a `tool` note.** The target note holds scope,
known issues and attack directions; the tool note holds the setup — clone path, package
manager, chains and funding, wallet requirements, submission gotchas. Write the tool
note when you set the program up, not later; it is the cheapest note in the vault and
the one that saves the most time on return.

## The novelty gate

`sift distill` exists to keep the vault small. Its rule: **store only what the
reasoning model does not already know.** A technique note explaining something Claude
can already explain is worse than useless — it costs retrieval tokens and dilutes
ranking, pushing genuinely rare material down the results.

Three things earn a place: post-cutoff research, obscure variants whose specifics the
model would fumble, and the user's own per-target results.

If you change `GATE_SYSTEM` in `src/sift/distill/gate.py`, re-run the calibration
tests. A silent gate regression turns the vault to noise.

### The same rule, applied at ingest: `SIFT_MODEL_CUTOFF`

The gate judges one candidate at a time and costs an API call. For a source whose
material is *dated*, the cheap version of the same judgement is the publication date:
anything older than the cutoff is, by construction, something the model trained on.
Every freshness source defaults its horizon to `get_settings().model_cutoff` and takes
`--since` to override — a new dated source should do the same rather than inventing
its own default.

This is not a size optimisation, it is a precision one. The Chromium corpus is the
worked example: `rule-of-2.md` has read the same way for years, so storing it back
teaches nothing and pushes rare material down the ranking. What *is* novel is the
diff — a paragraph added to `severity-guidelines.md` in July 2026 changes what is
filable. Hence `ingest chromium-docs` selects on "did the vendor touch this file since
the cutoff" and leads each note with that file's post-cutoff commit list.

**Do not add a deep-backfill default to a Google/Chromium source.** Measured: the
whole Project Zero archive is 240 posts and the newest is 2025-12-12, i.e. entirely
pre-cutoff — a P0 backfill source would add 240 notes of pure training data, which is
why there isn't one. `research` already carries the P0 feed for whatever they publish
next.

## Ingesting a target's vendor corpus

Three sources cover Google (`sift ingest google` runs all three). They write
`reference` notes — primary material quoted verbatim, never distilled, always stamped
with the revision or release it came from, because the point is to be citable back at
triage.

- `chromium-docs` — in-tree security/IPC docs, selected by post-cutoff change. Each
  note carries a **section index with line numbers at the pinned SHA**, because the
  citation that settles an argument is `faq.md:729`, not "the FAQ".
- `chromium-fixes` — security-fix commits. Filtered on the **subject line only**:
  matching bodies too was measured at 4,430 hits vs 1,174 over the same window, for
  much worse precision. Each note ends by saying the patched site is worthless and the
  *set of other sites relying on the same invariant* is the lead.
- `chrome-releases` — per-release security tables plus one rolling **reward ledger**
  (component × bug class × payout). `TBD` and `N/A` rewards are excluded from every
  median rather than counted as zero, and the ledger states its own window: a median
  over four months and one over ten years are different claims.

Both Chromium sources need `SIFT_CHROMIUM_SRC` and shell out to `git`. Neither raises
on a git failure — a missing checkout labels the notes `unknown` instead, because a
half-ingested corpus that says it succeeded is the failure mode that matters here.

## Conventions

- Settings are pydantic-settings fields with explicit `alias=` (no env prefix); add
  new ones to `.env.example` too. Everything reads `get_settings()`.
- CLI: heavy imports go *inside* command functions to keep startup fast. `cli.py`
  reconfigures stdout to UTF-8 before importing typer (Windows cp1252), hence its
  `E402` exemption.
- Tests import `sift.*` **inside** test bodies — the autouse `_isolated_env` fixture
  must set env vars before settings load. Hence the `E402` exemption for `tests/*`.
- Ingest sources are plain callables returning `Iterator[Note]`, run via
  `ingest.base.run_source`, which handles saving, batched indexing and `_state.json`.
- Keep parsing in pure underscore-private helpers, separate from I/O, so tests can
  run against fixture strings with no network.

## Ingestion: things that silently corrupt the corpus

Every rule below exists because a real note in the vault was wrong. Do not relax one
without re-measuring — several plausible-looking heuristics were tried and would have
destroyed good research.

- **Compression.** `brotli` and `zstandard` are hard dependencies. Without them httpx
  cannot decode `br`/`zstd`, and brotli-serving sites become undecoded binary stored
  as note text. This silently corrupted 2% of the corpus.
- **Extraction.** `extract_article` uses trafilatura (regex fallback). The regex alone
  left every navigation menu in the note body.
- **JS-rendered pages.** `extraction_failed` rejects a body under 2% of the raw HTML —
  real articles measure 0.062–0.374, failures 0.002–0.009. Nothing server-side fixes a
  JS-rendered page; trafilatura was tried and did no better.
- **Medium.** 403s every non-browser client, and that is 42% of PentesterLand losses.
  `ingest/medium.py` goes through Medium's own RSS feeds instead. **Do not work around
  the 403 by spoofing a browser.** Feeds only carry ~10 recent posts per author, so
  this fixes ongoing ingestion, not backfill.
- **Never delete notes on a title heuristic.** Judge the body. A title-based cleanup
  would have deleted "can I speak to your manager? hacking root EPP servers" (real,
  just lowercase), and a "needs 3 long lines" rule would have deleted 174 legitimate
  Assetnote advisories.
- **`looks_like_prose` is script-agnostic on purpose.** An ASCII-word version scored a
  Japanese writeup at 0.167 and would have discarded it.
- **Slug collisions.** Ids truncate to 80 chars, so similar long titles collide;
  `run_source` disambiguates and reports `collisions`. It used to overwrite silently
  and still report success.

## Gotchas

- **Schema changes to `ChunkRow` require `sift reindex --force`.**
- Never Ctrl-C `uv sync` mid-run — it has corrupted the venv before.
- `onnxruntime` is pinned `<1.29` (severe CPU inference regression in 1.29.x).
- Benign `WARN ... latest_version_hint.json: Access is denied` during indexing is a
  Windows temp-file race in LanceDB; writes land.
- `git push` / `gh` are classifier-blocked from Claude Code — the user pushes.
