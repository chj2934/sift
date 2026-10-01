# sift

[![CI](https://github.com/chj2934/sift/actions/workflows/ci.yml/badge.svg)](https://github.com/chj2934/sift/actions/workflows/ci.yml)

Obsidian-like vector memory for bug bounty hunting. A personal knowledge base of
disclosed reports, CVEs, attack techniques, target notes and your own findings —
stored as plain markdown, searchable semantically, and exposed to Claude Code as
MCP tools.

> **Phase 1** of the plan in [`DESIGN.md`](DESIGN.md): the memory + copilot layer.
> Phases 2–3 (recon agent, optional local fine-tune) come later.

## What it is

- **Vault**: `vault/<type>/<Title>.md` — YAML frontmatter + markdown body + `[[wikilinks]]`.
  Open it in Obsidian if you want the graph view; nothing here requires Obsidian. A note
  is identified by its frontmatter `id`, so renaming or moving the file is fine.
- **Index**: LanceDB, hybrid search (dense embeddings ∥ BM25 keyword), RRF-fused,
  optional cross-encoder rerank, optional 1-hop link-graph expansion.
- **Interface**: an MCP server (`sift mcp`) so Claude Code *is* the copilot, plus a CLI.

Embeddings run locally (fastembed/ONNX on CPU, or torch on GPU). No data leaves your
machine except the ingest fetches (public datasets, your own HackerOne API calls, and
any URL you ask `ingest url` / `capture_url` to capture) and `sift distill eval`, which
calls the Anthropic API. Apart from first-run model downloads, the MCP server only goes
online for `capture_url` (FastMCP's PyPI update check is off).

```
 sources                       vault (markdown)              index & retrieval
 ───────                       ────────────────              ─────────────────
 CISA KEV        ┐                                          ┌ dense vector (BGE)
 NVD CVEs        │              vault/cve/*.md              │ BM25 keyword (FTS)
 EPSS scores     │   normalize  vault/report/*.md           │ RRF fusion
 HackerOne (pub) ├─────────────▶ vault/technique/*.md ─────▶├ cross-encoder rerank
 HackerOne (you) │              vault/target/*.md    embed  │ [[wikilink]] graph walk
 Chromium src    │              vault/reference/*.md        │
 Chrome releases │                                          └▶ cited results
 your notes      ┘              [[wikilinks]] between them
                                          │                          │
                                          └──────────── sift CLI  /  MCP tools ──┘
                                                        (search_memory, remember, …)
```

### Example

```
$ sift search "unauthenticated file upload leads to webshell" -k 3

1. CVE-2024-55956 — Cleo Multiple Products Unauthenticated File Upload   cve  @Cleo
   https://nvd.nist.gov/vuln/detail/CVE-2024-55956
   Cleo Harmony / VLTrader / LexiCom allow unauthenticated file upload to
   autorun directories, enabling remote code execution (exploited by Cl0p).

2. CVE-2024-57968 — Advantive VeraCore Unrestricted File Upload            cve  @Advantive
3. CVE-2026-56291 — Balbooa Forms Unrestricted Upload → RCE               cve  @Balbooa
```

## Setup

```bash
uv sync                          # CPU-only (fastembed / ONNX)
uv sync --extra gpu              # + GPU embeddings (torch, bundles CUDA — needs only an NVIDIA driver)
uv run sift init                 # create vault/ + .env
# edit .env — add H1_API_USERNAME / H1_API_TOKEN (and optionally NVD_API_KEY)
```

Default model is **`bge-base-en-v1.5`** (768-dim) — fast enough on CPU. Embeddings
use the **GPU** automatically when `--extra gpu` is installed (`SIFT_EMBED_DEVICE=auto`);
in Phase 1 nothing else uses VRAM. GPU users can bump quality with
`SIFT_EMBED_MODEL=BAAI/bge-large-en-v1.5` (then `sift reindex --force`).

## Populate the memory

```bash
uv run sift ingest kev                    # CISA known-exploited CVEs (~1.3k, fast)
uv run sift ingest nvd --since 2024       # recent CVEs, web-app CWEs (slow without NVD key)
uv run sift ingest epss                   # add exploit-prediction scores to CVE notes
uv run sift ingest h1-public --limit 2000 # public HackerOne disclosed reports
uv run sift ingest h1-mine --hacktivity   # YOUR reports + the public hacktivity feed
uv run sift ingest research               # research feeds, newer than SIFT_MODEL_CUTOFF
uv run sift ingest url https://...        # one fresh article, verbatim
uv run sift ingest notes                  # index anything you hand-wrote into vault/
uv run sift status
```

First run downloads the embedding model (bge-base-en-v1.5, ~0.2 GB).

Re-running a source skips what is already stored (`--refresh` re-fetches), never
duplicates a note, and doesn't bring back what `sift prune` removed. KEV and NVD merge
into one note per CVE.

### Target-specific: Google / Chromium

```bash
uv run sift ingest google                 # all three of the below, one horizon
uv run sift ingest chromium-docs          # in-tree security docs the vendor CHANGED since the cutoff
uv run sift ingest chromium-fixes         # security-fix commits — the mechanism corpus
uv run sift ingest chrome-releases        # release security tables + the VRP reward ledger
```

The first two read a local Chromium checkout (`SIFT_CHROMIUM_SRC`) and stamp its HEAD
sha on every note, so a quote stays traceable to the revision it shipped with.

All three default their horizon to **`SIFT_MODEL_CUTOFF`**, because this corpus is
unusually well represented in training data: `rule-of-2.md` has read the same way for
years, and storing it back costs retrieval tokens to tell the reasoning model
something it can already recite. What is *not* training data is the change — a
paragraph added to `severity-guidelines.md` in July 2026 moves what is filable, and a
fix commit from last week names an invariant that was not being enforced. `sift ingest
chromium-docs --all` overrides this when you want the stable policy text verbatim,
with line anchors, to quote at triage.

## Search

```bash
uv run sift search "IDOR in GraphQL node id" -k 5 --links
uv run sift search "cache poisoning" --type report --program "Example Corp"
```

## Use from Claude Code

The repo ships `.mcp.json`. From this folder:

```bash
claude mcp list                  # should show "sift"
```

Then in a Claude Code session these tools are available. Ask things like *"what SVG
upload XSS bypasses are in my memory?"* or *"remember this technique: …"*.

| tool | what it does |
|---|---|
| `search_memory` | hybrid search, with type / CWE / program / quality filters and optional link expansion |
| `get_note` | one note by id, slug or path (optionally one section) |
| `list_notes` | browse by type, program, status, tag, source or date |
| `stats` | note counts, index size, last ingests, unreadable files |
| `remember` | save a finding, technique or target note; repeats append to the same note |
| `update_note` | correct a note in place (title, body, tags, …) |
| `capture_idea` | record a testing hypothesis *before* testing it |
| `resolve_idea` | record how it went: worked, failed or partial (a failure is a result) |
| `forget_note` | soft-delete into `vault/.trash` (undo: `sift trash restore <id>`) |
| `capture_url` | store one fresh article verbatim (public hosts only, post-cutoff unless forced) |

## Maintenance

```bash
uv run sift status               # counts, index health, last ingest runs
uv run sift doctor               # read-only: duplicate ids, unreadable notes, stale index rows
uv run sift reindex              # incremental; --force after changing the model
uv run sift compact              # compact the index and reclaim old versions' disk
uv run sift prune                # dry run; --yes quarantines to data/pruned/, --restore undoes
uv run sift trash list           # notes soft-deleted by forget_note
```

## Config (`.env`)

Every setting, with its default and a comment, is in [`.env.example`](.env.example)
(`sift init` copies it to `.env`; a blank value means the default). The ones you are
most likely to touch:

| var | meaning |
|---|---|
| `SIFT_VAULT_PATH` / `SIFT_DB_PATH` | where notes and the LanceDB index live |
| `SIFT_EMBED_MODEL` / `SIFT_EMBED_DEVICE` | embedding model (`bge-large` = better, GPU-recommended) and device |
| `SIFT_QUERY_DEVICE` | `cpu` = fast, VRAM-free query embeddings for the MCP server |
| `SIFT_MCP_WARMUP` / `SIFT_MCP_AUTO_SYNC` | MCP model warm-up and background index sync |
| `SIFT_MODEL_CUTOFF` | default horizon for every freshness source — older material is what the model already knows |
| `SIFT_CHROMIUM_SRC` | local Chromium `src` checkout, for `ingest chromium-docs` / `chromium-fixes` |
| `H1_API_USERNAME` / `H1_API_TOKEN` | HackerOne API ([token](https://hackerone.com/settings/api_token/edit)) |
| `NVD_API_KEY` | raises NVD rate limit ([request](https://nvd.nist.gov/developers/request-an-api-key)) |

## Tests

```bash
uv run python -m pytest          # offline: a fake embedder, temp vaults, no credentials
uv run ruff check src tests
uv run ruff format --check src tests
```

## Safety

Only operate against targets you are explicitly authorized to test (an active bug
bounty program's scope, your own lab, sanctioned CTFs). This tool stores and
retrieves knowledge; it does not send traffic to targets.

## License

MIT — see [LICENSE](LICENSE).
