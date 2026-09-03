# sift

[![CI](https://github.com/chj2934/sift/actions/workflows/ci.yml/badge.svg)](https://github.com/chj2934/sift/actions/workflows/ci.yml)

Obsidian-like vector memory for bug bounty hunting. A personal knowledge base of
disclosed reports, CVEs, attack techniques, target notes and your own findings —
stored as plain markdown, searchable semantically, and exposed to Claude Code as
MCP tools.

> **Phase 1** of the plan in [`DESIGN.md`](DESIGN.md): the memory + copilot layer.
> Phases 2–3 (recon agent, optional local fine-tune) come later.

## What it is

- **Vault**: `vault/<type>/<slug>.md` — YAML frontmatter + markdown body + `[[wikilinks]]`.
  Open it in Obsidian if you want the graph view; nothing here requires Obsidian.
- **Index**: LanceDB, hybrid search (dense embeddings ∥ BM25 keyword), RRF-fused,
  optional cross-encoder rerank, optional 1-hop link-graph expansion.
- **Interface**: an MCP server (`sift mcp`) so Claude Code *is* the copilot, plus a CLI.

Embeddings run locally (fastembed/ONNX on CPU, or torch on GPU). No data leaves your
machine except the ingest fetches (public datasets + your own HackerOne API calls).

```
 sources                       vault (markdown)              index & retrieval
 ───────                       ────────────────              ─────────────────
 CISA KEV        ┐                                          ┌ dense vector (BGE)
 NVD CVEs        │   normalize   vault/cve/*.md    chunk +   │ BM25 keyword (FTS)
 EPSS scores     ├──────────────▶ vault/report/*.md ────────▶├ RRF fusion
 HackerOne (pub) │               vault/technique/*.md  embed │ + cross-encoder rerank
 HackerOne (you) │               vault/target/*.md          │ + [[wikilink]] graph walk
 your notes      ┘               [[wikilinks]] between them  └▶ cited results
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
uv run sift ingest notes                  # index anything you hand-wrote into vault/
uv run sift status
```

First run downloads the embedding model (bge-base-en-v1.5, ~0.2 GB).

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

Then in a Claude Code session, the tools `search_memory`, `get_note`, `remember`,
`list_notes`, and `stats` are available. Ask things like *"what SVG upload XSS
bypasses are in my memory?"* or *"remember this technique: …"*.

## Config (`.env`)

| var | default | meaning |
|---|---|---|
| `SIFT_VAULT_PATH` | `./vault` | where notes live |
| `SIFT_DB_PATH` | `./data/lancedb` | LanceDB directory |
| `SIFT_EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | embedding model (`bge-large` = better, GPU-recommended) |
| `SIFT_EMBED_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `SIFT_RERANK` | `false` | enable cross-encoder reranking |
| `H1_API_USERNAME` / `H1_API_TOKEN` | – | HackerOne API ([token](https://hackerone.com/settings/api_token/edit)) |
| `NVD_API_KEY` | – | raises NVD rate limit ([request](https://nvd.nist.gov/developers/request-an-api-key)) |

## Tests

```bash
uv run pytest        # offline; uses a fake embedder for the search test
```

## Safety

Only operate against targets you are explicitly authorized to test (an active bug
bounty program's scope, your own lab, sanctioned CTFs). This tool stores and
retrieves knowledge; it does not send traffic to targets.

## License

MIT — see [LICENSE](LICENSE).
