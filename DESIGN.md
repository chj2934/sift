# Bug Bounty AI — Design & Options Analysis

> Draft v0.1 · 2026-09-02 · status: awaiting decisions before build

## 1. What you asked for

A custom AI for bug bounty hunting that can:

1. Act as an **interactive copilot** (reason about targets, suggest attack angles, explain findings, draft reports)
2. Act as an **autonomous agent with tools** (run recon/scanning, chain steps, triage output)
3. Have a **vector-based memory like Obsidian** (ingest disclosed reports, CVEs, writeups, your own notes; recall relevant knowledge on demand; add new knowledge over time)
4. Be **fine-tuned** on bug-bounty domain data

## 2. Your constraints

| Resource | Value | Implication |
|---|---|---|
| GPU | RTX 3070, **8 GB VRAM** | Can run quantized 7–8B models; can QLoRA-fine-tune ≤8B overnight. Cannot run/fine-tune 30B+ locally. |
| CPU | Ryzen 5 5600X (6c/12t) | Fine for embeddings, ingestion, orchestration. |
| RAM | 64 GB DDR4-3200 | Big win. Can hold large vector indexes in memory, run 70B GGUF on CPU (but slow, ~1 tok/s). |
| OS | Windows 11 | ML + security tooling is smoother on Linux → **WSL2 strongly recommended**. |
| Budget | $20–100/mo | Enough for frontier API as the "brain" + free local models for volume work. |
| Involvement | Build most of it together | Phased repo, you review and run each phase. |

## 3. The core misconception to clear up first

**Fine-tuning ≠ giving the model knowledge.**

- **Fine-tuning (LoRA/QLoRA)** changes *behavior, style, and format* — it teaches the model to "sound like" a bug bounty analyst, follow your report template, classify vuln types, know what to test given recon output. It is a poor and unreliable way to memorize facts (specific CVEs, specific technique details). Facts fade, hallucinate, or go stale.
- **RAG / vector memory** is how you inject *facts* — disclosed reports, CVE details, writeups. Retrieved fresh at query time, always current, cite-able, editable.

So the "train on disclosed reports and CVEs" goal is **mostly a RAG job, not a fine-tuning job.** Fine-tuning comes later and is optional — it polishes the local model's instincts and writing once you have a corpus and usage logs to build a training set from.

This reframes the priority order: **memory first, copilot second, agent third, fine-tune last.**

## 4. Component options & trade-offs

### 4.1 Reasoning engine (the "brain")

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Frontier API (Claude)** | Best reasoning for vuln analysis / code review / chaining; zero VRAM; cheap at low volume (~$0.01–0.10 per complex query) | Recurring cost; data leaves machine (fine for public data + your own targets you're authorized on; be careful with third-party client data) | **Primary brain** |
| **Local 8B (Ollama, Qwen3-8B / Llama 3.x)** | Free; private; offline | Noticeably weaker at multi-step security reasoning; 8 GB caps context + speed | **Secondary** — bulk/cheap work |
| **Hybrid (recommended)** | API does the hard thinking; local model does the volume (summarizing reports for ingestion, tagging, embeddings, draft text, scanner-output triage) → keeps monthly cost near the low end | More plumbing | ✅ **This** |
| **Local 70B on CPU (64 GB RAM, llama.cpp)** | Runs; fully private | ~0.5–2 tok/s — too slow for interactive use | Experiment only |

Every state-of-the-art autonomous bug-bounty system (XBOW, etc.) runs on frontier models. An 8 GB local-only build will feel weak. The hybrid is the sweet spot for your budget.

### 4.2 Memory store (vector DB)

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **LanceDB** | Embedded (no server), on-disk, scales to millions of vectors, native hybrid (vector + full-text) search, Windows-friendly, versioned tables | Newer, smaller community | ✅ **Recommended** — CVE + report corpus gets large |
| ChromaDB | Easiest to start, huge ecosystem | Heavier; less comfortable at large scale | Fine alternative if you want max simplicity |
| Qdrant | Best filtering + scale, sparse+dense vectors | Needs a running server (Docker) | Overkill for single-user local |
| pgvector | SQL, familiar, transactional | Need Postgres; slower at scale without index tuning | Only if you already run Postgres |
| FAISS (raw) | Fastest raw ANN | No persistence/metadata/hybrid — you build all of it | No |

### 4.3 Knowledge format (the "Obsidian-like" layer)

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Obsidian vault: markdown + YAML frontmatter + `[[wikilinks]]`** | Human-readable; you browse/edit/graph it *in Obsidian*; portable; git-versionable; frontmatter carries metadata (source, CVE id, vuln class, target, severity, date) | Need a parser/ingest step | ✅ **Recommended** |
| SQLite / JSON documents | Structured queries | Not human-friendly; no Obsidian graph | No |
| Vector chunks only (no source docs) | Simple | Loses provenance, can't edit/curate, no linking | No |

**How the "memory like Obsidian" actually works:**

```
Your vault (markdown notes, one per report/CVE/technique/target)
        │  ingest: parse frontmatter + [[links]], chunk body
        ▼
Embeddings (local: bge-m3)  ──►  LanceDB (vector + BM25 keyword index)
        │
Query time:
  1. hybrid search (semantic + exact-token, critical for CVE-IDs / header names / params)
  2. rerank top-k with a small cross-encoder (bge-reranker-v2-m3)
  3. graph expansion: pull notes linked via [[wikilinks]] from the hits  ← the "pull connected knowledge" part
  4. stuff into the brain's context with citations
Write-back:
  "remember this" → brain drafts a new note (frontmatter + links) → saved to vault → re-indexed
```

This is GraphRAG-lite: vector retrieval + link-graph walk. You get Obsidian's linking/graph *and* semantic recall.

### 4.4 Embeddings

| Option | Pros | Cons |
|---|---|---|
| **Local: bge-large-en-v1.5** (implemented) | Free, private, 1024-dim, strong English retrieval; GPU via torch bundle (only needs an NVIDIA driver) | ~1.5 GB VRAM during ingest — free in Phase 1 since no local LLM yet |
| API: Voyage / OpenAI embeddings | Slightly better quality, no VRAM | Tiny recurring cost, data leaves machine |

> Note: `bge-m3` isn't in `fastembed`'s model list, so Phase 1 ships `bge-large-en-v1.5`.
> On CPU, bulk ingest is slow (~20 docs/s) — use `bge-base`/`bge-small` or the GPU extra.

### 4.5 Fine-tuning (Phase 3, optional)

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **QLoRA on ≤8B via Unsloth**, on your 3070 | Fits ~6 GB VRAM for an 8B model; overnight run; teaches report style, vuln classification, "what to test" instincts | Doesn't add reliable facts; needs a good 1–5k-example dataset | ✅ **When ready** |
| Cloud QLoRA on 14–32B (RunPod, ~$1–2/hr) | Bigger, smarter base | Costs per run; data handling off-machine | If the 8B result isn't good enough |
| Full fine-tune | Deeper adaptation | Needs 40 GB+ VRAM | No |
| Skip fine-tuning, RAG only | Zero training risk; fastest to value | Local model stays generic in tone | Valid long-term choice |

**Prior art:** `BugTraceAI-CORE-Ultra-27B` on Hugging Face is a community model already SFT'd on ~2,500 disclosed reports + CVE writeups. Worth evaluating as a local model and/or a template for our own dataset. Dataset `Hacker0x01/hackerone_disclosed_reports` on HF is a ready ingest source.

### 4.6 Agent framework (Phase 2)

| Option | Pros | Cons |
|---|---|---|
| **Claude Agent SDK** | Solid tool loop, MCP-native, maintained, good with our RAG-as-MCP-server | Reasoning tied to Claude API |
| Custom Python agent loop | Full control, model-agnostic | You maintain it |
| Existing open agents (PentestGPT, Strix, etc.) | Prebuilt security prompting/flows | Opinionated; integration work; still need your memory bolted on |

## 5. Recommended architecture — phased

### Phase 1 — Knowledge base + copilot (the foundation) ⭐ start here
- Obsidian vault schema (frontmatter fields, note types, link conventions)
- Ingestion pipeline with pluggable fetchers:
  - HackerOne disclosed reports (HF dataset + Hacktivity)
  - CVE / NVD bulk feeds (CVE.org JSON, NVD 2.0 API)
  - Writeup collections (PentesterLand, GitHub "awesome" lists, PortSwigger research) — respecting robots/ToS
  - Your own notes (dropped into the vault)
  - Normalizer → clean markdown notes with metadata + auto-suggested `[[links]]`
- Index: local embeddings → LanceDB with hybrid search + reranker
- Query interface: a CLI/TUI chat that retrieves from the vault and calls Claude for reasoning; supports "remember this" write-back
- **Ship it as an MCP server** so Claude Code / Claude Desktop / Cursor can query the memory directly

### Phase 2 — Tool-augmented agent
- Recon tool wrappers (subfinder, httpx, naabu, katana, nuclei) behind a **scope guard** (allowlist of authorized hosts, rate limits, kill switch)
- Flow: recon → local-8B triage of noisy output → RAG pulls comparable past findings → Claude forms exploit hypotheses → **human approval gate** before any active/intrusive test
- Findings logged back into the vault as new notes

### Phase 3 — Fine-tune the local model
- Build SFT dataset from your corpus + Phase 1–2 usage logs (report-writing pairs, recon→test-plan pairs, vuln-classification)
- QLoRA on Qwen3-8B via Unsloth on the 3070, export GGUF, serve via Ollama
- Slot it in as the Phase 2 triage/draft model; re-evaluate vs. API for more tasks

## 6. Legal / ethical guardrails (built in from day one)
- Only operate against targets you're **explicitly authorized** for (active bug bounty program scope, your own lab, sanctioned CTFs)
- Scope allowlist enforced in code; out-of-scope hosts refused
- Rate limiting + identifiable traffic + respect program rules
- Active/intrusive actions require human confirmation in Phase 2
- Public data ingestion respects site ToS and robots.txt
- Not built for: mass/untargeted scanning, evasion, exfiltration

## 7. Open decisions (need your input before build)

1. **Authorization context** — enrolled bug bounty programs? CTFs? personal lab? (Shapes Phase 2; guardrails go in regardless.)
2. **The vault** — do you already use Obsidian? Should the AI's memory *be* your existing vault, or a separate vault it owns? Where on disk?
3. **Primary interface** — CLI/TUI chat, a local web UI, or mainly an MCP server you plug into Claude Code / Desktop?
4. **First data sources** — default is HackerOne disclosed + NVD CVEs + your notes. Add Bugcrowd/YesWeHack, PortSwigger research, Exploit-DB, specific writeup lists?
5. **Offline requirement** — does any part need to work with no API access? (Raises local-model priority.)
6. **WSL2** — OK to work inside WSL2 Ubuntu? (Strongly recommended for the tooling.)
7. **Language/stack** — default is Python 3.11+, `uv` for env management. Any preference against?

## 8. Rough cost picture
- Phase 1 running cost: ~$5–30/mo API (retrieval-augmented queries are cheap; embeddings are local/free)
- Phase 2: +$10–40/mo depending on agent run frequency
- Phase 3 training: $0 (local) or ~$5–20 one-off if we rent a bigger GPU
- Well within your $20–100/mo band
