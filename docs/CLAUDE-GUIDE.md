## SIFT — your long-term memory

sift is a local hybrid index (semantic vectors + BM25 keywords, fused by rank) over a
markdown vault, served to you as MCP tools. It holds what your training does not: the
operator's own findings, hypotheses and dead ends, per-program target and tool notes,
research published after your training cutoff, and vendor reference material pinned to a
revision. It also carries a bulk corpus of CVEs and disclosed reports — useful for
specifics, but weigh it below the operator's own notes.

**Search before you form a hypothesis, and record as you work.** It is fast: query it
whenever a past result could change what you do next.

### Tools

| tool | use it to |
|---|---|
| `search_memory` | find notes. Filters: `program`, `type`, `cwe`, `min_quality`; `expand_links` follows `[[links]]` one hop |
| `get_note` | read one note in full, or one `section` of it. **A search snippet is not a read** |
| `list_notes` | browse metadata by `program`, `type`, `status`, `tag`, `source`, `since` |
| `capture_idea` | record a hypothesis **the moment it forms, before testing it** |
| `resolve_idea` | record the outcome: `worked` / `partial` / `failed` |
| `remember` | save durable knowledge: a finding, a technique, a negative result |
| `update_note` | correct or extend an existing note in place |
| `forget_note` | remove a note that is wrong (soft delete) |
| `capture_url` | keep one post-cutoff article verbatim |
| `stats` | vault size, index health, last ingest per source |

### Searching

1. **Send several phrasings in one call.** `queries` takes 2–4 strings that are searched
   together and fused into one ranked list. Cover different angles:
   - **the exact identifiers** — function, contract, endpoint, parameter, header, CVE id,
     error string. The keyword side matches these literally;
   - **a plain-language description** of the behaviour or bug class;
   - **optionally, a sentence written as the note you hope exists.** It matches stored
     text better than a question does.

   ```
   search_memory(queries=[
       "redirect_uri",
       "OAuth redirect URI validation bypass",
       "the authorization server accepts a redirect_uri on an attacker-controlled subdomain",
   ], k=10)
   ```

   `query="..."` (one string) still works; use it for a single exact lookup.
2. **Read `matched_queries`.** With several queries, each hit lists the phrasings that
   would have returned it on their own (indices into `queries`). A note listed under
   several is the strongest match. One listed only under the identifier is an exact-term
   hit — check it is about the same thing. An empty list means only the combination
   surfaced it.
3. **Filter when you know the scope.** `program` is case-insensitive — also search once
   without it, because notes may sit under a sibling or vendor name. `cwe` is exact
   (`"CWE-79"`). `type` separates the operator's own material (`finding`, `target`,
   `tool`, `technique`) from bulk corpora (`cve`, `report`) and research (`writeup`,
   `reference`). `min_quality=70` drops thin bulk notes.
4. **Size the search to the job.** `k` defaults to 8. For a program sweep use `k=20` and
   `expand_links=true`.
5. **Read before you rely.** Open every note you will act on with `get_note`, by its
   `note_id` (`section=` reads one heading). Cite note ids when a past result shapes a
   decision.
6. **An empty result is not an answer yet.** Rephrase with identifiers, drop the filters,
   then conclude "not in memory" — and say that you did.

### Recording

- **Ideas have a lifecycle.** `capture_idea` before testing, `resolve_idea` after. A
  `failed` outcome is a result: "tried X, blocked by Y" stops the next session repeating
  it. `list_notes(status="hypothesis")` lists ideas that were never resolved.
- **What earns a note:** the operator's own results, post-cutoff research, and specifics
  you would otherwise get wrong — exact versions, line numbers, payloads, program rulings.
  Not general knowledge you already have; that only dilutes search.
- **Write it to be found.** Put the program, component and exact identifiers in the title
  and body; set `program`, `cwe` and `tags`; link related notes with `[[note_id]]`.
- **One note per thing.** `remember` never forks a note: the same title (same type and
  program) appends to the existing note, and `note_id=` appends to a given one. To
  rewrite, retitle or retag, use `update_note` — `append_md` for new evidence, `body_md`
  to replace. Ingested notes (CVEs, reports) need `force=True`, and a re-ingest may
  overwrite the change; prefer a linked note of your own.
- **Wrong is worse than missing.** When a note is disproved, `update_note` it to say so,
  or `forget_note` it with a `reason` — the file moves to `.trash/` and is tombstoned so
  an ingest cannot bring it back. Never forget a note just to tidy up.
- **`capture_url`** keeps one fresh article verbatim. It refuses articles published before
  your training cutoff (you already know them) unless `force=True`.

### Good to know

- Each Claude Code session starts its own sift server; the first search can take about a
  second while the embedding model warms up.
- If a sift call errors, say so and stop relying on memory for that question. Never
  continue as if it had returned nothing.
- The operator browses the same vault in Obsidian. Write through the tools, so the index
  stays in step with the files.
