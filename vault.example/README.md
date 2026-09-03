# vault.example

Sample notes showing the schema. Copy any of these into your real `vault/`
(which is gitignored) as a starting point, or just let `sift ingest ...` fill it.

## Note anatomy

```
vault/<type>/<slug>.md
```

- **Frontmatter** (YAML): `id`, `type`, `title` are required. Optional:
  `source, url, created, ingested, tags[], cwe[], severity, program, assets[], bounty, links[], extra{}`.
- **Body**: markdown. Normalized sections where possible
  (`## Summary`, `## Steps to Reproduce`, `## Impact`, `## Remediation`, `## Notes`).
- **Links**: `[[other-note-slug]]` anywhere in the body, and/or a `links:` list in frontmatter.
  Retrieval walks these 1 hop with `--links` / `expand_links=true`.

## Types

| type | what it holds |
|---|---|
| `report` | a disclosed bug bounty report (public dataset, hacktivity, or your own) |
| `cve` | a CVE entry (KEV, NVD), enriched with CVSS + EPSS |
| `technique` | a reusable methodology / attack pattern |
| `target` | a program: scope, auth setup, recon notes, an idea checklist |
| `finding` | your own session notes / working findings |
| `writeup` | an external blog post / writeup |
