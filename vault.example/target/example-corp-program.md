---
id: target-example-corp
type: target
title: "Example Corp — bug bounty program notes"
source: manual
program: Example Corp
url: https://hackerone.com/example-corp
assets:
  - "*.example.com"
  - api.example.com
  - app.example.com
tags:
  - target
  - active
---

## Scope

- In scope: `*.example.com`, mobile apps, `api.example.com`.
- Out of scope: `blog.example.com` (WordPress, managed), `status.example.com`.
- No automated scanning above 5 req/s. No social engineering. No DoS.

## Auth / setup

- Test accounts: request via program; use `+bbp` gmail aliases.
- Two roles matter: `member` and `org_admin`.

## Recon notes

- GraphQL at `api.example.com/graphql`, introspection enabled in staging only.
- Legacy REST at `api.example.com/v1/` — inconsistent authz (see [[idor-graphql-node-id-enumeration]]).

## Ideas to try

- [ ] Batch `nodes(ids:)` authorization on `api.example.com/graphql`
- [ ] SVG upload -> stored XSS (see [[example-h1-000000]])
- [ ] JWT `alg` confusion on `app.example.com` session tokens
