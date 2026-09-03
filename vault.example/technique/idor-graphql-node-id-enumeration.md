---
id: idor-graphql-node-id-enumeration
type: technique
title: IDOR via GraphQL global node IDs
source: manual
tags:
  - idor
  - graphql
  - authorization
cwe:
  - CWE-639
links:
  - graphql-introspection
---

## Idea

Relay-style GraphQL APIs expose a global `node(id: ID!)` field. IDs are often
base64 of `Type:integer` (e.g. `VXNlcjoxMjM` -> `User:123`). If object-level
authorization is enforced only in the top-level resolvers and not in `node`,
you can read arbitrary objects by decrementing/incrementing the integer.

## Steps to reproduce

1. Capture a legitimate `node(id: ...)` query from the app.
2. Base64-decode the ID; note the `Type:int` structure.
3. Re-encode with a different integer; replay.
4. Compare responses for objects you should not own.

## Impact

Horizontal (other users' data) or vertical (admin objects) authorization bypass.

## Notes

- Also try the `nodes(ids: [...])` batch field — often less guarded.
- Related: [[graphql-introspection]] to discover the type map first.
