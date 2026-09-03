---
id: example-h1-000000
type: report
title: "Stored XSS in comment field via SVG upload"
source: hackerone-public
url: https://hackerone.com/reports/000000
created: 2024-03-11
severity: high
program: Example Corp
cwe:
  - CWE-79
assets:
  - app.example.com
tags:
  - hackerone
  - disclosed
  - cross-site scripting (xss)
links:
  - svg-xss-upload-bypass
---

## Summary

The comment attachment upload accepts `image/svg+xml`. Uploaded SVGs are served
from the same origin with `Content-Type: image/svg+xml` and no CSP, so embedded
`<script>` executes in the context of `app.example.com` when another user opens
the attachment.

## Steps to Reproduce

1. Create a comment, attach `poc.svg` containing `<svg onload=alert(document.domain)>`.
2. Copy the attachment URL (`https://app.example.com/attachments/...`).
3. Open the URL as a second user -> script executes.

## Impact

Session theft / account takeover for any user who opens a malicious attachment.

## Remediation

Serve user uploads from a sandbox origin, force `Content-Disposition: attachment`,
and set `Content-Security-Policy: default-src 'none'`.
