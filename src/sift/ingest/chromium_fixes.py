"""Chromium's own security-fix commits -> `reference` notes.

The highest-volume source of genuinely post-cutoff Google knowledge. Every one of
these commits is the vendor stating, in their own words, that an invariant was not
being enforced somewhere - and the value of that is not the patched line. It is the
*mechanism*: the whole set of other sites where the same unenforced invariant applies.
That set is not published anywhere, and it is what a differential hunt is built from.

Why mine the log rather than read CVEs: a CVE exists only for bugs that shipped to
stable and got a reward row. The log carries the rest - the hardening commits, the
"reject this IPC from a bfcached frame" class, the guards added pre-emptively next to
a bug that *did* pay. Those are the ones that name an invariant without anyone having
been paid to find the next instance of it.

Local-only, one `git log` call over ``SIFT_CHROMIUM_SRC``. Scoped by default to
``SIFT_MODEL_CUTOFF`` and to the browser-process surfaces a renderer can reach;
measured on a real checkout, the filter keeps ~1.6% of commits in those paths.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sift.config import get_settings
from sift.ingest.base import clean_text
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

# Where a compromised renderer's messages actually land. Excluding third_party/ is
# deliberate: its fixes are overwhelmingly fuzzer-found memory bugs in parsers, which
# is the one class with a working automated oracle already pointed at it.
DEFAULT_PATHS: tuple[str, ...] = (
    "content/browser",
    "chrome/browser",
    "components",
    "services",
    "mojo",
    "ipc",
    "base/memory",
    "sandbox",
)

# Four mechanism families, each a tag on the note so a search can be narrowed to one.
# Matched against the subject line only: a subject states what the commit *did*,
# whereas the body mentions bug classes in passing and matching it pulled in 4x the
# volume at a fraction of the precision (948 -> 4,430 on a 6-month window).
CLASS_PATTERNS: dict[str, str] = {
    # Memory safety - the only class that clears Chrome's logic-bug reward ceiling.
    "memory-safety": (
        r"use[- ]after[- ]free|\bUAFs?\b|dangling|\bOOBs?\b|out[- ]of[- ]bounds"
        r"|buffer overflow|type confusion|integer overflow|double free"
        r"|uninitiali[sz]ed|miracleptr|raw_ptr|use after destr"
    ),
    # A boundary check being added is a boundary that was missing.
    "boundary-enforcement": (
        r"\b(reject|validate|verify|guard|restrict|deny|disallow|block|forbid)\w*\b"
        r".{0,40}\b(ipc|mojo|renderer|frame|origin|url|handle|token|message|request"
        r"|caller|process|navigation|binder|interface)\b"
    ),
    # The vendor naming the threat model outright.
    "threat-model": (
        r"\bsecurit\w+|compromis\w+|sandbox escape|spoof\w*|CVE-\d{4}"
        r"|site isolation|cross[- ]origin|process lock|privilege"
    ),
    # Object lifetime - where synchronous destruction bugs live.
    "lifetime": r"\bweakptr\b|lifetime|destroy\w*.{0,30}(during|while|reentran)|reentran",
}

# Automated rolls carry no reasoning, and a test-expectation update is not a fix.
# `Reland` is deliberately NOT dropped: a relanded security fix is still the fix, and
# its message usually explains what broke the first attempt.
DROP_SUBJECT = re.compile(
    r"^(roll\s|revert\s|\[fuchsia|update\s.{0,30}expectations?"
    r"|disable\s.{0,30}test|mark\s.{0,30}(flaky|failing))",
    re.IGNORECASE,
)

# Gerrit trailers: metadata, not reasoning. Stripped so the note body is the argument
# the author actually made.
TRAILER_RE = re.compile(
    r"^(Change-Id|Reviewed-by|Reviewed-on|Commit-Queue|Cr-Commit-Position|Cr-Branched-From"
    r"|Auto-Submit|Bug|Fixed|Fixes|Test|Tests|R|TBR|NOTRY|NOPRESUBMIT|Cq-[\w-]+|X-[\w-]+"
    r"|Signed-off-by|Co-authored-by|Binary-Size|Low-Coverage-Reason|Validate-Test-Flakiness)"
    r"\s*[:=]",
    re.IGNORECASE,
)
_BUG_RE = re.compile(r"^(?:Bug|Fixed|Fixes)\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_BUG_ID_RE = re.compile(r"\b(\d{6,10})\b")
_REC, _FLD, _END = "\x01", "\x1f", "\x02"
MAX_BODY_CHARS = 12_000
MAX_FILES = 40

_CLASS_RE = {name: re.compile(pat, re.IGNORECASE) for name, pat in CLASS_PATTERNS.items()}


@dataclass
class Commit:
    sha: str
    when: str
    author: str
    subject: str
    body: str
    files: list[str]


def classify(subject: str) -> list[str]:
    """Mechanism families this subject line matches; empty means "not a security fix".

    Subject-only by design - see `CLASS_PATTERNS`.
    """
    if not subject or DROP_SUBJECT.search(subject):
        return []
    return [name for name, rx in _CLASS_RE.items() if rx.search(subject)]


def strip_trailers(body: str) -> str:
    """The commit message without its Gerrit trailers."""
    kept = [line for line in body.splitlines() if not TRAILER_RE.match(line.strip())]
    return clean_text("\n".join(kept))


def bug_ids(body: str) -> list[str]:
    """Issue ids from the Bug:/Fixed: trailers, deduplicated, in order."""
    out: dict[str, None] = {}
    for line in _BUG_RE.findall(body):
        for bug in _BUG_ID_RE.findall(line):
            out.setdefault(bug, None)
    return list(out)


def parse_log(raw: str) -> Iterator[Commit]:
    """Parse the `git log` stream this module asks for.

    The format ends each message with `\\x02` precisely so the file list that
    `--name-only` appends can be told apart from the commit body - commit bodies
    contain blank lines of their own, so splitting on those would mangle both.
    """
    for record in raw.split(_REC):
        if not record.strip():
            continue
        head, _, filelist = record.partition(_END)
        parts = head.split(_FLD)
        if len(parts) < 5:
            continue
        sha, when, author, subject, body = parts[0], parts[1], parts[2], parts[3], parts[4]
        files = [line.strip() for line in filelist.splitlines() if line.strip()]
        yield Commit(
            sha=sha.strip(),
            when=when.strip(),
            author=author.strip(),
            subject=subject.strip(),
            body=body,
            files=files,
        )


def _area_tags(files: list[str]) -> list[str]:
    """Coarse surface tags from the touched paths, for narrowing a search."""
    areas: dict[str, None] = {}
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2 and parts[0] in {"content", "chrome", "services", "components"}:
            areas.setdefault(f"{parts[0]}-{parts[1]}", None)
        elif parts[0] in {"mojo", "ipc", "sandbox", "base"}:
            areas.setdefault(parts[0], None)
    return list(areas)[:6]


def to_note(commit: Commit, classes: list[str]) -> Note:
    bugs = bug_ids(commit.body)
    message = strip_trailers(commit.body)[:MAX_BODY_CHARS]
    files = commit.files[:MAX_FILES]

    blocks = [
        f"**{commit.subject}**",
        f"`{commit.sha[:12]}` · {commit.when} · {commit.author} · "
        + ", ".join(f"`{c}`" for c in classes),
    ]
    if bugs:
        blocks.append(
            "Bug: "
            + ", ".join(f"[{b}](https://issues.chromium.org/issues/{b})" for b in bugs)
            + "  \n(Restricted while the fix rolls out — a 403 here means the bug was "
            "security-tagged, which is itself the signal.)"
        )
    if message:
        blocks.append("## Commit message\n\n" + message)
    else:
        blocks.append(
            "## Commit message\n\n*(No body — the subject is the whole claim. Read the "
            "diff before trusting a mechanism read of this one.)*"
        )
    if files:
        blocks.append(
            f"## Files touched ({len(commit.files)})\n\n"
            + "\n".join(f"- `{f}`" for f in files)
            + ("\n- …" if len(commit.files) > MAX_FILES else "")
        )
    blocks.append(
        "## Generalise, do not re-walk\n\n"
        "The patched call site is fixed and worthless. The question is which *other* "
        "sites rely on the same invariant this commit had to start enforcing — that "
        "set is unpublished, and it is the lead."
    )

    created = None
    with contextlib.suppress(ValueError):
        created = date.fromisoformat(commit.when[:10])

    meta = Frontmatter(
        id=f"chromium-fix-{commit.sha[:12]}",
        type="reference",
        title=f"Chromium fix {commit.sha[:8]} — {commit.subject}"[:180],
        source="chromium-git",
        url=f"https://chromium.googlesource.com/chromium/src/+/{commit.sha}",
        created=created,
        program="Google Chrome",
        tags=["chromium", "google", "security-fix", *classes, *_area_tags(commit.files)],
        extra={
            "sha": commit.sha,
            "author": commit.author,
            "bug_ids": bugs,
            "classes": classes,
            "files_touched": len(commit.files),
        },
    )
    return Note(meta=meta, body="\n\n".join(blocks))


def _resolved_src() -> Path:
    src = get_settings().chromium_src
    if not src:
        raise RuntimeError(
            "SIFT_CHROMIUM_SRC is not set - point it at your Chromium `src` directory."
        )
    src = Path(src).expanduser()
    if not (src / ".git").exists():
        raise RuntimeError(f"{src} is not a git checkout - chromium-fixes needs the history")
    return src


def fetch_log(src: Path, since: date, paths: tuple[str, ...]) -> str:
    """The whole windowed log in one call - no `--grep` pre-filter.

    A `--grep` would cut the output roughly eightfold (50MB/17s down to a few), but
    git's pattern syntax is not Python's: `\\b` is a GNU extension in POSIX ERE and
    the alternations here lean on it. A pre-filter that quietly matched less than
    `CLASS_PATTERNS` would drop real commits while still reporting success, and a
    filter that cannot be shown to be a superset of the real one is not worth 15
    seconds. The subject filter in `classify` is the only selector.
    """
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                f"--since={since.isoformat()}",
                "--no-merges",
                "--date=short",
                "--name-only",
                f"--format={_REC}%H{_FLD}%ad{_FLD}%an{_FLD}%s{_FLD}%b{_END}",
                "--",
                *paths,
            ],
            cwd=src,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  ! chromium-fixes: git log failed: {exc}")
        return ""
    return out.stdout


def source(
    *,
    since: date | None = None,
    paths: tuple[str, ...] = DEFAULT_PATHS,
    classes: tuple[str, ...] = (),
    limit: int | None = None,
) -> Iterator[Note]:
    """Security-relevant commits since ``since`` (default: the model cutoff).

    ``classes`` narrows to named mechanism families (see `CLASS_PATTERNS`); empty
    means all four.
    """
    settings = get_settings()
    src = _resolved_src()
    horizon = since or settings.model_cutoff
    wanted = set(classes)

    raw = fetch_log(src, horizon, paths)
    if not raw:
        return

    seen = 0
    scanned = 0
    for commit in parse_log(raw):
        scanned += 1
        matched = classify(commit.subject)
        if not matched or (wanted and not wanted.intersection(matched)):
            continue
        yield to_note(commit, matched)
        seen += 1
        if limit and seen >= limit:
            break
    print(f"  .. chromium-fixes: {seen} security-relevant of {scanned} commits since {horizon}")
