"""Chromium's in-tree security and IPC documentation -> `reference` notes.

These documents are the vendor's written contract: what they consider a security bug,
what severity they assign it, which IPC idioms they consider safe, which pointer kinds
they consider mitigated. A hunt against Chrome argues *with* them.

**Only the ones they have changed since the model cutoff are ingested.** The stable
core of this corpus - rule-of-2, the severity guidelines, mojo.md - has read the same
way for years and is therefore training data; storing it back would cost retrieval
tokens and dilute ranking without teaching anything. What is *not* training data is
the edit: a paragraph added to the severity guidelines in July 2026 changes what is
filable, and nothing in the model knows about it. So each note leads with the
post-cutoff commit list for that file, then carries the current text in full for
context. Use ``--all`` when the question is "what does the vendor say about X" rather
than "what did the vendor change".

Local-only: reads the checkout at ``SIFT_CHROMIUM_SRC``. No network, no rate limit.

The section index on each note exists because the useful citation is
``docs/security/faq.md:729``, not "the FAQ" - and a line number only means something
against a named revision, which is why both are recorded.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from slugify import slugify

from sift.config import get_settings
from sift.ingest.base import clean_text
from sift.ingest.existing import attach_existing
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

log = logging.getLogger(__name__)

# Everything under docs/security is in scope by definition. The rest is curated: docs
# that decide a question a browser-process hunt actually asks.
DEFAULT_GLOBS: tuple[str, ...] = ("docs/security/**/*.md",)

CURATED: tuple[str, ...] = (
    # Mojo: the renderer->browser boundary itself.
    "docs/mojo_and_services.md",
    "docs/mojo_ipc_conversion.md",
    "mojo/README.md",
    "mojo/public/cpp/bindings/README.md",
    # Lifetime and pointer semantics - what decides whether a UAF is a security bug.
    "base/memory/raw_ptr.md",
    "docs/callback.md",
    "docs/threading_and_tasks.md",
    # Process model: the boundary a compromised renderer is meant to be trapped behind.
    "docs/process_model_and_site_isolation.md",
    "docs/design/sandbox.md",
    "sandbox/win/README.md",
    # Surfaces whose lifetime rules keep producing IPC-rejection fixes.
    "docs/bfcache.md",
    # Process: how a security fix reaches a release branch, which is what a closure
    # diff is actually measuring.
    "docs/process/merge_request.md",
    # Runtime verdicts: a memory finding is priced by what ASAN prints.
    "docs/asan.md",
)

# Chromium's longest doc (security/faq.md) is ~83k chars and all of it is load-bearing.
# The cap exists only to stop a pathological file becoming the whole index.
MAX_DOC_CHARS = 150_000

_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$")
# Record separators for the one `git log` call that drives the whole source.
_REC, _FLD = "\x01", "\x1f"
# Path fragment -> the axis it informs, so `search --type reference` can be narrowed.
# Docs under docs/security already get `security-doc` and are not listed again.
_AREA_TAGS: tuple[tuple[str, str], ...] = (
    ("mojo", "mojo"),
    ("raw_ptr", "lifetime"),
    ("callback", "lifetime"),
    ("threading", "lifetime"),
    ("sandbox", "sandbox"),
    ("site_isolation", "site-isolation"),
    ("process_model", "site-isolation"),
    ("bfcache", "lifetime"),
    ("asan", "memory-safety"),
    ("severity", "severity"),
    ("rule-of-2", "memory-safety"),
    ("compromised-renderer", "threat-model"),
)


@dataclass
class DocChange:
    """What the vendor did to one doc since the cutoff."""

    path: str
    commits: list[tuple[str, str, str]] = field(default_factory=list)  # (sha, date, subject)
    added: bool = False  # created after the cutoff - the whole document is new

    @property
    def last_touched(self) -> str:
        return self.commits[0][1] if self.commits else ""


def _git(src: Path, *args: str, timeout: int = 180) -> str:
    """Run git in the checkout, returning stdout ("" on any failure).

    Never raises: a missing .git or a git that is not on PATH is a reason to label the
    notes `unknown`, not to abandon the ingest.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=src,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("chromium-docs: git %s failed: %s", " ".join(args[:2]), exc)
        return ""
    return out.stdout


def revision(src: Path) -> str:
    """The checkout's HEAD sha, or "unknown"."""
    return _git(src, "rev-parse", "HEAD", timeout=30).strip() or "unknown"


def _resolved_src() -> Path:
    src = get_settings().chromium_src
    if not src:
        raise RuntimeError(
            "SIFT_CHROMIUM_SRC is not set - point it at your Chromium `src` directory."
        )
    src = Path(src).expanduser()
    if not (src / "docs").is_dir():
        raise RuntimeError(f"{src} does not look like a Chromium checkout (no docs/ directory)")
    return src


def parse_change_log(raw: str) -> dict[str, DocChange]:
    """Parse `git log --name-status` output into per-path change records.

    One git call covers every candidate path, which is why this has to demultiplex
    the result rather than asking per file - `git log` over a 1.8M-commit history is
    the expensive part, and doing it 70 times is the difference between a second and
    a minute.
    """
    changes: dict[str, DocChange] = {}
    for record in raw.split(_REC):
        record = record.strip("\n")
        if not record:
            continue
        head, _, tail = record.partition("\n")
        parts = head.split(_FLD)
        if len(parts) < 3:
            continue
        sha, when, subject = parts[0][:12], parts[1], parts[2]
        for line in tail.splitlines():
            if not line.strip():
                continue
            bits = line.split("\t")
            status, path = bits[0], bits[-1]  # a rename gives old+new; the new one wins
            entry = changes.setdefault(path, DocChange(path=path))
            entry.commits.append((sha, when, subject))
            if status.startswith("A"):
                entry.added = True
    return changes


def changes_since(src: Path, since: date, paths: list[str]) -> dict[str, DocChange]:
    """Every commit touching ``paths`` since ``since``, keyed by path, newest first."""
    if not paths:
        return {}
    raw = _git(
        src,
        "log",
        f"--since={since.isoformat()}",
        "--no-merges",
        "--date=short",
        "--name-status",
        f"--format={_REC}%H{_FLD}%ad{_FLD}%s",
        "--",
        *paths,
    )
    return parse_change_log(raw)


def _doc_title(relpath: str, text: str) -> str:
    """`Chromium: <the doc's own H1> (<path>)`.

    The path belongs in the title because the path is how these docs get cited, and
    because several Chromium docs are all called "README".
    """
    m = _H1_RE.search(text)
    heading = (
        clean_text(m.group(1)) if m else Path(relpath).stem.replace("_", " ").replace("-", " ")
    )
    heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading).strip()  # link-wrapped headings
    heading = re.sub(r"\s*\{#.*\}$", "", heading).strip()  # anchor suffixes
    return f"Chromium: {heading} ({relpath})"


def section_index(text: str) -> list[tuple[int, str, str]]:
    """(line number, marker, heading) for each H2-H4, 1-based.

    Line numbers are against the file as it exists in this checkout, which is what
    makes them citable - see the module docstring.
    """
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _HEADING_RE.match(line)
        if m:
            heading = re.sub(r"\s*\{#.*\}$", "", m.group(2)).strip()
            out.append((lineno, m.group(1), heading))
    return out


def _tags(relpath: str, change: DocChange | None) -> list[str]:
    tags = ["chromium", "google", "vendor-doc"]
    low = relpath.lower()
    if low.startswith("docs/security"):
        tags.append("security-doc")
    for fragment, tag in _AREA_TAGS:
        if fragment in low and tag not in tags:
            tags.append(tag)
    if change:
        tags.append("new-doc" if change.added else "changed-doc")
    return tags


def to_note(relpath: str, text: str, sha: str, change: DocChange | None = None) -> Note:
    truncated = len(text) > MAX_DOC_CHARS
    body_text = text[:MAX_DOC_CHARS]
    sections = section_index(body_text)

    header = [
        f"> Verbatim from the Chromium checkout at `{sha[:12]}`.",
        f"> Path: `{relpath}` — cite as `{Path(relpath).name}:<line>`.",
    ]
    if truncated:
        header.append(f"> **Truncated** at {MAX_DOC_CHARS} chars (source is {len(text)}).")

    blocks = ["\n".join(header)]

    # The edit is the novel part, so it goes first - both for a human skimming and
    # because the opening chunk is what a retrieval hit shows.
    if change and change.commits:
        what = "Added" if change.added else "Changed"
        blocks.append(
            f"## {what} since the cutoff — {len(change.commits)} commit(s), "
            f"last {change.last_touched}\n"
            + "\n".join(f"- `{sha_}` {when} — {subject}" for sha_, when, subject in change.commits)
            + (
                "\n\nThis document did not exist at training time; all of it is new."
                if change.added
                else "\n\nThe body below is the current text. Diff these commits to see "
                "exactly what moved — a doc edit on this corpus usually follows a "
                "policy decision, and the policy is the part worth knowing."
            )
        )

    blocks.append(body_text.strip())

    if sections:
        blocks.append(
            "## Section index (line numbers at this revision)\n"
            + "\n".join(
                f"- `{relpath}:{n}` — {'  ' * (len(mark) - 2)}{head}" for n, mark, head in sections
            )
        )

    meta = Frontmatter(
        id=f"chromium-doc-{slugify(relpath, max_length=90)}",
        type="reference",
        title=_doc_title(relpath, text),
        source="chromium-src",
        url=f"https://source.chromium.org/chromium/chromium/src/+/{sha}:{relpath}",
        program="Google Chrome",
        tags=_tags(relpath, change),
        extra={
            "chromium_path": relpath,
            "chromium_revision": sha,
            "doc_chars": len(text),
            "sections": len(sections),
            "commits_since_cutoff": len(change.commits) if change else 0,
            "last_touched": change.last_touched if change else "",
            "new_since_cutoff": bool(change and change.added),
        },
    )
    return Note(meta=meta, body="\n\n---\n\n".join(blocks))


def candidate_paths(src: Path, *, globs: tuple[str, ...] = DEFAULT_GLOBS) -> list[str]:
    """Repo-relative paths of every doc in scope, deduplicated, in stable order."""
    found: dict[str, None] = {}
    for pattern in globs:
        for path in sorted(src.glob(pattern)):
            if path.is_file():
                found.setdefault(path.relative_to(src).as_posix(), None)
    for rel in CURATED:
        if (src / rel).is_file():
            found.setdefault(rel, None)
    return list(found)


def source(
    *,
    since: date | None = None,
    all_docs: bool = False,
    limit: int | None = None,
    globs: tuple[str, ...] = DEFAULT_GLOBS,
) -> Iterator[Note]:
    """Docs the vendor has touched since ``since`` (default: the model cutoff).

    ``all_docs=True`` ingests the whole corpus regardless of recency - worth it when
    you want the exact wording and line anchors of the stable policy docs to quote at
    triage, but most of that text is material the reasoning model already has.
    """
    settings = get_settings()
    src = _resolved_src()
    horizon = since or settings.model_cutoff
    sha = revision(src)

    paths = candidate_paths(src, globs=globs)
    changes = changes_since(src, horizon, paths)
    selected = paths if all_docs else [p for p in paths if p in changes]

    if not selected:
        log.info(
            "chromium-docs: no doc under docs/security or the curated set changed since "
            "%s - nothing new to store. Use --all to take the whole corpus anyway.",
            horizon.isoformat(),
        )
        return

    vault = settings.resolved_vault()
    seen = 0
    for relpath in selected:
        try:
            text = (src / relpath).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("chromium-docs: cannot read %s: %s", relpath, exc)
            continue
        if len(clean_text(text)) < 400:
            continue  # a stub or a redirect page, not a document
        # The id is the doc's path, i.e. the document. A sync that changes both the
        # revision in its URL and its H1 is still an update of the same note.
        yield attach_existing(vault, to_note(relpath, text, sha, changes.get(relpath)))
        seen += 1
        if limit and seen >= limit:
            return
