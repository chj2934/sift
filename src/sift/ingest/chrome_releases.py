"""Chrome Releases security notes -> `reference` notes + one reward ledger.

Every Stable desktop release post carries the full security table: reward, bug id,
severity, CVE, bug class, component and reporter. That table is the only public,
per-component record of *what Google actually pays for*, and it is the evidence behind
any claim that a bug class is "vendor-proven" - five fixes in a component in nine
weeks is a measurement, not an impression.

Two kinds of note come out of this:

* one per release, holding that release's table - so a search for a component name
  finds every time the vendor shipped a fix there, with the reporter's name next to
  it (which is also the cheapest duplicate check available);
* one rolling **ledger** aggregating every row parsed, which answers the calibration
  question - what a High in this component has historically paid, and how much of a
  component's fix stream is internal ("Reported by Google") rather than bounty-earning.

Scoped to releases published since ``SIFT_MODEL_CUTOFF``. The format and the rough
bands are training data; *this quarter's* components, reporters and amounts are not,
and a decade-deep backfill would bury the months that carry information. Pass a
deeper ``--since`` when the question is historical rather than operational.

The feed is Blogger's JSON API. Post bodies are small and the Stable-updates label is
~1,565 posts total, so even a full backfill is only ~11 requests.
"""

from __future__ import annotations

import html
import re
import statistics
from collections.abc import Iterator
from datetime import date, datetime

import httpx
from slugify import slugify

from sift.config import get_settings
from sift.ingest.base import clean_text, have_note
from sift.vault.notes import Note
from sift.vault.schema import Frontmatter

FEED = "https://chromereleases.googleblog.com/feeds/posts/default"
# Security tables only ever appear on Stable and Extended Stable posts. Filtering by
# label rather than fetching all 6,375 posts cuts the backfill by 75%.
DEFAULT_LABELS: tuple[str, ...] = ("Stable updates",)
PAGE_SIZE = 150
UA = "sift-chrome-releases-ingest/0.1 (personal bug-bounty memory)"

# `[$3,000][ 550839154 ] High CVE-2026-93375: Incorrect reference resolution in
#  Tracing. Reported by M. Fauzan Wijaya (Gh05t666nero) on 2026-08-22`
#
# The bug id arrives wrapped in spaces because it is a link in the HTML. Reward is
# `$N`, `TBD` (reward not yet decided) or `N/A` (no reward - almost always an
# internally-found bug). The `$` is optional on all three: 2026 posts write `[TBD]`
# and 2015 posts write `[$TBD]`, and requiring it silently dropped every row of the
# older format. Older posts also say "Credit to X" rather than "Reported by X", and
# the trailing report date is absent before ~2016, so both tails are optional.
_ROW_RE = re.compile(
    r"\[\s*(?P<reward>\$?(?:[\d,]+|TBD)|N/?A)\s*\]\s*"
    r"\[\s*(?P<bug>\d+)\s*\]\s*"
    r"(?P<severity>Critical|High|Medium|Low)\s+"
    r"(?P<cve>CVE-\d{4}-\d+)\s*:\s*"
    r"(?P<desc>.+?)"
    r"(?:\.\s*(?:Reported by|Credit to)\s+(?P<reporter>.+?))?"
    r"(?:\s+on\s+(?P<reported_on>\d{4}-\d{2}-\d{2}))?"
    r"\s*(?=\[|$)",
    re.IGNORECASE | re.DOTALL,
)
# The last row of a table has no following `[` to bound it, so the reporter capture
# runs on into the post's closing prose. Two different shapes of that leak turned up
# on real posts and each needs its own cut:
#
#   "Saif El-Sherei. As usual, our ongoing internal security work..."  -> sentence break
#   "Google on 2026-05-17 We would also like to thank..."              -> no period at all
#
# The sentence cut must not fire after a single capital letter, or "M. Fauzan Wijaya"
# loses its surname.
_REPORTER_END_RE = re.compile(r"(?<![A-Z])\.\s+(?=[A-Z])")
# The report date, which the row regex only captures when a later row bounds it.
_REPORTER_DATE_RE = re.compile(r"^(?P<who>.*?)\s+on\s+(?P<when>\d{4}-\d{2}-\d{2})\b")
# Boilerplate that closes these posts, i.e. everything that is certainly not a name.
_CLOSING_PROSE = (
    "We would also like to thank",
    "As usual, our ongoing",
    "Many of our security bugs",
    "Interested in switching",
    "A list of all changes",
    "Note: Access to bug details",
)
_VERSION_RE = re.compile(r"\b(\d{2,3}\.\d+\.\d+\.\d+)\b")
_FIX_COUNT_RE = re.compile(r"includes?\s+(\d+)\s+security\s+fix", re.IGNORECASE)
_TAG_RE = re.compile(r"(?s)<[^>]+>")
_INTERNAL_REPORTERS = ("google", "chrome security", "internal audit", "clusterfuzz")


def _text(raw: str) -> str:
    """Flatten a post body to single-spaced text, keeping row boundaries.

    The rows are `<br>`-separated in the HTML; collapsing to one line is fine because
    `_ROW_RE` anchors on the bracketed reward, not on line starts.
    """
    txt = html.unescape(_TAG_RE.sub(" ", raw))
    # Non-breaking spaces are everywhere in these posts and would break `\s` matching
    # of the reward bracket.
    txt = txt.replace("\xa0", " ").replace("​", "")
    return re.sub(r"[ \t\r\n]+", " ", txt).strip()


def _split_desc(desc: str) -> tuple[str, str]:
    """"Use after free in Dawn" -> ("Use after free", "Dawn").

    Chrome's descriptions are uniformly `<bug class> in <component>`, and the split
    has to take the *last* " in " - "Inappropriate implementation in Extensions API"
    would otherwise yield the wrong half.
    """
    desc = clean_text(desc).rstrip(". ")
    parts = re.split(r"\s+in\s+", desc)
    if len(parts) < 2:
        return desc, ""
    return " in ".join(parts[:-1]).strip(), parts[-1].strip()


def _reward_value(reward: str) -> float | None:
    """Numeric USD, or None for TBD / N/A - which are not zero and must not average in."""
    digits = reward.lstrip("$").replace(",", "")
    if not digits.isdigit():
        return None
    return float(digits)


def _clean_reporter(raw: str | None) -> tuple[str, str]:
    """(reporter name, report date) with the post's trailing prose cut off.

    Returns the date too because on the final row of a table it is still inside the
    reporter capture - see `_REPORTER_DATE_RE`.
    """
    reporter = clean_text(raw or "")
    reported_on = ""
    m = _REPORTER_DATE_RE.match(reporter)
    if m:
        reporter, reported_on = m.group("who"), m.group("when")
    for marker in _CLOSING_PROSE:
        reporter = reporter.split(marker, 1)[0]
    reporter = _REPORTER_END_RE.split(reporter, maxsplit=1)[0]
    return reporter.rstrip(". ").strip(), reported_on


def parse_rows(body_html: str) -> list[dict]:
    """Every security row in one release post."""
    txt = _text(body_html)
    rows: list[dict] = []
    for m in _ROW_RE.finditer(txt):
        bug_class, component = _split_desc(m.group("desc"))
        reporter, trailing_date = _clean_reporter(m.group("reporter"))
        reward = m.group("reward").strip()
        if not reward.startswith("$"):
            reward = reward.upper().replace("NA", "N/A")
        rows.append(
            {
                "cve": m.group("cve").upper(),
                "severity": m.group("severity").capitalize(),
                "reward": reward,
                "reward_usd": _reward_value(reward),
                "bug_id": m.group("bug"),
                "bug_class": bug_class,
                "component": component,
                "reporter": reporter,
                "reported_on": m.group("reported_on") or trailing_date,
                "internal": any(k in reporter.lower() for k in _INTERNAL_REPORTERS),
            }
        )
    return rows


def _published(entry: dict) -> date | None:
    raw = (entry.get("published") or {}).get("$t")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return None


def _link(entry: dict) -> str:
    for link in entry.get("link", []):
        if link.get("rel") == "alternate":
            return link.get("href", "")
    return ""


def _release_title(rows: list[dict], version: str, channel: str, when: date | None) -> str:
    what = f"Chrome {version}" if version else "Chrome release"
    stamp = when.isoformat() if when else "undated"
    return f"{what} {channel} — {len(rows)} security fixes, {stamp}"


def release_note(entry: dict, rows: list[dict]) -> Note:
    body_html = (entry.get("content") or {}).get("$t", "")
    txt = _text(body_html)
    post_title = clean_text((entry.get("title") or {}).get("$t", "")) or "Chrome release"
    when = _published(entry)
    version_match = _VERSION_RE.search(txt)
    version = version_match.group(1) if version_match else ""
    channel = "Stable"
    if "extended stable" in post_title.lower():
        channel = "Extended Stable"
    elif "chromeos" in post_title.lower():
        channel = "ChromeOS"

    claimed = _FIX_COUNT_RE.search(txt)
    lines = [
        f"**{post_title}** — {when.isoformat() if when else 'undated'}"
        + (f", version `{version}`" if version else ""),
        "",
        "| CVE | Sev | Reward | Bug class | Component | Reporter | Bug |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cve']} | {r['severity']} | {r['reward']} | {r['bug_class']} | "
            f"{r['component']} | {r['reporter'] or '—'} | "
            f"[{r['bug_id']}](https://issues.chromium.org/issues/{r['bug_id']}) |"
        )
    # The post states its own fix count. When it disagrees with what parsed, say so on
    # the note rather than in a log line nobody reads - a silent regex drift here would
    # quietly shrink the ledger.
    if claimed and int(claimed.group(1)) != len(rows):
        lines += [
            "",
            f"> ⚠ The post says **{claimed.group(1)}** security fixes; {len(rows)} rows parsed. "
            "The remainder are fixes with no CVE row (internal audit, fuzzing, hardening).",
        ]
    ext = [r for r in rows if not r["internal"]]
    paid = [r["reward_usd"] for r in rows if r["reward_usd"]]
    lines += [
        "",
        f"External reporters: **{len(ext)}/{len(rows)}**. "
        f"Rewards stated: {len(paid)}"
        + (f", total ${sum(paid):,.0f}, max ${max(paid):,.0f}." if paid else "."),
    ]

    slug_basis = f"{version or post_title}-{when.isoformat() if when else 'x'}"
    meta = Frontmatter(
        id=f"chrome-release-{slugify(slug_basis, max_length=70)}",
        type="reference",
        title=_release_title(rows, version, channel, when),
        source="chrome-releases",
        url=_link(entry),
        created=when,
        program="Google Chrome",
        tags=["chromium", "google", "chrome-vrp", "release-notes", channel.lower().replace(" ", "-")],
        cwe=[],
        extra={
            "chrome_version": version,
            "channel": channel,
            "cve_count": len(rows),
            "claimed_fix_count": int(claimed.group(1)) if claimed else None,
            "components": sorted({r["component"] for r in rows if r["component"]}),
            "cves": [r["cve"] for r in rows],
        },
    )
    return Note(meta=meta, body="\n".join(lines))


_SEV_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _rank(rows: list[dict], key: str, top: int) -> list[tuple[str, int, list[str]]]:
    counts: dict[str, list[dict]] = {}
    for r in rows:
        val = r.get(key) or ""
        if val:
            counts.setdefault(val, []).append(r)
    ranked = sorted(counts.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]
    return [
        (
            name,
            len(rs),
            sorted({r["severity"] for r in rs}, key=lambda s: _SEV_ORDER.get(s, 9)),
        )
        for name, rs in ranked
    ]


def ledger_note(rows: list[dict], *, top: int = 40, horizon: date | None = None) -> Note:
    """One rolling note answering "what does Chrome pay, and for what?".

    Scoped to the ingest window, and it says so on the note: a median computed over
    four months of releases is a current signal, and one computed over ten years is a
    different claim entirely. Reading it as the latter when it is the former is how a
    calibration note starts lying.
    """
    dates = sorted(r["reported_on"] for r in rows if r["reported_on"])
    span = f"{dates[0]} to {dates[-1]}" if dates else "unknown span"
    ext = [r for r in rows if not r["internal"]]

    lines = [
        f"Parsed from **{len({r['cve'] for r in rows})}** CVE rows across Chrome Stable "
        f"release notes, reports dated {span}"
        + (f", releases from {horizon.isoformat()} onward." if horizon else "."),
        "",
        "## Reward by severity — stated rewards only",
        "",
        "`TBD` and `N/A` rows are excluded, not counted as zero: TBD means the panel had "
        "not decided at publication, N/A almost always means internally found.",
        "",
        "| Severity | Rows | Stated $ | Median | Max |",
        "|---|---|---|---|---|",
    ]
    for sev in ("Critical", "High", "Medium", "Low"):
        group = [r for r in rows if r["severity"] == sev]
        paid = sorted(r["reward_usd"] for r in group if r["reward_usd"])
        lines.append(
            f"| {sev} | {len(group)} | {len(paid)} | "
            + (f"${statistics.median(paid):,.0f} | ${max(paid):,.0f} |" if paid else "— | — |")
        )

    lines += [
        "",
        f"## Internal vs external — {len(ext)}/{len(rows)} rows had an external reporter",
        "",
        'A component whose fixes are mostly "Reported by Google" is one the vendor '
        "audits itself; a component with a long external tail is one where outside "
        "reports land. Both are useful, in opposite directions.",
        "",
        "## Components by fix count",
        "",
        "| Component | Fixes | External | Severities seen | Stated $ total |",
        "|---|---|---|---|---|",
    ]
    for name, count, sevs in _rank(rows, "component", top):
        group = [r for r in rows if r["component"] == name]
        paid = [r["reward_usd"] for r in group if r["reward_usd"]]
        lines.append(
            f"| {name} | {count} | {len([r for r in group if not r['internal']])} | "
            f"{', '.join(sevs)} | " + (f"${sum(paid):,.0f} |" if paid else "— |")
        )

    lines += ["", "## Bug classes by fix count", "", "| Bug class | Fixes | Severities seen |", "|---|---|---|"]
    for name, count, sevs in _rank(rows, "bug_class", top):
        lines.append(f"| {name} | {count} | {', '.join(sevs)} |")

    meta = Frontmatter(
        id="chrome-vrp-ledger",
        type="reference",
        title="Chrome VRP reward ledger — what the vendor paid, by component and bug class",
        source="chrome-releases",
        url="https://chromereleases.googleblog.com/",
        program="Google Chrome",
        tags=["chromium", "google", "chrome-vrp", "payout-calibration", "ledger"],
        extra={
            "rows": len(rows),
            "unique_cves": len({r["cve"] for r in rows}),
            "external_rows": len(ext),
            "reports_span": span,
            "ingest_horizon": horizon.isoformat() if horizon else None,
        },
    )
    return Note(meta=meta, body="\n".join(lines))


def _fetch_page(client: httpx.Client, label: str | None, start: int) -> list[dict]:
    url = FEED + (f"/-/{label}" if label else "")
    r = client.get(url, params={"alt": "json", "max-results": PAGE_SIZE, "start-index": start})
    r.raise_for_status()
    return r.json().get("feed", {}).get("entry", []) or []


def source(
    *,
    since: date | None = None,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    limit: int | None = None,
    ledger: bool = True,
) -> Iterator[Note]:
    """Release notes published on/after ``since`` (default: the model cutoff), then
    the rolling ledger.

    The horizon matters here. Chrome's *format* and its rough payout bands are
    training data; which components it has been fixing this quarter, who got paid for
    them and how much are not. Pulling a decade of releases would bury the few months
    that carry information.

    Posts already in the vault are re-parsed but not re-yielded: the fetch is one
    request per 150 posts either way, and the ledger needs every row in the window to
    stay honest. Dropping already-seen posts from the aggregate would make the ledger
    describe only the newest release.
    """
    settings = get_settings()
    vault = settings.resolved_vault()
    horizon = since or settings.model_cutoff
    all_rows: list[dict] = []
    emitted = 0
    seen_cves: set[str] = set()

    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": UA}) as client:
        for label in labels or (None,):
            start = 1
            while True:
                try:
                    entries = _fetch_page(client, label, start)
                except httpx.HTTPError as exc:
                    print(f"  ! chrome-releases: fetch failed at start-index {start}: {exc}")
                    break
                if not entries:
                    break
                stop = False
                for entry in entries:
                    when = _published(entry)
                    if when and when < horizon:
                        stop = True  # the feed is newest-first, so everything after is older
                        break
                    rows = parse_rows((entry.get("content") or {}).get("$t", ""))
                    if not rows:
                        continue  # Android/ChromeOS/beta post, or a release with no CVEs
                    for r in rows:
                        if r["cve"] not in seen_cves:
                            seen_cves.add(r["cve"])
                            all_rows.append(r)
                    note = release_note(entry, rows)
                    if have_note(vault, note.meta):
                        continue
                    yield note
                    emitted += 1
                    if limit and emitted >= limit:
                        stop = True
                        break
                if stop:
                    break
                start += PAGE_SIZE

    if ledger and all_rows:
        yield ledger_note(all_rows, horizon=horizon)
