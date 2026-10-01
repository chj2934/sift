"""Offline tests for the Google/Chromium sources.

Every assertion here corresponds to something that was actually wrong against real
data during development, which is why the fixtures are shaped the way they are:

* `[$TBD]` vs `[TBD]` — the dollar sign is optional, and requiring it silently
  dropped every row of the pre-2016 release format.
* the last row of a release table has no following `[` to bound it, so the reporter
  capture ran on into the post's closing paragraph, in two different shapes.
* `git log --name-only` appends the file list after a commit body that itself
  contains blank lines, so the body/file boundary needs an explicit terminator.
* `classify` matches the subject only; matching the body too was measured at 4x the
  volume for a fraction of the precision.
"""

from __future__ import annotations

from datetime import date

# --------------------------------------------------------------------------- #
# chrome-releases
# --------------------------------------------------------------------------- #

# The current format, trimmed to four rows. Bug ids arrive wrapped in spaces because
# they are links in the real post, and &nbsp; is everywhere.
RELEASE_2026 = """
<div>The Stable channel has been updated to 153.0.8010.52/.53 for Windows and Mac.
<b>Security Fixes and Rewards</b>
This update includes 4 security fixes.
[TBD][<a href="https://issues.chromium.org/issues/500417361"> 500417361 </a>]
Critical CVE-2026-93374: Use after free in Dawn. Reported by Florian Schweitzer on 2026-04-08<br>
[$3,000][<a href="x"> 550839154 </a>] High CVE-2026-93375: Incorrect reference
resolution in Tracing. Reported by M. Fauzan Wijaya (Gh05t666nero) on 2026-08-22<br>
[N/A][<a href="x"> 553130676 </a>] High CVE-2026-93373: Use after free in
Extensions&nbsp;API. Reported by Google on 2026-08-26<br>
[$7,000][<a href="x"> 560039872 </a>] Medium CVE-2026-93379: Incorrect authorization
in ORB. Reported by Google on 2026-05-17
We would also like to thank all security researchers that worked with us during the
development cycle to prevent security bugs from ever reaching the stable channel.</div>
"""

# The pre-2016 format: `[$TBD]`, "Credit to" instead of "Reported by", no report date,
# and a single row followed immediately by prose.
RELEASE_2015 = """
<div>This update includes 5 security fixes.
[$TBD][<a href="x"> 453279 </a>] High CVE-2015-1243: Use-after-free in DOM.
Credit to Saif El-Sherei. As usual, our ongoing internal security work was
responsible for a wide range of fixes.</div>
"""


def test_current_release_table_parses_every_field():
    from sift.ingest.chrome_releases import parse_rows

    rows = parse_rows(RELEASE_2026)
    assert len(rows) == 4
    first = rows[0]
    assert first["cve"] == "CVE-2026-93374"
    assert first["severity"] == "Critical"
    assert first["reward"] == "TBD"
    assert first["reward_usd"] is None  # TBD is not zero
    assert first["bug_id"] == "500417361"
    assert first["bug_class"] == "Use after free"
    assert first["component"] == "Dawn"
    assert first["reporter"] == "Florian Schweitzer"
    assert first["reported_on"] == "2026-04-08"
    assert first["internal"] is False


def test_reporter_with_an_initial_keeps_its_surname():
    from sift.ingest.chrome_releases import parse_rows

    row = parse_rows(RELEASE_2026)[1]
    assert row["reporter"] == "M. Fauzan Wijaya (Gh05t666nero)"
    assert row["reward_usd"] == 3000.0


def test_last_row_reporter_does_not_swallow_closing_prose():
    from sift.ingest.chrome_releases import parse_rows

    last = parse_rows(RELEASE_2026)[-1]
    assert last["reporter"] == "Google"
    assert last["internal"] is True
    # The date lives inside the reporter capture on the final row; it must be rescued.
    assert last["reported_on"] == "2026-05-17"


def test_dollar_prefixed_tbd_and_credit_to_still_parse():
    from sift.ingest.chrome_releases import parse_rows

    rows = parse_rows(RELEASE_2015)
    assert len(rows) == 1
    assert rows[0]["cve"] == "CVE-2015-1243"
    assert rows[0]["reward_usd"] is None
    assert rows[0]["reporter"] == "Saif El-Sherei"  # not "...El-Sherei. As usual, our..."


def test_component_split_takes_the_last_in():
    from sift.ingest.chrome_releases import _split_desc

    assert _split_desc("Inappropriate implementation in Extensions API") == (
        "Inappropriate implementation",
        "Extensions API",
    )
    assert _split_desc("Use after free in Dawn") == ("Use after free", "Dawn")
    # No " in " at all: the whole thing is the bug class, component unknown.
    assert _split_desc("Privilege escalation using service workers") == (
        "Privilege escalation using service workers",
        "",
    )


def test_release_note_flags_a_fix_count_mismatch():
    from sift.ingest.chrome_releases import parse_rows, release_note

    entry = {
        "title": {"$t": "Stable Channel Update for Desktop"},
        "published": {"$t": "2026-09-17T09:43:32.834-07:00"},
        "content": {"$t": RELEASE_2015},  # claims 5 fixes, one parseable row
        "link": [{"rel": "alternate", "href": "https://example.invalid/post"}],
    }
    note = release_note(entry, parse_rows(RELEASE_2015))
    # The warning has to name both numbers, or a regex drift that silently halves the
    # ledger reads as a normal ingest.
    assert "says **5**" in note.body
    assert "1 rows parsed" in note.body
    assert note.meta.type == "reference"
    assert note.meta.created == date(2026, 9, 17)
    assert "chrome-vrp" in note.meta.tags


def test_ledger_excludes_tbd_and_na_from_the_medians():
    from sift.ingest.chrome_releases import ledger_note, parse_rows

    note = ledger_note(parse_rows(RELEASE_2026), horizon=date(2026, 4, 1))
    assert note.meta.id == "chrome-vrp-ledger"  # stable id: it is rewritten, not added to
    # One Critical row, reward TBD -> no median can be claimed for Critical.
    critical = next(line for line in note.body.splitlines() if line.startswith("| Critical |"))
    assert critical.endswith("| — | — |")
    # The single stated High reward is $3,000.
    high = next(line for line in note.body.splitlines() if line.startswith("| High |"))
    assert "$3,000" in high
    assert note.meta.extra["ingest_horizon"] == "2026-04-01"


# --------------------------------------------------------------------------- #
# chromium-docs
# --------------------------------------------------------------------------- #

CHANGE_LOG = (
    "\x01abc123def456\x1f2026-09-16\x1fTweak the VRP FAQ\n"
    "M\tdocs/security/vrp-faq.md\n"
    "A\tdocs/security/ai-generated-security-bugs-faq.md\n"
    "\x01fed654cba321\x1f2026-07-28\x1fClarify Mojo review guidance\n"
    "M\tdocs/security/mojo.md\n"
    "R100\tdocs/security/old-name.md\tdocs/security/mojo-renamed.md\n"
)

DOC_TEXT = """# Severity Guidelines for Security Issues

Some preamble.

## Critical severity {#TOC-Critical-severity}

Body text.

### A sub-heading

More text.
"""


def test_change_log_demultiplexes_paths_and_detects_new_files():
    from sift.ingest.chromium_docs import parse_change_log

    changes = parse_change_log(CHANGE_LOG)
    assert set(changes) == {
        "docs/security/vrp-faq.md",
        "docs/security/ai-generated-security-bugs-faq.md",
        "docs/security/mojo.md",
        "docs/security/mojo-renamed.md",  # a rename reports under its new name
    }
    assert changes["docs/security/ai-generated-security-bugs-faq.md"].added is True
    assert changes["docs/security/vrp-faq.md"].added is False
    assert changes["docs/security/vrp-faq.md"].last_touched == "2026-09-16"
    assert changes["docs/security/vrp-faq.md"].commits[0][0] == "abc123def456"


def test_section_index_line_numbers_are_one_based():
    from sift.ingest.chromium_docs import section_index

    idx = section_index(DOC_TEXT)
    assert (5, "##", "Critical severity") in idx  # anchor suffix stripped
    assert (9, "###", "A sub-heading") in idx
    assert all(mark != "#" for _, mark, _ in idx)  # the H1 is the title, not a section


def test_doc_note_leads_with_the_change_and_tags_novelty():
    from sift.ingest.chromium_docs import DocChange, to_note

    change = DocChange(
        path="docs/security/severity-guidelines.md",
        commits=[("abc123def456", "2026-08-26", "Reclassify renderer-only bypasses")],
    )
    note = to_note("docs/security/severity-guidelines.md", DOC_TEXT, "0a9371a0935b", change)

    assert note.meta.type == "reference"
    assert note.meta.title == (
        "Chromium: Severity Guidelines for Security Issues "
        "(docs/security/severity-guidelines.md)"
    )
    assert "changed-doc" in note.meta.tags and "severity" in note.meta.tags
    assert "new-doc" not in note.meta.tags
    # The edit is the novel part, so it must precede the document text.
    assert note.body.index("Changed since the cutoff") < note.body.index("Some preamble")
    assert "Reclassify renderer-only bypasses" in note.body
    assert "`docs/security/severity-guidelines.md:5`" in note.body  # citable line anchor
    assert note.meta.extra["commits_since_cutoff"] == 1


def test_new_doc_says_the_whole_thing_is_new():
    from sift.ingest.chromium_docs import DocChange, to_note

    change = DocChange(path="docs/security/x.md", commits=[("a1b2c3", "2026-09-03", "Add")], added=True)
    note = to_note("docs/security/x.md", DOC_TEXT, "deadbeef", change)
    assert "new-doc" in note.meta.tags
    assert "did not exist at training time" in note.body
    assert note.meta.extra["new_since_cutoff"] is True


# --------------------------------------------------------------------------- #
# chromium-fixes
# --------------------------------------------------------------------------- #

# Two commits. The first body contains a blank line and a bulleted list, which is why
# the format needs the \x02 terminator to find where the file list starts.
FIX_LOG = (
    "\x01aaaaaaaaaaaaaaaa\x1f2026-09-18\x1fCarlos K\x1f"
    "Reject frame detach IPCs from pages in the BackForwardCache\x1f"
    "A renderer in the bfcache must not be able to detach a frame.\n"
    "\n"
    "Changes:\n"
    "- Reject the message and report a bad message.\n"
    "\n"
    "Bug: 514040146, 987654321\n"
    "Change-Id: Iabc123\n"
    "Reviewed-by: Someone <someone@chromium.org>\n"
    "Cr-Commit-Position: refs/heads/main@{#1}\n"
    "\x02"
    "content/browser/renderer_host/render_frame_host_impl.cc\n"
    "content/browser/renderer_host/render_frame_host_impl.h\n"
    "\x01bbbbbbbbbbbbbbbb\x1f2026-09-17\x1fDep Roller\x1f"
    "Roll src/third_party/dawn/ abc..def (3 revisions)\x1f"
    "Includes a use-after-free fix.\n"
    "\x02"
    "DEPS\n"
)


def test_parse_log_separates_a_multi_paragraph_body_from_the_file_list():
    from sift.ingest.chromium_fixes import parse_log

    commits = list(parse_log(FIX_LOG))
    assert len(commits) == 2
    first = commits[0]
    assert first.subject == "Reject frame detach IPCs from pages in the BackForwardCache"
    assert first.author == "Carlos K"
    assert first.when == "2026-09-18"
    assert first.files == [
        "content/browser/renderer_host/render_frame_host_impl.cc",
        "content/browser/renderer_host/render_frame_host_impl.h",
    ]
    assert "Changes:" in first.body  # the blank-line paragraph survived


def test_classify_reads_the_subject_not_the_body():
    from sift.ingest.chromium_fixes import classify, parse_log

    commits = list(parse_log(FIX_LOG))
    assert classify(commits[0].subject) == ["boundary-enforcement"]
    # The roll's *body* says "use-after-free" but its subject does not, and rolls carry
    # no reasoning: matching the body is what made this filter 4x noisier.
    assert classify(commits[1].subject) == []


def test_rolls_are_dropped_but_relands_are_kept():
    from sift.ingest.chromium_fixes import classify

    assert classify("Roll src/third_party/x: fix use-after-free") == []
    assert classify("Revert \"Validate the renderer-supplied origin\"") == []
    assert classify('Reland "Validate RenderFrameMetadata fields during Mojo deserialization"') == [
        "boundary-enforcement"
    ]
    assert classify("Fix UAF in CredentialManager task") == ["memory-safety"]
    assert classify("Update WPT expectations for a security test") == []


def test_trailers_are_stripped_and_bug_ids_extracted():
    from sift.ingest.chromium_fixes import bug_ids, parse_log, strip_trailers

    body = list(parse_log(FIX_LOG))[0].body
    assert bug_ids(body) == ["514040146", "987654321"]
    cleaned = strip_trailers(body)
    assert "Change-Id" not in cleaned
    assert "Reviewed-by" not in cleaned
    assert "Cr-Commit-Position" not in cleaned
    assert "must not be able to detach a frame" in cleaned


def test_fix_note_shape():
    from sift.ingest.chromium_fixes import classify, parse_log, to_note

    commit = list(parse_log(FIX_LOG))[0]
    note = to_note(commit, classify(commit.subject))
    assert note.meta.id == "chromium-fix-aaaaaaaaaaaa"
    assert note.meta.type == "reference"
    assert note.meta.created == date(2026, 9, 18)
    assert note.meta.url.endswith("aaaaaaaaaaaaaaaa")
    assert "security-fix" in note.meta.tags
    assert "content-browser" in note.meta.tags
    assert note.meta.extra["bug_ids"] == ["514040146", "987654321"]
    assert "issues.chromium.org/issues/514040146" in note.body


# --------------------------------------------------------------------------- #
# the `reference` type itself
# --------------------------------------------------------------------------- #


def test_prune_never_drops_a_reference_note():
    from sift.prune import classify as prune_classify
    from sift.vault.notes import Note
    from sift.vault.schema import Frontmatter

    note = Note(
        meta=Frontmatter(id="chromium-doc-x", type="reference", title="x"),
        body="the vendor's own wording",
    )
    verdict = prune_classify(note, keep_since_year=2024, report_quality_bar=60)
    assert verdict.keep is True


def test_reference_notes_outrank_bulk_material():
    from sift.quality import score_note
    from sift.vault.schema import Frontmatter

    ref = score_note(Frontmatter(id="a", type="reference", title="a"), "body")
    report = score_note(Frontmatter(id="b", type="report", title="b"), "body")
    assert ref > report
