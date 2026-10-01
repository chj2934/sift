"""Guards against storing pages the extractor failed on.

Each rule here exists because a real note in the vault was wrong. The measured values
in the comments are from that corpus, not invented.
"""

from __future__ import annotations

from datetime import date

import pytest

# --- extraction failure detection -------------------------------------------------
# Measured: real articles extract at 0.062-0.374 of raw HTML; known failures
# (soroush.me, labs.hakaioffsec.com, tracebit.com) at 0.0021-0.0085.


def test_js_rendered_page_is_detected():
    from sift.ingest.research import extraction_failed

    # labs.hakaioffsec.com: 377542 raw -> 801 chars of Portuguese nav menu.
    assert extraction_failed("x" * 801, "y" * 377542)
    # soroush.me: 229424 -> 815, and what came out was the blog index.
    assert extraction_failed("x" * 815, "y" * 229424)
    # tracebit.com: 58579 -> 500.
    assert extraction_failed("x" * 500, "y" * 58579)


def test_real_articles_pass():
    from sift.ingest.research import extraction_failed

    assert not extraction_failed("x" * 9981, "y" * 48216)  # portswigger, 0.207
    assert not extraction_failed("x" * 7343, "y" * 118663)  # doyensec, 0.062 - worst good
    assert not extraction_failed("x" * 15697, "y" * 41935)  # watchtowr, 0.374


def test_short_but_dense_page_passes():
    """A genuinely short post on a light page is not an extraction failure."""
    from sift.ingest.research import extraction_failed

    assert not extraction_failed("x" * 900, "y" * 6000)  # 0.15


def test_long_extract_is_exempt_from_the_ratio():
    """A long article inside a heavy SPA bundle got enough text out to be usable."""
    from sift.ingest.research import extraction_failed

    assert not extraction_failed("x" * 12000, "y" * 5_000_000)


def test_empty_raw_is_a_failure():
    from sift.ingest.research import extraction_failed

    assert extraction_failed("", "")


# --- publication date -------------------------------------------------------------
# Without this, every top10 note got 1 Jan of its nomination year: the Assetnote React
# Native article (February 2020) was stamped 2024-01-01 and ranked as recent.


def test_meta_published_time_wins():
    from sift.ingest.top10 import article_date

    html = '<meta property="article:published_time" content="2020-02-01T10:00:00Z">'
    assert article_date(html, 2024) == date(2020, 2, 1)


def test_json_ld_date_published():
    from sift.ingest.top10 import article_date

    assert article_date('"datePublished":"2021-11-30"', 2024) == date(2021, 11, 30)


def test_time_element_datetime():
    from sift.ingest.top10 import article_date

    assert article_date('<time datetime="2019-06-05">June 5</time>', 2024) == date(2019, 6, 5)


def test_falls_back_to_nomination_year():
    from sift.ingest.top10 import article_date

    assert article_date("<html>no dates here</html>", 2024) == date(2024, 1, 1)


@pytest.mark.parametrize("bad", ['<time datetime="1970-01-01">', '"datePublished":"2099-01-01"'])
def test_implausible_dates_are_rejected(bad):
    """The web is full of stray dates in unrelated markup - bound them."""
    from sift.ingest.top10 import article_date

    assert article_date(bad, 2024) == date(2024, 1, 1)


def test_malformed_date_does_not_raise():
    from sift.ingest.top10 import article_date

    assert article_date('<time datetime="2024-13-45">', 2024) == date(2024, 1, 1)


# --- link filtering regressions ---------------------------------------------------


def test_assetnote_advisories_are_not_filtered_by_length():
    """174 real Assetnote advisories extract to short lines. An earlier 'needs 3 long
    lines' rule would have deleted every one of them."""
    from sift.ingest.top10 import looks_like_prose

    advisory = "\n".join(["Advisory: Metabase Pre-Auth RCE (CVE-2023-38646)"] * 40)
    assert looks_like_prose(advisory)
