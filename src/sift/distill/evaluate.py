"""Score gate designs against hand-labelled verdicts.

The gate decides what a thousand articles become, so which design to ship is an
empirical question, not an argument. This runs each design over the labelled set and
reports agreement, plus the specific disagreements - those are what you read.

Labels live in `tests/fixtures/gate_labels.json` (url -> keep/drop), hand-assigned
over the first PortSwigger batch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sift.config import PROJECT_ROOT
from sift.distill.candidates import Candidate, best_copies, url_key
from sift.distill.gate import GateConfigError, Verdict, run_fatal
from sift.vault.notes import iter_notes

LABELS_PATH = PROJECT_ROOT / "tests" / "fixtures" / "gate_labels.json"

# A 400 can be specific to one candidate's content, so one is logged and skipped. Three
# in a row means the request itself is wrong (a bad model id or parameter), and the
# rest of the run would fail the same way.
MAX_CONSECUTIVE_BAD_REQUESTS = 3


@dataclass
class Score:
    name: str
    agree: int = 0
    total: int = 0
    false_keeps: list[str] = field(default_factory=list)  # gate kept, human dropped
    false_drops: list[str] = field(default_factory=list)  # gate dropped, human kept
    errors: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.agree / self.total if self.total else 0.0


def load_labels(path: Path | None = None) -> dict[str, str]:
    raw = json.loads((path or LABELS_PATH).read_text(encoding="utf-8"))
    return {r["url"]: r["decision"] for r in raw if r.get("url")}


def labelled_candidates(labels: dict[str, str], vault: Path) -> list[tuple[Candidate, str]]:
    """Pair each label with the copy of its article that production would gate.

    Same selection as `manual.collect` - the longest body per url key - but with no
    gated-url filter (the labelled articles are already judged, so filtering would empty
    the set) and no prefilter (eval scores the gate alone). Keying on the raw url used
    to score 5 of the 40 labelled articles on a different, shorter copy than production
    would send.
    """
    best, _ = best_copies(iter_notes(vault, note_type="writeup"))
    out = []
    for url, decision in labels.items():
        cand = best.get(url_key(url))
        if cand is not None:
            out.append((cand, decision))
    return out


def _is_bad_request(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 400


def score_gate(
    name: str,
    judge: Callable[..., Verdict],
    pairs: list[tuple[Candidate, str]],
    *,
    client=None,
    on_result: Callable[[str, Candidate, Verdict | None, str], None] | None = None,
) -> Score:
    """Run `judge` over the labelled pairs.

    One bad candidate never voids the run, but a run-fatal error (credentials, billing,
    an unknown model, a bad effort setting, or repeated 400s) raises GateConfigError on
    the spot - otherwise every remaining call fails identically and the report reads
    0/0.
    """
    s = Score(name=name)
    bad_requests = 0
    for cand, human in pairs:
        try:
            verdict = judge(cand, client=client)
        except Exception as exc:  # one bad candidate must not void the run
            fatal = run_fatal(exc)
            if fatal:
                raise GateConfigError(f"{name}: {fatal}") from exc
            bad_requests = bad_requests + 1 if _is_bad_request(exc) else 0
            if bad_requests >= MAX_CONSECUTIVE_BAD_REQUESTS:
                raise GateConfigError(
                    f"{name}: {bad_requests} consecutive 400 Bad Request replies - the "
                    f"request itself is invalid (check SIFT_GATE_MODEL): {exc}"
                ) from exc
            s.errors.append(f"{cand.title[:60]}: {exc}")
            if on_result:
                on_result(name, cand, None, human)
            continue
        bad_requests = 0
        s.total += 1
        got = "keep" if verdict.keep else "drop"
        if got == human:
            s.agree += 1
        elif got == "keep":
            s.false_keeps.append(f"{cand.title[:70]} :: {verdict.justification[:110]}")
        else:
            s.false_drops.append(f"{cand.title[:70]} :: {verdict.justification[:110]}")
        if on_result:
            on_result(name, cand, verdict, human)
    return s


def format_report(scores: list[Score]) -> str:
    lines = ["", "=" * 72, "GATE EVALUATION vs hand labels", "=" * 72, ""]
    for s in scores:
        lines.append(f"{s.name:22} {s.agree}/{s.total} = {s.accuracy:.0%}")
    lines.append("")
    for s in scores:
        lines.append(f"--- {s.name} ---")
        # A false drop is the expensive error: the material never reaches the vault
        # and nothing records that it was lost.
        lines.append(f"  false DROPS ({len(s.false_drops)}) - lost material, the costly error:")
        lines += [f"    - {d}" for d in s.false_drops] or ["    (none)"]
        lines.append(f"  false KEEPS ({len(s.false_keeps)}) - dilutes ranking:")
        lines += [f"    - {k}" for k in s.false_keeps] or ["    (none)"]
        if s.errors:
            lines.append(f"  errors ({len(s.errors)}):")
            lines += [f"    ! {e}" for e in s.errors]
        lines.append("")
    return "\n".join(lines)
