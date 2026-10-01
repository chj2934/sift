"""Novelty-gated distillation: keep only what the reasoning model doesn't already know.

The KB cannot make Opus smarter — it can only supply facts Opus lacks. So the
filter is the product: `gate.judge` rejects anything the model can already
explain, and only survivors become `technique` notes.
"""

from sift.distill.candidates import Candidate
from sift.distill.gate import KEEP_REASONS, GateConfigError, GateError, Verdict, judge

__all__ = ["Candidate", "GateConfigError", "GateError", "KEEP_REASONS", "Verdict", "judge"]
