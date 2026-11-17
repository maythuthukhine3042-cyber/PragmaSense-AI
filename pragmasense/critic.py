from __future__ import annotations
"""Tier 3: Metacognitive pragmatic critic (System 2).

Compares the register profile of the Thai source with that of a Myanmar
hypothesis, computes the affective divergence

    delta = | I_target - I_generated |

and writes a natural-language critique when the hypothesis is rejected.
"""
from dataclasses import dataclass, field


@dataclass
class Verdict:
    accepted: bool
    delta: float
    target_level: float
    generated_level: float
    missing_functions: list = field(default_factory=list)
    gender_error: str | None = None
    found_markers: list = field(default_factory=list)
    critique: str = ""

    def score(self):
        """Lower is better. Used by the convergence gate to compare hypotheses."""
        n_errors = len(self.missing_functions) + (1 if self.gender_error else 0)
        return (n_errors, round(self.delta, 3))


class PragmaticCritic:
    def __init__(self, kb, threshold=None):
        self.kb = kb
        self.threshold = kb.threshold if threshold is None else threshold

    def audit(self, target_profile, hypothesis):
        gen = self.kb.analyse_myanmar(hypothesis)
        delta = abs(target_profile.level - gen.level)
        missing = sorted(target_profile.functions - gen.functions)

        gender_error = None
        if target_profile.gender in ("male", "female") and gen.gender not in ("unknown", target_profile.gender):
            gender_error = (f"speaker is {target_profile.gender} but the translation uses a "
                            f"{'female' if target_profile.gender == 'male' else 'male'} politeness marker")

        accepted = delta <= self.threshold and not missing and gender_error is None
        v = Verdict(accepted=accepted, delta=delta, target_level=target_profile.level,
                    generated_level=gen.level, missing_functions=missing,
                    gender_error=gender_error, found_markers=gen.marker_texts())
        if not accepted:
            v.critique = self._write_critique(target_profile, gen, v)
        return v

    def _write_critique(self, target, gen, v):
        lines = []
        src = ", ".join(target.marker_texts()) or "none"
        out = ", ".join(gen.marker_texts()) or "none"
        lines.append(f"The Thai source has register level {target.level:.1f}/4 (particles: {src}). "
                     f"Your translation has level {gen.level:.1f}/4 (Myanmar markers found: {out}). "
                     f"Divergence = {v.delta:.2f}.")
        if gen.level < target.level - self.threshold:
            lines.append("The translation sounds more blunt / less polite than the source.")
        elif gen.level > target.level + self.threshold:
            msg = "The translation sounds more formal / polite than the source, which is rougher or more intimate."
            if gen.markers:
                msg += " Remove politeness markers that do not match the source tone."
            lines.append(msg)
        for fn in v.missing_functions:
            sug = ", ".join(self.kb.suggest(fn, target.gender))
            lines.append(f"Missing pragmatic function '{fn}'. Consider markers such as: {sug}.")
        if v.gender_error:
            sug = ", ".join(self.kb.suggest("deference", target.gender))
            lines.append(f"Gender mismatch: {v.gender_error}. Use {sug} instead.")
        return " ".join(lines)
