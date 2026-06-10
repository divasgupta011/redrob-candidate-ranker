"""Honeypot / internal-consistency detection.

The pool contains ~80 honeypots: profiles that are *internally impossible* rather
than merely weak (spec section 7 examples: "8 years of experience at a company
founded 3 years ago", "'expert' proficiency in 10 skills with 0 years used").
They're forced to relevance tier 0, and a submission with a honeypot rate > 10%
in its top 100 is disqualified.

The detector looks for **contradictions between fields**, not for poor fit. Two
design rules:

  * Precision over recall. A false positive evicts a genuine star and costs NDCG,
    so every "hard" check requires a real contradiction plus a generous buffer to
    absorb rounding/data noise. We don't need to catch all 80 -- only to keep them
    out of the top 100, and a honeypot only reaches the top 100 if it *looks*
    strong, which is exactly the kind built from these impossibilities.
  * The spec says a good ranker should "naturally avoid them"; we don't special-
    case ids, we just refuse to believe impossible profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .jdspec import JDSpec
from .schema import Candidate

_DEFAULT_REF = date(2026, 6, 1)


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


@dataclass(slots=True)
class HoneypotResult:
    is_honeypot: bool
    score: float                                  # higher = more contradictions
    reasons: list[str] = field(default_factory=list)


def detect_honeypot(c: Candidate, spec: JDSpec) -> HoneypotResult:
    ref = spec.reference_date or _DEFAULT_REF
    reasons: list[str] = []
    hard = 0
    soft = 0
    yoe = c.years_of_experience

    # A) per-role tenure exceeds the time actually elapsed since the role began
    #    (the canonical "8 years at a 3-year-old company" signature)
    for r in c.career_history:
        if not r.start_date:
            continue
        end = r.end_date or ref
        elapsed = _months_between(r.start_date, end)
        if elapsed < -1:
            reasons.append(f"role '{r.title} @ {r.company}' ends before it starts")
            hard += 1
        elif r.duration_months > elapsed + 18:
            reasons.append(f"role '{r.title} @ {r.company}': claims {r.duration_months}mo tenure "
                           f"but only {max(elapsed, 0)}mo elapsed since it began")
            hard += 1

    # B) stated total experience exceeds time since the earliest role began
    starts = [r.start_date for r in c.career_history if r.start_date]
    if starts:
        span_years = (ref - min(starts)).days / 365.25
        if yoe > span_years + 6:
            reasons.append(f"claims {yoe:.1f}y experience but earliest role began only "
                           f"{span_years:.1f}y ago")
            hard += 1

    # C) many high-proficiency skills with zero months of use
    #    ("'expert' proficiency in 10 skills with 0 years used")
    zero_expert = [s.name for s in c.skills if s.proficiency_rank >= 3 and s.duration_months == 0]
    if len(zero_expert) >= 5:
        reasons.append(f"{len(zero_expert)} expert/advanced skills with 0 months of use")
        hard += 1

    # (NB: we deliberately do NOT flag "skill duration > years of experience" -- a
    #  skill is commonly used before one's first job, e.g. Python learned at college,
    #  so that is normal data, not an impossible profile. An early version of this
    #  detector flagged ~2,800 candidates that way; removed.)

    # D) education that ends before it begins
    for e in c.education:
        if e.start_year and e.end_year and e.end_year < e.start_year:
            reasons.append(f"education '{e.degree}' ends ({e.end_year}) before it starts "
                           f"({e.start_year})")
            hard += 1

    # (NB: we also do NOT flag "last_active before signup". It's logically impossible,
    #  but this synthetic dataset generates the two signal dates independently, so ~7.5%
    #  of *normal* candidates trip it -- a data-generation artifact, not a honeypot.)

    # --- soft signals: corroborate but never flag on their own --------------
    total_years = sum(max(r.duration_months, 0) for r in c.career_history) / 12.0
    if len(c.career_history) >= 2 and total_years > yoe * 1.5 + 4:
        reasons.append(f"summed tenure ({total_years:.1f}y) far exceeds stated experience "
                       f"({yoe:.1f}y)")
        soft += 1
    if any(r.is_current and r.end_date for r in c.career_history):
        reasons.append("a role marked 'current' also carries an end date")
        soft += 1

    is_hp = hard >= 1 or soft >= 2
    return HoneypotResult(is_honeypot=is_hp, score=hard + 0.4 * soft, reasons=reasons)
