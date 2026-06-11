"""A JD-derived *proxy* gold relevance labelling (tiers 0-5).

The contest hides the real ground truth, so we reconstruct the JD's intent as a
relevance label. To keep this from being circular with the full ranker, the gold uses
only the JD's *headline* signals -- title family, career evidence, experience band,
basic availability, and honeypot status -- NOT the ranker's finer machinery
(skill-trust corroboration, location weighting, education, the exact behavioural
formula). Crucially it labels a *negative-title* profile low **regardless of its skills
list**, which is precisely where keyword baselines diverge from the JD's intent.

Caveats (stated honestly in the deck): this is a proxy, useful for *relative*
comparison; the one fully objective metric is honeypot-rate, which the contest itself
uses as a Stage-3 disqualifier.
"""
from __future__ import annotations

from redrob_ranker.honeypot import detect_honeypot
from redrob_ranker.jdspec import JDSpec
from redrob_ranker.schema import Candidate

_BASE_BY_FAMILY = {"core": 3, "adjacent": 2, "other": 1, "negative": 0}

_EVIDENCE_PHRASES = (
    "recommendation system", "recommendation engine", "recommender", "recsys",
    "search ranking", "ranking system", "search system", "search engine",
    "retrieval", "relevance", "personalization", "matching system", "learning to rank",
)


def gold_tier(c: Candidate, spec: JDSpec) -> int:
    if detect_honeypot(c, spec).is_honeypot:
        return 0
    tier = _BASE_BY_FAMILY[spec.title_family(c.current_title_lc)]

    career = c.career_text().lower()
    if any(p in career for p in _EVIDENCE_PHRASES):
        tier += 1

    y = c.years_of_experience
    if spec.exp_ideal_min <= y <= spec.exp_ideal_max:
        tier += 1
    elif not (spec.exp_min <= y <= spec.exp_max):
        tier -= 1

    # basic availability: a totally unreachable profile is worth less
    sig = c.signals
    if sig.recruiter_response_rate < 0.1 and not sig.open_to_work_flag:
        tier -= 1

    return max(0, min(5, tier))


def gold_relevance(candidates: list[Candidate], spec: JDSpec) -> dict[str, int]:
    return {c.candidate_id: gold_tier(c, spec) for c in candidates}
