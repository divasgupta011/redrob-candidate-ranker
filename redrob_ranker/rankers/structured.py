"""The structured ranker -- the decisive strategy.

Composes everything built in Steps 2-4 into one score:

    honeypot?  -> 0   (forced below every real candidate)
    otherwise  -> base_fit  x  disqualifier_multiplier  x  behavioral_modifier

base_fit already weights title/career fit, corroborated must-haves, experience,
location and education (features.py). This is the layer that resists keyword
stuffers and surfaces plain-language fits; the hybrid ranker reuses it verbatim
and merely blends a semantic term into the fit estimate.
"""
from __future__ import annotations

from ..behavioral import behavioral_modifier
from ..disqualifiers import apply_disqualifiers
from ..features import extract_features
from ..honeypot import detect_honeypot
from ..jdspec import JDSpec
from ..schema import Candidate
from .base import RankedCandidate, Ranker


class StructuredRanker(Ranker):
    name = "structured"

    def __init__(self, spec: JDSpec):
        self.spec = spec

    def score_one(self, c: Candidate) -> RankedCandidate:
        f = extract_features(c, self.spec)
        dq = apply_disqualifiers(c, self.spec, f)
        bh = behavioral_modifier(c, self.spec)
        hp = detect_honeypot(c, self.spec)
        score = 0.0 if hp.is_honeypot else f.base_fit * dq.multiplier * bh.modifier
        return RankedCandidate(c, score, {
            "features": f, "dq": dq, "behavioral": bh, "honeypot": hp,
            "fit": f.base_fit,
        })

    def rank(self, candidates: list[Candidate]) -> list[RankedCandidate]:
        return [self.score_one(c) for c in candidates]
