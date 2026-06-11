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

    def __init__(self, spec: JDSpec, use_honeypot: bool = True,
                 use_disqualifiers: bool = True, use_behavioral: bool = True):
        self.spec = spec
        # toggles exist so the eval harness can ablate each guard and measure its effect
        self.use_honeypot = use_honeypot
        self.use_disqualifiers = use_disqualifiers
        self.use_behavioral = use_behavioral

    def score_one(self, c: Candidate) -> RankedCandidate:
        f = extract_features(c, self.spec)
        dq = apply_disqualifiers(c, self.spec, f)
        bh = behavioral_modifier(c, self.spec)
        hp = detect_honeypot(c, self.spec)
        if self.use_honeypot and hp.is_honeypot:
            score = 0.0
        else:
            score = f.base_fit
            if self.use_disqualifiers:
                score *= dq.multiplier
            if self.use_behavioral:
                score *= bh.modifier
        return RankedCandidate(c, score, {
            "features": f, "dq": dq, "behavioral": bh, "honeypot": hp,
            "fit": f.base_fit,
        })

    def rank(self, candidates: list[Candidate]) -> list[RankedCandidate]:
        return [self.score_one(c) for c in candidates]
