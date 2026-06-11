"""Hybrid ranker -- the submission strategy.

Combines the two complementary signals:

  * structured fit (robust to keyword stuffers and honeypots), and
  * semantic similarity (recall for plain-language fits with no buzzwords),

by blending semantic into the *fit estimate only*, then applying the same
disqualifier / behavioral / honeypot machinery as the structured ranker:

    fit'   = (1 - sem_w) * base_fit + sem_w * semantic_sim
    score  = 0 if honeypot else fit' x disqualifier x behavioral

Blending into fit (rather than the final score) is deliberate: it lets semantic
*raise* an under-described real candidate, but it can never rescue a honeypot or a
disqualified profile, because those guards are applied afterwards. ``sem_w`` comes
from ``weights.semantic`` in the spec (small by design). With no embeddings
provided the semantic term is simply absent and this reduces to the structured
ranker -- so the rank step never fails for lack of the precomputed artifact.
"""
from __future__ import annotations

from ..jdspec import JDSpec
from ..schema import Candidate
from .base import RankedCandidate, Ranker
from .semantic import SemanticRanker
from .structured import StructuredRanker


class HybridRanker(Ranker):
    name = "hybrid"

    def __init__(self, spec: JDSpec, semantic: SemanticRanker | None = None):
        self.spec = spec
        self.structured = StructuredRanker(spec)
        self.semantic = semantic
        self.sem_w = float(spec.weights.get("semantic", 0.05)) if semantic is not None else 0.0

    def rank(self, candidates: list[Candidate]) -> list[RankedCandidate]:
        out = []
        for c in candidates:
            rc = self.structured.score_one(c)
            hp = rc.breakdown["honeypot"]
            base_fit = rc.breakdown["fit"]
            sem = self.semantic._sim(c.candidate_id) if self.semantic is not None else 0.0
            fit = (1.0 - self.sem_w) * base_fit + self.sem_w * sem
            if hp.is_honeypot:
                score = 0.0
            else:
                dq = rc.breakdown["dq"]
                bh = rc.breakdown["behavioral"]
                score = fit * dq.multiplier * bh.modifier
            rc.score = score
            rc.breakdown["semantic"] = sem
            rc.breakdown["fit_blended"] = fit
            out.append(rc)
        return out
