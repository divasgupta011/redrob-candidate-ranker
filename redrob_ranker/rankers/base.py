"""Common ranker interface.

Every strategy takes the candidate pool and returns a score per candidate, so the
evaluation harness can race them on identical input. The ``breakdown`` dict carries
strategy-specific diagnostics (features, disqualifiers, behavioural, honeypot,
similarity) that the reasoning generator later turns into prose -- computed once,
here, never recomputed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..schema import Candidate


@dataclass(slots=True)
class RankedCandidate:
    candidate: Candidate
    score: float
    breakdown: dict = field(default_factory=dict)


class Ranker(ABC):
    name: str = "base"

    @abstractmethod
    def rank(self, candidates: list[Candidate]) -> list[RankedCandidate]:
        """Score every candidate. Order of the returned list is not significant;
        the submission writer sorts and applies the tie-break."""
        raise NotImplementedError
