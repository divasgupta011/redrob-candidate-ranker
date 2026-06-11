"""Skill-keyword-count baseline -- the naivest possible ranker, and the one the
organizers' deliberately-bad ``sample_submission.csv`` effectively uses.

It scores each candidate purely by how many of the JD's must-have/nice-to-have skill
terms appear in their **skills list** -- ignoring title, career evidence, corroboration,
honeypots and behaviour entirely. This is the ranker that the dataset's keyword
stuffers are designed to beat: a Marketing Manager who lists 9 AI skills scores at the
top. We include it to make the keyword trap vivid in the evaluation.
"""
from __future__ import annotations

from ..jdspec import JDSpec
from ..schema import Candidate
from .base import RankedCandidate, Ranker


class SkillCountRanker(Ranker):
    name = "skill_count"

    def __init__(self, spec: JDSpec):
        # the set of JD skill surface-terms to count
        terms: set[str] = set()
        for cap in spec.must_haves + spec.nice_to_haves:
            terms.update(cap.skills)
        self.terms = terms

    def _count(self, c: Candidate) -> int:
        hits = 0
        for sk in c.skill_names_lc:
            if any(term == sk or term in sk or sk in term for term in self.terms):
                hits += 1
        return hits

    def rank(self, candidates: list[Candidate]) -> list[RankedCandidate]:
        out = [RankedCandidate(c, float(self._count(c)), {}) for c in candidates]
        mx = max((rc.score for rc in out), default=0.0) or 1.0
        for rc in out:
            rc.score /= mx        # normalize to [0,1] for comparability
        return out
