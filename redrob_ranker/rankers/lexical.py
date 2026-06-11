"""Lexical baseline: BM25 over JD-derived keywords.

This is the "what keyword filters do today" floor. The query is the bag of terms a
naive recruiter would extract from the JD (the must-have/nice-to-have skill terms
plus the target titles). BM25 then rewards candidates whose text repeats those
terms -- which is exactly why it walks into the keyword-stuffer trap: a Marketing
Manager who lists every AI skill scores well here. It exists to be *beaten*, and to
quantify how much the structured/hybrid rankers improve over it.

Pure-Python/numpy, no external BM25 dependency.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..jdspec import JDSpec
from ..schema import Candidate
from .base import RankedCandidate, Ranker

_TOKEN = re.compile(r"[a-z0-9+#.]+")
_STOP = {
    "the", "and", "for", "with", "you", "your", "our", "are", "from", "this", "that",
    "have", "has", "was", "were", "will", "out", "not", "but", "all", "any", "can",
    "who", "via", "per", "into", "over", "a", "an", "of", "to", "in", "on", "at", "is",
    "we", "i", "my", "me", "it", "as", "or", "by", "be",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) >= 2 and t not in _STOP]


class LexicalRanker(Ranker):
    name = "lexical"

    def __init__(self, spec: JDSpec, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        # build the JD keyword query from the rubric's surface terms
        terms: list[str] = []
        for cap in spec.must_haves + spec.nice_to_haves:
            terms.extend(cap.skills)
            terms.extend(cap.evidence)
        terms.extend(spec.titles_core)
        terms.extend(spec.titles_adjacent)
        query_tokens = []
        for phrase in terms:
            query_tokens.extend(_tokenize(phrase))
        # unique query terms (a set; BM25 sums idf-weighted tf over them)
        self.query: list[str] = sorted(set(query_tokens))

    def rank(self, candidates: list[Candidate]) -> list[RankedCandidate]:
        docs = [Counter(_tokenize(c.full_text())) for c in candidates]
        lengths = [sum(d.values()) for d in docs]
        n = len(docs)
        avgdl = (sum(lengths) / n) if n else 1.0

        qset = set(self.query)
        df = Counter()
        for d in docs:
            for term in d.keys() & qset:
                df[term] += 1
        idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in qset if df[t] > 0}

        out = []
        for c, d, dl in zip(candidates, docs, lengths):
            score = 0.0
            denom_len = self.k1 * (1 - self.b + self.b * dl / avgdl) if avgdl else self.k1
            for t in qset:
                tf = d.get(t, 0)
                if tf:
                    score += idf.get(t, 0.0) * (tf * (self.k1 + 1)) / (tf + denom_len)
            out.append(RankedCandidate(c, score, {"lexical_raw": score}))
        # normalize to [0,1] for comparability with the other rankers
        mx = max((rc.score for rc in out), default=0.0) or 1.0
        for rc in out:
            rc.score = rc.score / mx
            rc.breakdown["lexical_norm"] = rc.score
        return out
