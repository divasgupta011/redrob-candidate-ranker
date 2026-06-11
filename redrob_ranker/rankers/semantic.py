"""Dense semantic ranker: cosine similarity between candidate and JD embeddings.

This catches *plain-language* fits -- candidates who clearly did the work but never
use the buzzwords ("built the recommendation engine" vs "RAG/Pinecone"). On its own
it is still trap-prone (a fluent stuffer embeds close to the JD too), which is why
the submission uses it only as one term inside the hybrid ranker.

It is deliberately decoupled from *how* the vectors are produced: it takes a JD
vector and a candidate embedding lookup (candidate_id -> vector). ``precompute.py``
(Step 9) builds those offline with a local sentence-transformer and saves them as a
plain .npy artifact, so this stays pure numpy and runs in the no-network rank step.
"""
from __future__ import annotations

import numpy as np

from ..schema import Candidate
from .base import RankedCandidate, Ranker


class SemanticRanker(Ranker):
    name = "semantic"

    def __init__(self, jd_vector: np.ndarray, embeddings: dict[str, np.ndarray]):
        v = np.asarray(jd_vector, dtype=np.float32)
        self.jd = v / (np.linalg.norm(v) + 1e-8)
        self.embeddings = embeddings

    def _sim(self, cid: str) -> float:
        vec = self.embeddings.get(cid)
        if vec is None:
            return 0.0
        v = np.asarray(vec, dtype=np.float32)
        cos = float(self.jd @ (v / (np.linalg.norm(v) + 1e-8)))
        return (cos + 1.0) / 2.0          # map [-1,1] -> [0,1]

    def rank(self, candidates: list[Candidate]) -> list[RankedCandidate]:
        out = []
        for c in candidates:
            s = self._sim(c.candidate_id)
            out.append(RankedCandidate(c, s, {"semantic": s}))
        return out
