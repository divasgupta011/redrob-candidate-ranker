"""Ranking metrics matching the contest's scoring (spec section 4).

Composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10
"relevant" for P@10 / MAP means relevance tier >= 3.
"""
from __future__ import annotations

import math

RELEVANT_TIER = 3
COMPOSITE_WEIGHTS = {"ndcg@10": 0.50, "ndcg@50": 0.30, "map": 0.15, "p@10": 0.05}


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int) -> float:
    gains = [relevance.get(cid, 0) for cid in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int,
                   threshold: int = RELEVANT_TIER) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if relevance.get(cid, 0) >= threshold)
    return hits / k


def average_precision(ranked_ids: list[str], relevance: dict[str, float],
                      threshold: int = RELEVANT_TIER) -> float:
    total_relevant = sum(1 for v in relevance.values() if v >= threshold)
    if total_relevant == 0:
        return 0.0
    hits, running = 0, 0.0
    for i, cid in enumerate(ranked_ids, start=1):
        if relevance.get(cid, 0) >= threshold:
            hits += 1
            running += hits / i
    return running / min(total_relevant, len(ranked_ids))


def composite(ranked_ids: list[str], relevance: dict[str, float]) -> dict[str, float]:
    m = {
        "ndcg@10": ndcg_at_k(ranked_ids, relevance, 10),
        "ndcg@50": ndcg_at_k(ranked_ids, relevance, 50),
        "map": average_precision(ranked_ids, relevance),
        "p@10": precision_at_k(ranked_ids, relevance, 10),
    }
    m["composite"] = sum(COMPOSITE_WEIGHTS[k] * m[k] for k in COMPOSITE_WEIGHTS)
    return m
