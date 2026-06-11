"""Pluggable ranking strategies.

Every ranker implements the same interface (see ``base.Ranker``) so they can be
raced against each other by the evaluation harness:

    lexical    - BM25 over JD keywords (the naive baseline; falls for stuffers/honeypots)
    semantic   - dense embedding cosine similarity to the JD
    structured - the JD-rubric scorer (title/career evidence, trust-weighted skills,
                 disqualifiers, honeypot filter, behavioural modifier) -- the decisive layer
    hybrid     - semantic recall blended with structured scoring (the submission ranker)
"""
from .base import RankedCandidate, Ranker
from .hybrid import HybridRanker
from .keyword_count import SkillCountRanker
from .lexical import LexicalRanker
from .semantic import SemanticRanker
from .structured import StructuredRanker

__all__ = ["Ranker", "RankedCandidate", "SkillCountRanker", "LexicalRanker",
           "SemanticRanker", "StructuredRanker", "HybridRanker", "build_ranker"]


def build_ranker(name: str, spec, semantic: SemanticRanker | None = None) -> Ranker:
    """Factory used by the CLI/eval harness. ``semantic`` (if given) is reused for
    the hybrid blend; the baselines ignore it."""
    name = name.lower()
    if name == "skill_count":
        return SkillCountRanker(spec)
    if name == "lexical":
        return LexicalRanker(spec)
    if name == "structured":
        return StructuredRanker(spec)
    if name == "semantic":
        if semantic is None:
            raise ValueError("semantic ranker needs precomputed embeddings (run precompute.py)")
        return semantic
    if name == "hybrid":
        return HybridRanker(spec, semantic=semantic)
    raise ValueError(f"unknown ranker: {name!r}")
