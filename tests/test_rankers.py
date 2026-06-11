"""Ranker tests. The headline assertion: the lexical baseline ranks a keyword
stuffer at the top while the structured/hybrid rankers do not -- i.e. the whole
thesis of the project, encoded as a test."""
from __future__ import annotations

import numpy as np
import pytest

from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.rankers import (HybridRanker, LexicalRanker, SemanticRanker,
                                   StructuredRanker, build_ranker)
from redrob_ranker.schema import Candidate


@pytest.fixture(scope="module")
def spec():
    return load_jd_spec()


def _cand(cid, **parts) -> Candidate:
    d = {"candidate_id": cid, "profile": parts.pop("profile", {}),
         "career_history": parts.pop("career_history", []), "education": parts.pop("education", []),
         "skills": parts.pop("skills", []), "redrob_signals": parts.pop("redrob_signals", {})}
    return Candidate.from_dict(d)


def _real_engineer(cid="CAND_0000001"):
    return _cand(cid,
        profile={"current_title": "Machine Learning Engineer", "years_of_experience": 7,
                 "location": "Pune", "country": "India"},
        career_history=[{"company": "Flipkart", "title": "ML Engineer", "is_current": True,
                         "duration_months": 48, "industry": "Internet",
                         "description": "Built the search ranking and recommendation system; "
                                        "embeddings retrieval with FAISS, tracked NDCG in A/B tests."}],
        skills=[{"name": "Embeddings", "proficiency": "advanced", "endorsements": 25,
                 "duration_months": 40},
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 12,
                 "duration_months": 30}],
        redrob_signals={"last_active_date": "2026-05-20", "recruiter_response_rate": 0.7,
                        "open_to_work_flag": True, "notice_period_days": 20})


def _stuffer(cid="CAND_0000002"):
    """Marketing Manager who lists every AI keyword as a skill (0 months, no assessment)."""
    return _cand(cid,
        profile={"current_title": "Marketing Manager", "years_of_experience": 6,
                 "location": "Pune", "country": "India"},
        career_history=[{"company": "AdCo", "title": "Marketing Manager", "is_current": True,
                         "duration_months": 50, "industry": "Marketing",
                         "description": "Brand campaigns, budgets, growth marketing."}],
        skills=[{"name": s, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
                for s in ["Embeddings", "Retrieval", "Ranking", "FAISS", "Pinecone",
                          "Recommendation", "NLP", "Semantic Search", "Vector Database",
                          "Learning to Rank", "Information Retrieval", "Relevance"]],
        redrob_signals={"last_active_date": "2026-05-20", "recruiter_response_rate": 0.7,
                        "open_to_work_flag": True, "notice_period_days": 20})


def _plain_language_fit(cid="CAND_0000004"):
    """A genuine fit who describes the work in plain words and lists almost no
    buzzword skills -- the 'Tier-5' the JD warns keyword filters will miss."""
    return _cand(cid,
        profile={"current_title": "Software Engineer", "years_of_experience": 7,
                 "location": "Pune", "country": "India"},
        career_history=[{"company": "Flipkart", "title": "Software Engineer", "is_current": True,
                         "duration_months": 60, "industry": "Internet",
                         "description": "Built the recommendation system and search ranking that "
                                        "powers the product; improved relevance and validated it "
                                        "with online experiments."}],
        skills=[{"name": "Python", "proficiency": "advanced", "endorsements": 20,
                 "duration_months": 60}],
        redrob_signals={"last_active_date": "2026-05-20", "recruiter_response_rate": 0.7,
                        "open_to_work_flag": True, "notice_period_days": 20})


def test_lexical_falls_for_the_stuffer(spec):
    """The trap: BM25 ranks a keyword stuffer at/above a genuine plain-language fit,
    because it counts keyword occurrences and can't tell a listed skill from a
    described achievement."""
    cands = [_plain_language_fit(), _stuffer()]
    ranked = {rc.candidate.candidate_id: rc.score for rc in LexicalRanker(spec).rank(cands)}
    assert ranked["CAND_0000002"] >= ranked["CAND_0000004"]      # stuffer wins/ties -> trap


def test_structured_beats_the_stuffer(spec):
    """Structured reverses it: the plain-language fit wins decisively over the stuffer."""
    cands = [_plain_language_fit(), _stuffer()]
    ranked = {rc.candidate.candidate_id: rc.score for rc in StructuredRanker(spec).rank(cands)}
    assert ranked["CAND_0000004"] > ranked["CAND_0000002"]
    assert ranked["CAND_0000004"] > 2 * ranked["CAND_0000002"]   # decisively


def test_honeypot_forced_to_zero(spec):
    hp = _cand("CAND_0000003",
               profile={"current_title": "ML Engineer", "years_of_experience": 14},
               career_history=[{"company": "X", "title": "ML Engineer", "is_current": True,
                                "start_date": "2025-01-01", "end_date": None,
                                "duration_months": 16, "industry": "Internet",
                                "description": "ranking systems"}])
    rc = StructuredRanker(spec).score_one(hp)
    assert rc.score == 0.0 and rc.breakdown["honeypot"].is_honeypot


def test_semantic_orders_by_cosine(spec):
    jd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb = {"CAND_0000001": np.array([0.9, 0.1, 0.0]),   # close to JD
           "CAND_0000002": np.array([0.0, 1.0, 0.0])}   # orthogonal
    sem = SemanticRanker(jd, emb)
    scores = {rc.candidate.candidate_id: rc.score for rc in sem.rank([_real_engineer(),
                                                                      _stuffer()])}
    assert scores["CAND_0000001"] > scores["CAND_0000002"]


def test_hybrid_without_embeddings_equals_structured(spec):
    cands = [_real_engineer(), _stuffer()]
    s = {rc.candidate.candidate_id: rc.score for rc in StructuredRanker(spec).rank(cands)}
    h = {rc.candidate.candidate_id: rc.score for rc in HybridRanker(spec, semantic=None).rank(cands)}
    assert s == pytest.approx(h)


def test_hybrid_blends_semantic(spec):
    real = _real_engineer()
    jd = np.array([1.0, 0.0], dtype=np.float32)
    sem = SemanticRanker(jd, {real.candidate_id: np.array([1.0, 0.0])})  # perfect sim
    base = StructuredRanker(spec).score_one(real).score
    blended = HybridRanker(spec, semantic=sem).rank([real])[0]
    assert blended.breakdown["semantic"] == pytest.approx(1.0)
    assert blended.score >= base            # a real fit with high semantic sim is not hurt


def test_build_ranker_factory(spec):
    assert build_ranker("lexical", spec).name == "lexical"
    assert build_ranker("structured", spec).name == "structured"
    assert build_ranker("hybrid", spec).name == "hybrid"
    with pytest.raises(ValueError):
        build_ranker("semantic", spec)      # needs embeddings
