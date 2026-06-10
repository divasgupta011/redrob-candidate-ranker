"""Tests for feature extraction and disqualifiers -- focused on the behaviours that
make or break the ranking: corroboration beats keyword stuffing, career evidence
rescues plain-language fits, and disqualifiers fire precisely (not bluntly)."""
from __future__ import annotations

import pytest

from redrob_ranker.disqualifiers import apply_disqualifiers
from redrob_ranker.features import (
    education_fit, experience_fit, extract_features, location_fit, skill_trust,
)
from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.schema import Candidate, Skill


@pytest.fixture(scope="module")
def spec():
    return load_jd_spec()


def _cand(**profile_and_parts) -> Candidate:
    """Build a minimal candidate dict -> Candidate."""
    d = {
        "candidate_id": profile_and_parts.pop("candidate_id", "CAND_0000000"),
        "profile": profile_and_parts.pop("profile", {}),
        "career_history": profile_and_parts.pop("career_history", []),
        "education": profile_and_parts.pop("education", []),
        "skills": profile_and_parts.pop("skills", []),
        "redrob_signals": profile_and_parts.pop("redrob_signals", {}),
    }
    return Candidate.from_dict(d)


# ---- skill corroboration (the anti-stuffer core) --------------------------

def test_skill_trust_floors_a_stuffer(spec):
    stuffer = Skill("Embeddings", "expert", endorsements=0, duration_months=0)
    t = skill_trust(stuffer, assessment_lc={}, cfg=spec.skill_trust)
    assert t <= spec.skill_trust["base"] + 0.02   # no corroboration -> floor


def test_skill_trust_rewards_corroboration(spec):
    real = Skill("Embeddings", "advanced", endorsements=30, duration_months=30)
    t = skill_trust(real, assessment_lc={"embeddings": 85.0}, cfg=spec.skill_trust)
    assert t > 0.9


def test_expert_claim_alone_does_not_buy_trust(spec):
    """An 'expert' claim with 0 months used must not beat a used 'intermediate'."""
    claimed = Skill("Embeddings", "expert", endorsements=0, duration_months=0)
    used = Skill("Embeddings", "intermediate", endorsements=10, duration_months=24)
    assert skill_trust(used, {}, spec.skill_trust) > skill_trust(claimed, {}, spec.skill_trust)


# ---- capability coverage / evidence ---------------------------------------

def test_career_evidence_rescues_plain_language_fit(spec):
    """No buzzword skills, but the work history clearly describes building a
    recommendation system -> the capability should still be credited."""
    c = _cand(
        profile={"current_title": "Software Engineer", "years_of_experience": 7,
                 "location": "Pune", "country": "India"},
        career_history=[{
            "company": "ShopCo", "title": "Software Engineer", "is_current": True,
            "duration_months": 40, "industry": "Internet",
            "description": "Built the recommendation system and search ranking that powers the "
                           "product feed; ran A/B tests and tracked NDCG offline.",
        }],
        skills=[{"name": "Python", "proficiency": "advanced", "endorsements": 5,
                 "duration_months": 60}],
    )
    f = extract_features(c, spec)
    rec = next(m for m in f.must_have_matches if m.id == "ranking_search_rec_systems")
    assert rec.credit >= 0.85
    assert f.evidence_strength > 0.0


def test_keyword_stuffer_marketing_manager_scores_low(spec):
    """All the AI skills listed, but a Marketing Manager title and a marketing
    career with zero corroboration -> low fit (the planted trap)."""
    c = _cand(
        profile={"current_title": "Marketing Manager", "years_of_experience": 6,
                 "location": "Pune", "country": "India"},
        career_history=[{"company": "AdCo", "title": "Marketing Manager", "is_current": True,
                         "duration_months": 50, "industry": "Marketing",
                         "description": "Ran campaigns, managed budgets and brand strategy."}],
        skills=[{"name": s, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
                for s in ["Embeddings", "Retrieval", "Ranking", "NLP", "FAISS", "Pinecone"]],
    )
    f = extract_features(c, spec)
    assert f.title_career_fit < 0.2
    assert f.must_have_coverage < 0.35     # listed-but-uncorroborated earns little
    assert f.base_fit < 0.3


def test_experience_and_location_bands(spec):
    assert experience_fit(_cand(profile={"years_of_experience": 7}), spec) == 1.0   # ideal
    assert experience_fit(_cand(profile={"years_of_experience": 9}), spec) == 0.85  # band edge
    assert experience_fit(_cand(profile={"years_of_experience": 1}), spec) < 0.6    # too junior
    assert location_fit(_cand(profile={"location": "Pune", "country": "India"}), spec) == 1.0
    assert location_fit(_cand(profile={"location": "Berlin", "country": "Germany"}), spec) <= 0.4


# ---- disqualifiers fire precisely -----------------------------------------

def test_consulting_only_fires_then_exempts_on_product_role(spec):
    services = [{"company": "Infosys", "title": "Engineer", "duration_months": 36,
                 "industry": "IT Services", "is_current": True, "description": "delivery work"},
                {"company": "Wipro", "title": "Engineer", "duration_months": 36,
                 "industry": "IT Services", "is_current": False, "description": "delivery work"}]
    c = _cand(profile={"current_title": "Engineer", "years_of_experience": 6},
              career_history=services)
    f = extract_features(c, spec)
    assert "consulting_only" in apply_disqualifiers(c, spec, f).fired

    with_product = services + [{"company": "Flipkart", "title": "Engineer", "duration_months": 24,
                                "industry": "Internet", "is_current": False,
                                "description": "product engineering"}]
    c2 = _cand(profile={"current_title": "Engineer", "years_of_experience": 8},
               career_history=with_product)
    f2 = extract_features(c2, spec)
    assert "consulting_only" not in apply_disqualifiers(c2, spec, f2).fired


def test_title_chaser_requires_escalation(spec):
    hop = lambda title, cur=False: {"company": f"Co{title}", "title": title,
                                    "duration_months": 14, "industry": "Internet",
                                    "is_current": cur, "description": "ml work"}
    # short tenure WITH escalation junior -> senior -> principal => fires
    climbing = _cand(profile={"current_title": "Principal Engineer", "years_of_experience": 5},
                     career_history=[hop("Principal Engineer", True), hop("Senior Engineer"),
                                     hop("Junior Engineer")])
    f = extract_features(climbing, spec)
    assert "title_chaser" in apply_disqualifiers(climbing, spec, f).fired

    # short tenure but NO escalation (stays "Engineer") => does not fire
    flat = _cand(profile={"current_title": "ML Engineer", "years_of_experience": 5},
                 career_history=[hop("ML Engineer", True), hop("ML Engineer"), hop("ML Engineer")])
    f2 = extract_features(flat, spec)
    assert "title_chaser" not in apply_disqualifiers(flat, spec, f2).fired


def test_cv_without_nlp_fires(spec):
    c = _cand(profile={"current_title": "Computer Vision Engineer", "years_of_experience": 7},
              career_history=[{"company": "Visio", "title": "CV Engineer", "is_current": True,
                               "duration_months": 60, "industry": "Internet",
                               "description": "object detection and image segmentation models"}],
              skills=[{"name": "Computer Vision", "proficiency": "expert",
                       "endorsements": 20, "duration_months": 60}])
    f = extract_features(c, spec)
    assert "cv_speech_robotics_no_nlp" in apply_disqualifiers(c, spec, f).fired
