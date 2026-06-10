"""Honeypot detector tests: each planted impossibility is caught, and a normal
profile (including the legitimate 'skill used longer than years employed' case) is
NOT flagged -- precision is what protects NDCG."""
from __future__ import annotations

import pytest

from redrob_ranker.honeypot import detect_honeypot
from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.schema import Candidate


@pytest.fixture(scope="module")
def spec():
    return load_jd_spec()


def _cand(**parts) -> Candidate:
    d = {
        "candidate_id": parts.pop("candidate_id", "CAND_0000000"),
        "profile": parts.pop("profile", {}),
        "career_history": parts.pop("career_history", []),
        "education": parts.pop("education", []),
        "skills": parts.pop("skills", []),
        "redrob_signals": parts.pop("redrob_signals", {}),
    }
    return Candidate.from_dict(d)


def test_tenure_exceeds_elapsed_is_flagged(spec):
    # 8 years claimed at a role that started only ~2 years ago
    c = _cand(profile={"years_of_experience": 8, "current_title": "Engineer"},
              career_history=[{"company": "NewCo", "title": "Engineer", "is_current": True,
                               "start_date": "2024-06-01", "end_date": None,
                               "duration_months": 96, "industry": "Internet",
                               "description": "x"}])
    assert detect_honeypot(c, spec).is_honeypot


def test_experience_exceeds_career_span_is_flagged(spec):
    c = _cand(profile={"years_of_experience": 14, "current_title": "Analyst"},
              career_history=[{"company": "Co", "title": "Analyst", "is_current": True,
                               "start_date": "2025-01-01", "end_date": None,
                               "duration_months": 16, "industry": "Internet",
                               "description": "x"}])
    assert detect_honeypot(c, spec).is_honeypot


def test_many_zero_month_expert_skills_flagged(spec):
    c = _cand(profile={"years_of_experience": 6},
              skills=[{"name": f"Skill{i}", "proficiency": "expert",
                       "endorsements": 0, "duration_months": 0} for i in range(6)])
    assert detect_honeypot(c, spec).is_honeypot


def test_education_reversed_dates_flagged(spec):
    c = _cand(profile={"years_of_experience": 6},
              education=[{"institution": "U", "degree": "BS", "field_of_study": "CS",
                          "start_year": 2018, "end_year": 2015}])
    assert detect_honeypot(c, spec).is_honeypot


def test_active_before_signup_NOT_flagged(spec):
    """Regression guard: last_active < signup is logically impossible but is a
    data-generation artifact here (~7.5% of normal candidates), so it must NOT flag."""
    c = _cand(profile={"years_of_experience": 6},
              redrob_signals={"signup_date": "2025-01-01", "last_active_date": "2024-06-01"})
    assert not detect_honeypot(c, spec).is_honeypot


def test_normal_profile_not_flagged(spec):
    """A real profile: skill used 60 months but only 3 years employed (Python from
    college) must NOT be flagged -- that was the big false-positive class."""
    c = _cand(
        profile={"years_of_experience": 3, "current_title": "ML Engineer"},
        career_history=[{"company": "ProductCo", "title": "ML Engineer", "is_current": True,
                         "start_date": "2023-06-01", "end_date": None, "duration_months": 36,
                         "industry": "Internet", "description": "built ranking models"}],
        skills=[{"name": "Python", "proficiency": "expert", "endorsements": 20,
                 "duration_months": 60}],
        education=[{"institution": "IIT", "degree": "BTech", "field_of_study": "CS",
                    "start_year": 2016, "end_year": 2020}],
        redrob_signals={"signup_date": "2024-01-01", "last_active_date": "2026-05-01"},
    )
    assert not detect_honeypot(c, spec).is_honeypot
