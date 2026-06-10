"""Smoke + correctness tests for the data layer (schema, loader, jdspec)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.loader import load_sample_json
from redrob_ranker.schema import Candidate, parse_date

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample" / "sample_candidates.json"


@pytest.fixture(scope="module")
def candidates() -> list[Candidate]:
    return load_sample_json(SAMPLE)


@pytest.fixture(scope="module")
def spec():
    return load_jd_spec()


def test_sample_loads(candidates):
    assert len(candidates) == 50
    assert all(c.candidate_id.startswith("CAND_") for c in candidates)


def test_candidate_accessors(candidates):
    c = candidates[0]
    assert c.years_of_experience > 0
    assert c.current_title
    assert isinstance(c.skills, list) and len(c.skills) >= 1
    assert isinstance(c.career_history, list) and len(c.career_history) >= 1
    # signals are typed & safe
    assert 0.0 <= c.signals.recruiter_response_rate <= 1.0
    assert isinstance(c.signals.skill_assessment_scores, dict)


def test_parse_date_is_total():
    assert parse_date("2024-03-08") == date(2024, 3, 8)
    assert parse_date(None) is None
    assert parse_date("not-a-date") is None
    assert parse_date("") is None
    assert parse_date("2024") == date(2024, 1, 1)


def test_defensive_accessors_never_raise():
    # a deliberately broken / sparse record must not crash any accessor
    broken = Candidate.from_dict({"candidate_id": "CAND_9999999", "profile": None,
                                  "skills": [None, {"name": "X"}], "redrob_signals": "garbage"})
    assert broken.current_title == ""
    assert broken.years_of_experience == 0.0
    assert broken.signals.recruiter_response_rate == 0.0
    assert broken.skill_names_lc == {"x"}  # the None skill is skipped
    assert broken.full_text() is not None


def test_jd_spec_structure(spec):
    assert spec.exp_min == 5 and spec.exp_max == 9
    must_have_ids = {c.id for c in spec.must_haves}
    assert "embeddings_retrieval" in must_have_ids
    assert "ranking_search_rec_systems" in must_have_ids
    assert spec.weights["title_career_fit"] > spec.weights["education"]


def test_title_family_negative_wins(spec):
    # the decisive anti-stuffer rule: a marketing/HR title is 'negative' even if
    # other words appear; AI/ML titles are 'core'
    assert spec.title_family("marketing manager") == "negative"
    assert spec.title_family("hr manager") == "negative"
    assert spec.title_family("machine learning engineer") == "core"
    assert spec.title_family("recommendation systems engineer") == "core"
    assert spec.title_family("data scientist") == "adjacent"


def test_consulting_detection(spec):
    assert spec.is_consulting_company("infosys")
    assert spec.is_consulting_company("tata consultancy services")
    assert not spec.is_consulting_company("google")
