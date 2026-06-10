"""Behavioral modifier tests: the multiplier rewards reachable candidates, down-
weights dormant/unresponsive ones, and -- the key property -- cleanly separates
'behavioral twins' (identical resumes, opposite engagement)."""
from __future__ import annotations

import pytest

from redrob_ranker.behavioral import behavioral_modifier
from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.schema import Candidate


@pytest.fixture(scope="module")
def spec():
    return load_jd_spec()


def _cand(signals: dict) -> Candidate:
    return Candidate.from_dict({"candidate_id": "CAND_0000000", "profile": {},
                                "redrob_signals": signals})


ACTIVE = {
    "last_active_date": "2026-05-25", "recruiter_response_rate": 0.8, "open_to_work_flag": True,
    "interview_completion_rate": 0.9, "notice_period_days": 15, "profile_completeness_score": 95,
    "saved_by_recruiters_30d": 8, "verified_email": True, "verified_phone": True,
}
DORMANT = {
    "last_active_date": "2025-09-01", "recruiter_response_rate": 0.05, "open_to_work_flag": False,
    "interview_completion_rate": 0.1, "notice_period_days": 150, "profile_completeness_score": 40,
    "saved_by_recruiters_30d": 0, "verified_email": False, "verified_phone": False,
}


def test_active_candidate_keeps_score(spec):
    bh = behavioral_modifier(_cand(ACTIVE), spec)
    assert bh.availability > 0.85
    assert bh.modifier > 0.95            # barely penalised


def test_dormant_candidate_is_down_weighted(spec):
    bh = behavioral_modifier(_cand(DORMANT), spec)
    assert bh.availability < 0.3
    # takes close to the full configured haircut
    assert bh.modifier < 1.0 - 0.8 * float(spec.behavioral["weight"])
    assert any("response" in c for c in bh.concerns)


def test_behavioral_twins_are_separated(spec):
    """Same resume, opposite engagement -> a meaningful score gap."""
    active = behavioral_modifier(_cand(ACTIVE), spec).modifier
    dormant = behavioral_modifier(_cand(DORMANT), spec).modifier
    assert active - dormant > 0.15       # enough to reorder a tie


def test_modifier_bounded_by_weight(spec):
    w = float(spec.behavioral["weight"])
    for sig in (ACTIVE, DORMANT, {}):
        m = behavioral_modifier(_cand(sig), spec).modifier
        assert 1.0 - w - 1e-9 <= m <= 1.0 + 1e-9
