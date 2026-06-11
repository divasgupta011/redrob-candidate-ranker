"""Reasoning generator tests, mapped to the six Stage-4 checks: specific facts,
JD connection, honest concerns, no hallucination, variation, rank-consistency."""
from __future__ import annotations

import pytest

from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.loader import load_sample_json
from redrob_ranker.rankers import StructuredRanker
from redrob_ranker.reasoning import generate_reasoning
from redrob_ranker.schema import Candidate

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample" / "sample_candidates.json"


@pytest.fixture(scope="module")
def spec():
    return load_jd_spec()


@pytest.fixture(scope="module")
def ranked(spec):
    cands = load_sample_json(SAMPLE)
    r = StructuredRanker(spec).rank(cands)
    r.sort(key=lambda x: (-x.score, x.candidate.candidate_id))
    return r


def _cand(cid, **parts) -> Candidate:
    d = {"candidate_id": cid, "profile": parts.pop("profile", {}),
         "career_history": parts.pop("career_history", []), "education": parts.pop("education", []),
         "skills": parts.pop("skills", []), "redrob_signals": parts.pop("redrob_signals", {})}
    return Candidate.from_dict(d)


def test_reasoning_contains_specific_facts(ranked, spec):
    rc = ranked[0]
    txt = generate_reasoning(rc.candidate, spec, rc.score, rc.breakdown)
    assert str(int(rc.candidate.years_of_experience)) in txt or \
        f"{rc.candidate.years_of_experience:.1f}" in txt          # years referenced
    assert rc.candidate.current_title.split()[0] in txt           # title referenced
    assert len(txt) < 320                                         # 1-2 sentences


def test_no_hallucinated_skills(ranked, spec):
    """Named skills appear only in the strengths half (before 'Concern:'). Each such
    parenthesised skill token must exist on the candidate; numeric parentheticals are
    signal values (e.g. '70%', '150d'), not skills."""
    import re
    for rc in ranked[:15]:
        txt = generate_reasoning(rc.candidate, spec, rc.score, rc.breakdown)
        strengths = txt.split(" Concern:")[0]      # concerns may contain "(no visa sponsorship)" etc.
        names = {s.name.lower() for s in rc.candidate.skills}
        for inside in re.findall(r"\(([^)]*)\)", strengths):
            for tok in (t.strip() for t in inside.split(",")):
                if re.search(r"\d", tok) or not tok:
                    continue                       # signal value, not a skill
                assert tok.lower() in names, \
                    f"hallucinated skill {tok!r} for {rc.candidate.candidate_id}"


def test_reasonings_vary(ranked, spec):
    texts = [generate_reasoning(rc.candidate, spec, rc.score, rc.breakdown) for rc in ranked[:10]]
    assert len(set(texts)) == len(texts)                          # all 10 distinct


def test_rank_consistency_framing(ranked, spec):
    # framing may appear lowercased mid-sentence in one template variant
    top = generate_reasoning(ranked[0].candidate, spec, ranked[0].score, ranked[0].breakdown).lower()
    bottom = generate_reasoning(ranked[-1].candidate, spec, ranked[-1].score,
                                ranked[-1].breakdown).lower()
    assert any(w in top for w in ("excellent fit", "strong fit", "solid fit"))
    assert any(w in bottom for w in ("adjacent fit", "partial fit", "inconsistent"))


def test_honest_concern_surfaced(spec):
    """A candidate with a real gap (outside India) must have it acknowledged."""
    c = _cand("CAND_0000010",
              profile={"current_title": "ML Engineer", "years_of_experience": 7,
                       "location": "Berlin", "country": "Germany"},
              career_history=[{"company": "ProdCo", "title": "ML Engineer", "is_current": True,
                               "duration_months": 60, "industry": "Internet",
                               "description": "built recommendation system and ranking"}],
              skills=[{"name": "Python", "proficiency": "advanced", "endorsements": 10,
                       "duration_months": 60}],
              # clean engagement signals, so the surfaced concern is the location gap
              redrob_signals={"last_active_date": "2026-05-25", "recruiter_response_rate": 0.8,
                              "open_to_work_flag": True, "interview_completion_rate": 0.9,
                              "notice_period_days": 20, "profile_completeness_score": 95,
                              "verified_email": True, "verified_phone": True})
    txt = generate_reasoning(c, spec)
    assert "Concern" in txt and "India" in txt


def test_honeypot_reasoning_is_explicit(spec):
    hp = _cand("CAND_0000099",
               profile={"current_title": "ML Engineer", "years_of_experience": 14},
               career_history=[{"company": "X", "title": "ML Engineer", "is_current": True,
                                "start_date": "2025-01-01", "end_date": None,
                                "duration_months": 16, "industry": "Internet",
                                "description": "ranking"}])
    txt = generate_reasoning(hp, spec)
    assert "inconsistent" in txt.lower()
