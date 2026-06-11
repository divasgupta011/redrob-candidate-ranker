"""Tests for the eval metrics (known-value checks) and the gold labelling."""
from __future__ import annotations

import math

import pytest

from eval.gold import gold_tier
from eval.metrics import average_precision, ndcg_at_k, precision_at_k
from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.schema import Candidate


@pytest.fixture(scope="module")
def spec():
    return load_jd_spec()


def test_ndcg_perfect_is_one():
    rel = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0)


def test_ndcg_reversed_is_low():
    rel = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["c", "b", "a"], rel, 3) < ndcg_at_k(["a", "b", "c"], rel, 3)


def test_ndcg_known_value():
    # one relevant item (gain 1) placed at rank 2 -> DCG = 1/log2(3); IDCG = 1/log2(2)=1
    rel = {"x": 1, "y": 0}
    assert ndcg_at_k(["y", "x"], rel, 2) == pytest.approx(1 / math.log2(3))


def test_precision_at_k():
    rel = {"a": 3, "b": 0, "c": 4, "d": 2}
    # tier>=3 relevant: a, c  -> of top 4, 2 relevant
    assert precision_at_k(["a", "b", "c", "d"], rel, 4) == pytest.approx(0.5)


def test_average_precision():
    rel = {"a": 3, "b": 0, "c": 3}     # 2 relevant
    # ranking a,b,c: hits at rank1 (1/1) and rank3 (2/3) -> AP = (1 + 0.6667)/2
    assert average_precision(["a", "b", "c"], rel) == pytest.approx((1 + 2 / 3) / 2)


def _cand(cid, title, yoe, desc="", signals=None):
    return Candidate.from_dict({
        "candidate_id": cid,
        "profile": {"current_title": title, "years_of_experience": yoe},
        "career_history": [{"company": "C", "title": title, "is_current": True,
                            "duration_months": int(yoe * 12), "industry": "Internet",
                            "description": desc}],
        "redrob_signals": signals or {},
    })


def test_gold_negative_title_low_even_with_evidence(spec):
    """A Marketing Manager scores low even if the text mentions ranking -- the key
    way the gold encodes the JD's intent against keyword-stuffing."""
    c = _cand("CAND_0000001", "Marketing Manager", 6,
              "ran campaigns; mentions recommendation system in passing")
    assert gold_tier(c, spec) <= 1


def test_gold_core_with_evidence_high(spec):
    c = _cand("CAND_0000002", "Machine Learning Engineer", 7,
              "built the recommendation system and search ranking at scale")
    assert gold_tier(c, spec) >= 4
