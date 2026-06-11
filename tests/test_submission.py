"""Submission writer tests -- the proof is running the *official* validator on our
output, including the tricky tie case (rounding creates equal scores, which must be
ordered by candidate_id ascending)."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.rankers.base import RankedCandidate
from redrob_ranker.schema import Candidate
from redrob_ranker.submission import quick_check, select_top, write_submission

ROOT = Path(__file__).resolve().parent.parent


def _load_official_validator():
    """Import the bundled official validator from the repo root by path."""
    path = ROOT / "validate_submission.py"
    spec = importlib.util.spec_from_file_location("official_validator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate_submission


@pytest.fixture(scope="module")
def spec():
    return load_jd_spec()


def _ranked(n=150):
    out = []
    for i in range(1, n + 1):
        c = Candidate.from_dict({"candidate_id": f"CAND_{i:07d}",
                                 "profile": {"current_title": "ML Engineer",
                                             "years_of_experience": 6}})
        out.append(RankedCandidate(c, score=(n - i) / n, breakdown={}))
    return out


def test_output_passes_official_validator(tmp_path, spec):
    out = tmp_path / "team_test.csv"
    # 2-decimal rounding deliberately forces score ties to exercise the tie-break rule
    write_submission(_ranked(150), spec, out, top_n=100, score_decimals=2)
    errors = _load_official_validator()(str(out))
    assert errors == [], "official validator rejected our output:\n" + "\n".join(errors)


def test_exactly_100_rows_and_header(tmp_path, spec):
    out = tmp_path / "team_test.csv"
    write_submission(_ranked(150), spec, out, top_n=100)
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
    assert len(rows) == 101                              # header + 100
    assert [r[1] for r in rows[1:]] == [str(i) for i in range(1, 101)]


def test_quick_check_clean(spec):
    assert quick_check(select_top(_ranked(150), top_n=100, score_decimals=2)) == []


def test_tie_break_is_candidate_id_ascending(spec):
    # everyone tied on the same score -> ranks must follow candidate_id ascending
    ranked = [RankedCandidate(
        Candidate.from_dict({"candidate_id": f"CAND_{i:07d}", "profile": {}}), 0.5, {})
        for i in (5, 1, 3, 2, 4)]
    rows = select_top(ranked, top_n=5, score_decimals=2)
    ids = [rc.candidate.candidate_id for _, rc in rows]
    assert ids == [f"CAND_{i:07d}" for i in (1, 2, 3, 4, 5)]
