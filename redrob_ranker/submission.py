"""Top-N submission CSV writer -- enforces the official validator's exact rules.

The validator (``validate_submission.py``, bundled and copied to the repo root) is
stricter than the prose spec in one place: for *equal* scores, the earlier rank must
have the smaller candidate_id. Rounding scores to a fixed precision can create such
ties, so the order of operations matters:

  1. pick the true top-N by full-precision score (tie-break candidate_id asc),
  2. round the scores,
  3. re-sort the N by (rounded score desc, candidate_id asc),
  4. assign ranks 1..N.

Done in that order, the written file satisfies "score non-increasing by rank" AND
"equal score -> candidate_id ascending" no matter how the rounding falls.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from .jdspec import JDSpec
from .reasoning import generate_reasoning

HEADER = ["candidate_id", "rank", "score", "reasoning"]
_CID_RE = re.compile(r"^CAND_[0-9]{7}$")


def select_top(ranked, top_n: int = 100, score_decimals: int = 6):
    """Return ``[(rounded_score, RankedCandidate), ...]`` ready to write."""
    by_full = sorted(ranked, key=lambda rc: (-rc.score, rc.candidate.candidate_id))[:top_n]
    rounded = [(round(rc.score, score_decimals), rc) for rc in by_full]
    rounded.sort(key=lambda t: (-t[0], t[1].candidate.candidate_id))
    return rounded


def write_submission(ranked, spec: JDSpec, out_path, top_n: int = 100,
                     score_decimals: int = 6, with_reasoning: bool = True) -> Path:
    out_path = Path(out_path)
    rows = select_top(ranked, top_n=top_n, score_decimals=score_decimals)
    if len(rows) < top_n:
        # the real submission needs exactly top_n; sandbox demos on a small sample
        # legitimately produce fewer -- warn rather than crash.
        print(f"[submission] WARNING: only {len(rows)} candidates available "
              f"(< requested {top_n}); writing what we have.")

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for rank, (rscore, rc) in enumerate(rows, start=1):
            c = rc.candidate
            reason = generate_reasoning(c, spec, rscore, rc.breakdown) if with_reasoning else ""
            writer.writerow([c.candidate_id, rank, f"{rscore:.{score_decimals}f}", reason])
    return out_path


def quick_check(rows) -> list[str]:
    """A fast internal sanity check mirroring the official validator's core rules.
    Returns a list of problems (empty == ok). Use the official validator for the
    authoritative check; this is for early failure during development."""
    errs = []
    prev_score, prev_cid = None, None
    seen_ids = set()
    for rank, (rscore, rc) in enumerate(rows, start=1):
        cid = rc.candidate.candidate_id
        if not _CID_RE.match(cid):
            errs.append(f"row {rank}: bad candidate_id {cid!r}")
        if cid in seen_ids:
            errs.append(f"row {rank}: duplicate candidate_id {cid!r}")
        seen_ids.add(cid)
        if prev_score is not None:
            if rscore > prev_score + 1e-12:
                errs.append(f"row {rank}: score {rscore} > previous {prev_score} (not non-increasing)")
            elif abs(rscore - prev_score) <= 1e-12 and cid < prev_cid:
                errs.append(f"row {rank}: equal score but candidate_id {cid} < previous {prev_cid}")
        prev_score, prev_cid = rscore, cid
    return errs
