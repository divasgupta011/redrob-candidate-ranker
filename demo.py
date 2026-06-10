#!/usr/bin/env python3
"""Step-by-step inspector for the ranking pipeline.

Run any stage on the sample pool (or any candidate file) and see what it produces.
This is a teaching/debugging tool -- it grows a new sub-command as each pipeline
stage is built, and later doubles as the hosted sandbox demo.

Examples
--------
    python demo.py data                      # parse the sample, show summary stats
    python demo.py candidate CAND_0000001    # deep-dive one candidate end-to-end
    python demo.py candidate 0               # ...by index too
    python demo.py features --top 15         # rank the sample by structured fit

Use a different pool with --candidates:
    python demo.py features --candidates data/raw/challenge/candidates.jsonl --top 20
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# make stdout utf-8 on Windows consoles (the data has em-dashes, accents, etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from redrob_ranker.disqualifiers import apply_disqualifiers
from redrob_ranker.features import extract_features, skill_trust
from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.loader import iter_candidates, load_sample_json

DEFAULT_SAMPLE = Path(__file__).resolve().parent / "data" / "sample" / "sample_candidates.json"


def _load(path: Path, limit: int | None):
    """Load candidates from a .json array or a .jsonl(.gz) stream."""
    if path.suffix == ".json":
        cands = load_sample_json(path)
        return cands[:limit] if limit else cands
    return list(iter_candidates(path, limit=limit))


def _rule(c: str = "-", n: int = 92) -> str:
    return c * n


# ---------------------------------------------------------------------------
# sub-command: data
# ---------------------------------------------------------------------------

def cmd_data(args, spec):
    cands = _load(args.candidates, args.limit)
    print(f"Loaded {len(cands)} candidates from {args.candidates.name}\n")

    titles = Counter(c.current_title for c in cands)
    print("Current-title distribution:")
    for t, n in titles.most_common(12):
        print(f"  {n:3d}  {t}")
    fams = Counter(spec.title_family(c.current_title_lc) for c in cands)
    print("\nTitle-family classification (core/adjacent fit the role; negative = stuffer risk):")
    for fam in ("core", "adjacent", "other", "negative"):
        print(f"  {fam:9s} {fams.get(fam, 0)}")

    c = cands[0]
    print(f"\nExample record  {c.candidate_id}  ({c.name})")
    print(f"  title   : {c.current_title} @ {c.current_company} [{c.current_industry}]")
    print(f"  exp     : {c.years_of_experience} yrs   location: {c.location}, {c.country}")
    print(f"  skills  : {len(c.skills)}   roles: {len(c.career_history)}   "
          f"education: {len(c.education)}")
    print(f"  active  : {c.signals.last_active_date}   resp_rate: {c.signals.recruiter_response_rate}"
          f"   open_to_work: {c.signals.open_to_work_flag}")


# ---------------------------------------------------------------------------
# sub-command: candidate (deep dive through every stage)
# ---------------------------------------------------------------------------

def cmd_candidate(args, spec):
    cands = _load(args.candidates, None)
    sel = args.id
    cand = None
    if sel.isdigit() and not sel.startswith("CAND"):
        idx = int(sel)
        if 0 <= idx < len(cands):
            cand = cands[idx]
    if cand is None:
        cand = next((c for c in cands if c.candidate_id == sel), None)
    if cand is None:
        print(f"Candidate '{sel}' not found in {args.candidates.name}")
        return

    c = cand
    print(_rule("="))
    print(f"{c.candidate_id}   {c.name}")
    print(_rule("="))
    print(f"Title : {c.current_title} @ {c.current_company} ({c.current_company_size}, "
          f"{c.current_industry})")
    print(f"Exp   : {c.years_of_experience} yrs    Location: {c.location}, {c.country}")
    print(f"Headline: {c.headline}")
    if c.summary:
        print(f"Summary : {c.summary[:240]}{'...' if len(c.summary) > 240 else ''}")

    print("\nCareer history (newest first):")
    for r in c.career_history:
        cur = " [current]" if r.is_current else ""
        print(f"  - {r.title} @ {r.company} ({r.duration_months}mo, {r.industry}){cur}")
        if r.description:
            print(f"      {r.description[:150]}{'...' if len(r.description) > 150 else ''}")

    assessment_lc = {str(k).lower(): float(v) for k, v in c.signals.skill_assessment_scores.items()}
    print("\nSkills (with computed trust = how much we believe the claim):")
    for s in sorted(c.skills, key=lambda s: -skill_trust(s, assessment_lc, spec.skill_trust))[:12]:
        t = skill_trust(s, assessment_lc, spec.skill_trust)
        a = assessment_lc.get(s.name_lc)
        astr = f", assess {a:.0f}" if a is not None else ""
        print(f"  {t:4.2f}  {s.name} ({s.proficiency}, {s.duration_months}mo, "
              f"{s.endorsements} end{astr})")

    f = extract_features(c, spec)
    dq = apply_disqualifiers(c, spec, f)
    print("\nFeature sub-scores:")
    print(f"  title_career_fit  : {f.title_career_fit:4.2f}")
    print(f"  must_have_coverage: {f.must_have_coverage:4.2f}")
    for m in f.must_have_matches:
        ev = f"{m.evidence_hits} evidence-hits" if m.evidence_hits else "no evidence"
        sk = (", skills: " + ", ".join(m.matched_skills[:3])) if m.matched_skills else ""
        print(f"      - {m.id:28s} credit {m.credit:4.2f}  ({ev}{sk})")
    print(f"  nice_to_have_score: {f.nice_to_have_score:4.2f}")
    print(f"  experience_fit    : {f.experience_fit:4.2f}")
    print(f"  location_fit      : {f.location_fit:4.2f}")
    print(f"  education_fit      : {f.education_fit:4.2f}")
    print(f"  evidence_strength : {f.evidence_strength:4.2f}")
    print(f"  career: {f.career.num_jobs} jobs, avg tenure "
          f"{f.career.avg_tenure_months/12:.1f}y, product-role={f.career.has_product_role}, "
          f"all-services={f.career.all_services}, current-IC={f.career.current_is_ic}")

    print(f"\n  base_fit (weighted)      : {f.base_fit:4.3f}")
    if dq.fired:
        print(f"  disqualifiers fired      : {', '.join(dq.fired)}  (x{dq.multiplier:.2f})")
        for note in dq.notes:
            print(f"      - {note}")
    else:
        print("  disqualifiers fired      : none (x1.00)")
    print(f"  STRUCTURED SCORE         : {f.base_fit * dq.multiplier:4.3f}")


# ---------------------------------------------------------------------------
# sub-command: features (rank the pool by structured fit)
# ---------------------------------------------------------------------------

def cmd_features(args, spec):
    cands = _load(args.candidates, args.limit)
    rows = []
    for c in cands:
        f = extract_features(c, spec)
        dq = apply_disqualifiers(c, spec, f)
        rows.append((f.base_fit * dq.multiplier, c, f, dq))
    rows.sort(key=lambda r: -r[0])

    print(f"Structured ranking of {len(rows)} candidates "
          f"(score = base_fit x disqualifier_multiplier)\n")
    print(f"{'#':>3} {'score':>6} {'base':>5} {'dqx':>4} {'title':26} {'yoe':>4} "
          f"{'cov':>4} {'tcf':>4} {'ev':>3}  fired")
    print(_rule())
    for i, (score, c, f, dq) in enumerate(rows[:args.top], 1):
        print(f"{i:3d} {score:6.3f} {f.base_fit:5.3f} {dq.multiplier:4.2f} "
              f"{c.current_title[:25]:26} {c.years_of_experience:4.1f} "
              f"{f.must_have_coverage:4.2f} {f.title_career_fit:4.2f} {f.evidence_strength:3.1f}  "
              f"{','.join(dq.fired) or '-'}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, default=DEFAULT_SAMPLE,
                    help="candidate file (.json array or .jsonl[.gz] stream)")
    ap.add_argument("--spec", type=Path, default=None, help="path to jd_spec.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("data", help="parse the pool and show summary stats")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("candidate", help="deep-dive one candidate through every stage")
    p.add_argument("id", help="candidate_id (CAND_XXXXXXX) or row index")
    p.set_defaults(func=cmd_candidate)

    p = sub.add_parser("features", help="rank the pool by structured fit")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--limit", type=int, default=None, help="only load first N candidates")
    p.set_defaults(func=cmd_features)

    args = ap.parse_args()
    spec = load_jd_spec(args.spec)
    args.func(args, spec)


if __name__ == "__main__":
    main()
