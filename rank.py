#!/usr/bin/env python3
"""Produce the top-100 submission CSV from the candidate pool.

This is the single reproduce command referenced in the README / submission metadata:

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

It runs fully offline on CPU: it streams the pool, scores every candidate with the
hybrid ranker (structured fit + a pre-computed semantic term, if the embedding
artifact is present), writes a validator-compliant CSV, and -- by default -- runs the
official validator on the result. No network, no GPU, no LLM calls.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from redrob_ranker.embeddings import load_precomputed
from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.loader import iter_candidates
from redrob_ranker.rankers import HybridRanker
from redrob_ranker.rankers.semantic import SemanticRanker
from redrob_ranker.submission import quick_check, select_top, write_submission

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, required=True,
                    help="candidates.jsonl or .jsonl.gz")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--spec", type=Path, default=None, help="path to jd_spec.yaml")
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"),
                    help="dir with pre-computed embeddings (optional)")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--no-reasoning", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    spec = load_jd_spec(args.spec)

    candidates = list(iter_candidates(args.candidates))
    t_load = time.time()
    print(f"[rank] loaded {len(candidates)} candidates in {t_load - t0:.1f}s")

    pre = load_precomputed(args.artifacts)
    if pre is not None:
        semantic = SemanticRanker(pre[0], pre[1])
        print(f"[rank] using pre-computed embeddings ({len(pre[1])} vectors) -> hybrid")
    else:
        semantic = None
        print("[rank] no embedding artifact found -> structured-only (hybrid fallback)")

    ranker = HybridRanker(spec, semantic=semantic)
    ranked = ranker.rank(candidates)
    t_rank = time.time()
    print(f"[rank] scored {len(ranked)} candidates in {t_rank - t_load:.1f}s")

    # early internal sanity check before writing
    problems = quick_check(select_top(ranked, top_n=args.top))
    if problems:
        print("[rank] internal check found issues:")
        for p in problems[:10]:
            print("   -", p)

    write_submission(ranked, spec, args.out, top_n=args.top,
                     with_reasoning=not args.no_reasoning)
    t_write = time.time()
    print(f"[rank] wrote {args.out} in {t_write - t_rank:.1f}s")
    print(f"[rank] TOTAL ranking step: {t_write - t_load:.1f}s "
          f"(budget: 5 min) | end-to-end incl. load: {t_write - t0:.1f}s")

    if not args.no_validate:
        try:
            from validate_submission import validate_submission
            errs = validate_submission(str(args.out))
            if errs:
                print(f"[rank] VALIDATION FAILED ({len(errs)} issue(s)):")
                for e in errs:
                    print("   -", e)
                return 1
            print("[rank] official validator: submission is VALID ✓")
        except ImportError:
            print("[rank] (validator not found at repo root; skipping self-check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
