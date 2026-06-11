"""Race the rankers and report metrics, honeypot-rate, and ablations.

Usage:
    python -m eval.evaluate --candidates data/raw/challenge/candidates.jsonl
    python -m eval.evaluate --limit 20000          # faster subset
    python -m eval.evaluate                          # defaults to the bundled sample

For each ranker we report the contest metrics against the proxy gold set, plus two
objective behaviours: honeypot-rate in the top-100 (the Stage-3 DQ filter) and the
share of the top-10 that are 'negative-title' profiles (keyword-stuffer risk).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from redrob_ranker.embeddings import load_precomputed
from redrob_ranker.honeypot import detect_honeypot
from redrob_ranker.jdspec import load_jd_spec
from redrob_ranker.loader import iter_candidates, load_sample_json
from redrob_ranker.rankers import (HybridRanker, LexicalRanker, SemanticRanker,
                                   SkillCountRanker, StructuredRanker)

from eval.gold import gold_relevance
from eval.metrics import composite

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_candidates.json"


def _ranked_ids(ranker, candidates):
    r = ranker.rank(candidates)
    r.sort(key=lambda x: (-x.score, x.candidate.candidate_id))
    return r


def _honeypot_rate(rcs, spec, k):
    top = rcs[:k]
    if not top:
        return 0.0
    return sum(1 for rc in top if detect_honeypot(rc.candidate, spec).is_honeypot) / len(top)


def _negative_title_rate(rcs, spec, k):
    top = rcs[:k]
    if not top:
        return 0.0
    return sum(1 for rc in top
               if spec.title_family(rc.candidate.current_title_lc) == "negative") / len(top)


def _row(name, rcs, gold, spec):
    ids = [rc.candidate.candidate_id for rc in rcs]
    m = composite(ids, gold)
    return {
        "ranker": name,
        "composite": m["composite"], "ndcg@10": m["ndcg@10"], "ndcg@50": m["ndcg@50"],
        "map": m["map"], "p@10": m["p@10"],
        "honeypot%@100": 100 * _honeypot_rate(rcs, spec, 100),
        "neg-title%@10": 100 * _negative_title_rate(rcs, spec, 10),
    }


def _print_table(rows):
    cols = ["ranker", "composite", "ndcg@10", "ndcg@50", "map", "p@10",
            "honeypot%@100", "neg-title%@10"]
    print(f"{'ranker':12} {'compos':>7} {'ndcg10':>7} {'ndcg50':>7} {'map':>6} "
          f"{'p@10':>6} {'hp%100':>7} {'neg%10':>7}")
    print("-" * 72)
    for r in rows:
        print(f"{r['ranker']:12} {r['composite']:7.3f} {r['ndcg@10']:7.3f} "
              f"{r['ndcg@50']:7.3f} {r['map']:6.3f} {r['p@10']:6.3f} "
              f"{r['honeypot%@100']:7.1f} {r['neg-title%@10']:7.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, default=SAMPLE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    spec = load_jd_spec()
    if args.candidates.suffix == ".json":
        candidates = load_sample_json(args.candidates)
        if args.limit:
            candidates = candidates[: args.limit]
    else:
        candidates = list(iter_candidates(args.candidates, limit=args.limit))
    print(f"Evaluating on {len(candidates)} candidates from {args.candidates.name}\n")

    gold = gold_relevance(candidates, spec)
    pre = load_precomputed(args.artifacts)
    semantic = SemanticRanker(pre[0], pre[1]) if pre else None

    rankers = [
        ("skill_count", SkillCountRanker(spec)),
        ("lexical", LexicalRanker(spec)),
        ("structured", StructuredRanker(spec)),
        ("hybrid", HybridRanker(spec, semantic=semantic)),
    ]
    if semantic is not None:
        rankers.insert(2, ("semantic", semantic))

    rows = []
    for name, ranker in rankers:
        t = time.time()
        rcs = _ranked_ids(ranker, candidates)
        rows.append(_row(name, rcs, gold, spec))
        print(f"  ran {name:12} in {time.time() - t:5.1f}s")

    print("\n=== Ranker comparison (vs proxy gold; honeypot% is the objective one) ===")
    _print_table(rows)

    # --- ablations: turn off one structured guard at a time --------------------
    print("\n=== Ablations (structured ranker, one guard removed) ===")
    ablations = [
        ("full", StructuredRanker(spec)),
        ("-honeypot_gate", StructuredRanker(spec, use_honeypot=False)),
        ("-disqualifiers", StructuredRanker(spec, use_disqualifiers=False)),
        ("-behavioral", StructuredRanker(spec, use_behavioral=False)),
    ]
    abl_rows = [_row(name, _ranked_ids(r, candidates), gold, spec) for name, r in ablations]
    _print_table(abl_rows)


if __name__ == "__main__":
    main()
