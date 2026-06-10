# Redrob — Intelligent Candidate Discovery & Ranking

Rank a 100,000-candidate pool against a single job description **the way a recruiter
would** — by understanding the role and reading careers, not by counting keywords.

> Built for the *Intelligent Candidate Discovery & Ranking Challenge*. The dataset is
> deliberately adversarial: keyword-stuffers, plain-language top-tier candidates, and
> ~80 internally-impossible "honeypot" profiles. A keyword/embedding matcher gets
> trapped; a system that *reads profiles* does not.

---

## The core insight

The job description is a scoring rubric in disguise. It tells us, explicitly:

- **What actually matters** — production retrieval/ranking experience, shipped systems
  at *product* companies, evaluation rigor — proven by **career history**, not by a
  skills list.
- **What's a trap** — *"a candidate who has all the AI keywords listed as skills but
  whose title is 'Marketing Manager' is not a fit, no matter how perfect their skill
  list looks."*
- **What to down-weight** — *"a perfect-on-paper candidate who hasn't logged in for 6
  months and has a 5% recruiter response rate is, for hiring purposes, not actually
  available."*

So the ranker is **rule-forward and evidence-based**: title/career evidence dominates,
skills only count when corroborated (duration + endorsements + assessment scores),
internally-impossible profiles are filtered, and real platform behaviour modulates the
final score.

## Approach: four rankers, raced and evaluated

| Ranker | Idea | Why it's here |
|---|---|---|
| `lexical` | BM25 over JD keywords | The naive baseline — demonstrably falls for stuffers/honeypots |
| `semantic` | Dense embedding cosine vs the JD | Catches *plain-language* fits with no buzzwords |
| `structured` | The JD rubric: trust-weighted skills, title/career evidence, disqualifier penalties, honeypot filter, behavioural modifier | The decisive layer |
| `hybrid` | Semantic recall blended with structured scoring | **The submission ranker** |

Because the ground truth is hidden, we evaluate with **proxies**: honeypot-avoidance
rate, a small hand-labelled gold set (obvious ideal / stuffer / honeypot cases), and
ablations showing the title/career signal beating raw keyword count.

## Compute compliance (hard constraints)

The ranking step satisfies the spec's limits: **CPU-only, ≤ 5 min on 100k, ≤ 16 GB RAM,
no GPU, no network, ≤ 5 GB disk.** No hosted LLM is ever called during ranking.
Embeddings are **pre-computed offline** (`precompute.py`) and loaded as plain numpy
arrays. Any LLM assistance is confined to *offline* JD distillation and development.

## Repository layout

```
rank.py                 # reproduce command entry point (candidates.jsonl -> submission.csv)
precompute.py           # offline: embed the pool -> compact .npy artifact
config/jd_spec.yaml     # the distilled, weighted JD rubric (the "understanding" artifact)
redrob_ranker/          # the package
  schema.py             #   defensive typed accessors over a raw candidate dict
  loader.py             #   streaming reader for candidates.jsonl(.gz)
  jdspec.py             #   load the rubric + skill/title ontology
  features.py           #   per-candidate fit features
  honeypot.py           #   internal-consistency / impossible-profile detection
  behavioral.py         #   availability/engagement multiplier
  reasoning.py          #   deterministic, fact-grounded reasoning strings
  submission.py         #   top-100 CSV writer enforcing the validator's exact rules
  pipeline.py           #   end-to-end orchestration
  rankers/              #   lexical | semantic | structured | hybrid
eval/                   # gold set + NDCG/MAP/honeypot-rate harness + ablations
scripts/extract_docs.py # bundle .docx -> text helper
data/sample/            # 50-candidate public sample (for tests + the sandbox demo)
tests/                  # unit tests
deck/                   # approach deck -> PDF
```

## Quickstart

```bash
# 1. Install runtime deps (tiny: numpy + pyyaml)
pip install -r requirements.txt

# 2. (Offline, one-time) pre-compute pool embeddings for the hybrid ranker
pip install -r requirements-precompute.txt
python precompute.py --candidates data/raw/challenge/candidates.jsonl --out artifacts/

# 3. Produce the top-100 submission (the timed, offline ranking step)
python rank.py --candidates data/raw/challenge/candidates.jsonl --out submission.csv

# 4. Validate against the official format rules
python data/raw/challenge/validate_submission.py submission.csv
```

> `rank.py` also runs in a **structured-only** mode with no embedding artifact present,
> so reproduction never fails for lack of the pre-computed file.

## Status

Built step by step (see git history). Current: project scaffolding. Next: data layer +
JD-spec rubric. See the task list in the PR/commit trail for the roadmap.

## Data & ethics

The candidate pool is synthetic and provided by the organizers (`[PUB]`). The full
`candidates.jsonl` (≈487 MB) is **not** committed; a 50-candidate sample lives under
`data/sample/`. No candidate data is sent to any external service.
