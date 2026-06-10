"""
redrob_ranker
=============

Intelligent candidate discovery & ranking for the Redrob hackathon.

The package ranks a 100k-candidate pool against a single job description the way a
recruiter would: it reads career history and corroborated skills (not raw keywords),
filters internally-inconsistent "honeypot" profiles, and modulates by real platform
behaviour (is the candidate actually reachable and available?).

Design constraints baked in (see ``submission_spec``):
  * the ranking step runs offline, CPU-only, <= 5 min on the 100k pool, no network;
  * any LLM use is confined to *offline* job-description distillation, never the
    live ranking path.

Sub-modules:
  schema      - typed, defensive accessors over a raw candidate dict
  loader      - streaming reader for candidates.jsonl(.gz)
  jdspec      - the distilled, weighted JD rubric + skill/title ontology
  features    - per-candidate fit features (title/career evidence, trust-weighted skills, ...)
  honeypot    - internal-consistency / "subtly impossible profile" detection
  behavioral  - availability/engagement multiplier
  rankers/    - lexical, semantic, structured, hybrid (pluggable)
  reasoning   - deterministic, fact-grounded reasoning strings
  submission  - top-100 CSV writer enforcing the validator's exact rules
  pipeline    - end-to-end orchestration
"""

__version__ = "0.1.0"
