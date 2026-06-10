"""Pluggable ranking strategies.

Every ranker implements the same interface (see ``base.Ranker``) so they can be
raced against each other by the evaluation harness:

    lexical    - BM25 over JD keywords (the naive baseline; falls for stuffers/honeypots)
    semantic   - dense embedding cosine similarity to the JD
    structured - the JD-rubric scorer (title/career evidence, trust-weighted skills,
                 disqualifiers, honeypot filter, behavioural modifier) -- the decisive layer
    hybrid     - semantic recall blended with structured scoring (the submission ranker)
"""
