"""Evaluation harness: ranking metrics, a JD-derived gold set, and a head-to-head
comparison of the rankers (NDCG/MAP/P@k + honeypot-rate + ablations).

No hidden ground truth is provided by the contest, so the gold set here is a
documented *proxy* reconstruction of the JD's intent -- useful for relative
comparison and as a sanity check, with honeypot-rate (the contest's own Stage-3
filter) as the one fully objective number.
"""
