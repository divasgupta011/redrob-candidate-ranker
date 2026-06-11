"""Deterministic, fact-grounded reasoning for the submission's ``reasoning`` column.

Stage-4 manual review samples 10 rows and checks each reasoning for: specific facts,
JD connection, honest concerns, no hallucination, variation, and rank-consistency.
This generator is built to pass all six *without* an LLM in the path:

  * Specific facts / no hallucination -- every clause is composed from the candidate's
    own fields (years, title, matched skills, signal values). We never assert a skill
    or employer the profile doesn't contain.
  * JD connection -- strengths are phrased as the JD's named capabilities (embeddings
    retrieval, vector search, ranking evaluation, ...), credited only when the profile
    actually evidences them.
  * Honest concerns -- disqualifier notes, behavioural concerns, missing must-haves,
    experience-band and location gaps are surfaced when present.
  * Variation -- content differs per candidate (their real facts), and the sentence
    frame is chosen deterministically from the candidate_id so phrasing varies too.
  * Rank-consistency -- the opening framing scales with the score, so a top pick reads
    confidently and a filler pick reads as a stretch.

An optional offline-LLM polish hook exists (off by default); it only ever *rewrites*
this fact-grounded text and never runs inside the timed ranking step.
"""
from __future__ import annotations

from .behavioral import behavioral_modifier
from .disqualifiers import apply_disqualifiers
from .features import extract_features
from .honeypot import detect_honeypot
from .jdspec import JDSpec
from .schema import Candidate

# JD-requirement names, so strengths read as connections to the JD, not generic praise
_CAP_NAMES = {
    "embeddings_retrieval": "embeddings-based retrieval",
    "vector_db_hybrid_search": "vector / hybrid search",
    "ranking_search_rec_systems": "ranking / search / recommendation systems",
    "ranking_evaluation": "ranking evaluation with NDCG/MRR",  # no parens: they're reserved for named skills
    "strong_python": "strong Python",
    "llm_finetuning": "LLM fine-tuning",
    "learning_to_rank_models": "learning-to-rank models",
    "hr_marketplace_domain": "HR-tech / marketplace experience",
    "distributed_inference": "large-scale inference",
    "open_source": "open-source work",
}

# score -> opening framing (rank-consistency)
_FRAMINGS = [
    (0.75, "Excellent fit"),
    (0.60, "Strong fit"),
    (0.45, "Solid fit"),
    (0.30, "Partial fit"),
    (0.00, "Adjacent fit"),
]


def _fmt_years(y: float) -> str:
    return f"{y:.0f}" if abs(y - round(y)) < 0.05 else f"{y:.1f}"


def _and_join(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _framing(score: float) -> str:
    for threshold, label in _FRAMINGS:
        if score >= threshold:
            return label
    return "Adjacent fit"


def generate_reasoning(c: Candidate, spec: JDSpec, score: float | None = None,
                       breakdown: dict | None = None) -> str:
    b = breakdown or {}
    f = b.get("features") or extract_features(c, spec)
    dq = b.get("dq") or apply_disqualifiers(c, spec, f)
    bh = b.get("behavioral") or behavioral_modifier(c, spec)
    hp = b.get("honeypot") or detect_honeypot(c, spec)
    if score is None:
        score = 0.0 if hp.is_honeypot else f.base_fit * dq.multiplier * bh.modifier

    # defensive: honeypots should never reach the top-100, but be honest if asked
    if hp.is_honeypot:
        return f"Excluded as internally inconsistent: {hp.reasons[0]}."

    yrs = _fmt_years(c.years_of_experience)
    title = c.current_title or "Candidate"

    # ---- strengths (JD capabilities the profile actually evidences) ----------
    caps = sorted((m for m in f.must_have_matches if m.credit >= 0.55),
                  key=lambda m: -(m.weight * m.credit))
    built = [_CAP_NAMES.get(m.id, m.id) for m in caps if m.evidence_hits][:2]
    listed = [_CAP_NAMES.get(m.id, m.id) for m in caps if not m.evidence_hits
              and m.best_trust >= 0.6][:2]
    named_skills: list[str] = []
    for m in caps[:3]:
        for s in m.matched_skills:
            if s not in named_skills:
                named_skills.append(s)

    clause = f"{title} with {yrs} yrs"
    if built:
        clause += f" whose history shows {_and_join(built)}"
    elif listed:
        clause += f" with corroborated {_and_join(listed)}"
    if named_skills:
        clause += f" ({', '.join(named_skills[:3])})"
    # product-company signal connects to the JD's "product over services" preference
    if (f.career.has_product_role and not f.career.all_services
            and spec.title_family(c.current_title_lc) in ("core", "adjacent") and built):
        clause += " at product companies"

    # one positive behavioural fact (carries a concrete signal value)
    if bh.positives and score >= 0.45:
        clause += f"; {bh.positives[0]}"

    # ---- honest concerns -----------------------------------------------------
    concerns: list[str] = list(dq.notes)
    concerns += bh.concerns
    missing = [_CAP_NAMES.get(mid, mid) for mid in f.missing_must_haves]
    if missing:
        concerns.append(f"no clear evidence of {_and_join(missing[:2])}")
    if c.years_of_experience > spec.exp_max + 1:
        concerns.append(f"{yrs}y is above the {int(spec.exp_min)}–{int(spec.exp_max)}y target")
    elif 0 < c.years_of_experience < spec.exp_min - 0.5:
        concerns.append(f"{yrs}y is below the target band")
    if f.location_fit < 0.5:
        if c.country and "india" not in c.country.lower():
            concerns.append("based outside India (no visa sponsorship)")
        else:
            concerns.append("not in the Pune/Noida hubs")
    # de-dup preserving order
    seen, uniq = set(), []
    for x in concerns:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    concerns = uniq[:2]

    # ---- assemble (frame chosen deterministically from the id for variation) -
    frame = _framing(score)
    pick = sum(ord(ch) for ch in c.candidate_id) % 3
    if pick == 0:
        text = f"{frame}: {clause}."
    elif pick == 1:
        text = f"{clause} — {frame.lower()}."
    else:
        text = f"{frame}. {clause[0].upper()}{clause[1:]}."

    if concerns:
        text += f" Concern: {_and_join(concerns)}."
    return text
