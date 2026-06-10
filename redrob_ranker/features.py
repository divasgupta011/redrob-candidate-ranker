"""Per-candidate fit features for the structured ranker.

This module turns a :class:`Candidate` into a small set of interpretable sub-scores
in [0, 1], plus the diagnostics the reasoning generator needs (which capabilities
matched, which are missing, career stats).

The central idea -- the thing that defeats keyword stuffers -- is *corroboration*:
a listed skill is believed only in proportion to how long it was used, how many
endorsements it has, and what the candidate actually scored on the Redrob
assessment for it. And *career evidence* (what the role descriptions say they
built) outweighs the skills list entirely, which is how plain-language top-tier
candidates -- who never write "RAG" or "Pinecone" -- still surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .jdspec import Capability, JDSpec
from .schema import Candidate, Skill

# NB: anything role-specific (title families, services-industry terms, relevant
# education fields, scores) is loaded from config/jd_spec.yaml via JDSpec, so the
# engine here is JD-agnostic -- swap the YAML to retarget to a different role.


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------------------
# diagnostics structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CapabilityMatch:
    id: str
    weight: float
    credit: float                 # [0,1] how well this capability is covered
    matched_skills: list[str] = field(default_factory=list)
    best_trust: float = 0.0
    evidence_hits: int = 0


@dataclass(slots=True)
class CareerStats:
    num_jobs: int
    total_months: int
    avg_tenure_months: float
    earliest_start: Optional[date]
    has_product_role: bool
    all_services: bool
    current_is_ic: bool


@dataclass(slots=True)
class CandidateFeatures:
    title_career_fit: float
    must_have_coverage: float
    must_have_matches: list[CapabilityMatch]
    missing_must_haves: list[str]
    nice_to_have_score: float
    nice_matches: list[CapabilityMatch]
    experience_fit: float
    location_fit: float
    education_fit: float
    career: CareerStats
    evidence_strength: float       # overall AI/ranking career-evidence signal [0,1]
    base_fit: float                # weighted combo (pre disqualifier/behavioral/honeypot)


# ---------------------------------------------------------------------------
# skill corroboration ("trust")
# ---------------------------------------------------------------------------

def skill_trust(skill: Skill, assessment_lc: dict[str, float], cfg: dict) -> float:
    """How much to believe a single listed skill, in [0, 1].

    A bare listing earns ``base``; corroboration (duration, endorsements,
    assessment score, and proficiency *only when actually used*) lifts it toward 1.
    A "0 months / no assessment / expert" entry -- the stuffer signature -- stays at
    the floor.
    """
    base = float(cfg.get("base", 0.2))
    dur_full = float(cfg.get("duration_full_months", 18))
    end_full = float(cfg.get("endorsement_full", 15))
    assess_full = float(cfg.get("assessment_full", 70))
    w_dur = float(cfg.get("w_duration", 0.30))
    w_end = float(cfg.get("w_endorsement", 0.20))
    w_assess = float(cfg.get("w_assessment", 0.25))
    w_prof = float(cfg.get("w_proficiency", 0.10))

    channels: list[tuple[float, float]] = []
    channels.append((w_dur, clip(skill.duration_months / dur_full)))
    channels.append((w_end, clip(skill.endorsements / end_full)))
    # proficiency is cheap to claim, so it only counts when the skill was actually used
    if skill.duration_months > 0:
        channels.append((w_prof, skill.proficiency_rank / 4.0))
    score = assessment_lc.get(skill.name_lc)
    if score is not None:
        channels.append((w_assess, clip(score / assess_full)))

    wsum = sum(w for w, _ in channels)
    corro = (sum(w * v for w, v in channels) / wsum) if wsum > 0 else 0.0
    return clip(base + (1.0 - base) * corro)


# ---------------------------------------------------------------------------
# capability coverage
# ---------------------------------------------------------------------------

def _skill_matches_term(skill_lc: str, term: str) -> bool:
    # word-ish containment in either direction (handles "embeddings" vs "openai embeddings")
    return term == skill_lc or term in skill_lc or skill_lc in term


def _capability_match(cap: Capability, c: Candidate, assessment_lc: dict[str, float],
                      career_lc: str, trust_cfg: dict) -> CapabilityMatch:
    matched: list[str] = []
    best_trust = 0.0
    for sk in c.skills:
        if any(_skill_matches_term(sk.name_lc, term) for term in cap.skills):
            matched.append(sk.name)
            t = skill_trust(sk, assessment_lc, trust_cfg)
            if t > best_trust:
                best_trust = t

    evidence_hits = sum(1 for phrase in cap.evidence if phrase in career_lc)

    # Career evidence dominates (it shows they DID the thing); corroborated skills
    # carry the rest. A capability proven by evidence is near-fully credited even
    # with no matching skill listed (the plain-language-Tier-5 case).
    evidence_component = 0.9 if evidence_hits >= 1 else 0.0
    if evidence_hits >= 2:
        evidence_component = 1.0
    credit = clip(max(evidence_component, best_trust) + (0.1 if evidence_hits and best_trust > 0.4 else 0.0))
    return CapabilityMatch(id=cap.id, weight=cap.weight, credit=credit,
                           matched_skills=matched, best_trust=best_trust,
                           evidence_hits=evidence_hits)


def _coverage(caps: list[Capability], matches: list[CapabilityMatch]) -> float:
    wsum = sum(c.weight for c in caps) or 1.0
    return sum(m.weight * m.credit for m in matches) / wsum


# ---------------------------------------------------------------------------
# career statistics (also consumed by disqualifiers + reasoning)
# ---------------------------------------------------------------------------

def _is_services_role(role, spec: JDSpec) -> bool:
    if spec.is_consulting_company(role.company_lc):
        return True
    ind = role.industry.lower()
    return any(t in ind for t in spec.services_industry_terms)


def career_stats(c: Candidate, spec: JDSpec) -> CareerStats:
    roles = c.career_history
    num_jobs = len(roles)
    total_months = sum(max(r.duration_months, 0) for r in roles)
    avg_tenure = (total_months / num_jobs) if num_jobs else 0.0
    starts = [r.start_date for r in roles if r.start_date]
    earliest = min(starts) if starts else None
    services_flags = [_is_services_role(r, spec) for r in roles]
    has_product = any(not s for s in services_flags) if roles else False
    all_services = all(services_flags) if roles else False

    nonic = tuple(spec.disqualifiers.get("no_recent_code", {}).get("nonic_titles", []))
    cur = c.current_title_lc
    current_is_ic = not any(t.strip() in cur for t in nonic)

    return CareerStats(num_jobs=num_jobs, total_months=total_months,
                       avg_tenure_months=avg_tenure, earliest_start=earliest,
                       has_product_role=has_product, all_services=all_services,
                       current_is_ic=current_is_ic)


# ---------------------------------------------------------------------------
# individual fit dimensions
# ---------------------------------------------------------------------------

def title_career_fit(c: Candidate, spec: JDSpec, evidence_strength: float) -> float:
    fam = spec.title_family_scores
    current_score = fam[spec.title_family(c.current_title_lc)]
    hist_best = 0.0
    for r in c.career_history:
        hist_best = max(hist_best, fam[spec.title_family(r.title_lc)])
    title_score = max(current_score, 0.85 * hist_best)
    # evidence in the actual work history can lift an adjacent/plain-language profile
    return clip(0.65 * title_score + 0.45 * evidence_strength)


def experience_fit(c: Candidate, spec: JDSpec) -> float:
    y = c.years_of_experience
    if spec.exp_ideal_min <= y <= spec.exp_ideal_max:
        return 1.0
    if spec.exp_min <= y <= spec.exp_max:
        return 0.85
    if y < spec.exp_min:
        return clip(0.85 - (spec.exp_min - y) * 0.15, 0.25, 0.85)
    return clip(0.85 - (y - spec.exp_max) * 0.07, 0.40, 0.85)  # senior overshoot: gentle


def location_fit(c: Candidate, spec: JDSpec) -> float:
    loc = c.location.lower()
    country = c.country.lower()
    relocate = c.signals.willing_to_relocate
    in_india = ("india" in country) or any(city in loc for city in spec.loc_primary | spec.loc_acceptable)
    if country and not in_india:
        return 0.40 if relocate else 0.25      # outside India: no visa sponsorship
    if any(city in loc for city in spec.loc_primary):
        return 1.0                              # Pune / Noida
    if any(city in loc for city in spec.loc_acceptable):
        return 0.85                             # other Tier-1 Indian city
    return 0.70 if relocate else 0.55           # India, elsewhere


def education_fit(c: Candidate, spec: JDSpec) -> float:
    if not c.education:
        return spec.edu_default
    tier = max(spec.edu_tier_scores.get(e.tier, 0.5) for e in c.education)
    field_bonus = spec.edu_field_bonus if any(
        any(f in e.field_of_study.lower() for f in spec.edu_relevant_fields) for e in c.education
    ) else 0.0
    return clip(0.9 * tier + field_bonus)


# ---------------------------------------------------------------------------
# top-level extraction
# ---------------------------------------------------------------------------

def extract_features(c: Candidate, spec: JDSpec) -> CandidateFeatures:
    assessment_lc = {str(k).lower().strip(): float(v)
                     for k, v in c.signals.skill_assessment_scores.items()}
    career_lc = c.career_text().lower()

    must_matches = [_capability_match(cap, c, assessment_lc, career_lc, spec.skill_trust)
                    for cap in spec.must_haves]
    nice_matches = [_capability_match(cap, c, assessment_lc, career_lc, spec.skill_trust)
                    for cap in spec.nice_to_haves]

    must_cov = _coverage(spec.must_haves, must_matches)
    nice_cov = _coverage(spec.nice_to_haves, nice_matches)
    missing = [m.id for m in must_matches if m.credit < 0.4]

    # overall career-evidence signal: total must-have evidence hits, saturating
    total_evidence = sum(m.evidence_hits for m in must_matches)
    evidence_strength = clip(total_evidence / 4.0)

    tcf = title_career_fit(c, spec, evidence_strength)
    exp = experience_fit(c, spec)
    loc = location_fit(c, spec)
    edu = education_fit(c, spec)
    stats = career_stats(c, spec)

    w = spec.weights
    parts = {
        "title_career_fit": (w.get("title_career_fit", 0.34), tcf),
        "must_have_coverage": (w.get("must_have_coverage", 0.30), must_cov),
        "experience_fit": (w.get("experience_fit", 0.11), exp),
        "nice_to_haves": (w.get("nice_to_haves", 0.08), nice_cov),
        "education": (w.get("education", 0.05), edu),
        "location": (w.get("location", 0.07), loc),
    }
    wsum = sum(wt for wt, _ in parts.values()) or 1.0
    base_fit = sum(wt * val for wt, val in parts.values()) / wsum

    return CandidateFeatures(
        title_career_fit=tcf, must_have_coverage=must_cov, must_have_matches=must_matches,
        missing_must_haves=missing, nice_to_have_score=nice_cov, nice_matches=nice_matches,
        experience_fit=exp, location_fit=loc, education_fit=edu, career=stats,
        evidence_strength=evidence_strength, base_fit=base_fit,
    )
