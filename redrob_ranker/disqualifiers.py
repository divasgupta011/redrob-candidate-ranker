"""The JD's explicit hard negatives, as a fit-score penalty.

The JD has an unusually direct "Things we explicitly do NOT want" section plus a
list of disqualifiers. We don't hard-drop candidates on these (the band is "soft"
and signals can rescue), but we down-weight them, exactly as the JD frames it
("we will probably not move forward"). Each rule returns a multiplier <= 1; the
fired rules are reported so the reasoning generator can be honest about concerns.

Parameters come from ``config/jd_spec.yaml`` (``disqualifiers``); the *logic* lives
here so it stays auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .features import CandidateFeatures
from .jdspec import JDSpec
from .schema import Candidate

# Fallback hands-on terms if the spec omits no_recent_code.code_evidence.
_CODE_EVIDENCE_DEFAULT = ("build", "built", "implement", "develop", "wrote", "code", "coded",
                          "python", "shipped", "deployed", "designed", "engineer")

# seniority ladder for detecting title *escalation* (role-agnostic; the JD's chaser criterion)
_SENIORITY_KEYWORDS = (
    (5, ("principal", "vp ", "vice president", "head of", "director", "cto", "chief")),
    (4, ("staff", "lead", "architect")),
    (3, ("senior", "sr ", "sr.")),
    (1, ("junior", "jr ", "jr.", "associate", "intern", "trainee", "graduate")),
)


def _seniority_rank(title_lc: str) -> int:
    for rank, kws in _SENIORITY_KEYWORDS:
        if any(k in title_lc for k in kws):
            return rank
    return 2  # plain "Engineer", "Manager", etc.


def _title_escalated(c: Candidate) -> bool:
    """True if seniority climbed across the career (Senior -> Staff -> Principal-style).

    career_history is newest-first in the data, so the *earliest* role is last.
    """
    roles = c.career_history
    if len(roles) < 2:
        return False
    earliest_rank = _seniority_rank(roles[-1].title_lc)
    current_rank = _seniority_rank(c.current_title_lc or roles[0].title_lc)
    return (current_rank - earliest_rank) >= 2


@dataclass(slots=True)
class DisqualifierResult:
    multiplier: float
    fired: list[str] = field(default_factory=list)     # machine ids
    notes: list[str] = field(default_factory=list)     # short human phrases for reasoning


def _any(terms, *haystacks: str) -> bool:
    return any(any(t in h for h in haystacks) for t in terms)


def apply_disqualifiers(c: Candidate, spec: JDSpec, feats: CandidateFeatures) -> DisqualifierResult:
    dq = spec.disqualifiers
    skills_lc = " ".join(c.skill_names_lc)
    career_lc = c.career_text().lower()
    profile_lc = c.profile_text().lower()
    cur_role_desc = (c.career_history[0].description.lower()
                     if c.career_history and c.career_history[0].is_current else "")
    stats = feats.career

    fired: list[str] = []
    notes: list[str] = []
    mult = 1.0

    # 1) title-chaser: job-hops ~1.5yr WHILE climbing the title ladder. Both conditions
    #    are required, so a genuine engineer who simply changed jobs isn't punished.
    tc = dq.get("title_chaser", {})
    escalation_required = bool(tc.get("require_title_escalation", True))
    if (stats.num_jobs >= int(tc.get("min_jobs_to_judge", 3))
            and 0 < stats.avg_tenure_months < float(tc.get("max_avg_tenure_months", 18))
            and (not escalation_required or _title_escalated(c))):
        mult *= float(tc.get("penalty", 0.70))
        fired.append("title_chaser")
        notes.append(f"short tenure while climbing titles (~{stats.avg_tenure_months/12:.1f}y/role "
                     f"across {stats.num_jobs} jobs)")

    # 2) consulting-only: entire career in services, no product-company role
    co = dq.get("consulting_only", {})
    if stats.num_jobs >= 2 and stats.all_services and not stats.has_product_role:
        mult *= float(co.get("penalty", 0.60))
        fired.append("consulting_only")
        notes.append("career entirely at services/consulting firms (no product-company role)")

    # 3) research-only: every employer academic/research, no production evidence
    ro = dq.get("research_only", {})
    research_terms = tuple(ro.get("research_employers", []))
    prod_terms = tuple(ro.get("production_evidence", []))
    if c.career_history and research_terms:
        research_roles = sum(
            1 for r in c.career_history
            if _any(research_terms, r.company_lc, r.industry.lower(), r.title_lc)
        )
        if research_roles == stats.num_jobs and not _any(prod_terms, career_lc):
            mult *= float(ro.get("penalty", 0.55))
            fired.append("research_only")
            notes.append("pure-research background with no visible production deployment")

    # 4) recent-LLM-wrapper-only: shallow AI that's just LangChain/OpenAI calls
    rw = dq.get("recent_llm_wrapper_only", {})
    if (_any(tuple(rw.get("wrapper_terms", [])), skills_lc, profile_lc, career_lc)
            and feats.must_have_coverage < 0.45 and feats.evidence_strength < 0.25):
        mult *= float(rw.get("penalty", 0.50))
        fired.append("recent_llm_wrapper_only")
        notes.append("AI experience looks like recent LLM-API wrappers without deeper retrieval/ranking work")

    # 5) framework-enthusiast: tutorials/demos rather than systems
    fe = dq.get("framework_enthusiast", {})
    fe_terms = tuple(fe.get("terms", []))
    if sum(1 for t in fe_terms if t in career_lc or t in profile_lc) >= 2 and feats.must_have_coverage < 0.5:
        mult *= float(fe.get("penalty", 0.75))
        fired.append("framework_enthusiast")
        notes.append("profile reads as tutorials/demos rather than shipped systems")

    # 6) no recent hands-on code: senior moved fully into a non-IC role
    nc = dq.get("no_recent_code", {})
    code_terms = tuple(nc.get("code_evidence", [])) or _CODE_EVIDENCE_DEFAULT
    if not stats.current_is_ic and not _any(code_terms, cur_role_desc):
        mult *= float(nc.get("penalty", 0.70))
        fired.append("no_recent_code")
        notes.append("current role is non-IC with no recent hands-on coding signal")

    # 7) CV/speech/robotics without NLP/IR
    cv = dq.get("cv_speech_robotics_no_nlp", {})
    domain = tuple(cv.get("domain_terms", []))
    nlp_ir = tuple(cv.get("nlp_ir_terms", []))
    if _any(domain, skills_lc, career_lc) and not _any(nlp_ir, skills_lc, career_lc):
        mult *= float(cv.get("penalty", 0.55))
        fired.append("cv_speech_robotics_no_nlp")
        notes.append("primary expertise is vision/speech/robotics without NLP/IR")

    return DisqualifierResult(multiplier=max(mult, 0.10), fired=fired, notes=notes)
