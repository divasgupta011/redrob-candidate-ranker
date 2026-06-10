"""Defensive, typed accessors over a raw candidate dict.

The dataset is adversarial: ~80 honeypots carry *deliberately impossible* values
(8 years at a 3-year-old company, "expert" skills with 0 months used, dates that
don't add up). So every accessor here is total -- it returns a sensible default
rather than raising, and it preserves the raw value so downstream consistency
checks (see ``honeypot.py``) can still detect the contradictions.

Nothing in this module makes a judgement about a candidate; it only parses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from functools import cached_property
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# small safe coercion helpers
# ---------------------------------------------------------------------------

def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _s(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def parse_date(value: Any) -> Optional[date]:
    """Parse an ISO 'YYYY-MM-DD' date, tolerating junk/None. Returns None on failure."""
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    for fmt, ln in (("%Y-%m-%d", 10), ("%Y/%m/%d", 10), ("%Y-%m", 7), ("%Y", 4)):
        try:
            return datetime.strptime(s[:ln], fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# sub-objects
# ---------------------------------------------------------------------------

_PROF_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}


@dataclass(slots=True)
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int

    @property
    def name_lc(self) -> str:
        return self.name.lower().strip()

    @property
    def proficiency_rank(self) -> int:
        return _PROF_RANK.get(self.proficiency.lower().strip(), 0)

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            name=_s(d.get("name")),
            proficiency=_s(d.get("proficiency")),
            endorsements=_i(d.get("endorsements")),
            duration_months=_i(d.get("duration_months")),
        )


@dataclass(slots=True)
class Role:
    company: str
    title: str
    start_date: Optional[date]
    end_date: Optional[date]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

    @property
    def company_lc(self) -> str:
        return self.company.lower().strip()

    @property
    def title_lc(self) -> str:
        return self.title.lower().strip()

    @classmethod
    def from_dict(cls, d: dict) -> "Role":
        return cls(
            company=_s(d.get("company")),
            title=_s(d.get("title")),
            start_date=parse_date(d.get("start_date")),
            end_date=parse_date(d.get("end_date")),
            duration_months=_i(d.get("duration_months")),
            is_current=bool(d.get("is_current")),
            industry=_s(d.get("industry")),
            company_size=_s(d.get("company_size")),
            description=_s(d.get("description")),
        )


@dataclass(slots=True)
class Education:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str]
    tier: str

    @classmethod
    def from_dict(cls, d: dict) -> "Education":
        return cls(
            institution=_s(d.get("institution")),
            degree=_s(d.get("degree")),
            field_of_study=_s(d.get("field_of_study")),
            start_year=_i(d.get("start_year")),
            end_year=_i(d.get("end_year")),
            grade=d.get("grade") if isinstance(d.get("grade"), str) else None,
            tier=_s(d.get("tier"), "unknown"),
        )


@dataclass(slots=True)
class Signals:
    """The 23 redrob_signals, with safe defaults. -1 sentinels are preserved."""

    raw: dict = field(default_factory=dict)

    def _g(self, key: str, default: Any) -> Any:
        v = self.raw.get(key)
        return default if v is None else v

    # engagement / availability
    @property
    def profile_completeness_score(self) -> float:
        return _f(self._g("profile_completeness_score", 0.0))

    @property
    def signup_date(self) -> Optional[date]:
        return parse_date(self.raw.get("signup_date"))

    @property
    def last_active_date(self) -> Optional[date]:
        return parse_date(self.raw.get("last_active_date"))

    @property
    def open_to_work_flag(self) -> bool:
        return bool(self.raw.get("open_to_work_flag"))

    @property
    def profile_views_received_30d(self) -> int:
        return _i(self._g("profile_views_received_30d", 0))

    @property
    def applications_submitted_30d(self) -> int:
        return _i(self._g("applications_submitted_30d", 0))

    @property
    def recruiter_response_rate(self) -> float:
        return _f(self._g("recruiter_response_rate", 0.0))

    @property
    def avg_response_time_hours(self) -> float:
        return _f(self._g("avg_response_time_hours", 0.0))

    @property
    def skill_assessment_scores(self) -> dict:
        v = self.raw.get("skill_assessment_scores")
        return v if isinstance(v, dict) else {}

    @property
    def connection_count(self) -> int:
        return _i(self._g("connection_count", 0))

    @property
    def endorsements_received(self) -> int:
        return _i(self._g("endorsements_received", 0))

    @property
    def notice_period_days(self) -> int:
        return _i(self._g("notice_period_days", 90))

    @property
    def expected_salary_lpa(self) -> tuple[float, float]:
        v = self.raw.get("expected_salary_range_inr_lpa") or {}
        return _f(v.get("min")), _f(v.get("max"))

    @property
    def preferred_work_mode(self) -> str:
        return _s(self._g("preferred_work_mode", "flexible")).lower()

    @property
    def willing_to_relocate(self) -> bool:
        return bool(self.raw.get("willing_to_relocate"))

    @property
    def github_activity_score(self) -> float:
        return _f(self._g("github_activity_score", -1.0), -1.0)

    @property
    def search_appearance_30d(self) -> int:
        return _i(self._g("search_appearance_30d", 0))

    @property
    def saved_by_recruiters_30d(self) -> int:
        return _i(self._g("saved_by_recruiters_30d", 0))

    @property
    def interview_completion_rate(self) -> float:
        return _f(self._g("interview_completion_rate", 0.0))

    @property
    def offer_acceptance_rate(self) -> float:
        return _f(self._g("offer_acceptance_rate", -1.0), -1.0)

    @property
    def verified_email(self) -> bool:
        return bool(self.raw.get("verified_email"))

    @property
    def verified_phone(self) -> bool:
        return bool(self.raw.get("verified_phone"))

    @property
    def linkedin_connected(self) -> bool:
        return bool(self.raw.get("linkedin_connected"))


# ---------------------------------------------------------------------------
# top-level candidate
# ---------------------------------------------------------------------------

@dataclass  # NOT slots=True: cached_property below needs a __dict__ to memoize into
class Candidate:
    raw: dict
    candidate_id: str
    # profile
    name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

    @cached_property
    def skills(self) -> list[Skill]:
        return [Skill.from_dict(s) for s in self.raw.get("skills", []) if isinstance(s, dict)]

    @cached_property
    def career_history(self) -> list[Role]:
        return [Role.from_dict(r) for r in self.raw.get("career_history", []) if isinstance(r, dict)]

    @cached_property
    def education(self) -> list[Education]:
        return [Education.from_dict(e) for e in self.raw.get("education", []) if isinstance(e, dict)]

    @cached_property
    def certifications(self) -> list[dict]:
        return [c for c in self.raw.get("certifications", []) if isinstance(c, dict)]

    @cached_property
    def languages(self) -> list[dict]:
        return [l for l in self.raw.get("languages", []) if isinstance(l, dict)]

    @cached_property
    def signals(self) -> Signals:
        sig = self.raw.get("redrob_signals")
        return Signals(sig if isinstance(sig, dict) else {})

    # ---- convenience views --------------------------------------------------

    @property
    def current_title_lc(self) -> str:
        return self.current_title.lower().strip()

    @cached_property
    def skill_names_lc(self) -> set[str]:
        return {s.name_lc for s in self.skills}

    @cached_property
    def all_titles_lc(self) -> list[str]:
        """Current + historical titles, lowercased (for title-family matching)."""
        out = [self.current_title_lc] if self.current_title_lc else []
        out.extend(r.title_lc for r in self.career_history if r.title_lc)
        return out

    def career_text(self) -> str:
        """Concatenated role descriptions (where the *real* evidence lives)."""
        return " ".join(r.description for r in self.career_history if r.description)

    def profile_text(self) -> str:
        """Headline + summary + titles (the self-description)."""
        parts = [self.headline, self.summary, self.current_title]
        parts.extend(r.title for r in self.career_history)
        return " ".join(p for p in parts if p)

    def full_text(self) -> str:
        """Everything textual -- used by the lexical/semantic rankers."""
        parts = [self.profile_text(), self.career_text(), " ".join(s.name for s in self.skills)]
        return " ".join(p for p in parts if p)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        p = d.get("profile") or {}
        return cls(
            raw=d,
            candidate_id=_s(d.get("candidate_id")),
            name=_s(p.get("anonymized_name")),
            headline=_s(p.get("headline")),
            summary=_s(p.get("summary")),
            location=_s(p.get("location")),
            country=_s(p.get("country")),
            years_of_experience=_f(p.get("years_of_experience")),
            current_title=_s(p.get("current_title")),
            current_company=_s(p.get("current_company")),
            current_company_size=_s(p.get("current_company_size")),
            current_industry=_s(p.get("current_industry")),
        )


def candidates_from_iterable(rows: Iterable[dict]) -> Iterable[Candidate]:
    for d in rows:
        yield Candidate.from_dict(d)
