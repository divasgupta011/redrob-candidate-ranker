"""Load and expose the distilled JD rubric (``config/jd_spec.yaml``).

This wraps the YAML in a typed object with pre-normalized lookups so the hot path
(feature extraction over 100k candidates) does no per-candidate string munging of
the spec itself. All decision *logic* lives in the feature/honeypot/behavioral
modules; this module only structures the data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .schema import parse_date

DEFAULT_SPEC_PATH = Path(__file__).resolve().parent.parent / "config" / "jd_spec.yaml"


@dataclass(slots=True)
class Capability:
    """A must-have or nice-to-have skill cluster."""

    id: str
    weight: float
    skills: tuple[str, ...]
    evidence: tuple[str, ...]


def _lc_tuple(xs: Any) -> tuple[str, ...]:
    if not xs:
        return ()
    return tuple(str(x).lower().strip() for x in xs)


class JDSpec:
    """Typed view over jd_spec.yaml."""

    def __init__(self, data: dict):
        self.raw = data
        role = data.get("role", {})
        self.role_title: str = role.get("title", "")
        self.company: str = role.get("company", "")
        exp = role.get("experience", {})
        self.exp_min: float = float(exp.get("min_years", 0))
        self.exp_max: float = float(exp.get("max_years", 99))
        self.exp_ideal_min: float = float(exp.get("ideal_min", self.exp_min))
        self.exp_ideal_max: float = float(exp.get("ideal_max", self.exp_max))
        self.applied_ml_years_ideal: float = float(exp.get("applied_ml_years_ideal", 0))

        loc = data.get("location", {})
        self.loc_primary: set[str] = set(_lc_tuple(loc.get("primary")))
        self.loc_acceptable: set[str] = set(_lc_tuple(loc.get("acceptable_in_country")))
        self.loc_countries: set[str] = set(_lc_tuple(loc.get("country")))
        self.relocation_helps: bool = bool(loc.get("relocation_helps", True))

        titles = data.get("titles", {})
        self.titles_core: tuple[str, ...] = _lc_tuple(titles.get("core"))
        self.titles_adjacent: tuple[str, ...] = _lc_tuple(titles.get("adjacent"))
        self.titles_negative: tuple[str, ...] = _lc_tuple(titles.get("negative"))

        self.must_haves: list[Capability] = self._capabilities(data.get("must_haves", []))
        self.nice_to_haves: list[Capability] = self._capabilities(data.get("nice_to_haves", []))

        self.disqualifiers: dict = data.get("disqualifiers", {})
        self.behavioral: dict = data.get("behavioral", {})
        self.skill_trust: dict = data.get("skill_trust", {})
        self.weights: dict = data.get("weights", {})
        self.ideal_profile: dict = data.get("ideal_profile", {})

        # consulting-company set, normalized once
        self.consulting_companies: set[str] = set(
            _lc_tuple(self.disqualifiers.get("consulting_only", {}).get("companies"))
        )

        ref = self.behavioral.get("reference_date")
        self.reference_date: date | None = parse_date(ref) if ref else None

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _capabilities(items: list[dict]) -> list[Capability]:
        out = []
        for it in items:
            out.append(
                Capability(
                    id=str(it.get("id", "")),
                    weight=float(it.get("weight", 1.0)),
                    skills=_lc_tuple(it.get("skills")),
                    evidence=_lc_tuple(it.get("evidence")),
                )
            )
        return out

    def title_family(self, title_lc: str) -> str:
        """Classify a (lowercased) title as 'core' | 'adjacent' | 'negative' | 'other'.

        Negative is checked first so a 'Marketing Manager' is never accidentally
        rescued by an adjacent/core substring elsewhere.
        """
        if not title_lc:
            return "other"
        for term in self.titles_negative:
            if term in title_lc:
                return "negative"
        for term in self.titles_core:
            if term in title_lc:
                return "core"
        for term in self.titles_adjacent:
            if term in title_lc:
                return "adjacent"
        return "other"

    def is_consulting_company(self, company_lc: str) -> bool:
        if not company_lc:
            return False
        return any(c in company_lc for c in self.consulting_companies)

    def w(self, key: str, default: float = 0.0) -> float:
        return float(self.weights.get(key, default))


def load_jd_spec(path: str | Path | None = None) -> JDSpec:
    p = Path(path) if path else DEFAULT_SPEC_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return JDSpec(data)
