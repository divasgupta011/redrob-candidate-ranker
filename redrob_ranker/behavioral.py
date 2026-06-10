"""Behavioral availability modifier.

Static fit answers "could this person do the job?". The Redrob signals answer a
different, equally important question: "can we actually hire them right now?" The
JD is blunt about it -- a perfect-on-paper candidate who hasn't logged in for six
months and replies to 5% of recruiters is, for hiring purposes, not available, and
should be down-weighted.

We fold the relevant signals into an ``availability`` score in [0, 1], then turn it
into a *multiplier* on the fit score:

    modifier = 1 - weight * (1 - availability)

so a fully-available candidate is unchanged (x1.0) and a fully-unavailable one
takes at most a ``weight`` haircut (x0.70 by default). This makes the signals a
real differentiator -- e.g. it cleanly separates "behavioral twins" (identical
resumes, opposite engagement) -- without letting them dominate genuine fit.

All thresholds/weights come from ``config/jd_spec.yaml`` (``behavioral``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .features import clip
from .jdspec import JDSpec
from .schema import Candidate

_DEFAULT_REF = date(2026, 6, 1)


@dataclass(slots=True)
class BehavioralResult:
    modifier: float                 # multiply the fit score by this
    availability: float             # [0,1] raw availability
    components: dict                # per-signal sub-scores (diagnostics)
    positives: list[str] = field(default_factory=list)   # for reasoning
    concerns: list[str] = field(default_factory=list)


def _recency_score(c: Candidate, ref: date, full_days: float, zero_days: float) -> tuple[float, int]:
    la = c.signals.last_active_date
    if la is None:
        return 0.3, -1
    days = (ref - la).days
    if days <= full_days:
        return 1.0, max(days, 0)
    if days >= zero_days:
        return 0.0, days
    return clip(1.0 - (days - full_days) / (zero_days - full_days)), days


def _notice_score(days: int, great: float, ok: float) -> float:
    if days <= great:
        return 1.0
    if days <= ok:
        return clip(1.0 - 0.5 * (days - great) / max(ok - great, 1), 0.5, 1.0)
    return clip(0.5 - 0.3 * (days - ok) / max(180 - ok, 1), 0.2, 0.5)


def behavioral_modifier(c: Candidate, spec: JDSpec) -> BehavioralResult:
    b = spec.behavioral
    ref = spec.reference_date or _DEFAULT_REF
    comp_w = b.get("components", {})
    weight = float(b.get("weight", 0.30))
    sig = c.signals

    recency, days = _recency_score(c, ref, float(b.get("recency_days_full", 30)),
                                   float(b.get("recency_days_zero", 180)))
    rr_good = float(b.get("response_rate_good", 0.5))
    response = clip(sig.recruiter_response_rate / rr_good) if rr_good else 0.0
    open_to_work = 1.0 if sig.open_to_work_flag else 0.0
    interview = clip(sig.interview_completion_rate)
    notice = _notice_score(sig.notice_period_days, float(b.get("notice_period_great", 30)),
                           float(b.get("notice_period_ok", 90)))
    completeness = clip(sig.profile_completeness_score / 100.0)
    saved = clip(sig.saved_by_recruiters_30d / 10.0)
    verified = (int(sig.verified_email) + int(sig.verified_phone)) / 2.0

    components = {
        "recency": recency,
        "recruiter_response_rate": response,
        "open_to_work_flag": open_to_work,
        "interview_completion_rate": interview,
        "notice_period": notice,
        "profile_completeness_score": completeness,
        "saved_by_recruiters_30d": saved,
        "verified": verified,
    }
    wsum = sum(float(comp_w.get(k, 0.0)) for k in components) or 1.0
    availability = sum(float(comp_w.get(k, 0.0)) * v for k, v in components.items()) / wsum
    modifier = 1.0 - weight * (1.0 - availability)

    # human-readable notes for the reasoning generator
    positives, concerns = [], []
    if days == -1:
        concerns.append("no recent activity recorded")
    elif recency >= 0.9:
        positives.append("recently active")
    elif days >= 150:
        concerns.append(f"inactive for ~{days // 30} months")
    if sig.recruiter_response_rate >= rr_good:
        positives.append(f"responsive to recruiters ({sig.recruiter_response_rate:.0%})")
    elif sig.recruiter_response_rate <= float(b.get("response_rate_floor", 0.1)):
        concerns.append(f"very low recruiter response rate ({sig.recruiter_response_rate:.0%})")
    if sig.open_to_work_flag:
        positives.append("open to work")
    if sig.notice_period_days > float(b.get("notice_period_ok", 90)):
        concerns.append(f"long notice period ({sig.notice_period_days}d)")
    elif sig.notice_period_days <= float(b.get("notice_period_great", 30)):
        positives.append(f"short notice ({sig.notice_period_days}d)")

    return BehavioralResult(modifier=modifier, availability=availability, components=components,
                            positives=positives, concerns=concerns)
