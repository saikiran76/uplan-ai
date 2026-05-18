"""
Uplan Deterministic Rule Engine

Pure math on SemanticGraph nodes. No LLM calls.
Each rule produces a RuleFinding with severity, field path, and expected vs actual values.
Thresholds come from VisaContext (jurisdiction-aware) when available,
falling back to config.py defaults.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

import config
from encoding.schema import SemanticGraph
from rules.context_profiles import VisaContext


def _alpha(ctx: Optional[VisaContext]) -> float:
    return ctx.spike_ratio_alpha if ctx else config.ALPHA_SPIKE_RATIO

def _epsilon(ctx: Optional[VisaContext]) -> float:
    return ctx.income_balance_epsilon if ctx else config.EPSILON_INCOME_COHERENCE

def _gap_warn(ctx: Optional[VisaContext]) -> int:
    return ctx.gap_warn_days if ctx else config.DELTA_GAP_WARN_DAYS

def _gap_crit(ctx: Optional[VisaContext]) -> int:
    return ctx.gap_crit_days if ctx else config.DELTA_GAP_CRIT_DAYS


class RuleFinding(BaseModel):
    """Output of a single rule check."""
    rule_id: str
    severity: str = Field(description="'info', 'warning', or 'critical'")
    field_path: str = Field(description="Dot-path to the relevant graph field")
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


def _check_spike_ratio(graph: SemanticGraph, ctx: Optional[VisaContext] = None) -> list[RuleFinding]:
    """FIN-001: Flag financial spikes exceeding α × average."""
    alpha = _alpha(ctx)
    findings = []
    for spike in graph.financial.spikes:
        severity = "critical" if spike.label is None else "warning"
        findings.append(RuleFinding(
            rule_id="FIN-001",
            severity=severity,
            field_path=f"financial.spikes[date={spike.date}]",
            message=(
                f"Transaction on {spike.date} is {spike.ratio_to_average}× the average credit"
                f"{' — NO SOURCE LABEL (unlabeled deposit)' if spike.label is None else f' — labeled: {spike.label}'}"
            ),
            expected=f"ratio ≤ {alpha}",
            actual=f"ratio = {spike.ratio_to_average}",
        ))
    return findings


def _check_unlabeled_deposits(graph: SemanticGraph, ctx: Optional[VisaContext] = None) -> list[RuleFinding]:
    """FIN-002: Flag any unlabeled deposits."""
    if graph.financial.unlabeled_deposit_count > 0:
        return [RuleFinding(
            rule_id="FIN-002",
            severity="warning",
            field_path="financial.unlabeled_deposit_count",
            message=f"{graph.financial.unlabeled_deposit_count} deposit(s) have no source label",
            expected="0 unlabeled deposits",
            actual=str(graph.financial.unlabeled_deposit_count),
        )]
    return []


def _check_income_balance_coherence(graph: SemanticGraph, ctx: Optional[VisaContext] = None) -> list[RuleFinding]:
    """FIN-003: Check if declared income × months ≈ balance delta."""
    epsilon = _epsilon(ctx)
    fin = graph.financial
    if fin.avg_monthly_income and fin.opening_balance is not None and fin.closing_balance is not None:
        balance_delta = fin.closing_balance - fin.opening_balance
        # Estimate months from transaction count (rough: ~30 txns/month)
        est_months = max(fin.transaction_count / 30, 1)
        expected_delta = fin.avg_monthly_income * est_months
        if expected_delta > 0:
            deviation = abs(balance_delta - expected_delta) / expected_delta
            if deviation > epsilon:
                return [RuleFinding(
                    rule_id="FIN-003",
                    severity="warning" if deviation < 0.5 else "critical",
                    field_path="financial.closing_balance vs avg_monthly_income",
                    message=(
                        f"Balance delta ({balance_delta:,.0f}) deviates {deviation:.0%} "
                        f"from expected ({expected_delta:,.0f} over ~{est_months:.0f} months)"
                    ),
                    expected=f"deviation ≤ {epsilon:.0%}",
                    actual=f"deviation = {deviation:.0%}",
                )]
    return []


def _check_min_balance(graph: SemanticGraph, ctx: Optional[VisaContext] = None) -> list[RuleFinding]:
    """FIN-004: Check if closing balance meets jurisdiction minimum."""
    if ctx and ctx.min_balance is not None and graph.financial.closing_balance is not None:
        if graph.financial.closing_balance < ctx.min_balance:
            shortfall = ctx.min_balance - graph.financial.closing_balance
            return [RuleFinding(
                rule_id="FIN-004",
                severity="critical",
                field_path="financial.closing_balance",
                message=(
                    f"Closing balance ({graph.financial.closing_balance:,.0f} {ctx.balance_currency}) "
                    f"below minimum requirement ({ctx.min_balance:,.0f} {ctx.balance_currency}). "
                    f"Shortfall: {shortfall:,.0f} {ctx.balance_currency}"
                ),
                expected=f"balance ≥ {ctx.min_balance:,.0f} {ctx.balance_currency}",
                actual=f"{graph.financial.closing_balance:,.0f} {ctx.balance_currency}",
            )]
    return []


def _check_name_mismatch(graph: SemanticGraph, ctx: Optional[VisaContext] = None) -> list[RuleFinding]:
    """IDN-001: Flag cross-document name mismatches."""
    if not graph.identity.cross_doc_name_match:
        return [RuleFinding(
            rule_id="IDN-001",
            severity="critical",
            field_path="identity.cross_doc_name_match",
            message=(
                f"Name mismatch across documents: {graph.identity.name_variants}. "
                f"Possible transliteration issues: {graph.identity.transliteration_flags}"
            ),
            expected="Consistent name across all documents",
            actual=f"{len(graph.identity.name_variants)} variants found",
        )]
    return []


def _check_sponsor_coverage(graph: SemanticGraph, ctx: Optional[VisaContext] = None) -> list[RuleFinding]:
    """SPN-001: Check if sponsor income supports applicant coverage."""
    if graph.sponsor and not graph.sponsor.income_supports_coverage:
        return [RuleFinding(
            rule_id="SPN-001",
            severity="critical",
            field_path="sponsor.income_supports_coverage",
            message="Sponsor's declared income is insufficient to cover applicant costs",
            expected="income_supports_coverage = True",
            actual="income_supports_coverage = False",
        )]
    return []


def _check_enrollment_funds(graph: SemanticGraph, ctx: Optional[VisaContext] = None) -> list[RuleFinding]:
    """ENR-001: Check if funds cover full program stay."""
    if graph.enrollment and not graph.enrollment.funds_cover_full_stay:
        return [RuleFinding(
            rule_id="ENR-001",
            severity="critical",
            field_path="enrollment.funds_cover_full_stay",
            message="Available funds do not cover the full program cost and stay duration",
            expected="funds_cover_full_stay = True",
            actual="funds_cover_full_stay = False",
        )]
    return []


def _check_enrollment_dates(graph: SemanticGraph, ctx: Optional[VisaContext] = None) -> list[RuleFinding]:
    """ENR-002: Check if CoE dates align with visa window."""
    if graph.enrollment and not graph.enrollment.coe_date_matches_visa:
        return [RuleFinding(
            rule_id="ENR-002",
            severity="warning",
            field_path="enrollment.coe_date_matches_visa",
            message="Enrollment start date does not align with visa application window",
            expected="coe_date_matches_visa = True",
            actual="coe_date_matches_visa = False",
        )]
    return []


def _check_chronology(graph: SemanticGraph, ctx: Optional[VisaContext] = None) -> list[RuleFinding]:
    """TMP-001: Flag chronological inconsistencies."""
    if not graph.temporal.chronology_valid:
        return [RuleFinding(
            rule_id="TMP-001",
            severity="critical",
            field_path="temporal.chronology_valid",
            message="Document dates contain logically impossible chronology",
            expected="chronology_valid = True",
            actual="chronology_valid = False",
        )]
    return []


def _check_employment_gaps(graph: SemanticGraph, ctx: Optional[VisaContext] = None) -> list[RuleFinding]:
    """TMP-002: Flag employment gaps."""
    warn_days = _gap_warn(ctx)
    findings = []
    for gap in graph.temporal.gap_flags:
        findings.append(RuleFinding(
            rule_id="TMP-002",
            severity=gap.severity,
            field_path=f"temporal.gap_flags[{gap.period_start}–{gap.period_end}]",
            message=f"Employment gap of {gap.gap_days} days ({gap.period_start} to {gap.period_end})",
            expected=f"gap ≤ {warn_days} days",
            actual=f"{gap.gap_days} days",
        ))
    return findings


# ── Public API ───────────────────────────────────────────────────────

ALL_RULES = [
    _check_spike_ratio,
    _check_unlabeled_deposits,
    _check_income_balance_coherence,
    _check_min_balance,
    _check_name_mismatch,
    _check_sponsor_coverage,
    _check_enrollment_funds,
    _check_enrollment_dates,
    _check_chronology,
    _check_employment_gaps,
]


def run_rules(
    graph: SemanticGraph,
    context: Optional[VisaContext] = None,
) -> list[RuleFinding]:
    """
    Execute all deterministic rules against the semantic graph.
    Uses VisaContext for jurisdiction-aware thresholds when available.
    Returns a list of findings sorted by severity (critical first).
    """
    findings: list[RuleFinding] = []
    for rule_fn in ALL_RULES:
        findings.extend(rule_fn(graph, context))

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_order.get(f.severity, 3))
    return findings
