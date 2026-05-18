"""
Uplan Graph Builder — Merge per-page extractions into a single SemanticGraph.

Takes list[PageExtraction] (flat, per-page) and produces a SemanticGraph
(rich, cross-document). This is pure Python — no LLM calls.

Key operations:
  - Identity deduplication (lowercase + strip normalization)
  - Financial aggregation (time-series, spike detection)
  - Temporal construction (date ranges, gap detection)
  - Edge building (cross-node coherence checks)
  - Token counting (via Gemini count_tokens API)
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Optional

from config import ALPHA_SPIKE_RATIO, PRO_MODEL, client
from encoding.schema import (
    EdgeType,
    EnrollmentNode,
    FinancialNode,
    GapFlag,
    GraphEdge,
    IdentityNode,
    PageExtraction,
    PageType,
    SemanticGraph,
    SpikeEntry,
    SponsorNode,
    TemporalNode,
)


def _normalize_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace for name deduplication."""
    return " ".join(name.lower().strip().split())


def _build_identity(pages: list[PageExtraction]) -> IdentityNode:
    """Merge identity information across all pages.

    Priority: passport > applicant_name on affidavit > person_name on non-affidavit.
    DOB priority: passport DOB > applicant_dob from affidavit > declarant DOB (last resort).
    """
    raw_names: list[str] = []
    applicant_names: list[str] = []   # From affidavit applicant_name field
    sponsor_names: list[str] = []     # Declarant names (person_name on affidavit)
    dob: Optional[str] = None
    applicant_dob: Optional[str] = None   # From affidavit's family member table
    declarant_dob: Optional[str] = None   # The sponsor's own DOB (fallback only)
    nationality: Optional[str] = None
    passport_no: Optional[str] = None

    for p in pages:
        if p.page_type == PageType.AFFIDAVIT:
            # On affidavit: person_name = declarant/sponsor, applicant_name = beneficiary
            if p.applicant_name:
                applicant_names.append(p.applicant_name)
            if p.person_name:
                sponsor_names.append(p.person_name)
            # applicant_dob is the son/daughter's DOB from the family table
            if p.applicant_dob and not applicant_dob:
                applicant_dob = p.applicant_dob
            # date_of_birth on affidavit is the DECLARANT's (sponsor's) DOB
            if p.date_of_birth and not declarant_dob:
                declarant_dob = p.date_of_birth
        else:
            if p.person_name:
                raw_names.append(p.person_name)
            if p.date_of_birth and not dob:
                dob = p.date_of_birth
        if p.nationality and not nationality:
            nationality = p.nationality
        if p.passport_number and not passport_no:
            passport_no = p.passport_number

    # DOB priority: non-affidavit (passport) > applicant_dob from affidavit > declarant DOB
    resolved_dob = dob or applicant_dob
    # Only fall back to declarant DOB if nothing else available (and log it)
    if resolved_dob is None and declarant_dob:
        print(f"  [WARN] identity: no applicant DOB found, falling back to declarant DOB {declarant_dob}")
        resolved_dob = declarant_dob

    # Name priority: non-affidavit > applicant_name from affidavit > declarant name
    candidate_names = raw_names or applicant_names or sponsor_names

    # Deduplicate names via normalization
    seen_normalized: dict[str, str] = {}
    for name in candidate_names:
        norm = _normalize_name(name)
        if norm not in seen_normalized:
            seen_normalized[norm] = name

    unique_variants = list(seen_normalized.values())
    canonical = unique_variants[0] if unique_variants else None

    transliteration_flags = []
    if len(seen_normalized) > 1:
        transliteration_flags = [
            f"'{v}' vs '{unique_variants[0]}'" for v in unique_variants[1:]
        ]

    return IdentityNode(
        name_variants=unique_variants,
        canonical_name=canonical,
        date_of_birth=resolved_dob,
        nationality=nationality,
        passport_number=passport_no,
        transliteration_flags=transliteration_flags,
        cross_doc_name_match=len(seen_normalized) <= 1,
    )


def _build_financial(pages: list[PageExtraction]) -> FinancialNode:
    """Aggregate financial data across bank statements, payslips, and affidavits.

    For affidavits:
      - declared_liquid_assets (savings + FD only) -> closing_balance
      - declared_annual_income / 12               -> avg_monthly_income
      - declared_property_value                   -> financial.declared_property_value (non-liquid)
      - declared_movable_assets                   -> financial.declared_movable_assets (non-liquid)
      - source_is_affidavit = True                -> agents know 0 transactions is expected
    """
    currency: Optional[str] = None
    all_opening: list[float] = []
    all_closing: list[float] = []
    all_salaries: list[float] = []
    all_credits: list[float] = []
    all_debits: list[float] = []
    unlabeled_count = 0
    total_transactions = 0
    has_affidavit = False
    property_value: Optional[float] = None
    movable_assets: Optional[float] = None

    for p in pages:
        if p.currency and not currency:
            currency = p.currency

        if p.page_type == PageType.AFFIDAVIT:
            has_affidavit = True
            # Use declared_liquid_assets (bank+FD only) as closing balance
            liquid = p.declared_liquid_assets
            # Fall back to deprecated declared_assets_total if new field missing
            if liquid is None:
                liquid = p.declared_assets_total
            if liquid is not None:
                all_closing.append(liquid)

            # Annual income -> monthly equivalent
            annual = p.declared_annual_income
            if annual is None:
                annual = p.sponsor_declared_income
            if annual is not None:
                all_salaries.append(annual / 12.0)

            # Non-liquid wealth
            if p.declared_property_value is not None and property_value is None:
                property_value = p.declared_property_value
            if p.declared_movable_assets is not None and movable_assets is None:
                movable_assets = p.declared_movable_assets
        else:
            if p.opening_balance is not None:
                all_opening.append(p.opening_balance)
            if p.closing_balance is not None:
                all_closing.append(p.closing_balance)
            if p.monthly_salary is not None:
                all_salaries.append(p.monthly_salary)

        for txn in p.transactions:
            total_transactions += 1
            if txn.transaction_type.value == "credit":
                all_credits.append(txn.amount)
                if txn.label is None:
                    unlabeled_count += 1
            else:
                all_debits.append(txn.amount)

    total_credits = sum(all_credits)
    total_debits = sum(all_debits)
    avg_monthly = mean(all_salaries) if all_salaries else None

    # Spike detection: credits > ALPHA x average credit
    spikes: list[SpikeEntry] = []
    if all_credits:
        avg_credit = mean(all_credits)
        if avg_credit > 0:
            for p in pages:
                for txn in p.transactions:
                    if txn.transaction_type.value == "credit":
                        ratio = txn.amount / avg_credit
                        if ratio > ALPHA_SPIKE_RATIO:
                            spikes.append(SpikeEntry(
                                date=txn.date,
                                amount=txn.amount,
                                label=txn.label,
                                ratio_to_average=round(ratio, 2),
                            ))

    return FinancialNode(
        currency=currency,
        opening_balance=all_opening[0] if all_opening else None,
        closing_balance=all_closing[-1] if all_closing else None,
        avg_monthly_income=avg_monthly,
        total_credits=total_credits,
        total_debits=total_debits,
        transaction_count=total_transactions,
        spikes=spikes,
        unlabeled_deposit_count=unlabeled_count,
        declared_property_value=property_value,
        declared_movable_assets=movable_assets,
        source_is_affidavit=has_affidavit,
    )


def _build_temporal(pages: list[PageExtraction]) -> TemporalNode:
    """Build timeline from document dates and employment periods."""
    date_ranges: list[dict] = []
    emp_start: Optional[str] = None
    emp_end: Optional[str] = None

    for p in pages:
        if p.period_start or p.period_end:
            date_ranges.append({
                "source": p.page_type.value,
                "page": p.page_number,
                "start": p.period_start,
                "end": p.period_end,
            })
        if p.employment_start and not emp_start:
            emp_start = p.employment_start
        if p.employment_end and not emp_end:
            emp_end = p.employment_end

    return TemporalNode(
        doc_date_ranges=date_ranges,
        employment_start=emp_start,
        employment_end=emp_end,
        # Gap detection and chronology validation require date parsing.
        # For PoC: flag if no employment dates found at all.
        chronology_valid=True,  # Will be refined by rule engine
    )


def _build_sponsor(pages: list[PageExtraction]) -> Optional[SponsorNode]:
    """Extract sponsor data if any sponsor pages exist."""
    for p in pages:
        if p.page_type in (PageType.SPONSOR_LETTER, PageType.AFFIDAVIT):
            if p.sponsor_name or p.sponsor_declared_income is not None:
                return SponsorNode(
                    sponsor_name=p.sponsor_name,
                    relationship=p.sponsor_relationship,
                    declared_income=p.sponsor_declared_income,
                    income_currency=p.sponsor_income_currency,
                )
    return None


def _build_enrollment(pages: list[PageExtraction]) -> Optional[EnrollmentNode]:
    """Extract enrollment data if any enrollment pages exist."""
    for p in pages:
        if p.page_type == PageType.ENROLLMENT_LETTER:
            if p.institution_name or p.program_cost is not None:
                return EnrollmentNode(
                    institution_name=p.institution_name,
                    program_name=p.program_name,
                    program_cost=p.program_cost,
                    duration_months=p.program_duration_months,
                    enrollment_start=p.enrollment_start,
                )
    return None


def _build_edges(
    identity: IdentityNode,
    financial: FinancialNode,
    temporal: TemporalNode,
    sponsor: Optional[SponsorNode],
    enrollment: Optional[EnrollmentNode],
) -> list[GraphEdge]:
    """Build typed edges representing cross-node relationships."""
    edges: list[GraphEdge] = []

    # Identity → Financial: do we have both identity and financial data?
    if identity.canonical_name and financial.transaction_count > 0:
        edges.append(GraphEdge(
            edge_type=EdgeType.INCOME_LINK,
            source_node="identity",
            target_node="financial",
            coherent=True,  # Refined by rule engine
            detail=None,
        ))

    # Identity → Temporal: employment dates present?
    if identity.canonical_name and temporal.employment_start:
        edges.append(GraphEdge(
            edge_type=EdgeType.DATE_MATCH,
            source_node="identity",
            target_node="temporal",
            coherent=temporal.chronology_valid,
            detail=None if temporal.chronology_valid else "Chronology invalid",
        ))

    # Identity → Sponsor: sponsor relationship exists?
    if sponsor and sponsor.sponsor_name:
        edges.append(GraphEdge(
            edge_type=EdgeType.SPONSOR_LINK,
            source_node="identity",
            target_node="sponsor",
            coherent=sponsor.income_supports_coverage,
            detail=None if sponsor.income_supports_coverage else "Sponsor income insufficient",
        ))

    # Identity → Enrollment: enrollment data exists?
    if enrollment and enrollment.institution_name:
        edges.append(GraphEdge(
            edge_type=EdgeType.FUNDS_CHECK,
            source_node="identity",
            target_node="enrollment",
            coherent=enrollment.funds_cover_full_stay,
            detail=None if enrollment.funds_cover_full_stay else "Funds insufficient for program",
        ))

    # Financial → Temporal: salary-to-deposit coherence
    if financial.avg_monthly_income and financial.transaction_count > 0:
        edges.append(GraphEdge(
            edge_type=EdgeType.INCOME_TO_BALANCE,
            source_node="financial",
            target_node="temporal",
            coherent=True,  # Refined by rule engine
            detail=None,
        ))

    return edges


def _count_graph_tokens(graph: SemanticGraph) -> int:
    """Count tokens of the graph JSON using Gemini's tokenizer."""
    graph_json = graph.model_dump_json()
    try:
        response = client.models.count_tokens(
            model=PRO_MODEL,
            contents=graph_json,
        )
        return response.total_tokens
    except Exception as e:
        # Fallback: rough estimate (1 token ≈ 4 chars)
        print(f"  ! Token counting failed ({e}), using estimate")
        return len(graph_json) // 4


def build_graph(pages: list[PageExtraction]) -> SemanticGraph:
    """
    Merge per-page extractions into a single SemanticGraph.

    This is pure Python — no LLM calls except for token counting.
    The output graph is ~1,200 tokens regardless of input document size.
    """
    print("  -> Building identity node...")
    identity = _build_identity(pages)

    print("  -> Building financial node...")
    financial = _build_financial(pages)

    print("  -> Building temporal node...")
    temporal = _build_temporal(pages)

    print("  -> Building sponsor node...")
    sponsor = _build_sponsor(pages)

    print("  -> Building enrollment node...")
    enrollment = _build_enrollment(pages)

    print("  -> Building edges...")
    edges = _build_edges(identity, financial, temporal, sponsor, enrollment)

    # Estimate raw token count: ~2,000 tokens per page of scanned document
    estimated_raw = len(pages) * 2000

    graph = SemanticGraph(
        identity=identity,
        financial=financial,
        temporal=temporal,
        sponsor=sponsor,
        enrollment=enrollment,
        edges=edges,
        source_page_count=len(pages),
        estimated_raw_tokens=estimated_raw,
        source_doc_types=list({p.page_type.value for p in pages}),
    )

    # 4. Final compression metrics
    print("  -> Counting graph tokens...")
    graph.token_count = _count_graph_tokens(graph)

    return graph
