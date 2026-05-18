"""
Uplan Semantic Graph Schema

Two-tier design:
  1. PageExtraction — FLAT model used as Gemini Flash response_schema.
     No Optional[ComplexModel] nesting. Safe for SDK schema conversion.
  2. SemanticGraph — RICH model built in Python by graph_builder.py.
     Never used as response_schema. Contains full typed node hierarchy.

This file is the typed contract between every layer of Uplan.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# TIER 1 — Flat extraction model (Gemini Flash response_schema)
# ═══════════════════════════════════════════════════════════════════════


class PageType(str, Enum):
    BANK_STATEMENT = "bank_statement"
    PAYSLIP = "payslip"
    TAX_RETURN = "tax_return"
    PASSPORT = "passport"
    SPONSOR_LETTER = "sponsor_letter"
    EMPLOYMENT_LETTER = "employment_letter"
    ENROLLMENT_LETTER = "enrollment_letter"
    AFFIDAVIT = "affidavit"
    UNKNOWN = "unknown"


class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"


class Transaction(BaseModel):
    """Single financial transaction row. Flat — safe for response_schema."""
    date: str = Field(description="Transaction date as it appears on the document")
    description: str = Field(description="Transaction description or narration text")
    amount: float = Field(description="Transaction amount as a number, no currency symbol")
    transaction_type: TransactionType = Field(description="credit for incoming, debit for outgoing")
    label: Optional[str] = Field(
        default=None,
        description="Source label if identifiable (e.g. 'salary', 'transfer'). null if unlabeled.",
    )


class PageExtraction(BaseModel):
    """
    Flat per-page extraction result from Gemini Flash.
    Used as response_schema — all fields are primitives or list[simple_model].
    No Optional[ComplexModel] to avoid SDK schema conversion failures.
    """

    page_number: int = Field(description="1-indexed page number within the document")
    page_type: PageType = Field(description="Document type detected on this page")
    raw_text: Optional[str] = Field(default=None, description="Raw text extracted via PyMuPDF fast-path")

    # -- Identity fields (passport, ID pages) ---------------------------------
    person_name: Optional[str] = Field(
        default=None,
        description=(
            "Full name of the PERSON who created/signed this document. "
            "On a passport/payslip = the applicant. On an affidavit = the declarant/sponsor."
        ),
    )
    date_of_birth: Optional[str] = Field(
        default=None,
        description=(
            "Date of birth of person_name (the signer/declarant). "
            "On a passport = the applicant's DOB. On an affidavit = the SPONSOR's DOB, NOT the applicant's."
        ),
    )
    applicant_dob: Optional[str] = Field(
        default=None,
        description=(
            "[AFFIDAVIT ONLY] Date of birth of the VISA APPLICANT (the beneficiary being sponsored). "
            "Extract from the family member table — the son/daughter entry, NOT the declarant's own row. "
            "e.g. if affidavit lists 'Korada Sai Kiran (Son), DOB: 02-06-2003', this = '02-06-2003'."
        ),
    )
    nationality: Optional[str] = Field(default=None, description="Nationality or citizenship")
    passport_number: Optional[str] = Field(default=None, description="Passport or ID number")

    # ── Financial fields (bank statements, payslips) ────────────────
    currency: Optional[str] = Field(default=None, description="3-letter currency code (e.g. JPY, USD)")
    opening_balance: Optional[float] = Field(default=None, description="Opening/starting balance on this page")
    closing_balance: Optional[float] = Field(default=None, description="Closing/ending balance on this page")
    transactions: list[Transaction] = Field(default_factory=list, description="All visible transactions")

    # ── Income / Employment fields ──────────────────────────────────
    employer_name: Optional[str] = Field(default=None, description="Employer or company name")
    job_title: Optional[str] = Field(default=None, description="Job title or designation")
    monthly_salary: Optional[float] = Field(default=None, description="Monthly salary amount")
    annual_income: Optional[float] = Field(default=None, description="Annual income amount")

    # ── Temporal fields ─────────────────────────────────────────────
    document_date: Optional[str] = Field(default=None, description="Date of issuance of this document")
    period_start: Optional[str] = Field(default=None, description="Start of the period this page covers")
    period_end: Optional[str] = Field(default=None, description="End of the period this page covers")
    employment_start: Optional[str] = Field(default=None, description="Employment start date if stated")
    employment_end: Optional[str] = Field(default=None, description="Employment end date if stated")

    # -- Sponsor fields -------------------------------------------------------
    sponsor_name: Optional[str] = Field(
        default=None,
        description="Financial sponsor's name — the person DECLARING/SIGNING the document",
    )
    sponsor_relationship: Optional[str] = Field(default=None, description="Relationship to applicant (e.g. 'Father', 'Mother')")
    sponsor_declared_income: Optional[float] = Field(default=None, description="Sponsor's total declared annual income from ALL sources combined")
    sponsor_income_currency: Optional[str] = Field(default=None, description="Currency of sponsor income")

    # -- Affidavit-specific fields (affidavit documents) ---------------------
    # An affidavit is signed BY the sponsor FOR the applicant.
    # person_name  = the declarant/sponsor (signs it, their DOB goes in date_of_birth)
    # applicant_name = the beneficiary (the actual visa applicant)
    # applicant_dob  = the beneficiary's DOB (from the family member table)
    applicant_name: Optional[str] = Field(
        default=None,
        description=(
            "[AFFIDAVIT ONLY] Name of the visa APPLICANT (beneficiary), distinct from the sponsor/declarant. "
            "Extract from phrases like 'father of X', 'for my son X', 'studies of X'. "
            "This is the person whose visa is being applied for."
        ),
    )
    declared_liquid_assets: Optional[float] = Field(
        default=None,
        description=(
            "[AFFIDAVIT ONLY] Sum of BANK/FINANCIAL account balances ONLY: "
            "savings accounts + fixed deposits + LIC + investment accounts. "
            "Do NOT include property value, land, gold, or physical assets here. "
            "e.g. SBI savings 90978 + SBI savings 154603 + Union Bank 154603 + SBI FD 1023668 + LIC 162290 + Bajaj 158519 = 1744664."
        ),
    )
    declared_movable_assets: Optional[float] = Field(
        default=None,
        description=(
            "[AFFIDAVIT ONLY] Value of movable physical assets: cash, gold, silver ornaments, jewelry. "
            "Do NOT include bank accounts or property/land here. "
            "e.g. 'Cash, Gold & Silver ornaments: Rs. 25,00,000' → 2500000."
        ),
    )
    declared_property_value: Optional[float] = Field(
        default=None,
        description=(
            "[AFFIDAVIT ONLY] Total declared value of immovable property: residential flats, land, agricultural land. "
            "e.g. 'Properties owned: Rs. 1,63,60,000' → 16360000."
        ),
    )
    declared_annual_income: Optional[float] = Field(
        default=None,
        description=(
            "[AFFIDAVIT ONLY] SUM of ALL annual income lines from ALL sources. "
            "Add salary + business + rent + other income. "
            "e.g. 300000 + 552000 + 348000 = 1200000."
        ),
    )
    # Keep for backward compat — deprecated in favor of declared_liquid_assets
    declared_assets_total: Optional[float] = Field(
        default=None,
        description="Deprecated. Use declared_liquid_assets instead.",
    )

    # ── Enrollment fields ───────────────────────────────────────────
    institution_name: Optional[str] = Field(default=None, description="Educational institution name")
    program_name: Optional[str] = Field(default=None, description="Program or course name")
    program_cost: Optional[float] = Field(default=None, description="Total program cost/tuition")
    program_duration_months: Optional[int] = Field(default=None, description="Program duration in months")
    enrollment_start: Optional[str] = Field(default=None, description="Enrollment or term start date")

    # ── Tax fields ──────────────────────────────────────────────────
    tax_year: Optional[int] = Field(default=None, description="Tax assessment year")
    taxable_income: Optional[float] = Field(default=None, description="Total taxable income reported")
    tax_paid: Optional[float] = Field(default=None, description="Total tax paid or deducted")

    # ── Anomaly signals ─────────────────────────────────────────────
    anomaly_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Observable irregularities: 'large_unlabeled_deposit', "
            "'date_format_inconsistent', 'possible_alteration', "
            "'balance_discontinuity', 'name_spelling_variation'"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# TIER 2 — Rich graph model (built in Python, NOT used as response_schema)
# ═══════════════════════════════════════════════════════════════════════


class SpikeEntry(BaseModel):
    """A financial spike — an anomalously large transaction."""
    date: str
    amount: float
    label: Optional[str] = None  # null = unlabeled = strongest anomaly signal
    ratio_to_average: float = Field(description="spike_amount / avg_monthly_credit")


class GapFlag(BaseModel):
    """An employment or temporal gap."""
    period_start: str
    period_end: str
    gap_days: int
    severity: str = Field(description="'warning' or 'critical'")


class IdentityNode(BaseModel):
    """Central node: applicant identity across all documents."""
    name_variants: list[str] = Field(default_factory=list, description="All name spellings found")
    canonical_name: Optional[str] = Field(default=None, description="Normalized canonical name")
    date_of_birth: Optional[str] = None
    nationality: Optional[str] = None
    passport_number: Optional[str] = None
    transliteration_flags: list[str] = Field(
        default_factory=list,
        description="Name variant mismatches that may be transliteration issues",
    )
    cross_doc_name_match: bool = Field(
        default=True,
        description="False if names across documents don't reconcile",
    )


class FinancialNode(BaseModel):
    """Financial summary across all bank statements, payslips, and affidavits."""
    currency: Optional[str] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None       # Bank stmt closing OR liquid assets from affidavit
    avg_monthly_income: Optional[float] = None
    total_credits: float = 0.0
    total_debits: float = 0.0
    transaction_count: int = 0
    spikes: list[SpikeEntry] = Field(default_factory=list)
    unlabeled_deposit_count: int = 0
    income_percentile: Optional[float] = Field(
        default=None,
        description="Contextual income percentile (hardcoded thresholds for PoC)",
    )
    # Affidavit-sourced wealth (non-liquid, for context only)
    declared_property_value: Optional[float] = Field(
        default=None,
        description="Declared immovable property value from affidavit (not liquid)",
    )
    declared_movable_assets: Optional[float] = Field(
        default=None,
        description="Declared gold/silver/cash value from affidavit",
    )
    source_is_affidavit: bool = Field(
        default=False,
        description="True if financial data comes from a declared affidavit (no transactions expected)",
    )


class TemporalNode(BaseModel):
    """Timeline coherence across all documents."""
    doc_date_ranges: list[dict] = Field(
        default_factory=list,
        description="[{'source': 'bank_stmt', 'start': '...', 'end': '...'}]",
    )
    employment_start: Optional[str] = None
    employment_end: Optional[str] = None
    visa_window_start: Optional[str] = None
    visa_window_end: Optional[str] = None
    gap_flags: list[GapFlag] = Field(default_factory=list)
    chronology_valid: bool = Field(
        default=True,
        description="False if document dates are logically impossible",
    )


class SponsorNode(BaseModel):
    """Sponsor details from sponsor letter / affidavit."""
    sponsor_name: Optional[str] = None
    relationship: Optional[str] = None
    declared_income: Optional[float] = None
    income_currency: Optional[str] = None
    income_supports_coverage: bool = Field(
        default=True,
        description="False if sponsor income is insufficient for applicant costs",
    )
    jurisdiction: Optional[str] = None


class EnrollmentNode(BaseModel):
    """Enrollment details from CoE / admission letter."""
    institution_name: Optional[str] = None
    program_name: Optional[str] = None
    program_cost: Optional[float] = None
    program_cost_currency: Optional[str] = None
    duration_months: Optional[int] = None
    enrollment_start: Optional[str] = None
    funds_cover_full_stay: bool = Field(
        default=True,
        description="False if applicant funds don't cover program cost + living",
    )
    coe_date_matches_visa: bool = Field(
        default=True,
        description="False if enrollment date doesn't align with visa window",
    )


class EdgeType(str, Enum):
    INCOME_LINK = "income_link"          # Identity ↔ Financial
    DATE_MATCH = "date_match"            # Identity ↔ Temporal
    SPONSOR_LINK = "sponsor_link"        # Identity ↔ Sponsor
    FUNDS_CHECK = "funds_check"          # Identity ↔ Enrollment
    INCOME_TO_BALANCE = "income_to_balance"  # Payslip income ↔ Bank deposit


class GraphEdge(BaseModel):
    """Typed relationship between graph nodes."""
    edge_type: EdgeType
    source_node: str = Field(description="e.g. 'identity', 'financial'")
    target_node: str = Field(description="e.g. 'financial', 'sponsor'")
    coherent: bool = Field(description="True if the relationship is consistent")
    detail: Optional[str] = Field(
        default=None,
        description="Explanation if incoherent (e.g. 'salary on payslip not found in bank deposits')",
    )


class SemanticGraph(BaseModel):
    """
    The complete semantic graph -- the core data structure of Uplan.
    Built by graph_builder.py from a list[PageExtraction].
    Consumed by the rule engine and specialist agents.
    ~1,200 tokens vs. ~60,000 tokens of raw document text.
    """
    identity: IdentityNode = Field(default_factory=IdentityNode)
    financial: FinancialNode = Field(default_factory=FinancialNode)
    temporal: TemporalNode = Field(default_factory=TemporalNode)
    sponsor: Optional[SponsorNode] = None
    enrollment: Optional[EnrollmentNode] = None
    edges: list[GraphEdge] = Field(default_factory=list)
    token_count: int = Field(default=0, description="Token count of this graph's JSON representation")
    source_page_count: int = Field(default=0, description="Total pages processed from source documents")
    estimated_raw_tokens: int = Field(default=0, description="Estimated token count of raw documents")
    source_doc_types: list[str] = Field(
        default_factory=list,
        description=(
            "All PageType values seen across processed pages. "
            "Agents use this to contextualize findings — e.g. if 'affidavit' is present and "
            "'bank_statement' is absent, zero transactions is EXPECTED not suspicious."
        ),
    )
