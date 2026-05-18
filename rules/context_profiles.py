"""
Uplan Visa Context Profiles

Jurisdiction-aware threshold lookup. Hardcoded for 4 visa categories in the PoC,
designed to be RAG-fed from immigration policy PDFs in production.

The VisaContext object gets passed into the rule engine and injected into every
agent prompt — same algebra, different α/ε/δ values per visa+country combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisaContext:
    """Threshold set for a specific visa_type + destination combination."""
    visa_type: str
    destination: str
    applicant_country: Optional[str] = None

    # Financial thresholds
    min_balance: Optional[float] = None
    balance_currency: str = "USD"
    spike_ratio_alpha: float = 3.0
    income_balance_epsilon: float = 0.20  # Tolerance for salary vs deposit mismatch

    # Temporal thresholds
    income_months_required: int = 6
    gap_warn_days: int = 30
    gap_crit_days: int = 90

    # Sponsor thresholds
    sponsor_coverage_multiplier: float = 1.2

    # Required document types for this visa category
    required_doc_types: list[str] = field(default_factory=list)

    # Context string for agent prompts
    jurisdiction_context: str = ""

    def to_dict(self) -> dict:
        """Serialize for state storage and agent prompt injection."""
        return {
            "visa_type": self.visa_type,
            "destination": self.destination,
            "applicant_country": self.applicant_country,
            "min_balance": self.min_balance,
            "balance_currency": self.balance_currency,
            "spike_ratio_alpha": self.spike_ratio_alpha,
            "income_balance_epsilon": self.income_balance_epsilon,
            "income_months_required": self.income_months_required,
            "gap_warn_days": self.gap_warn_days,
            "gap_crit_days": self.gap_crit_days,
            "sponsor_coverage_multiplier": self.sponsor_coverage_multiplier,
            "required_doc_types": self.required_doc_types,
            "jurisdiction_context": self.jurisdiction_context,
        }


# ── Hardcoded profiles for PoC ──────────────────────────────────────

PROFILES: dict[tuple[str, str], VisaContext] = {
    ("student", "japan"): VisaContext(
        visa_type="student",
        destination="japan",
        min_balance=2_000_000,
        balance_currency="JPY",
        spike_ratio_alpha=3.0,
        income_balance_epsilon=0.15,
        income_months_required=6,
        gap_warn_days=30,
        gap_crit_days=90,
        sponsor_coverage_multiplier=1.2,
        required_doc_types=[
            "passport",
            "bank_statement",
            "payslip",
            "enrollment_letter",
        ],
        jurisdiction_context=(
            "Japanese Student (COE) visa requirements: "
            "Minimum bank balance of ¥2,000,000. "
            "6 months of consecutive bank statements required. "
            "Certificate of Eligibility (COE) must match visa application dates. "
            "All large deposits must have documented sources. "
            "Sponsor must demonstrate income ≥1.2× total program costs."
        ),
    ),
    ("student", "uk"): VisaContext(
        visa_type="student",
        destination="uk",
        min_balance=12_006,  # 9 months × £1,334
        balance_currency="GBP",
        spike_ratio_alpha=2.5,
        income_balance_epsilon=0.10,
        income_months_required=3,
        gap_warn_days=14,
        gap_crit_days=30,
        sponsor_coverage_multiplier=1.0,
        required_doc_types=[
            "passport",
            "bank_statement",
            "enrollment_letter",
        ],
        jurisdiction_context=(
            "UK Student (Tier 4/Student Route) visa requirements: "
            "Minimum funds of £1,334/month for up to 9 months (£12,006 total) "
            "for courses outside London. "
            "Funds must be held for 28 consecutive days. "
            "CAS letter required from licensed sponsor institution."
        ),
    ),
    ("student", "us"): VisaContext(
        visa_type="student",
        destination="us",
        min_balance=None,  # Full tuition + living — varies by institution
        balance_currency="USD",
        spike_ratio_alpha=3.0,
        income_balance_epsilon=0.20,
        income_months_required=6,
        gap_warn_days=30,
        gap_crit_days=90,
        sponsor_coverage_multiplier=1.0,
        required_doc_types=[
            "passport",
            "bank_statement",
            "enrollment_letter",
            "sponsor_letter",
        ],
        jurisdiction_context=(
            "US F-1 Student visa requirements: "
            "Must demonstrate funding for full tuition + living expenses. "
            "I-20 form from SEVP-certified school required. "
            "Bank statements must show sufficient funds for first year minimum. "
            "Sponsor's I-134 Affidavit of Support needed if not self-funded."
        ),
    ),
    ("work", "japan"): VisaContext(
        visa_type="work",
        destination="japan",
        min_balance=500_000,
        balance_currency="JPY",
        spike_ratio_alpha=4.0,
        income_balance_epsilon=0.20,
        income_months_required=3,
        gap_warn_days=30,
        gap_crit_days=60,
        sponsor_coverage_multiplier=1.0,
        required_doc_types=[
            "passport",
            "employment_letter",
            "tax_return",
            "bank_statement",
        ],
        jurisdiction_context=(
            "Japanese Work visa (Engineer/Specialist in Humanities) requirements: "
            "Employment contract or offer letter from Japanese company required. "
            "Tax returns demonstrating income history. "
            "Company must be registered as a visa sponsor."
        ),
    ),
}

# ── Default profile (generic fallback) ──────────────────────────────

_DEFAULT = VisaContext(
    visa_type="general",
    destination="unknown",
    min_balance=None,
    balance_currency="USD",
    spike_ratio_alpha=3.0,
    income_balance_epsilon=0.20,
    income_months_required=6,
    gap_warn_days=30,
    gap_crit_days=90,
    sponsor_coverage_multiplier=1.2,
    required_doc_types=["passport", "bank_statement"],
    jurisdiction_context="General visa requirements. No jurisdiction-specific thresholds applied.",
)


def get_context(visa_type: str, destination: str) -> VisaContext:
    """Look up the threshold set for a visa_type + destination pair."""
    key = (visa_type.lower().strip(), destination.lower().strip())
    return PROFILES.get(key, _DEFAULT)


def list_available_profiles() -> list[dict]:
    """List all available visa profiles for UI dropdowns."""
    return [
        {
            "visa_type": ctx.visa_type,
            "destination": ctx.destination,
            "label": f"{ctx.visa_type.title()} → {ctx.destination.title()}",
        }
        for ctx in PROFILES.values()
    ]


# ── Destination / visa type options for UI ──────────────────────────

DESTINATIONS = ["Japan", "UK", "US", "Canada", "Australia", "Germany"]
VISA_TYPES = ["Student", "Work", "Tourist", "Business", "Dependent"]
APPLICANT_COUNTRIES = [
    "India", "China", "Philippines", "Vietnam", "Indonesia",
    "Bangladesh", "Nepal", "Pakistan", "Sri Lanka", "Nigeria",
    "Brazil", "Mexico", "South Korea", "Thailand", "Other",
]
