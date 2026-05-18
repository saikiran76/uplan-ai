"""
Narrative Synthesis Agent

Receives ALL 5 specialist agent findings + rule engine findings + full semantic graph.
Produces: verdict, risk score, rejection narrative, rebuttal guidance, citations.
Includes exponential backoff retry for rate limits.
"""

from __future__ import annotations

import json
import random
import time
from typing import Optional

from pydantic import BaseModel, Field

from config import (
    PRO_MODEL,
    API_RETRY_ATTEMPTS,
    API_RETRY_BASE_DELAY,
    API_RETRY_MAX_DELAY,
    client,
)
from orchestrator.state import UplanState


class NarrativeVerdict(BaseModel):
    """Final cross-document verdict -- the output of the entire Uplan pipeline."""
    verdict: str = Field(description="'PASS', 'CONDITIONAL', 'INCOMPLETE_DOSSIER', or 'FAIL'")
    risk_score: float = Field(description="0.0 = clean application, 1.0 = near-certain rejection")
    missing_documents: list[str] = Field(
        default_factory=list,
        description="Document types that are missing from the dossier (e.g. 'passport', 'enrollment_letter')",
    )
    critical_issues: list[str] = Field(default_factory=list, description="Bullet-point critical issues found WITHIN provided documents")
    warning_issues: list[str] = Field(default_factory=list, description="Bullet-point warnings")
    rejection_narrative: str = Field(
        description="Pre-submission audit narrative — formal, specific, distinguishing anomalies from missing docs"
    )
    rebuttal_guidance: list[str] = Field(
        default_factory=list,
        description="For each issue: what additional document or explanation would resolve it",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Graph node paths supporting each finding (e.g. 'financial.spikes[0]')",
    )


# All document types that constitute a complete dossier
COMPLETE_DOSSIER_TYPES = [
    "passport", "bank_statement", "payslip", "tax_return",
    "sponsor_letter", "employment_letter", "enrollment_letter", "affidavit",
]


class NarrativeAgent:
    """Cross-document synthesis agent -- the final reasoning step."""

    def run(self, state: UplanState) -> dict:
        graph = state["semantic_graph"]
        doc_types = graph.get("source_doc_types", [])
        source_is_affidavit = graph.get("financial", {}).get("source_is_affidavit", False)

        # Compute dossier completeness
        present = set(doc_types)
        missing = [d for d in COMPLETE_DOSSIER_TYPES if d not in present]
        is_partial = len(missing) > 0

        doc_context_note = ""
        if source_is_affidavit and "bank_statement" not in doc_types:
            doc_context_note = (
                "\n\nDOCUMENT CONTEXT: The financial data comes ONLY from an affidavit (sworn declaration). "
                "An affidavit legitimately has zero bank transactions -- it declares point-in-time asset values. "
                "Do NOT cite 'zero transactions' as evidence of fraud. Instead, note that actual bank statements "
                "are MISSING and must be submitted."
            )

        partial_dossier_instructions = ""
        if is_partial:
            partial_dossier_instructions = f"""

CRITICAL CONTEXT — PARTIAL DOSSIER:
The user has uploaded ONLY these document types: {doc_types}
The following document types are MISSING: {missing}

RULES FOR PARTIAL DOSSIER:
1. You MUST set verdict to "INCOMPLETE_DOSSIER" (not FAIL) because the user has not submitted all documents yet.
2. List all missing document types in the missing_documents field.
3. DO NOT treat missing data nodes (null passport, null enrollment, null sponsor) as rejection-worthy anomalies.
   These are simply documents the user has not uploaded yet.
4. ONLY flag as "critical" issues that represent genuine anomalies WITHIN the provided documents
   (e.g., smurfing patterns, unexplained spikes, forged signatures, balance discontinuities).
5. The risk_score should reflect the risk of the PROVIDED documents only, not penalize for missing docs.
   A clean bank statement with no anomalies but missing passport = risk 0.2-0.3, NOT 0.99.
6. In rejection_narrative, frame it as a pre-submission audit report, NOT a visa denial letter.
   Example tone: "The provided bank statement shows X anomalies that require attention before submission.
   Additionally, the following documents must be uploaded to complete the dossier: [list]."
7. In rebuttal_guidance, list actionable steps to complete the dossier and address any real anomalies."""

        prompt = f"""You are an adversarial pre-submission auditor. Your job is to identify gaps and anomalies
BEFORE the applicant submits their visa package to the embassy. You are helping them strengthen their case,
not judging them.

DOCUMENT TYPES PROCESSED: {doc_types}
{doc_context_note}
{partial_dossier_instructions}

SEMANTIC GRAPH (full document picture):
{json.dumps(graph, indent=2)}

RULE ENGINE FINDINGS:
{json.dumps(state.get("rule_findings", []), indent=2)}

SPECIALIST AGENT FINDINGS:
{json.dumps(state.get("agent_findings", []), indent=2)}

Your task:
1. Synthesise ALL findings into a single coherent verdict.
2. Distinguish between:
   a) ACTUAL ANOMALIES found within provided documents (these are critical/warning)
   b) MISSING INFORMATION because documents haven't been uploaded yet (these are info-level gaps)
3. Write the rejection_narrative as a pre-submission audit: formal, specific, citing document evidence.
   Separate the "Anomalies Found" section from the "Documents Still Required" section.
4. For each issue, write one rebuttal_guidance item: what additional document or action would resolve it.
5. Risk score: 0.0 = clean provided documents, 1.0 = severe anomalies in provided documents.
   Do NOT inflate risk score just because documents are missing.
6. In citations, list the exact graph node paths that support your findings (e.g. 'financial.spikes[0].amount').

Respond ONLY with NarrativeVerdict JSON."""

        t0 = time.time()
        last_err = None
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=PRO_MODEL,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": NarrativeVerdict,
                    },
                )
                v = response.parsed
                elapsed = time.time() - t0

                # CP6 -- log narrative synthesis
                logger = state.get("_debug_logger")
                if logger:
                    logger.cp6_narrative(prompt, v.model_dump(), elapsed)
                    logger.close()

                return {"verdict": v.model_dump(), "risk_score": v.risk_score}
            except Exception as e:
                last_err = e
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_rate_limit and attempt < API_RETRY_ATTEMPTS - 1:
                    delay = min(
                        API_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1),
                        API_RETRY_MAX_DELAY,
                    )
                    print(f"    [WAIT] narrative: rate limited, retry in {delay:.0f}s (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})")
                    time.sleep(delay)
                elif not is_rate_limit:
                    break

        raise last_err
