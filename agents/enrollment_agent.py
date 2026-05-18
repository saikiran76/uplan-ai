"""
Enrollment Specialist Agent

Probes: program cost vs available balance, CoE date vs visa dates,
study duration feasibility, institution verification signals.
"""

from __future__ import annotations

import json

from agents.base_agent import BaseSpecialistAgent


class EnrollmentAgent(BaseSpecialistAgent):
    agent_id = "enrollment_agent"
    focus_nodes = ["enrollment", "financial", "temporal"]

    def _build_prompt(self, graph_subset: dict, rule_findings: list[dict], jurisdiction: str) -> str:
        ctx_block = f"\nJURISDICTION CONTEXT:\n{jurisdiction}\n" if jurisdiction else ""
        dossier_ctx = graph_subset.get("_document_context", {})
        partial_note = dossier_ctx.get("partial_dossier_instructions", "")
        return f"""You are an adversarial pre-submission auditor reviewing enrollment documentation.
Your task: identify enrollment/financial inconsistencies in the PROVIDED documents and flag missing
enrollment documents as informational gaps.
{ctx_block}
DOSSIER COMPLETENESS: {partial_note}

ENROLLMENT DATA:
{json.dumps(graph_subset.get("enrollment"), indent=2)}

FINANCIAL DATA:
{json.dumps(graph_subset.get("financial"), indent=2)}

TEMPORAL DATA:
{json.dumps(graph_subset.get("temporal"), indent=2)}

RULE ENGINE FLAGS (already detected):
{json.dumps(rule_findings, indent=2)}

Analyse the following — be specific, cite exact values:
1. Does the applicant's closing_balance cover program_cost? Show the arithmetic.
   Include a living expense estimate (~$1,500/month × duration_months) if relevant.
2. Does enrollment_start align with the visa application timeline?
3. Is the program_duration_months realistic for the stated program?
4. If funds_cover_full_stay is false, quantify the shortfall exactly.
5. If no enrollment data exists (null), classify as 'info' severity — enrollment documents
   may not have been uploaded yet, or this may be a non-student visa.
   Do NOT treat missing enrollment as a critical rejection issue.

Respond ONLY with the AgentFinding JSON schema. Every anomaly must include the exact field_path."""
