"""
Financial Specialist Agent

Probes: spike explanation, income-balance coherence, unlabeled deposits,
cross-document income discrepancies (payslip vs bank deposit amounts).
"""

from __future__ import annotations

import json

from agents.base_agent import BaseSpecialistAgent


class FinancialAgent(BaseSpecialistAgent):
    agent_id = "financial_agent"
    focus_nodes = ["financial", "identity"]

    def _build_prompt(self, graph_subset: dict, rule_findings: list[dict], jurisdiction: str) -> str:
        ctx_block = f"\nJURISDICTION CONTEXT:\n{jurisdiction}\n" if jurisdiction else ""
        return f"""You are an adversarial immigration officer reviewing financial documents.
Your task: identify every financial inconsistency that could justify visa rejection.
{ctx_block}
FINANCIAL DATA:
{json.dumps(graph_subset.get("financial"), indent=2)}

APPLICANT IDENTITY:
{json.dumps(graph_subset.get("identity"), indent=2)}

RULE ENGINE FLAGS (already detected):
{json.dumps(rule_findings, indent=2)}

Analyse the following — be specific, cite exact values:
1. For each spike in spikes[]: is there a plausible source? If unlabeled, flag as critical.
2. Does avg_monthly_income × visa_duration cover declared costs? Show the arithmetic.
3. Does the balance trajectory make sense given declared income? Flag any impossible accumulation.
4. Are there cross-document income discrepancies (payslip vs bank deposit amounts)?

Respond ONLY with the AgentFinding JSON schema. Every anomaly must include the exact field_path."""
