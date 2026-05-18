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
        dossier_ctx = graph_subset.get("_document_context", {})
        partial_note = dossier_ctx.get("partial_dossier_instructions", "")
        return f"""You are an adversarial pre-submission auditor reviewing financial documents.
Your task: identify genuine financial anomalies in the PROVIDED documents. Focus on patterns that
indicate fund manipulation, smurfing, or unexplained activity — NOT on missing documents.
{ctx_block}
DOSSIER COMPLETENESS: {partial_note}

FINANCIAL DATA:
{json.dumps(graph_subset.get("financial"), indent=2)}

APPLICANT IDENTITY:
{json.dumps(graph_subset.get("identity"), indent=2)}

RULE ENGINE FLAGS (already detected):
{json.dumps(rule_findings, indent=2)}

Analyse the following — be specific, cite exact values:
1. For each spike in spikes[]: is there a plausible source? If unlabeled, flag as warning or critical
   based on the severity of the spike ratio and clustering pattern.
2. Does avg_monthly_income × visa_duration cover declared costs? Show the arithmetic.
   If income data is missing because payslips haven't been uploaded, note this as 'info' not 'critical'.
3. Does the balance trajectory make sense given declared income? Flag any impossible accumulation.
4. Are there cross-document income discrepancies (payslip vs bank deposit amounts)?
5. If the closing balance is below a threshold, note the current documented balance and suggest
   the user upload additional accounts to meet the requirement — do NOT issue a hard rejection.
6. Aggregate balances across ALL provided accounts before assessing financial thresholds.

SEVERITY GUIDE:
- 'critical': Genuine anomalies WITHIN the data (smurfing clusters, forged amounts, impossible balances)
- 'warning': Suspicious patterns that need explanation (large unlabeled deposits, unusual spikes)
- 'info': Missing data because documents haven't been uploaded yet

CRITICAL RULE — SMURFING vs PASS-THROUGH:
Before flagging inbound deposits as 'smurfing' or 'fund parking', you MUST look at the subsequent
outbound debits within a 24-48 hour window. If a large deposit is followed almost immediately by
outbound transfers of similar amounts to the same recipient (a 'pass-through' pattern), it is
NOT smurfing, because the funds are not inflating the closing balance.

Example of PASS-THROUGH (NOT smurfing):
  Aug 17: 30,000 IN → Aug 17: 28,000 OUT to same payee → net impact: +2,000
  Aug 18: 30,000 IN → Aug 18: 30,000 OUT to same payee → net impact: 0

This is a transit/conduit pattern where the account is used as a pipeline. The balance barely moves.
Only flag as smurfing if the deposits ACCUMULATE and the closing balance grows significantly
as a result of the clustered deposits WITHOUT corresponding outflows.

Assess the NET IMPACT of each transaction cluster before classifying it.

Respond ONLY with the AgentFinding JSON schema. Every anomaly must include the exact field_path."""
