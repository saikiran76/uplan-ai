"""
Sponsor Specialist Agent

Probes: sponsor income vs total trip cost, relationship plausibility,
jurisdiction verification, income documentation sufficiency.
"""

from __future__ import annotations

import json

from agents.base_agent import BaseSpecialistAgent


class SponsorAgent(BaseSpecialistAgent):
    agent_id = "sponsor_agent"
    focus_nodes = ["sponsor", "identity", "financial"]

    def _build_prompt(self, graph_subset: dict, rule_findings: list[dict], jurisdiction: str) -> str:
        ctx_block = f"\nJURISDICTION CONTEXT:\n{jurisdiction}\n" if jurisdiction else ""
        dossier_ctx = graph_subset.get("_document_context", {})
        partial_note = dossier_ctx.get("partial_dossier_instructions", "")
        return f"""You are an adversarial pre-submission auditor reviewing sponsor documentation.
Your task: identify sponsorship inconsistencies in the PROVIDED documents and flag missing sponsor
documents as informational gaps.
{ctx_block}
DOSSIER COMPLETENESS: {partial_note}

SPONSOR DATA:
{json.dumps(graph_subset.get("sponsor"), indent=2)}

APPLICANT IDENTITY:
{json.dumps(graph_subset.get("identity"), indent=2)}

APPLICANT FINANCIAL DATA:
{json.dumps(graph_subset.get("financial"), indent=2)}

RULE ENGINE FLAGS (already detected):
{json.dumps(rule_findings, indent=2)}

Analyse the following — be specific, cite exact values:
1. Does the sponsor's declared_income realistically cover the applicant's total costs
   (tuition + living expenses + travel)? Show the arithmetic.
2. Is the stated relationship plausible? (e.g. "uncle" sponsoring a non-relative is suspicious)
3. If income_supports_coverage is false, quantify the shortfall.
4. Does the sponsor's income_currency match the jurisdiction where they claim to reside?
5. If no sponsor data exists (null), classify as 'info' severity — the sponsor documents
   have not been uploaded yet. The applicant may be self-funded or sponsor docs are pending.
   Do NOT treat missing sponsor as a critical rejection issue.

Respond ONLY with the AgentFinding JSON schema. Every anomaly must include the exact field_path."""
