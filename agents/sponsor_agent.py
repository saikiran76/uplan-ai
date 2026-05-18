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
        return f"""You are an adversarial immigration officer reviewing sponsor documentation.
Your task: identify every sponsorship inconsistency that could justify visa rejection.
{ctx_block}
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
5. If no sponsor data exists (null), note that the application lacks financial sponsorship
   documentation — this may be acceptable if self-funded.

Respond ONLY with the AgentFinding JSON schema. Every anomaly must include the exact field_path."""
