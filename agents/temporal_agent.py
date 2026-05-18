"""
Temporal Specialist Agent

Probes: employment gap arithmetic, date range overlaps, visa window fit,
chronological consistency across documents.
"""

from __future__ import annotations

import json

from agents.base_agent import BaseSpecialistAgent


class TemporalAgent(BaseSpecialistAgent):
    agent_id = "temporal_agent"
    focus_nodes = ["temporal", "financial"]

    def _build_prompt(self, graph_subset: dict, rule_findings: list[dict], jurisdiction: str) -> str:
        ctx_block = f"\nJURISDICTION CONTEXT:\n{jurisdiction}\n" if jurisdiction else ""
        return f"""You are an adversarial immigration officer reviewing the timeline of an application.
Your task: identify every temporal inconsistency that could justify visa rejection.
{ctx_block}
TEMPORAL DATA:
{json.dumps(graph_subset.get("temporal"), indent=2)}

FINANCIAL DATA (for date cross-reference):
{json.dumps(graph_subset.get("financial"), indent=2)}

RULE ENGINE FLAGS (already detected):
{json.dumps(rule_findings, indent=2)}

Analyse the following — be specific, cite exact dates:
1. Do employment_start and employment_end form a continuous period? Flag any gaps.
2. Do the doc_date_ranges overlap logically? A bank statement ending after an
   employment letter's issue date is normal; the reverse is suspicious.
3. If visa_window_start/end exist, does the employment and financial history
   cover the required pre-application period (typically 3-6 months)?
4. Check chronology_valid — if false, explain what dates are logically impossible.
5. Are transaction dates within the stated employment period?

Respond ONLY with the AgentFinding JSON schema. Every anomaly must include the exact field_path."""
