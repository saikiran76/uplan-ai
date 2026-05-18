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
        dossier_ctx = graph_subset.get("_document_context", {})
        partial_note = dossier_ctx.get("partial_dossier_instructions", "")
        return f"""You are an adversarial pre-submission auditor reviewing the timeline of an application.
Your task: identify temporal inconsistencies in the PROVIDED documents. Missing temporal data due to
un-uploaded documents should be flagged as 'info', not 'critical'.
{ctx_block}
DOSSIER COMPLETENESS: {partial_note}

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
6. If employment dates are null because employment letters haven't been uploaded,
   classify as 'info' severity — not a critical temporal anomaly.

CRITICAL RULE — BANK STATEMENT DATE GAPS:
Bank statements only record days with active transactions. Gaps of days or even weeks between
transaction dates are COMPLETELY NORMAL human behavior — people do not transact every single day.
Do NOT flag date gaps as 'missing days', 'discontinuities', or 'suspicious gaps' UNLESS:
  a) The running mathematical balance breaks (e.g., closing balance on page N does not match
     opening balance on page N+1), OR
  b) Page numbers are explicitly missing from the sequence (e.g., page 5 jumps to page 7).
A person who buys something on April 24th and then next on May 7th is normal. Do NOT flag this.

Respond ONLY with the AgentFinding JSON schema. Every anomaly must include the exact field_path."""
