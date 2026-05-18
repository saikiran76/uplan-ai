"""
Identity Specialist Agent

Probes: name variant analysis, transliteration mismatches,
cross-document identity coherence, passport-to-document name matching.
"""

from __future__ import annotations

import json

from agents.base_agent import BaseSpecialistAgent


class IdentityAgent(BaseSpecialistAgent):
    agent_id = "identity_agent"
    focus_nodes = ["identity"]

    def _build_prompt(self, graph_subset: dict, rule_findings: list[dict], jurisdiction: str) -> str:
        ctx_block = f"\nJURISDICTION CONTEXT:\n{jurisdiction}\n" if jurisdiction else ""
        return f"""You are an adversarial immigration officer reviewing identity documents.
Your task: identify every identity inconsistency that could justify visa rejection.
{ctx_block}
IDENTITY DATA:
{json.dumps(graph_subset.get("identity"), indent=2)}

RULE ENGINE FLAGS (already detected):
{json.dumps(rule_findings, indent=2)}

Analyse the following — be specific, cite exact field paths:
1. Are all name_variants plausibly the same person? Consider transliteration rules
   (e.g. Japanese: family-first vs given-first, romanisation differences).
2. If cross_doc_name_match is false, is this a genuine mismatch or an expected
   transliteration variation? Rate the severity.
3. Check for date_of_birth consistency — is it the same across all documents?
4. Does the passport_number match the identity page expectations for the stated nationality?

Respond ONLY with the AgentFinding JSON schema. Every anomaly must include the exact field_path."""
