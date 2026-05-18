"""
Uplan Base Specialist Agent

Abstract base class for all 5 domain agents.
Handles: graph subset extraction, rule filtering, Gemini Pro call with
structured output, and exponential backoff retry for rate limits.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
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


class Anomaly(BaseModel):
    """A single anomaly detected by a specialist agent."""
    field_path: str = Field(description="Dot-path to the relevant graph field, e.g. 'financial.spikes[0].amount'")
    severity: str = Field(description="'critical', 'warning', or 'info'")
    explanation: str = Field(description="Plain English explanation, 1-2 sentences")
    rule_id: Optional[str] = Field(default=None, description="Links back to rule engine finding if applicable")


class AgentFinding(BaseModel):
    """Structured output from each specialist agent."""
    agent_id: str = Field(description="Identifier of the agent producing this finding")
    anomalies: list[Anomaly] = Field(default_factory=list, description="All detected anomalies")
    confidence: float = Field(description="Agent confidence in its assessment, 0.0-1.0")
    agent_verdict: str = Field(description="'pass', 'flag', or 'critical'")
    summary: str = Field(description="1-sentence summary for the narrative agent")


class BaseSpecialistAgent(ABC):
    """
    Abstract base for domain specialist agents.
    Subclasses override agent_id, focus_nodes, and _build_prompt().
    """
    agent_id: str
    focus_nodes: list[str]  # Which keys to pull from semantic_graph

    def _extract_focus(self, graph: dict) -> dict:
        """Extract only the graph nodes relevant to this agent."""
        return {k: graph[k] for k in self.focus_nodes if k in graph and graph[k]}

    def _relevant_rules(self, rule_findings: list[dict]) -> list[dict]:
        """Filter rule findings to this agent's primary domain."""
        prefix = self.focus_nodes[0]  # Primary domain
        return [r for r in rule_findings if r.get("field_path", "").startswith(prefix)]

    @abstractmethod
    def _build_prompt(self, graph_subset: dict, rule_findings: list[dict], jurisdiction: str) -> str:
        """Build the domain-specific reasoning prompt."""
        pass

    def _call_with_retry(self, prompt: str) -> AgentFinding:
        """Call Gemini Pro with exponential backoff retry on rate limits."""
        last_err = None
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=PRO_MODEL,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": AgentFinding,
                    },
                )
                return response.parsed
            except Exception as e:
                last_err = e
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_rate_limit and attempt < API_RETRY_ATTEMPTS - 1:
                    delay = min(
                        API_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1),
                        API_RETRY_MAX_DELAY,
                    )
                    print(f"    [WAIT] {self.agent_id}: rate limited, retry in {delay:.0f}s (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})")
                    time.sleep(delay)
                elif not is_rate_limit:
                    break

        raise last_err

    def run(self, state: UplanState) -> dict:
        """Execute this agent: extract focus -> build prompt -> Gemini Pro -> return finding."""
        graph = state["semantic_graph"]
        graph_subset = self._extract_focus(graph)
        relevant_rules = self._relevant_rules(state.get("rule_findings", []))

        # Inject document type context so agents understand what they're looking at
        doc_types = graph.get("source_doc_types", [])
        source_is_affidavit = graph.get("financial", {}).get("source_is_affidavit", False)

        jurisdiction = ""
        if state.get("visa_context"):
            jurisdiction = state["visa_context"].get("jurisdiction_context", "")

        # Augment graph_subset with document context
        graph_subset["_document_context"] = {
            "source_doc_types": doc_types,
            "source_is_affidavit": source_is_affidavit,
            "note": (
                "IMPORTANT: If source_is_affidavit=True and 'bank_statement' is NOT in source_doc_types, "
                "then transaction_count=0 is EXPECTED and NORMAL. An affidavit declares static balances, "
                "it does NOT have transaction history. Do NOT flag zero transactions as 'funds parking' "
                "when the only financial document is an affidavit. Instead flag the ABSENCE of bank statements."
            ) if source_is_affidavit and "bank_statement" not in doc_types else "Standard multi-document analysis.",
        }

        prompt = self._build_prompt(graph_subset, relevant_rules, jurisdiction)
        logger = state.get("_debug_logger")

        # CP4 -- log exact prompt sent to Pro
        if logger:
            logger.cp4_agent_prompt(self.agent_id, prompt, self.focus_nodes)

        t0 = time.time()
        parse_error = None
        try:
            finding = self._call_with_retry(prompt)
        except Exception as e:
            parse_error = str(e)
            raise

        finding.agent_id = self.agent_id
        elapsed = time.time() - t0

        # CP5 -- log agent response
        if logger:
            logger.cp5_agent_response(self.agent_id, finding.model_dump(), elapsed, parse_error)

        return {
            "agent_findings": [finding.model_dump()],
            "completed_agents": [self.agent_id],
        }
