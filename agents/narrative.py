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
    verdict: str = Field(description="'PASS', 'CONDITIONAL', or 'FAIL'")
    risk_score: float = Field(description="0.0 = clean application, 1.0 = near-certain rejection")
    critical_issues: list[str] = Field(default_factory=list, description="Bullet-point critical issues")
    warning_issues: list[str] = Field(default_factory=list, description="Bullet-point warnings")
    rejection_narrative: str = Field(
        description="What the immigration officer would write -- formal, specific, citing evidence"
    )
    rebuttal_guidance: list[str] = Field(
        default_factory=list,
        description="For each issue: what additional document or explanation would resolve it",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Graph node paths supporting each finding (e.g. 'financial.spikes[0]')",
    )


class NarrativeAgent:
    """Cross-document synthesis agent -- the final reasoning step."""

    def run(self, state: UplanState) -> dict:
        graph = state["semantic_graph"]
        doc_types = graph.get("source_doc_types", [])
        source_is_affidavit = graph.get("financial", {}).get("source_is_affidavit", False)

        doc_context_note = ""
        if source_is_affidavit and "bank_statement" not in doc_types:
            doc_context_note = (
                "\n\nDOCUMENT CONTEXT: The financial data comes ONLY from an affidavit (sworn declaration). "
                "An affidavit legitimately has zero bank transactions -- it declares point-in-time asset values. "
                "Do NOT cite 'zero transactions' as evidence of fraud. Instead, note that actual bank statements "
                "are MISSING and must be submitted. The declared liquid assets and income should be treated as "
                "sponsor's declared financial standing, not as bank account history."
            )

        prompt = f"""You are a senior immigration officer writing the official assessment.

DOCUMENT TYPES PROCESSED: {doc_types}
{doc_context_note}

SEMANTIC GRAPH (full document picture):
{json.dumps(graph, indent=2)}

RULE ENGINE FINDINGS:
{json.dumps(state.get("rule_findings", []), indent=2)}

SPECIALIST AGENT FINDINGS:
{json.dumps(state.get("agent_findings", []), indent=2)}

Your task:
1. Synthesise ALL findings into a single coherent verdict.
2. A finding from ANY agent with severity "critical" = FAIL unless directly contradicted by another agent.
3. Write the rejection_narrative as an officer would: formal, specific, citing document evidence.
4. For each issue, write one rebuttal_guidance item: what additional document would resolve it.
5. Risk score: 0.0 = clean application, 1.0 = near-certain rejection.
6. In citations, list the exact graph node paths that support your verdict (e.g. 'financial.spikes[0].amount').

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
