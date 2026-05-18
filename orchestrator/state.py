"""
Uplan LangGraph State Schema

TypedDict with Annotated reducers for parallel agent output merging.
operator.add ensures lists from parallel Send nodes are concatenated, not overwritten.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from typing_extensions import TypedDict


class UplanState(TypedDict):
    # Input (purged after encode node)
    document_bytes: Optional[list[bytes]]

    # Encoding output
    semantic_graph: Optional[dict]          # SemanticGraph.model_dump()
    encoding_metadata: Optional[dict]       # token_count, page_count, duration_ms

    # Rule engine output
    rule_findings: Annotated[list[dict], operator.add]

    # Agent outputs — operator.add merges lists from parallel nodes
    agent_findings: Annotated[list[dict], operator.add]
    completed_agents: Annotated[list[str], operator.add]

    # Visa context — jurisdiction-aware thresholds
    visa_context: Optional[dict]           # VisaContext.to_dict()
    document_checklist: list[dict]         # [{label, status, validation_result}]

    # Synthesis
    verdict: Optional[dict]
    risk_score: Optional[float]

    # Privacy gate flag
    raw_purged: bool
