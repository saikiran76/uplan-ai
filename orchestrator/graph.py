"""
Uplan LangGraph Orchestrator

StateGraph with SEQUENTIAL agent execution to respect free-tier rate limits.
Pipeline: ingest -> encode -> purge_raw -> rule_check -> agents_sequential -> narrative -> END

Design note: The Send API fan-out is architecturally correct but fires 5 concurrent
Gemini Pro calls, which instantly triggers 429 RESOURCE_EXHAUSTED on free tier.
Sequential execution with inter-agent delay keeps us under the RPM ceiling.
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from agents.enrollment_agent import EnrollmentAgent
from agents.financial_agent import FinancialAgent
from agents.identity_agent import IdentityAgent
from agents.narrative import NarrativeAgent
from agents.sponsor_agent import SponsorAgent
from agents.temporal_agent import TemporalAgent
from encoding.extractor import extract_pages
from encoding.graph_builder import build_graph
from encoding.schema import SemanticGraph
from orchestrator.state import UplanState
from rules.engine import run_rules
from config import INTER_AGENT_DELAY


# -- Node functions -------------------------------------------------------


def ingest(state: UplanState) -> dict:
    """Intake gate -- document_bytes already in state from caller."""
    return {}


def encode(state: UplanState) -> dict:
    """
    Structural encoding: render each page as PNG -> Gemini Flash extraction -> graph merge.
    Handles multiple PDFs in document_bytes list.
    Wires CP0 + CP1 + CP2 + CP3 debug checkpoints.
    """
    import os
    from debug.logger import DebugLogger

    debug_enabled = os.environ.get("UPLAN_DEBUG", "1") != "0"
    logger = DebugLogger() if debug_enabled else None

    t0 = time.time()
    all_pages = []

    for i, pdf_bytes in enumerate(state["document_bytes"]):
        # CP0 -- document ingestion
        if logger:
            logger.cp0_ingestion(f"document_{i + 1}.pdf", pdf_bytes)

        all_pages.extend(extract_pages(pdf_bytes, logger=logger))

    graph = build_graph(all_pages)

    # CP3 -- graph builder output
    if logger:
        logger.cp3_graph(graph.model_dump(), len(all_pages))

    return {
        "semantic_graph": graph.model_dump(),
        "encoding_metadata": {
            "token_count": graph.token_count,
            "page_count": graph.source_page_count,
            "estimated_raw_tokens": graph.estimated_raw_tokens,
            "duration_ms": int((time.time() - t0) * 1000),
            "source_doc_types": graph.source_doc_types,
            "source_is_affidavit": graph.financial.source_is_affidavit,
        },
        "_debug_logger": logger,  # passed through state for agent CP4/CP5 logging
    }


def purge_raw(state: UplanState) -> dict:
    """Privacy gate: destroy raw document bytes. Agents never see them."""
    return {"document_bytes": None, "raw_purged": True}


def rule_check(state: UplanState) -> dict:
    """Run deterministic rule algebra on the semantic graph with jurisdiction context."""
    graph = SemanticGraph.model_validate(state["semantic_graph"])

    # Reconstruct VisaContext from state if available
    visa_ctx = None
    if state.get("visa_context"):
        from rules.context_profiles import VisaContext
        ctx_dict = state["visa_context"]
        visa_ctx = VisaContext(**{k: v for k, v in ctx_dict.items() if k != "jurisdiction_context"})
        visa_ctx.jurisdiction_context = ctx_dict.get("jurisdiction_context", "")

    findings = run_rules(graph, context=visa_ctx)
    return {"rule_findings": [f.model_dump() for f in findings]}


# -- Sequential agent execution (rate-limit safe) -------------------------


AGENT_CLASSES = [
    ("identity_agent", IdentityAgent),
    ("financial_agent", FinancialAgent),
    ("temporal_agent", TemporalAgent),
    ("sponsor_agent", SponsorAgent),
    ("enrollment_agent", EnrollmentAgent),
]


def run_agents_sequential(state: UplanState) -> dict:
    """
    Run all 5 specialist agents SEQUENTIALLY with inter-agent delay.

    This replaces the Send API parallel fan-out to stay within
    Google AI Studio free-tier rate limits (~15 RPM for Pro).
    Each agent call takes ~5-15s, plus a configurable delay between calls.
    """
    all_findings = []
    completed = []

    for i, (agent_id, agent_cls) in enumerate(AGENT_CLASSES):
        if i > 0:
            print(f"  [DELAY] Waiting {INTER_AGENT_DELAY:.0f}s before next agent (rate limit)...")
            time.sleep(INTER_AGENT_DELAY)

        print(f"  [AGENT] Running {agent_id} ({i + 1}/{len(AGENT_CLASSES)})...")
        t0 = time.time()

        try:
            result = agent_cls().run(state)
            all_findings.extend(result.get("agent_findings", []))
            completed.extend(result.get("completed_agents", []))
            elapsed = time.time() - t0
            print(f"  [DONE] {agent_id} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [FAIL] {agent_id} failed after {elapsed:.1f}s: {e}")
            # Don't crash the whole pipeline -- skip this agent
            completed.append(agent_id)

    return {
        "agent_findings": all_findings,
        "completed_agents": completed,
    }


def narrative_synthesis(state: UplanState) -> dict:
    # Brief delay before narrative to avoid back-to-back Pro calls
    print(f"  [DELAY] Waiting {INTER_AGENT_DELAY:.0f}s before narrative synthesis...")
    time.sleep(INTER_AGENT_DELAY)
    return NarrativeAgent().run(state)


# -- Graph assembly -------------------------------------------------------


def build_uplan_graph():
    g = StateGraph(UplanState)

    g.add_node("ingest", ingest)
    g.add_node("encode", encode)
    g.add_node("purge_raw", purge_raw)
    g.add_node("rule_check", rule_check)
    g.add_node("run_agents", run_agents_sequential)
    g.add_node("narrative_synthesis", narrative_synthesis)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "encode")
    g.add_edge("encode", "purge_raw")
    g.add_edge("purge_raw", "rule_check")
    g.add_edge("rule_check", "run_agents")
    g.add_edge("run_agents", "narrative_synthesis")
    g.add_edge("narrative_synthesis", END)

    return g.compile()


uplan_graph = build_uplan_graph()
