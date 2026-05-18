"""
Uplan -- End-to-End Pipeline Test

Usage:
    python test_pipeline.py <path_to_pdf>

Runs the full LangGraph pipeline:
    ingest -> encode -> purge -> rules -> agents (sequential) -> narrative

Validates:
    - Privacy purge (raw_purged == True)
    - All 5 agents complete (completed_agents has 5 entries)
    - Verdict and risk score produced
"""

from __future__ import annotations

import json
import sys
import time


def _safe(text: str) -> str:
    """Strip non-ASCII from model output to avoid Windows cp1252 crashes."""
    return text.encode("ascii", errors="replace").decode("ascii")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print(f"\n{'=' * 60}")
    print(f"  UPLAN END-TO-END PIPELINE TEST")
    print(f"  Input: {pdf_path}")
    print(f"{'=' * 60}\n")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    from orchestrator.graph import uplan_graph
    from rules.context_profiles import get_context

    # Default test context: Student visa to Japan
    context = get_context("student", "japan")

    t0 = time.time()
    result = uplan_graph.invoke({
        "document_bytes": [pdf_bytes],
        "rule_findings": [],
        "agent_findings": [],
        "completed_agents": [],
        "raw_purged": False,
        "semantic_graph": None,
        "encoding_metadata": None,
        "verdict": None,
        "risk_score": None,
        "visa_context": context.to_dict(),
        "document_checklist": [{"label": "bank_statement", "status": "uploaded"}],
    })
    total_time = time.time() - t0

    # -- Validation --------------------------------------------------------
    print(f"\n{'-' * 60}")
    print(f"  VALIDATION")
    print(f"{'-' * 60}")

    purged = result["raw_purged"]
    agents_done = result["completed_agents"]
    verdict = result.get("verdict")
    risk = result.get("risk_score")
    meta = result.get("encoding_metadata", {})

    print(f"  [OK] Privacy purge:     {'PASS' if purged else 'FAIL'} (raw_purged={purged})")
    print(f"  [OK] Agents completed:  {len(agents_done)}/5 -- {agents_done}")
    print(f"  [OK] Verdict:           {verdict.get('verdict') if verdict else 'MISSING'}")
    print(f"  [OK] Risk score:        {risk}")
    print(f"  [OK] Total time:        {total_time:.1f}s")

    # -- Compression stats -------------------------------------------------
    if meta:
        print(f"\n{'-' * 60}")
        print(f"  COMPRESSION")
        print(f"{'-' * 60}")
        print(f"  Pages:               {meta.get('page_count')}")
        print(f"  Est. raw tokens:     {meta.get('estimated_raw_tokens', 0):,}")
        print(f"  Graph tokens:        {meta.get('token_count', 0):,}")
        raw = meta.get("estimated_raw_tokens", 0)
        graph = meta.get("token_count", 1)
        if raw > 0:
            print(f"  Reduction:           {(1 - graph / raw) * 100:.1f}%")

    # -- Rule findings -----------------------------------------------------
    rules = result.get("rule_findings", [])
    print(f"\n{'-' * 60}")
    print(f"  RULE FINDINGS ({len(rules)})")
    print(f"{'-' * 60}")
    for r in rules:
        icon = "[CRIT]" if r["severity"] == "critical" else "[WARN]"
        print(f"  {icon} [{r['rule_id']}] {_safe(r['message'])}")

    # -- Agent findings ----------------------------------------------------
    agent_findings = result.get("agent_findings", [])
    print(f"\n{'-' * 60}")
    print(f"  AGENT FINDINGS ({len(agent_findings)})")
    print(f"{'-' * 60}")
    for af in agent_findings:
        print(f"\n  -- {af['agent_id']} ({af['agent_verdict']}, confidence={af['confidence']}) --")
        print(f"     {_safe(af['summary'])}")
        for anomaly in af.get("anomalies", []):
            icon = "[CRIT]" if anomaly["severity"] == "critical" else "[WARN]" if anomaly["severity"] == "warning" else "[INFO]"
            print(f"     {icon} {anomaly['field_path']}: {_safe(anomaly['explanation'])}")

    # -- Verdict -----------------------------------------------------------
    if verdict:
        print(f"\n{'=' * 60}")
        print(f"  FINAL VERDICT: {verdict['verdict']} (risk={risk})")
        print(f"{'=' * 60}")
        if verdict.get("rejection_narrative"):
            print(f"\n  Rejection Narrative:")
            print(f"  {_safe(verdict['rejection_narrative'])}")
        if verdict.get("rebuttal_guidance"):
            print(f"\n  Rebuttal Guidance:")
            for i, r in enumerate(verdict["rebuttal_guidance"], 1):
                print(f"    {i}. {_safe(r)}")

    # -- Save full result --------------------------------------------------
    output_path = pdf_path.rsplit(".", 1)[0] + "_result.json"
    serializable = {k: v for k, v in result.items() if k != "document_bytes"}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  [OK] Full result saved to: {output_path}")


if __name__ == "__main__":
    main()
