"""
Uplan — Standalone Encoding Test

Usage:
    python test_encoding.py <path_to_pdf>

Runs the full encoding pipeline (extract pages → build graph → run rules)
on a single PDF and prints the SemanticGraph JSON + compression stats.

This validates the core innovation before wiring LangGraph.
"""

from __future__ import annotations

import json
import sys
import time


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_encoding.py <path_to_pdf>")
        print("  e.g. python test_encoding.py ./test_files/bank_statement.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print(f"\n{'═' * 60}")
    print(f"  UPLAN STRUCTURAL ENCODING TEST")
    print(f"  Input: {pdf_path}")
    print(f"{'═' * 60}\n")

    # Read PDF bytes
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    print(f"  PDF size: {len(pdf_bytes):,} bytes\n")

    # ── Stage 1: Per-page extraction ─────────────────────────────────
    print("─── STAGE 1: Per-page entity extraction (Gemini Flash) ───")
    from encoding.extractor import extract_pages

    t0 = time.time()
    pages = extract_pages(pdf_bytes)
    extraction_time = time.time() - t0

    print(f"\n  Extracted {len(pages)} pages in {extraction_time:.1f}s")
    for p in pages:
        print(f"    Page {p.page_number}: {p.page_type.value}"
              f" | txns={len(p.transactions)}"
              f" | flags={p.anomaly_flags}")

    # ── Stage 2: Graph building ──────────────────────────────────────
    print(f"\n─── STAGE 2: Graph merge ───")
    from encoding.graph_builder import build_graph

    t1 = time.time()
    graph = build_graph(pages)
    build_time = time.time() - t1

    print(f"\n  Graph built in {build_time:.1f}s")

    # ── Stage 3: Rule engine ─────────────────────────────────────────
    print(f"\n─── STAGE 3: Deterministic rule engine ───")
    from rules.engine import run_rules

    findings = run_rules(graph)
    print(f"  {len(findings)} rule finding(s):")
    for f in findings:
        icon = "🔴" if f.severity == "critical" else "🟡" if f.severity == "warning" else "🔵"
        print(f"    {icon} [{f.rule_id}] {f.message}")

    # ── Compression stats (your strongest demo moment) ───────────────
    print(f"\n{'═' * 60}")
    print(f"  COMPRESSION STATS")
    print(f"{'═' * 60}")
    print(f"  Source pages:          {graph.source_page_count}")
    print(f"  Est. raw tokens:       {graph.estimated_raw_tokens:,}")
    print(f"  Graph tokens:          {graph.token_count:,}")
    if graph.estimated_raw_tokens > 0:
        reduction = (1 - graph.token_count / graph.estimated_raw_tokens) * 100
        ratio = graph.estimated_raw_tokens / max(graph.token_count, 1)
        print(f"  Token reduction:       {reduction:.1f}%")
        print(f"  Compression ratio:     {ratio:.0f}:1")
    print(f"  Extraction time:       {extraction_time:.1f}s")
    print(f"  Build time:            {build_time:.1f}s")
    print(f"  Total pipeline:        {extraction_time + build_time:.1f}s")

    # ── Output graph JSON ────────────────────────────────────────────
    graph_json = graph.model_dump(mode="json")
    print(f"\n{'═' * 60}")
    print(f"  SEMANTIC GRAPH JSON")
    print(f"{'═' * 60}")
    print(json.dumps(graph_json, indent=2, default=str))

    # Also save to file
    output_path = pdf_path.rsplit(".", 1)[0] + "_graph.json"
    with open(output_path, "w") as f:
        json.dump(graph_json, f, indent=2, default=str)
    print(f"\n  ✓ Graph saved to: {output_path}")


if __name__ == "__main__":
    main()
