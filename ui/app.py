"""
Uplan — Streamlit Entrypoint

Step 0: Context collection (country, visa type) + document upload with validation
Pipeline: validate → encode → rules → agents → narrative → report

Run: streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

# Ensure project root is on path (when run from ui/ directory)
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from ui.components import (
    render_agent_findings,
    render_compression_stats,
    render_graph_summary,
    render_narrative,
    render_rule_findings,
    render_verdict,
)
from ui.intake import run_intake

# ── Page Config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Uplan — Immigration Document Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }
    code, .stCode, pre {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    .block-container {
        padding-top: 2rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-top: 0;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 12px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────────────

st.markdown('<p class="hero-title">🔬 Uplan</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    "Adversarial Immigration Document Intelligence — "
    "Structural Encoding · Semantic Graph · Multi-Agent Reasoning"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Sidebar Info ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Stages")
    st.markdown(
        """
        **0** — Context + document intake
        **1** — Document validation gate
        **2** — Page segmentation (PyMuPDF)
        **3** — Entity extraction (Gemini Flash, async)
        **4** — Graph merge (Python)
        **5** — Rule engine (deterministic, context-aware)
        **6** — 5 specialist agents (Gemini Pro, parallel)
        **7** — Narrative synthesis (Gemini Pro)
        """
    )
    st.markdown("---")
    st.caption(
        "Privacy: raw documents are purged after encoding. "
        "Agents see only the semantic graph (~1,200 tokens)."
    )

# ── Step 0: Intake Flow ────────────────────────────────────────────

intake_result = run_intake()

if not intake_result:
    st.markdown("---")
    st.markdown("## How It Works")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            ### 📄 → 🧬
            **Structural Encoding**

            Raw documents (~60k tokens) → typed semantic graph
            (~1,200 tokens) — **98% reduction**.
            """
        )
    with col2:
        st.markdown(
            """
            ### ⚖️ → 🕵️
            **Adversarial Analysis**

            Deterministic rules + 5 specialist AI agents
            interrogate the graph for inconsistencies.
            """
        )
    with col3:
        st.markdown(
            """
            ### 📋 → 🛡️
            **Actionable Verdict**

            Officer-grade rejection narrative + specific
            rebuttal guidance for each issue.
            """
        )

    st.info("☝️ Select your visa context and upload documents above to begin.")

else:
    context, uploads = intake_result

    st.markdown("---")
    st.success(
        f"✅ {len(uploads)} document(s) ready | "
        f"Context: **{context.visa_type.title()} → {context.destination.title()}** | "
        f"Applicant: **{context.applicant_country}**"
    )

    # ── Validate + Run Pipeline ─────────────────────────────────────
    run_button = st.button("🚀 Run Uplan Analysis", type="primary", use_container_width=True)

    if run_button:
        progress = st.progress(0, text="Initializing...")
        status = st.status("Running Uplan pipeline...", expanded=True)

        with status:
            # Step 1: Document validation gate
            st.write("🔍 Validating uploads against expected labels...")
            progress.progress(5, text="Validating documents...")

            from validation.doc_validator import validate_all_uploads

            validations = validate_all_uploads(uploads)
            rejected = [v for v in validations if not v["accepted"]]

            if rejected:
                for v in rejected:
                    st.error(
                        f"❌ **{v['label']}**: Detected as "
                        f"'{v['validation'].actual_type}' — "
                        f"{v['validation'].reason}"
                    )
                st.warning("Some documents failed validation. Re-upload the correct files.")
                status.update(label="Validation failed", state="error")
                st.stop()

            for v in validations:
                st.write(
                    f"  ✓ {v['label']}: confirmed as "
                    f"{v['validation'].actual_type} "
                    f"(confidence: {v['validation'].confidence:.0%})"
                )

            # Step 2: Run full pipeline
            st.write("🧬 Running structural encoding + agent reasoning...")
            progress.progress(15, text="Running pipeline...")

            from orchestrator.graph import uplan_graph

            all_bytes = [u["pdf_bytes"] for u in uploads]
            checklist = [
                {"label": u["label"], "filename": u.get("filename", ""), "status": "validated"}
                for u in uploads
            ]

            t0 = time.time()
            try:
                result = uplan_graph.invoke({
                    "document_bytes": all_bytes,
                    "rule_findings": [],
                    "agent_findings": [],
                    "completed_agents": [],
                    "raw_purged": False,
                    "semantic_graph": None,
                    "encoding_metadata": None,
                    "verdict": None,
                    "risk_score": None,
                    "visa_context": context.to_dict(),
                    "document_checklist": checklist,
                })
                total_time = time.time() - t0
                progress.progress(100, text="Pipeline complete!")
                st.write(f"✅ Complete in {total_time:.1f}s")
                status.update(label="Pipeline complete!", state="complete")

            except Exception as e:
                progress.progress(100, text="Pipeline failed!")
                st.error(f"Pipeline error: {e}")
                status.update(label="Pipeline failed!", state="error")
                st.stop()

        st.session_state["result"] = result
        st.session_state["total_time"] = total_time
        st.session_state["context"] = context.to_dict()

    # ── Display Results ─────────────────────────────────────────────
    if "result" in st.session_state:
        result = st.session_state["result"]
        total_time = st.session_state.get("total_time", 0)

        st.markdown("---")

        tab_overview, tab_rules, tab_agents, tab_narrative, tab_graph = st.tabs(
            ["📊 Overview", "⚖️ Rules", "🕵️ Agents", "📝 Narrative", "🧬 Graph"]
        )

        with tab_overview:
            meta = result.get("encoding_metadata", {})
            if meta:
                render_compression_stats(meta)

            st.markdown("---")

            verdict = result.get("verdict", {})
            risk = result.get("risk_score", 0)
            if verdict:
                render_verdict(verdict, risk)

            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⏱️ Total Time", f"{total_time:.1f}s")
            col2.metric("🕵️ Agents", f"{len(result.get('completed_agents', []))}/5")
            col3.metric(
                "🔍 Findings",
                f"{len(result.get('rule_findings', [])) + len(result.get('agent_findings', []))}",
            )
            ctx = st.session_state.get("context", {})
            col4.metric("🌍 Context", f"{ctx.get('visa_type', '?')} → {ctx.get('destination', '?')}")

        with tab_rules:
            render_rule_findings(result.get("rule_findings", []))

        with tab_agents:
            render_agent_findings(result.get("agent_findings", []))

        with tab_narrative:
            if result.get("verdict"):
                render_narrative(result["verdict"])
            else:
                st.info("No narrative verdict available.")

        with tab_graph:
            if result.get("semantic_graph"):
                render_graph_summary(result["semantic_graph"])
            else:
                st.info("No semantic graph available.")

        st.markdown("---")
        st.download_button(
            "📥 Download Full Result JSON",
            data=json.dumps(
                {k: v for k, v in result.items() if k != "document_bytes"},
                indent=2,
                default=str,
            ),
            file_name="uplan_result.json",
            mime="application/json",
        )
