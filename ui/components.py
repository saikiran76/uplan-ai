"""
Uplan UI — Reusable Streamlit Components

Renders: compression stats, findings dashboard, verdict card, narrative report.
"""

from __future__ import annotations

import streamlit as st


# ── Compression Stats (strongest demo moment) ───────────────────────


def render_compression_stats(metadata: dict) -> None:
    """Hero metric: the 60k → 1,200 token reduction."""
    raw = metadata.get("estimated_raw_tokens", 0)
    graph = metadata.get("token_count", 1)
    pages = metadata.get("page_count", 0)
    duration = metadata.get("duration_ms", 0)

    reduction = (1 - graph / max(raw, 1)) * 100
    ratio = raw / max(graph, 1)

    st.markdown("### 🔬 Structural Encoding — Token Compression")

    cols = st.columns(4)
    cols[0].metric("📄 Pages Processed", f"{pages}")
    cols[1].metric("📝 Raw Tokens (est.)", f"{raw:,}")
    cols[2].metric("🧬 Graph Tokens", f"{graph:,}")
    cols[3].metric("📉 Reduction", f"{reduction:.1f}%")

    # Progress bar visualizing the compression
    st.progress(min(reduction / 100, 1.0))
    st.caption(
        f"**{ratio:.0f}:1 compression** — {pages} pages of raw documents condensed "
        f"into a {graph:,}-token semantic graph in {duration / 1000:.1f}s. "
        f"Fits any 4k+ context window."
    )


# ── Rule Findings ───────────────────────────────────────────────────


def render_rule_findings(findings: list[dict]) -> None:
    """Display deterministic rule engine results with severity badges."""
    st.markdown("### ⚖️ Deterministic Rule Engine")

    if not findings:
        st.success("✅ No rule violations detected.")
        return

    critical = [f for f in findings if f["severity"] == "critical"]
    warnings = [f for f in findings if f["severity"] == "warning"]
    info = [f for f in findings if f["severity"] == "info"]

    if critical:
        st.error(f"🔴 {len(critical)} critical finding(s)")
    if warnings:
        st.warning(f"🟡 {len(warnings)} warning(s)")
    if info:
        st.info(f"🔵 {len(info)} informational finding(s)")

    for f in findings:
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(f["severity"], "⚪")
        with st.expander(f"{icon} [{f['rule_id']}] {f['message']}", expanded=f["severity"] == "critical"):
            col1, col2 = st.columns(2)
            col1.markdown(f"**Expected:** `{f.get('expected', 'N/A')}`")
            col2.markdown(f"**Actual:** `{f.get('actual', 'N/A')}`")
            st.caption(f"Field: `{f['field_path']}`")


# ── Agent Findings ──────────────────────────────────────────────────


def render_agent_findings(findings: list[dict]) -> None:
    """Display specialist agent reasoning results."""
    st.markdown("### 🕵️ Specialist Agent Analysis")

    if not findings:
        st.info("No agent findings available.")
        return

    for af in findings:
        verdict_color = {
            "pass": "🟢",
            "flag": "🟡",
            "critical": "🔴",
        }.get(af.get("agent_verdict", ""), "⚪")

        agent_label = af["agent_id"].replace("_", " ").title()
        confidence = af.get("confidence", 0)

        with st.expander(
            f"{verdict_color} {agent_label} — {af['agent_verdict'].upper()} "
            f"(confidence: {confidence:.0%})",
            expanded=af.get("agent_verdict") == "critical",
        ):
            st.markdown(f"**Summary:** {af.get('summary', 'N/A')}")

            anomalies = af.get("anomalies", [])
            if anomalies:
                for anomaly in anomalies:
                    sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                        anomaly["severity"], "⚪"
                    )
                    st.markdown(
                        f"- {sev_icon} **`{anomaly['field_path']}`**: {anomaly['explanation']}"
                    )
                    if anomaly.get("rule_id"):
                        st.caption(f"  Linked rule: `{anomaly['rule_id']}`")
            else:
                st.success("No anomalies detected by this agent.")


# ── Verdict Card ────────────────────────────────────────────────────


def render_verdict(verdict: dict, risk_score: float) -> None:
    """Hero card: final verdict with risk gauge."""
    st.markdown("---")
    st.markdown("## 📋 Final Assessment")

    verdict_text = verdict.get("verdict", "UNKNOWN")
    verdict_styles = {
        "PASS": ("success", "✅"),
        "CONDITIONAL": ("warning", "⚠️"),
        "FAIL": ("error", "❌"),
    }
    style, icon = verdict_styles.get(verdict_text, ("info", "❓"))

    # Verdict + Risk score side by side
    col1, col2 = st.columns([2, 1])
    with col1:
        getattr(st, style)(f"{icon} Verdict: **{verdict_text}**")
    with col2:
        risk_color = "🔴" if risk_score > 0.7 else "🟡" if risk_score > 0.3 else "🟢"
        st.metric(f"{risk_color} Risk Score", f"{risk_score:.2f}")

    # Risk progress bar
    st.progress(min(risk_score, 1.0))

    # Issues summary
    critical = verdict.get("critical_issues", [])
    warnings = verdict.get("warning_issues", [])

    if critical:
        st.markdown("#### 🔴 Critical Issues")
        for issue in critical:
            st.markdown(f"- {issue}")

    if warnings:
        st.markdown("#### 🟡 Warnings")
        for issue in warnings:
            st.markdown(f"- {issue}")


# ── Narrative Report ────────────────────────────────────────────────


def render_narrative(verdict: dict) -> None:
    """Full rejection narrative and rebuttal guidance."""
    rejection = verdict.get("rejection_narrative", "")
    rebuttals = verdict.get("rebuttal_guidance", [])
    citations = verdict.get("citations", [])

    if rejection:
        st.markdown("### 📝 Officer's Assessment")
        st.markdown(
            f'<div style="background: #1a1a2e; border-left: 4px solid #e94560; '
            f'padding: 16px; border-radius: 4px; font-family: serif; '
            f'line-height: 1.8;">{rejection}</div>',
            unsafe_allow_html=True,
        )

    if rebuttals:
        st.markdown("### 🛡️ Rebuttal Guidance")
        st.caption("How the applicant can address each issue:")
        for i, r in enumerate(rebuttals, 1):
            st.markdown(f"**{i}.** {r}")

    if citations:
        st.markdown("### 🔗 Evidence Citations")
        st.caption("Graph node paths supporting the verdict:")
        for c in citations:
            st.code(c, language=None)


# ── Semantic Graph Summary ──────────────────────────────────────────


def render_graph_summary(graph: dict) -> None:
    """Quick overview of what the semantic graph contains."""
    st.markdown("### 🧬 Semantic Graph Structure")

    identity = graph.get("identity", {})
    financial = graph.get("financial", {})
    temporal = graph.get("temporal", {})
    sponsor = graph.get("sponsor")
    enrollment = graph.get("enrollment")
    edges = graph.get("edges", [])

    cols = st.columns(5)
    cols[0].metric("👤 Identity", f"{len(identity.get('name_variants', []))} names")
    cols[1].metric("💰 Financial", f"{financial.get('transaction_count', 0)} txns")
    cols[2].metric("📅 Temporal", f"{len(temporal.get('doc_date_ranges', []))} ranges")
    cols[3].metric("🤝 Sponsor", "Present" if sponsor else "None")
    cols[4].metric("🎓 Enrollment", "Present" if enrollment else "None")

    # Edge coherence
    coherent = sum(1 for e in edges if e.get("coherent"))
    incoherent = len(edges) - coherent
    if incoherent > 0:
        st.warning(f"⚠️ {incoherent}/{len(edges)} cross-document edges are **incoherent**")
    elif edges:
        st.success(f"✅ All {len(edges)} cross-document relationships are coherent")

    # Raw graph JSON in expander
    with st.expander("🔍 View Raw Graph JSON"):
        st.json(graph)
