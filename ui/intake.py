"""
Uplan UI — Intake Flow (Step 0)

Three-path document intake:
  Path A: Auto-generated checklist from visa context (BYOG RAG in production)
  Path B: User pastes their own checklist
  Path C: Free upload with document type label required

Returns (VisaContext, list[UploadedDoc]) when ready, None if still collecting.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from rules.context_profiles import (
    APPLICANT_COUNTRIES,
    DESTINATIONS,
    VISA_TYPES,
    VisaContext,
    get_context,
)


def _render_context_collection() -> Optional[VisaContext]:
    """Step 0: Collect applicant country, destination, visa type."""
    st.markdown("### 🌍 Application Context")
    st.caption("Tell us about the visa application — this determines which rules and thresholds apply.")

    col1, col2, col3 = st.columns(3)
    with col1:
        applicant_country = st.selectbox(
            "Applicant Country",
            options=APPLICANT_COUNTRIES,
            index=0,
            help="Country of citizenship/residence of the applicant",
        )
    with col2:
        destination = st.selectbox(
            "Destination Country",
            options=DESTINATIONS,
            index=0,
            help="Country the applicant is applying to",
        )
    with col3:
        visa_type = st.selectbox(
            "Visa Type",
            options=VISA_TYPES,
            index=0,
            help="Type of visa being applied for",
        )

    context = get_context(visa_type, destination)
    context.applicant_country = applicant_country

    # Show resolved thresholds
    with st.expander("📋 Resolved Thresholds for This Context", expanded=False):
        st.json(context.to_dict())

    return context


def _render_checklist_upload(context: VisaContext) -> Optional[list[dict]]:
    """
    Three-path upload flow.
    Returns list of {"label": str, "pdf_bytes": bytes} when files are ready.
    """
    st.markdown("### 📁 Document Upload")

    path = st.radio(
        "Choose your upload method:",
        options=[
            "📋 Use auto-generated checklist",
            "📝 Paste my own checklist",
            "📤 Free upload (I know what I'm uploading)",
        ],
        index=0,
        horizontal=True,
    )

    uploads: list[dict] = []

    if path.startswith("📋"):
        # ── Path A: Auto-generated checklist from context profile ───
        required = context.required_doc_types
        if not required:
            st.warning("No required documents defined for this visa profile.")
            required = ["passport", "bank_statement"]

        st.markdown(
            f"**Required documents for {context.visa_type.title()} → "
            f"{context.destination.title()}:**"
        )

        for doc_type in required:
            label = doc_type.replace("_", " ").title()
            file = st.file_uploader(
                f"📄 {label}",
                type=["pdf"],
                key=f"checklist_{doc_type}",
                help=f"Upload your {label}",
            )
            if file:
                uploads.append({
                    "label": doc_type,
                    "pdf_bytes": file.read(),
                    "filename": file.name,
                })

        # Show progress
        uploaded_count = len(uploads)
        total_required = len(required)
        st.progress(
            uploaded_count / max(total_required, 1),
            text=f"{uploaded_count}/{total_required} required documents uploaded",
        )

    elif path.startswith("📝"):
        # ── Path B: User pastes their own checklist ─────────────────
        checklist_text = st.text_area(
            "Paste your document checklist (one item per line):",
            placeholder="Bank Statement\nPassport\nEmployment Letter\nSponsor Letter",
            height=150,
        )

        if checklist_text.strip():
            items = [line.strip() for line in checklist_text.strip().split("\n") if line.strip()]
            for item in items:
                # Normalize to snake_case for internal use
                normalized = item.lower().replace(" ", "_")
                file = st.file_uploader(
                    f"📄 {item}",
                    type=["pdf"],
                    key=f"custom_{normalized}",
                )
                if file:
                    uploads.append({
                        "label": normalized,
                        "pdf_bytes": file.read(),
                        "filename": file.name,
                    })

    else:
        # ── Path C: Free upload with label dropdown ─────────────────
        st.caption("Upload files and label each one with its document type.")
        free_files = st.file_uploader(
            "Upload documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="free_upload",
        )

        if free_files:
            doc_type_options = [
                "bank_statement", "payslip", "tax_return", "passport",
                "sponsor_letter", "employment_letter", "enrollment_letter",
                "affidavit", "other",
            ]
            for i, file in enumerate(free_files):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {file.name}")
                with col2:
                    label = st.selectbox(
                        "Type",
                        options=doc_type_options,
                        key=f"label_{i}",
                        label_visibility="collapsed",
                    )
                uploads.append({
                    "label": label,
                    "pdf_bytes": file.read(),
                    "filename": file.name,
                })

    return uploads if uploads else None


def run_intake() -> Optional[tuple[VisaContext, list[dict]]]:
    """
    Complete intake flow: context collection + document upload.

    Returns:
        (VisaContext, list[{"label": str, "pdf_bytes": bytes, "filename": str}])
        when the user has provided context and uploaded files.
        None if still collecting.
    """
    context = _render_context_collection()
    st.markdown("---")
    uploads = _render_checklist_upload(context)

    if uploads:
        return context, uploads
    return None
