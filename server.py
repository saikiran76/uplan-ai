"""
Uplan FastAPI Server
====================
Serves the UI and exposes backend API endpoints.

Routes:
  GET  /                  -> serves ui/index.html
  POST /api/validate      -> quick Flash doc-type check (per upload)
  POST /api/analyze       -> full pipeline with SSE progress stream
  GET  /api/health        -> health check

SSE event schema for /api/analyze:
  {"type": "stage_start", "stage": "encode", "message": "..."}
  {"type": "stage_done",  "stage": "encode", "elapsed_s": 12.3, "data": {...}}
  {"type": "agent_start", "agent": "identity_agent"}
  {"type": "agent_done",  "agent": "identity_agent", "elapsed_s": 8.1}
  {"type": "complete",    "result": {...}}
  {"type": "error",       "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import AsyncGenerator

import fitz
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# -- Path setup so local modules are importable --
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import client, FLASH_MODEL
from encoding.extractor import extract_pages
from encoding.graph_builder import build_graph
from encoding.schema import PageExtraction, PageType
from agents.base_agent import AgentFinding
from agents.financial_agent import FinancialAgent
from agents.identity_agent import IdentityAgent
from agents.temporal_agent import TemporalAgent
from agents.sponsor_agent import SponsorAgent
from agents.enrollment_agent import EnrollmentAgent
from agents.narrative import NarrativeAgent
from rules.engine import run_rules
from google.genai import types

app = FastAPI(title="Uplan API", version="1.0.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Static UI --
UI_DIR = ROOT / "ui"


@app.get("/")
async def serve_ui():
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_flash": os.environ.get("FLASH_MODEL", "gemini-2.5-flash")}


# ─────────────────────────────────────────────────────────────────
# Quick document type validation endpoint
# Called per-upload in Step 2. Returns detected page_type + match.
# ─────────────────────────────────────────────────────────────────

VALIDATE_PROMPT = """You are a document classifier for immigration visa applications.
Look at this document page and classify it. Return ONLY a JSON object with:
{
  "detected_type": "<one of: passport, bank_statement, payslip, tax_return, sponsor_letter, employment_letter, enrollment_letter, affidavit, unknown>",
  "confidence": <0.0-1.0>,
  "reason": "<one sentence explaining the classification>"
}"""

TYPE_DISPLAY = {
    "passport": "Passport",
    "bank_statement": "Bank Statement",
    "payslip": "Payslip",
    "tax_return": "Tax Return",
    "sponsor_letter": "Sponsor Letter",
    "employment_letter": "Employment Letter",
    "enrollment_letter": "Enrollment Letter / COE",
    "affidavit": "Affidavit",
    "unknown": "Unknown",
    "supporting_docs": "Supporting Document",
    "cas_letter": "CAS Letter",
    "i20_letter": "I-20 Form",
    "acceptance_letter": "Acceptance Letter",
    "proof_of_funds": "Proof of Funds",
    "cos_letter": "Certificate of Sponsorship",
    "itinerary": "Travel Itinerary",
    "financial_affidavit": "Financial Affidavit",
}

# Map UI slot types -> acceptable Flash types (some are aliases)
SLOT_TYPE_MAP = {
    "bank_statement": ["bank_statement"],
    "payslip": ["payslip"],
    "passport": ["passport"],
    "enrollment_letter": ["enrollment_letter"],
    "sponsor_letter": ["sponsor_letter", "affidavit"],
    "employment_letter": ["employment_letter"],
    "affidavit": ["affidavit", "sponsor_letter"],
    "financial_affidavit": ["affidavit", "sponsor_letter"],
    "cas_letter": ["enrollment_letter"],
    "i20_letter": ["enrollment_letter"],
    "acceptance_letter": ["enrollment_letter"],
    "cos_letter": ["employment_letter", "sponsor_letter"],
    "itinerary": ["unknown"],  # No strict validation needed
    "supporting_docs": None,   # Accept anything
    "proof_of_funds": ["bank_statement", "affidavit"],
    "tax_return": ["tax_return"],
}


@app.post("/api/validate")
async def validate_document(
    file: UploadFile = File(...),
    expected_type: str = Form(...),
):
    """
    Validate a single uploaded document against its expected type.
    Returns: {match: bool, detected: str, confidence: float, reason: str}
    """
    try:
        pdf_bytes = await file.read()

        # Render first page to PNG
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(dpi=100)  # Low DPI for quick validation
        png_bytes = pix.tobytes("png")
        doc.close()

        # Call Flash for type classification
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=FLASH_MODEL,
                contents=[
                    VALIDATE_PROMPT,
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                ],
                config={"response_mime_type": "application/json"},
            )
        )

        try:
            result = json.loads(response.text)
            detected = result.get("detected_type", "unknown")
            confidence = float(result.get("confidence", 0.5))
            reason = result.get("reason", "")
        except Exception:
            detected = "unknown"
            confidence = 0.3
            reason = "Classification parsing failed"

        # Check match
        acceptable = SLOT_TYPE_MAP.get(expected_type)
        if acceptable is None:
            # "supporting_docs" — accept anything
            match = True
        else:
            match = detected in acceptable

        return JSONResponse({
            "match": match,
            "detected": detected,
            "detected_display": TYPE_DISPLAY.get(detected, detected),
            "expected_display": TYPE_DISPLAY.get(expected_type, expected_type),
            "confidence": confidence,
            "reason": reason,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────
# Full pipeline analysis with SSE progress streaming
# ─────────────────────────────────────────────────────────────────

AGENT_CLASSES = [
    ("identity_agent", IdentityAgent),
    ("financial_agent", FinancialAgent),
    ("temporal_agent", TemporalAgent),
    ("sponsor_agent", SponsorAgent),
    ("enrollment_agent", EnrollmentAgent),
]


def _sse(event_dict: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(event_dict, default=str)}\n\n"


async def _run_pipeline(
    pdf_bytes_list: list[bytes],
    context: dict,
    debug: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Async generator that runs the full Uplan pipeline and yields SSE events.
    Each stage yields start + done events. Final event is 'complete' with full result.
    """
    loop = asyncio.get_event_loop()
    inter_agent_delay = float(os.environ.get("INTER_AGENT_DELAY", "1.0"))

    try:
        # ── Stage 1: Structural encoding ──────────────────────────────
        yield _sse({"type": "stage_start", "stage": "encode",
                    "message": "Rendering pages and running Flash extraction..."})
        t0 = time.time()

        all_pages: list[PageExtraction] = []
        for pdf_bytes in pdf_bytes_list:
            pages = await loop.run_in_executor(None, extract_pages, pdf_bytes)
            all_pages.extend(pages)

        graph = await loop.run_in_executor(None, build_graph, all_pages)
        encode_elapsed = time.time() - t0

        yield _sse({
            "type": "stage_done", "stage": "encode",
            "elapsed_s": round(encode_elapsed, 1),
            "data": {
                "page_count": graph.source_page_count,
                "token_count": graph.token_count,
                "raw_tokens": graph.estimated_raw_tokens,
                "doc_types": graph.source_doc_types,
                "compression_pct": round((1 - graph.token_count / max(graph.estimated_raw_tokens, 1)) * 100, 1),
            }
        })

        # ── Stage 2: Privacy purge ────────────────────────────────────
        yield _sse({"type": "stage_start", "stage": "purge",
                    "message": "Purging raw document bytes from memory..."})
        pdf_bytes_list = []  # Purge -- raw bytes no longer referenced
        yield _sse({"type": "stage_done", "stage": "purge", "elapsed_s": 0.0})

        # ── Stage 3: Rule engine ──────────────────────────────────────
        yield _sse({"type": "stage_start", "stage": "rules",
                    "message": "Applying 9 deterministic rules to semantic graph..."})
        t0 = time.time()

        # Build visa context from intake form
        visa_context_dict = None
        try:
            from rules.context_profiles import VisaContext, build_context
            dest = context.get("dest", "")
            visa_type = context.get("type", "student")
            visa_context_dict = build_context(dest, visa_type)
        except Exception:
            pass

        rule_findings_raw = await loop.run_in_executor(
            None,
            lambda: run_rules(graph, context=None)
        )
        rule_findings = [f.model_dump() for f in rule_findings_raw]
        rules_elapsed = time.time() - t0

        yield _sse({
            "type": "stage_done", "stage": "rules",
            "elapsed_s": round(rules_elapsed, 1),
            "data": {"finding_count": len(rule_findings)}
        })

        # ── Stages 4-8: Specialist agents ────────────────────────────
        graph_dict = graph.model_dump()
        agent_findings: list[dict] = []
        completed_agents: list[str] = []

        state = {
            "semantic_graph": graph_dict,
            "rule_findings": rule_findings,
            "agent_findings": [],
            "completed_agents": [],
            "visa_context": visa_context_dict,
            "document_checklist": [],
            "verdict": None,
            "risk_score": None,
            "raw_purged": True,
            "_debug_logger": None,
        }

        for i, (agent_id, agent_cls) in enumerate(AGENT_CLASSES):
            if i > 0:
                await asyncio.sleep(inter_agent_delay)

            yield _sse({"type": "agent_start", "agent": agent_id})
            t0 = time.time()

            try:
                result = await loop.run_in_executor(None, lambda cls=agent_cls, s=state: cls().run(s))
                agent_findings.extend(result.get("agent_findings", []))
                completed_agents.extend(result.get("completed_agents", []))
                state["agent_findings"] = agent_findings
                state["completed_agents"] = completed_agents
                agent_elapsed = time.time() - t0

                # Count anomalies at this agent's findings
                this_agent_findings = result.get("agent_findings", [])
                anomaly_count = sum(len(f.get("anomalies", [])) for f in this_agent_findings)

                yield _sse({
                    "type": "agent_done", "agent": agent_id,
                    "elapsed_s": round(agent_elapsed, 1),
                    "verdict": this_agent_findings[0].get("agent_verdict") if this_agent_findings else None,
                    "anomaly_count": anomaly_count,
                })

            except Exception as e:
                yield _sse({"type": "agent_error", "agent": agent_id, "message": str(e)})
                completed_agents.append(agent_id)

        # ── Stage 9: Narrative synthesis ──────────────────────────────
        await asyncio.sleep(inter_agent_delay)
        yield _sse({"type": "stage_start", "stage": "narrative",
                    "message": "Synthesising cross-document verdict and rebuttal guidance..."})
        t0 = time.time()

        state["agent_findings"] = agent_findings
        narrative_result = await loop.run_in_executor(None, lambda: NarrativeAgent().run(state))
        narrative_elapsed = time.time() - t0

        yield _sse({
            "type": "stage_done", "stage": "narrative",
            "elapsed_s": round(narrative_elapsed, 1),
        })

        # ── Final: assemble result ────────────────────────────────────
        verdict = narrative_result.get("verdict", {})
        risk_score = narrative_result.get("risk_score", 1.0)

        final_result = {
            "verdict": verdict.get("verdict", "FAIL"),
            "risk_score": risk_score,
            "rejection_narrative": verdict.get("rejection_narrative", ""),
            "rebuttal_guidance": verdict.get("rebuttal_guidance", []),
            "citations": verdict.get("citations", []),
            "critical_issues": verdict.get("critical_issues", []),
            "agent_findings": agent_findings,
            "rule_findings": rule_findings,
            "graph": {
                "identity": graph_dict.get("identity", {}),
                "financial": graph_dict.get("financial", {}),
                "sponsor": graph_dict.get("sponsor"),
                "enrollment": graph_dict.get("enrollment"),
                "temporal": graph_dict.get("temporal", {}),
                "source_doc_types": graph_dict.get("source_doc_types", []),
                "token_count": graph_dict.get("token_count", 0),
                "page_count": graph_dict.get("source_page_count", 0),
                "estimated_raw_tokens": graph_dict.get("estimated_raw_tokens", 0),
            },
            "purged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        yield _sse({"type": "complete", "result": final_result})

    except Exception as e:
        tb = traceback.format_exc()
        yield _sse({"type": "error", "message": str(e), "traceback": tb[:800]})


@app.post("/api/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    context: str = Form(default="{}"),
):
    """
    Full pipeline analysis with SSE progress streaming.
    Accepts multipart form:
      - files: one or more PDF uploads
      - context: JSON string with {from, dest, type}
    """
    try:
        ctx = json.loads(context)
    except Exception:
        ctx = {}

    pdf_bytes_list = []
    for f in files:
        data = await f.read()
        pdf_bytes_list.append(data)

    if not pdf_bytes_list:
        raise HTTPException(400, "No files uploaded")

    async def generate():
        async for chunk in _run_pipeline(pdf_bytes_list, ctx):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  Uplan server starting on http://0.0.0.0:{port}")
    print(f"  Flash model : {os.environ.get('FLASH_MODEL', 'gemini-2.5-flash')}")
    print(f"  Pro model   : {os.environ.get('PRO_MODEL', 'gemini-2.5-pro')}")
    print(f"  Debug mode  : {os.environ.get('UPLAN_DEBUG', '1')}\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("DEV", "0") == "1",
        log_level="info",
    )
