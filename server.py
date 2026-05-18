"""
Uplan FastAPI Server
====================
Serves the UI and exposes backend API endpoints.

Routes:
  GET  /                  -> serves ui/index.html
  POST /api/validate      -> quick Flash doc-type check (per upload)
  POST /api/analyze       -> full pipeline with SSE progress stream (legacy)
  POST /api/start         -> start async pipeline, returns task_id immediately
  GET  /api/status/{id}   -> poll task progress + result
  GET  /api/health        -> health check
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
import uuid
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
    return {"status": "ok", "model_flash": os.environ.get("FLASH_MODEL", "gemini-3-flash-preview")}


# ─────────────────────────────────────────────────────────────────
# In-memory Task Store for async polling
# ─────────────────────────────────────────────────────────────────

_tasks: dict[str, dict] = {}

# Auto-purge completed tasks older than 30 minutes
TASK_TTL_SECONDS = 1800


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
        file_bytes = await file.read()
        filename = getattr(file, "filename", "").lower()

        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_bytes = file_bytes
            mime_type = "image/jpeg" if filename.endswith((".jpg", ".jpeg")) else "image/png"
        else:
            # Assume PDF and render first page to PNG
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page = doc[0]
            pix = page.get_pixmap(dpi=100)  # Low DPI for quick validation
            img_bytes = pix.tobytes("png")
            doc.close()
            mime_type = "image/png"

        # Call Flash for type classification
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=FLASH_MODEL,
                contents=[
                    VALIDATE_PROMPT,
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


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
            "missing_documents": verdict.get("missing_documents", []),
            "rejection_narrative": verdict.get("rejection_narrative", ""),
            "rebuttal_guidance": verdict.get("rebuttal_guidance", []),
            "citations": verdict.get("citations", []),
            "critical_issues": verdict.get("critical_issues", []),
            "warning_issues": verdict.get("warning_issues", []),
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


async def _run_pipeline_to_store(
    task_id: str,
    pdf_bytes_list: list[bytes],
    context: dict,
) -> None:
    """
    Runs the full pipeline and writes progress events into _tasks[task_id].
    This is the background coroutine launched by POST /api/start.
    """
    task = _tasks[task_id]

    def emit(event_dict: dict):
        """Append an event to the task's progress log."""
        event_dict["_ts"] = time.time()
        task["progress"].append(event_dict)
        task["current_stage"] = event_dict.get("stage", event_dict.get("agent", task["current_stage"]))

    loop = asyncio.get_event_loop()
    inter_agent_delay = float(os.environ.get("INTER_AGENT_DELAY", "1.0"))

    try:
        # ── Stage 1: Structural encoding
        emit({"type": "stage_start", "stage": "encode", "message": "Rendering pages and running Flash extraction..."})
        t0 = time.time()

        all_pages: list[PageExtraction] = []
        for pdf_bytes in pdf_bytes_list:
            pages = await loop.run_in_executor(None, extract_pages, pdf_bytes)
            all_pages.extend(pages)

        graph = await loop.run_in_executor(None, build_graph, all_pages)
        encode_elapsed = time.time() - t0

        emit({
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

        # ── Stage 2: Privacy purge
        emit({"type": "stage_start", "stage": "purge", "message": "Purging raw document bytes from memory..."})
        pdf_bytes_list = []
        emit({"type": "stage_done", "stage": "purge", "elapsed_s": 0.0})

        # ── Stage 3: Rule engine
        emit({"type": "stage_start", "stage": "rules", "message": "Applying 9 deterministic rules to semantic graph..."})
        t0 = time.time()

        visa_context_dict = None
        try:
            from rules.context_profiles import VisaContext, build_context
            dest = context.get("dest", "")
            visa_type = context.get("type", "student")
            visa_context_dict = build_context(dest, visa_type)
        except Exception:
            pass

        rule_findings_raw = await loop.run_in_executor(
            None, lambda: run_rules(graph, context=None)
        )
        rule_findings = [f.model_dump() for f in rule_findings_raw]
        rules_elapsed = time.time() - t0

        emit({
            "type": "stage_done", "stage": "rules",
            "elapsed_s": round(rules_elapsed, 1),
            "data": {"finding_count": len(rule_findings)}
        })

        # ── Stages 4-8: Specialist agents
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

            emit({"type": "agent_start", "agent": agent_id})
            t0 = time.time()

            try:
                result = await loop.run_in_executor(None, lambda cls=agent_cls, s=state: cls().run(s))
                agent_findings.extend(result.get("agent_findings", []))
                completed_agents.extend(result.get("completed_agents", []))
                state["agent_findings"] = agent_findings
                state["completed_agents"] = completed_agents
                agent_elapsed = time.time() - t0

                this_agent_findings = result.get("agent_findings", [])
                anomaly_count = sum(len(f.get("anomalies", [])) for f in this_agent_findings)

                emit({
                    "type": "agent_done", "agent": agent_id,
                    "elapsed_s": round(agent_elapsed, 1),
                    "verdict": this_agent_findings[0].get("agent_verdict") if this_agent_findings else None,
                    "anomaly_count": anomaly_count,
                })
            except Exception as e:
                emit({"type": "agent_error", "agent": agent_id, "message": str(e)})
                completed_agents.append(agent_id)

        # ── Stage 9: Narrative synthesis
        await asyncio.sleep(inter_agent_delay)
        emit({"type": "stage_start", "stage": "narrative", "message": "Synthesising cross-document verdict and rebuttal guidance..."})
        t0 = time.time()

        state["agent_findings"] = agent_findings
        narrative_result = await loop.run_in_executor(None, lambda: NarrativeAgent().run(state))
        narrative_elapsed = time.time() - t0

        emit({"type": "stage_done", "stage": "narrative", "elapsed_s": round(narrative_elapsed, 1)})

        # ── Final: assemble result
        verdict = narrative_result.get("verdict", {})
        risk_score = narrative_result.get("risk_score", 1.0)

        final_result = {
            "verdict": verdict.get("verdict", "FAIL"),
            "risk_score": risk_score,
            "missing_documents": verdict.get("missing_documents", []),
            "rejection_narrative": verdict.get("rejection_narrative", ""),
            "rebuttal_guidance": verdict.get("rebuttal_guidance", []),
            "citations": verdict.get("citations", []),
            "critical_issues": verdict.get("critical_issues", []),
            "warning_issues": verdict.get("warning_issues", []),
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

        emit({"type": "complete", "result": final_result})
        task["status"] = "complete"
        task["result"] = final_result
        task["completed_at"] = time.time()

    except Exception as e:
        tb = traceback.format_exc()
        emit({"type": "error", "message": str(e), "traceback": tb[:800]})
        task["status"] = "error"
        task["error"] = str(e)
        task["completed_at"] = time.time()


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
# Async Polling Endpoints (Task 2)
# ─────────────────────────────────────────────────────────────────

def _purge_stale_tasks():
    """Remove completed/errored tasks older than TTL."""
    now = time.time()
    stale = [tid for tid, t in _tasks.items()
             if t.get("completed_at") and (now - t["completed_at"]) > TASK_TTL_SECONDS]
    for tid in stale:
        del _tasks[tid]


@app.post("/api/start")
async def start_analysis(
    files: list[UploadFile] = File(...),
    context: str = Form(default="{}"),
):
    """
    Start async pipeline. Returns task_id immediately (HTTP 202).
    The pipeline runs in the background. Poll GET /api/status/{task_id} for progress.
    """
    _purge_stale_tasks()

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

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "running",
        "progress": [],
        "current_stage": None,
        "result": None,
        "error": None,
        "started_at": time.time(),
        "completed_at": None,
    }

    # Launch the pipeline as a background coroutine
    asyncio.create_task(_run_pipeline_to_store(task_id, pdf_bytes_list, ctx))

    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "status": "running"},
    )


@app.get("/api/status/{task_id}")
async def get_task_status(
    task_id: str,
    since: int = 0,
):
    """
    Poll task progress. Returns current status + new progress events since index `since`.
    Query param `since` is the index of the last event the client has seen.
    This avoids sending the entire progress array on every poll.
    """
    if task_id not in _tasks:
        raise HTTPException(404, f"Task {task_id} not found")

    task = _tasks[task_id]
    elapsed = time.time() - task["started_at"]

    return {
        "task_id": task_id,
        "status": task["status"],
        "current_stage": task["current_stage"],
        "elapsed_s": round(elapsed, 1),
        "progress": task["progress"][since:],
        "progress_total": len(task["progress"]),
        "result": task["result"],
        "error": task["error"],
    }


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  Uplan server starting on http://0.0.0.0:{port}")
    print(f"  Flash model : {os.environ.get('FLASH_MODEL', 'gemini-3-flash-preview')}")
    print(f"  Pro model   : {os.environ.get('PRO_MODEL', 'gemini-3.1-pro-preview')}")
    print(f"  Debug mode  : {os.environ.get('UPLAN_DEBUG', '1')}\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("DEV", "0") == "1",
        log_level="info",
    )
