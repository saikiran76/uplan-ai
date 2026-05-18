"""
Uplan Debug Logger -- Full Pipeline Observability (CP0-CP6)

Writes structured debug artifacts to debug/<run_id>/ directory.
Each checkpoint captures a specific pipeline stage for forensic inspection.

Checkpoints:
  CP0 -- Document ingestion (file metadata, is_scanned detection)
  CP1 -- Per-page raw content (extracted text or PNG size)
  CP2 -- Flash extraction output per page (raw JSON + non-null fields)
  CP3 -- Graph builder merge (final graph + per-node summary)
  CP4 -- Agent prompt (exact string sent to Pro, token estimate)
  CP5 -- Agent raw response (parsed AgentFinding + any parse errors)
  CP6 -- Narrative synthesis input + output

Usage:
  from debug.logger import DebugLogger
  logger = DebugLogger(run_id="my_run")  # or auto-generates timestamp ID
  logger.cp0_ingestion(pdf_path, pdf_bytes)
  logger.cp1_page_content(page_idx, page_type, text_or_size)
  ...
  logger.close()  # writes summary index
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF


class DebugLogger:
    """
    Writes structured debug artifacts for each pipeline stage.
    All output goes to debug/<run_id>/ relative to the project root.
    """

    def __init__(self, run_id: Optional[str] = None, base_dir: Optional[str] = None):
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.start_time = time.time()

        # Base dir: debug/<run_id>/ relative to project root
        project_root = Path(__file__).parent.parent
        if base_dir:
            self.out_dir = Path(base_dir) / run_id
        else:
            self.out_dir = project_root / "debug" / run_id

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._log: list[dict] = []
        print(f"  [DEBUG] Logger initialized -> {self.out_dir}")

    def _write(self, filename: str, data: Any, as_text: bool = False) -> Path:
        """Write data to a file in the debug output directory."""
        path = self.out_dir / filename
        if as_text:
            path.write_text(str(data), encoding="utf-8")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        return path

    def _checkpoint(self, cp: str, summary: dict) -> None:
        """Record a checkpoint entry in the run log."""
        entry = {"checkpoint": cp, "timestamp": datetime.now(timezone.utc).isoformat(), **summary}
        self._log.append(entry)
        print(f"  [DEBUG] {cp} logged")

    # -----------------------------------------------------------------------
    # CP0 -- Document Ingestion
    # -----------------------------------------------------------------------

    def cp0_ingestion(self, file_path: str, pdf_bytes: bytes) -> dict:
        """
        Log document intake metadata.
        Detects whether document is electronic (has text layer) or scanned.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)

        pages_info = []
        is_scanned = True  # assume scanned until we find text
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            char_count = len(text)
            if char_count > 50:
                is_scanned = False
            pages_info.append({
                "page": i + 1,
                "char_count": char_count,
                "has_text_layer": char_count > 20,
            })
        doc.close()

        meta = {
            "file_path": file_path,
            "file_size_bytes": len(pdf_bytes),
            "page_count": page_count,
            "is_electronic": not is_scanned,
            "is_scanned": is_scanned,
            "pages": pages_info,
        }
        self._write("cp0_ingestion.json", meta)
        self._checkpoint("CP0", {"file": os.path.basename(file_path), "pages": page_count, "electronic": not is_scanned})
        return meta

    # -----------------------------------------------------------------------
    # CP1 -- Per-page raw content
    # -----------------------------------------------------------------------

    def cp1_page_content(self, page_idx: int, pdf_bytes: bytes, png_bytes: bytes) -> None:
        """
        Log raw content for a single page.
        For electronic PDFs: first 500 chars of extracted text.
        For all: PNG size in KB.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_idx]
        raw_text = page.get_text("text").strip()
        doc.close()

        info = {
            "page": page_idx + 1,
            "png_size_kb": round(len(png_bytes) / 1024, 1),
            "has_text_layer": len(raw_text) > 20,
            "text_char_count": len(raw_text),
            "text_preview": raw_text[:500] if raw_text else None,
        }
        self._write(f"cp1_page_{page_idx + 1:02d}_raw.json", info)

    # -----------------------------------------------------------------------
    # CP2 -- Flash extraction output per page
    # -----------------------------------------------------------------------

    def cp2_extraction(self, page_idx: int, extraction_dict: dict, elapsed_s: float) -> None:
        """
        Log what Flash returned for a single page.
        Highlights non-null fields and lists what was silently dropped (null).
        """
        non_null = {k: v for k, v in extraction_dict.items()
                    if v is not None and v != [] and v != 0 and v is not False}
        null_fields = [k for k, v in extraction_dict.items()
                       if v is None or v == [] or (isinstance(v, float) and v == 0.0)]

        output = {
            "page": page_idx + 1,
            "elapsed_s": round(elapsed_s, 2),
            "page_type_detected": extraction_dict.get("page_type"),
            "non_null_fields": non_null,
            "null_or_empty_fields": null_fields,
            "raw_extraction": extraction_dict,
        }
        self._write(f"cp2_page_{page_idx + 1:02d}_extraction.json", output)
        self._checkpoint("CP2", {
            "page": page_idx + 1,
            "type": extraction_dict.get("page_type"),
            "non_null_count": len(non_null),
            "elapsed_s": round(elapsed_s, 2),
        })

    # -----------------------------------------------------------------------
    # CP3 -- Graph builder merge
    # -----------------------------------------------------------------------

    def cp3_graph(self, graph_dict: dict, pages_count: int) -> None:
        """
        Log the final SemanticGraph after merge.
        Writes full graph JSON + a human-readable summary of each node.
        """
        self._write("cp3_graph.json", graph_dict)

        # Human-readable summary
        identity = graph_dict.get("identity", {})
        financial = graph_dict.get("financial", {})
        sponsor = graph_dict.get("sponsor") or {}
        doc_types = graph_dict.get("source_doc_types", [])

        summary_lines = [
            f"=== GRAPH SUMMARY (run: {self.run_id}) ===",
            f"Source pages: {pages_count}  |  Doc types: {doc_types}",
            f"Token count: {graph_dict.get('token_count')}  |  Compression: {graph_dict.get('estimated_raw_tokens', 0):,} -> {graph_dict.get('token_count', 0):,}",
            "",
            "--- IDENTITY ---",
            f"  canonical_name:  {identity.get('canonical_name')}",
            f"  date_of_birth:   {identity.get('date_of_birth')}",
            f"  nationality:     {identity.get('nationality')}",
            f"  passport_number: {identity.get('passport_number')}",
            f"  name_variants:   {identity.get('name_variants')}",
            "",
            "--- FINANCIAL ---",
            f"  source_is_affidavit:   {financial.get('source_is_affidavit')}",
            f"  currency:              {financial.get('currency')}",
            f"  closing_balance:       {financial.get('closing_balance')}",
            f"  avg_monthly_income:    {financial.get('avg_monthly_income')}",
            f"  transaction_count:     {financial.get('transaction_count')}",
            f"  declared_property:     {financial.get('declared_property_value')}",
            f"  declared_movables:     {financial.get('declared_movable_assets')}",
            "",
            "--- SPONSOR ---",
            f"  sponsor_name:     {sponsor.get('sponsor_name')}",
            f"  relationship:     {sponsor.get('relationship')}",
            f"  declared_income:  {sponsor.get('declared_income')}",
            "",
            f"  enrollment:  {'present' if graph_dict.get('enrollment') else 'null'}",
            f"  edges:       {[e.get('edge_type') for e in graph_dict.get('edges', [])]}",
        ]

        summary_text = "\n".join(summary_lines)
        self._write("cp3_graph_summary.txt", summary_text, as_text=True)
        self._checkpoint("CP3", {
            "canonical_name": identity.get("canonical_name"),
            "doc_types": doc_types,
            "closing_balance": financial.get("closing_balance"),
            "token_count": graph_dict.get("token_count"),
        })

    # -----------------------------------------------------------------------
    # CP4 -- Agent prompt
    # -----------------------------------------------------------------------

    def cp4_agent_prompt(self, agent_id: str, prompt: str, graph_subset_keys: list[str]) -> None:
        """Log the exact prompt string sent to the Pro model for an agent."""
        char_count = len(prompt)
        estimated_tokens = char_count // 4

        output = {
            "agent_id": agent_id,
            "graph_subset_keys": graph_subset_keys,
            "estimated_input_tokens": estimated_tokens,
            "prompt_char_count": char_count,
            "prompt": prompt,
        }
        self._write(f"cp4_{agent_id}_prompt.json", output)
        self._checkpoint("CP4", {
            "agent": agent_id,
            "estimated_tokens": estimated_tokens,
            "graph_keys": graph_subset_keys,
        })

    # -----------------------------------------------------------------------
    # CP5 -- Agent raw response
    # -----------------------------------------------------------------------

    def cp5_agent_response(self, agent_id: str, finding_dict: dict, elapsed_s: float,
                           parse_error: Optional[str] = None) -> None:
        """Log what the Pro model returned for an agent call."""
        output = {
            "agent_id": agent_id,
            "elapsed_s": round(elapsed_s, 2),
            "parse_error": parse_error,
            "verdict": finding_dict.get("agent_verdict"),
            "confidence": finding_dict.get("confidence"),
            "anomaly_count": len(finding_dict.get("anomalies", [])),
            "summary": finding_dict.get("summary"),
            "full_finding": finding_dict,
        }
        self._write(f"cp5_{agent_id}_response.json", output)
        self._checkpoint("CP5", {
            "agent": agent_id,
            "verdict": finding_dict.get("agent_verdict"),
            "confidence": finding_dict.get("confidence"),
            "elapsed_s": round(elapsed_s, 2),
            "parse_error": parse_error,
        })

    # -----------------------------------------------------------------------
    # CP6 -- Narrative synthesis
    # -----------------------------------------------------------------------

    def cp6_narrative(self, prompt: str, verdict_dict: dict, elapsed_s: float,
                      parse_error: Optional[str] = None) -> None:
        """Log the narrative synthesis prompt and final verdict output."""
        output = {
            "elapsed_s": round(elapsed_s, 2),
            "parse_error": parse_error,
            "verdict": verdict_dict.get("verdict"),
            "risk_score": verdict_dict.get("risk_score"),
            "critical_issues_count": len(verdict_dict.get("critical_issues", [])),
            "prompt_char_count": len(prompt),
            "prompt": prompt,
            "full_verdict": verdict_dict,
        }
        self._write("cp6_narrative.json", output)
        self._checkpoint("CP6", {
            "verdict": verdict_dict.get("verdict"),
            "risk_score": verdict_dict.get("risk_score"),
            "elapsed_s": round(elapsed_s, 2),
        })

    # -----------------------------------------------------------------------
    # Close / Summary
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Write the run summary index file."""
        total_elapsed = round(time.time() - self.start_time, 1)
        summary = {
            "run_id": self.run_id,
            "total_elapsed_s": total_elapsed,
            "checkpoints": self._log,
            "output_dir": str(self.out_dir),
        }
        self._write("run_summary.json", summary)
        print(f"  [DEBUG] Run complete: {len(self._log)} checkpoints in {total_elapsed}s")
        print(f"  [DEBUG] Debug artifacts: {self.out_dir}")
