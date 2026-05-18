"""
Uplan Document Validation Gate

Flash-based upload validation: checks declared label vs. detected content
before any page enters the extraction pipeline.
Surfaces mismatch errors to UI immediately.
Includes exponential backoff retry for rate limits.
"""

from __future__ import annotations

import random
import time
from typing import Optional

import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from config import (
    FLASH_MODEL,
    PAGE_DPI,
    API_RETRY_ATTEMPTS,
    API_RETRY_BASE_DELAY,
    API_RETRY_MAX_DELAY,
    client,
)
from google.genai import types


class ValidationResult(BaseModel):
    """Structured output from the document validation check."""
    actual_type: str = Field(description="What type of document this actually is")
    matches_expected: bool = Field(description="True if it matches the expected label")
    confidence: float = Field(description="0.0-1.0 confidence in classification")
    reason: str = Field(description="Why it matches or doesn't match")


VALIDATION_PROMPT = """You are a document classification expert for immigration visa applications.

This document was uploaded as: "{expected_label}"

Examine this page and determine:
1. What type of document is this actually? Use one of these categories:
   bank_statement, payslip, tax_return, passport, sponsor_letter,
   employment_letter, enrollment_letter, affidavit, unknown
2. Does it match the expected label "{expected_label}"?
3. How confident are you (0.0-1.0)?
4. Brief reason for your classification.

Be strict: a bank statement is NOT a payslip, a passport is NOT an employment letter.
But allow reasonable equivalences: "bank statement" matches "bank_statement",
"salary slip" matches "payslip", "admission letter" matches "enrollment_letter"."""


def validate_upload(
    pdf_bytes: bytes,
    expected_label: str,
) -> ValidationResult:
    """
    Validate a single uploaded PDF against its expected document type.
    Uses only the first page for speed -- classification is visible on page 1.
    Retries with exponential backoff on rate limit errors.

    Args:
        pdf_bytes: Raw PDF file bytes.
        expected_label: What the user (or checklist) says this document is.

    Returns:
        ValidationResult with match status and reason.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Only need first page for type classification
    page = doc[0]
    pix = page.get_pixmap(dpi=PAGE_DPI)
    png_bytes = pix.tobytes("png")
    doc.close()

    last_err = None
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=FLASH_MODEL,
                contents=[
                    VALIDATION_PROMPT.format(expected_label=expected_label),
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ValidationResult,
                },
            )
            return response.parsed
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

            if is_rate_limit and attempt < API_RETRY_ATTEMPTS - 1:
                delay = min(
                    API_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1),
                    API_RETRY_MAX_DELAY,
                )
                print(f"  [WAIT] Validation: rate limited, retry in {delay:.0f}s (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})")
                time.sleep(delay)
            elif not is_rate_limit:
                break

    # Fail open for validation -- don't block the pipeline on classification errors
    return ValidationResult(
        actual_type="unknown",
        matches_expected=True,  # Fail open
        confidence=0.0,
        reason=f"Validation failed: {last_err}",
    )


def validate_all_uploads(
    uploads: list[dict],
) -> list[dict]:
    """
    Validate a batch of uploads against their expected labels.
    Runs sequentially to respect rate limits.

    Args:
        uploads: List of {"label": str, "pdf_bytes": bytes}

    Returns:
        List of {"label": str, "validation": ValidationResult, "accepted": bool}
    """
    results = []
    for i, upload in enumerate(uploads):
        if i > 0:
            # Small delay between validation calls
            time.sleep(2.0)
        result = validate_upload(upload["pdf_bytes"], upload["label"])
        results.append({
            "label": upload["label"],
            "validation": result,
            "accepted": result.matches_expected or result.confidence < 0.5,
        })
    return results
