"""
Uplan Structural Encoder -- Per-Page Entity Extraction

Uses PyMuPDF (fitz) to render each page as PNG, then sends to Gemini Flash
with structured output (response_schema=PageExtraction) for typed extraction.

ASYNC with THROTTLING: Pages are extracted concurrently via asyncio.gather()
but rate-limited by a semaphore (MAX_CONCURRENT_PAGES, default=2) to stay
within Google AI Studio free tier limits. Exponential backoff retries on 429.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
from typing import Callable, Optional

import fitz  # PyMuPDF
from google.genai import types

from config import (
    FLASH_MODEL,
    PAGE_DPI,
    MAX_CONCURRENT_PAGES,
    API_RETRY_ATTEMPTS,
    API_RETRY_BASE_DELAY,
    API_RETRY_MAX_DELAY,
    client,
)
from encoding.schema import PageExtraction


# -- Extraction Prompt -----------------------------------------------------

EXTRACTION_PROMPT = """You are a forensic immigration document analyst performing typed entity extraction.

TASK: Examine this single page from an immigration visa application document package.
Extract ALL structured data visible on this page into the provided JSON schema.

CRITICAL RULES:
1. EXTRACT ONLY what is explicitly visible on this page. Never infer, guess, or fabricate.
2. If a field is not present on this page, set it to null.
3. Classify page_type based on observable content:
   - "bank_statement": Account transactions, balances, bank headers
   - "payslip": Salary/wage payment details, pay period
   - "tax_return": Annual tax filing, assessment, income declaration
   - "passport": Identity page with name, DOB, nationality, photo
   - "sponsor_letter": Letter from a financial sponsor declaring support
   - "employment_letter": Letter confirming employment, position, salary
   - "enrollment_letter": Admission/enrollment letter from an institution
   - "affidavit": Sworn financial or support declaration signed by a sponsor/parent
   - "unknown": Cannot determine document type

4. FOR AFFIDAVIT PAGES -- READ THIS CAREFULLY:
   An affidavit is signed BY a sponsor (parent/guardian) FOR a visa applicant.

   FIELD MAPPING for affidavits:
   - person_name       = the DECLARANT who signs (e.g. "KORADA SATYANARAYANA" -- the father)
   - date_of_birth     = the DECLARANT/SPONSOR's own DOB (their personal info, e.g. "21/02/1971")
   - applicant_name    = the BENEFICIARY being sponsored -- extract from "father of X", "for my son X"
   - applicant_dob     = the APPLICANT's DOB -- from the family member table, find the Son/Daughter row
                         EXAMPLE: If table shows "Korada Sai Kiran (Son), Date of Birth: 02-06-2003"
                         then applicant_dob = "02-06-2003"  (NOT the father's 21/02/1971)
   - sponsor_name      = same as person_name
   - sponsor_relationship = relationship declared (e.g. "Father")

   ASSET FIELD MAPPING -- split assets by category:
   - declared_liquid_assets = SUM of BANK and FINANCIAL accounts ONLY:
       savings accounts + fixed deposits + LIC policies + investment accounts
       e.g. 90978 + 154603 + 154603 + 1023668 + 162290 + 158519 = 1744661
       DO NOT include cash/gold/silver/property in this field
   - declared_movable_assets = cash + gold + silver ornaments + jewelry ONLY
       e.g. "Cash, Gold & Silver ornaments: Rs. 25,00,000" -> 2500000
   - declared_property_value = immovable property ONLY: flats, land, agricultural land
       e.g. "Properties owned: Rs. 1,63,60,000" -> 16360000
   - declared_annual_income = SUM of ALL annual income sources:
       salary + business income + rental income + other
       e.g. 300000 + 552000 + 348000 = 1200000
   - sponsor_declared_income = same value as declared_annual_income
   - currency = "INR" if amounts are in Rupees

5. For financial transactions on bank statements: extract EVERY visible row.
   Mark as "credit" for incoming money, "debit" for outgoing.
6. For all amounts: extract the numeric value only.
   Strip commas, currency symbols, dashes. "1,54,603.98" -> 154603.98, "10,23,668-40" -> 1023668.40
7. In anomaly_flags, note observable irregularities:
   - "large_unlabeled_deposit", "date_format_inconsistent", "possible_alteration",
     "balance_discontinuity", "name_spelling_variation"

This is a SINGLE page. Extract only what THIS page shows."""


# -- Rate-limited async extraction -----------------------------------------

# Global semaphore — throttles concurrent API calls across all pages
_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    """Lazy-init semaphore (must be created inside the event loop)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
    return _semaphore


async def _extract_single_page_async(
    page_idx: int,
    pdf_bytes: bytes,
    png_bytes: bytes | None,
    fast_text: str | None,
    total: int,
    logger=None,
) -> PageExtraction:
    """Extract typed entities from a single page via Gemini Flash (async).
    Uses Text-to-Flash hybrid if fast_text is provided, otherwise Vision.
    Uses semaphore for concurrency control and exponential backoff for retries."""
    
    # Deduplication check
    if not png_bytes and not fast_text:
        print(f"  [DEDUPED] Page {page_idx + 1}/{total} -- skipped identical image")
        return PageExtraction(page_number=page_idx + 1, page_type="unknown")

    sem = _get_semaphore()

    # CP1 -- log raw page content before sending to Flash
    if logger and png_bytes:
        logger.cp1_page_content(page_idx, pdf_bytes, png_bytes)

    async with sem:
        start = time.time()
        last_err = None

        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                if fast_text:
                    contents = [
                        EXTRACTION_PROMPT,
                        "Parse the following document text into the standard JSON schema:\n\n[TEXT]\n" + fast_text
                    ]
                else:
                    contents = [
                        EXTRACTION_PROMPT,
                        types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                    ]

                response = await client.aio.models.generate_content(
                    model=FLASH_MODEL,
                    contents=contents,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": PageExtraction,
                    },
                )
                extraction: PageExtraction = response.parsed
                extraction.page_number = page_idx + 1
                
                # If we bypassed Vision, attach the raw text to the schema as requested
                if fast_text:
                    extraction.raw_text = fast_text

                elapsed = time.time() - start
                mode = "TEXT" if fast_text else "VISION"
                print(f"  [OK] Page {page_idx + 1}/{total} -- {extraction.page_type.value} [{mode}] ({elapsed:.1f}s)")

                # CP2 -- log Flash extraction output
                if logger:
                    logger.cp2_extraction(page_idx, extraction.model_dump(), elapsed)

                return extraction

            except Exception as e:
                last_err = e
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_rate_limit and attempt < API_RETRY_ATTEMPTS - 1:
                    delay = None
                    # Try to extract exact wait time from header or error string
                    try:
                        if hasattr(e, "response") and e.response and hasattr(e.response, "headers"):
                            retry_after = e.response.headers.get("Retry-After")
                            if retry_after:
                                delay = float(retry_after)
                    except Exception:
                        pass
                    
                    if delay is None:
                        m = re.search(r"Wait (\d+) seconds", err_str)
                        if m:
                            delay = float(m.group(1))
                        else:
                            delay = min(API_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), API_RETRY_MAX_DELAY)
                    
                    delay += 1.5 # Buffer
                    print(f"  [WAIT] Page {page_idx + 1}: rate limited, retry in {delay:.1f}s (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})")
                    await asyncio.sleep(delay)
                elif not is_rate_limit:
                    # Non-rate-limit error -- don't retry
                    break

        # All retries exhausted or non-retryable error
        elapsed = time.time() - start
        print(f"  [FAIL] Page {page_idx + 1}: extraction failed -- {last_err} ({elapsed:.1f}s)")
        return PageExtraction(page_number=page_idx + 1, page_type="unknown")


async def _extract_pages_async(
    pdf_bytes: bytes,
    *,
    on_page=None,
    logger=None,
) -> list[PageExtraction]:
    """
    Extract typed entities from every page of a PDF.

    Pages run concurrently but throttled by semaphore (MAX_CONCURRENT_PAGES)
    to stay within API rate limits.
    """
    # Reset semaphore for each extraction run (new event loop context)
    global _semaphore
    _semaphore = None

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    # Process pages: Text Fast-Path vs Image Deduplication
    page_tasks: list[tuple[int, bytes | None, str | None]] = []
    seen_hashes = set()

    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text()

        if len(text.strip()) > 100:
            # Text-heavy page -> Fast path (skip PNG)
            page_tasks.append((page_idx, None, text))
        else:
            # Scanned / Image-heavy page -> Rasterize to PNG
            pix = page.get_pixmap(dpi=PAGE_DPI)
            png_bytes = pix.tobytes("png")
            img_hash = hashlib.sha256(png_bytes).hexdigest()

            if img_hash in seen_hashes:
                # Exact duplicate page (e.g. blank back page)
                page_tasks.append((page_idx, None, None))
            else:
                seen_hashes.add(img_hash)
                page_tasks.append((page_idx, png_bytes, None))

    doc.close()

    if on_page:
        on_page(0, total_pages)

    print(f"  Extracting {total_pages} pages (max {MAX_CONCURRENT_PAGES} concurrent)...")

    # Fire extraction calls -- semaphore controls concurrency
    tasks = [
        _extract_single_page_async(idx, pdf_bytes, png, txt, total_pages, logger)
        for idx, png, txt in page_tasks
    ]
    results = await asyncio.gather(*tasks)

    if on_page:
        on_page(total_pages, total_pages)

    return list(results)


# -- Sync wrapper ----------------------------------------------------------


def extract_pages(
    pdf_bytes: bytes,
    *,
    on_page=None,
    logger=None,
) -> list[PageExtraction]:
    """
    Sync wrapper for async extraction.
    Pass a DebugLogger instance to enable CP1/CP2 checkpoint logging.
    """
    coro = _extract_pages_async(pdf_bytes, on_page=on_page, logger=logger)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
