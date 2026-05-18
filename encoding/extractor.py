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
import random
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
   - "affidavit": Sworn financial or support declaration
   - "unknown": Cannot determine document type
4. For financial transactions: extract EVERY visible row with date, description, amount.
   Mark as "credit" for incoming money, "debit" for outgoing.
   If a transaction has no clear source description, set label to null.
5. For all amounts: extract the numeric value only. Set the currency field separately.
6. In anomaly_flags, note observable irregularities:
   - "large_unlabeled_deposit": Significant credit with no source description
   - "date_format_inconsistent": Mixed date formats on the same page
   - "possible_alteration": Signs of digital manipulation, whiteout, or overwriting
   - "balance_discontinuity": Running balance doesn't match transaction arithmetic
   - "name_spelling_variation": Name spelled differently than expected

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
    png_bytes: bytes,
    total: int,
) -> PageExtraction:
    """Extract typed entities from a single page PNG via Gemini Flash (async).
    Uses semaphore for concurrency control and exponential backoff for retries."""
    sem = _get_semaphore()

    async with sem:
        start = time.time()
        last_err = None

        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                response = await client.aio.models.generate_content(
                    model=FLASH_MODEL,
                    contents=[
                        EXTRACTION_PROMPT,
                        types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                    ],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": PageExtraction,
                    },
                )
                extraction: PageExtraction = response.parsed
                extraction.page_number = page_idx + 1
                elapsed = time.time() - start
                print(f"  [OK] Page {page_idx + 1}/{total} -- {extraction.page_type.value} ({elapsed:.1f}s)")
                return extraction

            except Exception as e:
                last_err = e
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_rate_limit and attempt < API_RETRY_ATTEMPTS - 1:
                    delay = min(
                        API_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1),
                        API_RETRY_MAX_DELAY,
                    )
                    print(f"  [WAIT] Page {page_idx + 1}: rate limited, retry in {delay:.0f}s (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})")
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
    on_page: Optional[Callable[[int, int], None]] = None,
) -> list[PageExtraction]:
    """
    Extract typed entities from every page of a PDF.

    Pages run concurrently but throttled by semaphore (MAX_CONCURRENT_PAGES)
    to stay within free-tier rate limits.
    """
    # Reset semaphore for each extraction run (new event loop context)
    global _semaphore
    _semaphore = None

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    # Render all pages to PNG (CPU-bound, sequential -- very fast)
    page_images: list[tuple[int, bytes]] = []
    for page_idx in range(total_pages):
        pix = doc[page_idx].get_pixmap(dpi=PAGE_DPI)
        page_images.append((page_idx, pix.tobytes("png")))
    doc.close()

    if on_page:
        on_page(0, total_pages)

    print(f"  Extracting {total_pages} pages (max {MAX_CONCURRENT_PAGES} concurrent)...")

    # Fire extraction calls -- semaphore controls concurrency
    tasks = [
        _extract_single_page_async(idx, png, total_pages)
        for idx, png in page_images
    ]
    results = await asyncio.gather(*tasks)

    if on_page:
        on_page(total_pages, total_pages)

    return list(results)


# -- Sync wrapper ----------------------------------------------------------


def extract_pages(
    pdf_bytes: bytes,
    *,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> list[PageExtraction]:
    """
    Sync wrapper for async extraction.

    Handles event loop scenarios:
    - No running loop: uses asyncio.run() directly
    - Running loop (Streamlit): creates new loop in thread
    """
    coro = _extract_pages_async(pdf_bytes, on_page=on_page)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g. Streamlit)
        # Run async code in a separate thread with its own loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
