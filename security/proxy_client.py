"""
Uplan — Lobster Trap Proxy Client (Phase 2 — Scaffolded)

When wired, this rewrites the Gemini API base URL to route through
Lobster Trap's DPI layer for prompt injection detection, PII exfil
scanning, and YAML policy enforcement.

Not active in Phase 1 — all Gemini calls go direct to Vertex AI / AI Studio.
"""

from __future__ import annotations

import os
from typing import Optional


# Lobster Trap proxy URL — set when the proxy is deployed
LOBSTER_TRAP_URL = os.environ.get("LOBSTER_TRAP_URL", None)


def get_proxy_base_url() -> Optional[str]:
    """
    Returns the Lobster Trap proxy URL if configured, else None.
    When None, the google-genai client uses its default endpoint.
    """
    return LOBSTER_TRAP_URL


def is_proxy_active() -> bool:
    """Check if Lobster Trap proxy is configured and reachable."""
    return LOBSTER_TRAP_URL is not None


# ── Future: wrap google-genai client to route through proxy ─────────
#
# The google-genai SDK doesn't natively support custom base URLs.
# When Lobster Trap is wired in, the approach is:
#
# 1. Run Lobster Trap as an OpenAI-compatible proxy
# 2. Use the openai SDK (not google-genai) pointed at Lobster Trap
# 3. Lobster Trap forwards to Gemini after DPI inspection
#
# This requires switching from google-genai to openai SDK for all
# Gemini calls — a straightforward swap since Gemini supports the
# OpenAI-compatible API format.
