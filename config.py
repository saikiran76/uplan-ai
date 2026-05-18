"""
Uplan -- Central Configuration
Initializes the Gemini client and exposes model names + rule thresholds.
Uses Google AI Studio directly via API key.
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# -- Client Initialization ------------------------------------------------
# Using Google AI Studio directly via API key.
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# -- Model Selection -------------------------------------------------------
# Available models confirmed via models.list():
#   Extraction: gemini-3-flash-preview (fast, vision-capable)
#   Reasoning:  gemini-3.1-pro-preview (smart, structured output)
FLASH_MODEL = os.environ.get("FLASH_MODEL", "gemini-3-flash-preview")
PRO_MODEL = os.environ.get("PRO_MODEL", "gemini-3.1-pro-preview")

# -- Rate Limit / Retry Config --------------------------------------------
# With paid credits, rate limits are generous. Keeping retry logic as safety net.
MAX_CONCURRENT_PAGES = int(os.environ.get("MAX_CONCURRENT_PAGES", "5"))
API_RETRY_ATTEMPTS = int(os.environ.get("API_RETRY_ATTEMPTS", "3"))
API_RETRY_BASE_DELAY = float(os.environ.get("API_RETRY_BASE_DELAY", "2.0"))
API_RETRY_MAX_DELAY = float(os.environ.get("API_RETRY_MAX_DELAY", "30.0"))
INTER_AGENT_DELAY = float(os.environ.get("INTER_AGENT_DELAY", "1.0"))

# -- Rule Engine Thresholds ------------------------------------------------
# Hardcoded for PoC. Will be RAG-fed from policy store in production.
ALPHA_SPIKE_RATIO = float(os.environ.get("ALPHA_SPIKE_RATIO", "3.0"))
EPSILON_INCOME_COHERENCE = float(os.environ.get("EPSILON_INCOME_COHERENCE", "0.20"))
DELTA_GAP_WARN_DAYS = int(os.environ.get("DELTA_GAP_WARN_DAYS", "30"))
DELTA_GAP_CRIT_DAYS = int(os.environ.get("DELTA_GAP_CRIT_DAYS", "90"))

# -- Page Rendering --------------------------------------------------------
PAGE_DPI = int(os.environ.get("PAGE_DPI", "150"))
