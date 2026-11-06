"""Central settings and thresholds.

Everything a reviewer might ask "why this number?" about lives here, in one place.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env (gitignored) if present; real env vars still win

# ---- matcher tolerances ----------------------------------------------------
# A difference of up to this many paise is treated as rounding, not an exception.
AMOUNT_TOLERANCE_PAISE = 1
SETTLEMENT_TOLERANCE_PAISE = 1

# Razorpay-style settlement cycle: payout is due this long after capture (T+2).
SETTLEMENT_DELAY_DAYS = 2

# Money actions above this amount (in paise) will need double confirmation (policy phase).
LARGE_AMOUNT_PAISE = 10_000 * 100

# ---- LLM -------------------------------------------------------------------
DEFAULT_MODEL = "claude-sonnet-5-5"

# USD per 1M tokens (input, output): base tier, prompts under 100K tokens, no caching.
# Sources (Nov 2026): Anthropic's Sonnet 5.5 page and third-party price trackers.
# RE-CHECK on the Anthropic pricing page before quoting a cost in the README.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

MAX_TOKENS = 4096  # leaves room for the model's own reasoning plus the finding
REQUEST_TIMEOUT_S = 60
MAX_ATTEMPTS = 3  # tries per request; backoff 1s, 2s between them
MAX_STEPS = 6  # tool-use turns per exception before we give up
COST_CAP_USD = 1.0  # per run; remaining exceptions become NEEDS_HUMAN


def get_model() -> str:
    return os.getenv("RECO_MODEL") or DEFAULT_MODEL


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call. Unknown models fail loudly: never guess a price."""
    if model not in PRICES_PER_MTOK:
        raise ValueError(
            f"No price for model {model!r}. Add it to PRICES_PER_MTOK in config.py "
            "(check Anthropic's pricing page)."
        )
    price_in, price_out = PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
