"""Central settings and thresholds.

Matcher tolerances live here (not buried in rules) so they are easy to explain, test and
change. Env vars, the model name and token prices are added in a later phase.
"""

# A difference of up to this many paise is treated as rounding, not an exception.
AMOUNT_TOLERANCE_PAISE = 1
SETTLEMENT_TOLERANCE_PAISE = 1

# Razorpay-style settlement cycle: payout is due this long after capture (T+2).
SETTLEMENT_DELAY_DAYS = 2

# Money actions above this amount (in paise) will need double confirmation (policy phase).
LARGE_AMOUNT_PAISE = 10_000 * 100
