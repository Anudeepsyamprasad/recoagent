# Architecture

Orders + Razorpay data -> deterministic matcher -> exceptions -> Claude agent (read-only tools)
-> structured finding -> policy (safe / money / low confidence) -> idempotent executor
-> audit log -> evaluation vs ground truth.

(Expand this after the pipeline works.)
