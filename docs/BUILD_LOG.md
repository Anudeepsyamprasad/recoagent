# Build log

Format: time (IST) | decision | why | what changed.

- 2026-11-05 | Project kickoff | Shortlisted for Razorpay AI Builder; form needs an agent project | Plan agreed, repo scaffolded
- 2026-11-05 | Stay on custom tool-use loop with replay cassettes | Transparent, testable, free CI and demo | Architecture fixed
- 2026-11-05 | Schemas + seeded generator | Planted 8 exception types and 4 benign near-misses with ground truth, so precision and recall are measurable | schemas.py, generate.py, tests
- 2026-11-05 | Ingest + deterministic matcher | Rules close clean orders; exceptions carry factual signals, not categories (root cause is the agent's job) | ingest.py, matcher.py, config.py
- 2026-11-06 | LLM client, tools, agent loop | Record/replay cassettes; no forced tool_choice (current models reject it); setup errors stop the run, transient errors become per-exception findings | llm.py, tools.py, agent.py, audit.py, cli.py
- 2026-11-06 | Policy, ledger, eval, CI gate | Money actions always need a human; >Rs10,000 needs the amount typed back; gate fails on unrecorded prompt changes | policy.py, executor.py, eval.py, eval_gate.py, ci.yml
