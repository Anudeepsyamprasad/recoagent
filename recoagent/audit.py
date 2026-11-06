"""Append-only JSONL audit log: one line per event, never rewritten.

Every LLM call, tool call, finding, policy decision, approval, execution and error goes
here, so any outcome can be traced back to what the model saw and did.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

EVENTS = frozenset(
    {"llm_call", "tool_call", "finding", "policy_decision", "approval", "execution", "error"}
)


class AuditLog:
    def __init__(self, path: Path | str | None, run_id: str, *, model: str | None = None,
                 prompt_version: str | None = None):
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.model = model
        self.prompt_version = prompt_version
        self.events: list[dict] = []  # in-memory copy, handy for tests and the UI
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, *, exception_id: str | None = None, actor: str = "agent",
            payload: dict | None = None, input_tokens: int | None = None,
            output_tokens: int | None = None, cost_usd: float | None = None) -> dict:
        if event not in EVENTS:
            raise ValueError(f"unknown audit event {event!r}")
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "exception_id": exception_id,
            "event": event,
            "actor": actor,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "payload": payload or {},
        }
        self.events.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:  # append only, never "w"
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def read_events(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
