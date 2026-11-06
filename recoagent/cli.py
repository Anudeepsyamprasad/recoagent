"""Command line: python -m recoagent.cli <generate | run | smoke>

  generate   write a synthetic dataset (same as python -m data.generate)
  run        match, then investigate every exception with the agent
  smoke      two tiny API calls that check your key, model and request shape (about $0.01)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from recoagent.agent import PROMPT_VERSION, clean_blocks, run_investigations
from recoagent.audit import AuditLog
from recoagent.ingest import load_dataset
from recoagent.llm import LLMClient, LLMError
from recoagent.matcher import run_matcher
from recoagent.tools import ToolBox


def run_pipeline(data_dir: Path | str, llm: LLMClient, out_root: Path | str = "runs",
                 limit: int | None = None, run_id: str | None = None) -> dict:
    """Load, match, investigate, and write findings.json + audit.jsonl for the run."""
    ds = load_dataset(data_dir)
    result = run_matcher(ds)
    exceptions = result.exceptions[:limit] if limit else result.exceptions

    run_id = run_id or "run-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root) / run_id
    audit = AuditLog(run_dir / "audit.jsonl", run_id, model=llm.model,
                     prompt_version=PROMPT_VERSION)
    findings = run_investigations(exceptions, ToolBox(ds), llm, audit)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "findings.json").write_text(
        json.dumps([f.model_dump(mode="json") for f in findings], indent=2), encoding="utf-8")
    return {"run_id": run_id, "run_dir": str(run_dir), "stats": result.stats,
            "exceptions": exceptions, "findings": findings,
            "cost_usd": llm.total_cost_usd, "live_calls": llm.live_calls}


def _print_run(summary: dict) -> None:
    print(f"Matcher: {summary['stats']}")
    print(f"{'ID':9} {'KEY':14} {'CATEGORY':22} {'CONF':>5}  ACTION")
    for exc, f in zip(summary["exceptions"], summary["findings"], strict=True):
        print(f"{f.exception_id:9} {exc.key:14} {f.category.value:22} "
              f"{f.confidence:5.2f}  {f.proposed_action.type.value}"
              f"{'  [SUSPICIOUS]' if f.suspicious_content else ''}")
    print(f"\nTotal cost: ${summary['cost_usd']:.4f}  (live API calls: {summary['live_calls']})")
    print(f"Wrote {summary['run_dir']}/findings.json and audit.jsonl")


def smoke(model: str | None) -> int:
    llm = LLMClient(mode="live", model=model)
    tools = [{"name": "echo", "description": "Echo text back.",
              "input_schema": {"type": "object", "properties": {"text": {"type": "string"}},
                               "required": ["text"]}}]
    messages: list[dict] = [{"role": "user", "content":
                             "Call the echo tool once with text 'ping', then reply: done."}]
    try:
        first = llm.create(system="You are a connectivity test.", tools=tools,
                           messages=messages, max_tokens=1024)
        calls = [b for b in first.content if b.get("type") == "tool_use"]
        if not calls:
            print("FAIL: the model did not call the tool. Content types:",
                  [b.get("type") for b in first.content])
            return 1
        messages += [
            {"role": "assistant", "content": clean_blocks(first.content)},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0]["id"],
                                          "content": "ping"}]},
        ]
        second = llm.create(system="You are a connectivity test.", tools=tools,
                            messages=messages, max_tokens=1024)
    except LLMError as err:
        print(f"FAIL: {err}")
        return 1
    print(f"OK: model {llm.model}, tool round trip worked "
          f"({first.input_tokens + second.input_tokens} in / "
          f"{first.output_tokens + second.output_tokens} out tokens, ${llm.total_cost_usd:.4f})")
    print("Block types seen:", sorted({b.get('type') for b in first.content + second.content}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recoagent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="write a synthetic dataset")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--n-orders", type=int, default=200)
    gen.add_argument("--out", default="data/sample")

    run = sub.add_parser("run", help="match and investigate")
    run.add_argument("--data", default="data/sample")
    run.add_argument("--mode", choices=["replay", "record", "live"],
                     default=os.getenv("RECO_MODE", "replay"))
    run.add_argument("--limit", type=int, default=None, help="only the first N exceptions")
    run.add_argument("--model", default=None)
    run.add_argument("--cassettes", default="cassettes")
    run.add_argument("--out", default="runs")

    sm = sub.add_parser("smoke", help="check API key, model and request shape")
    sm.add_argument("--model", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "generate":
        from data.generate import generate

        summary = generate(args.seed, args.n_orders, args.out)
        print(f"Wrote {summary['orders']} orders to {summary['out']}")
        return 0
    if args.cmd == "smoke":
        return smoke(args.model)

    llm = LLMClient(mode=args.mode, cassette_dir=args.cassettes, model=args.model)
    try:
        summary = run_pipeline(args.data, llm, args.out, args.limit)
    except LLMError as err:
        print(f"ERROR: {err}")
        return 1
    _print_run(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
