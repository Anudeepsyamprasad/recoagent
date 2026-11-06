"""Command line: python -m recoagent.cli <command>

  generate   write a synthetic dataset (same as python -m data.generate)
  run        match, investigate every exception, apply the policy, fill the ledger
  queue      list money actions waiting for a human
  approve    approve (or --reject) a queued action; large amounts need --confirm-amount
  eval       score a replayed run against ground truth and write reports/
  smoke      two tiny API calls that check your key, model and request shape (about $0.01)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from recoagent.agent import PROMPT_VERSION, clean_blocks, run_investigations
from recoagent.audit import AuditLog
from recoagent.eval import evaluate, write_reports
from recoagent.executor import Ledger, LedgerError
from recoagent.ingest import load_dataset
from recoagent.llm import LLMClient, LLMError
from recoagent.matcher import run_matcher
from recoagent.policy import decide_all
from recoagent.tools import ToolBox


def run_pipeline(data_dir: Path | str, llm: LLMClient, out_root: Path | str = "runs",
                 limit: int | None = None, run_id: str | None = None,
                 automation_on: bool | None = None) -> dict:
    """Load, match, investigate, apply the policy, fill the ledger, write the run folder."""
    ds = load_dataset(data_dir)
    result = run_matcher(ds)
    exceptions = result.exceptions[:limit] if limit else result.exceptions

    run_id = run_id or "run-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root) / run_id
    audit = AuditLog(run_dir / "audit.jsonl", run_id, model=llm.model,
                     prompt_version=PROMPT_VERSION)
    findings = run_investigations(exceptions, ToolBox(ds), llm, audit)

    decisions = decide_all(findings, exceptions, ds, audit.events, automation_on)
    ledger = Ledger(run_dir / "ledger.json", audit)
    for finding, decision in zip(findings, decisions, strict=True):
        audit.log("policy_decision", exception_id=finding.exception_id, actor="policy",
                  payload=decision.model_dump(mode="json"))
        ledger.submit(finding, decision)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "findings.json").write_text(
        json.dumps([f.model_dump(mode="json") for f in findings], indent=2), encoding="utf-8")
    (run_dir / "decisions.json").write_text(
        json.dumps([d.model_dump(mode="json") for d in decisions], indent=2), encoding="utf-8")
    return {"run_id": run_id, "run_dir": str(run_dir), "stats": result.stats, "ds": ds,
            "exceptions": exceptions, "findings": findings, "decisions": decisions,
            "ledger": ledger, "events": audit.events, "cost_usd": llm.total_cost_usd,
            "live_calls": llm.live_calls, "model": llm.model}


def evaluate_dataset(data_dir: Path | str, llm: LLMClient, label: str) -> dict:
    """Run the full pipeline in a throwaway folder and score it against ground truth."""
    truth_path = Path(data_dir) / "ground_truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"{truth_path} not found (generate the data first)")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))["records"]
    with tempfile.TemporaryDirectory() as tmp:
        summary = run_pipeline(data_dir, llm, tmp, run_id="eval")
    return evaluate(
        truth, summary["exceptions"], summary["findings"], summary["decisions"],
        list(summary["ledger"].entries.values()), cost_usd=summary["cost_usd"], label=label,
        model=summary["model"], prompt_version=PROMPT_VERSION, n_orders=len(summary["ds"].orders),
    )


def _print_run(summary: dict) -> None:
    print(f"Matcher: {summary['stats']}")
    print(f"{'ID':9} {'KEY':14} {'CATEGORY':22} {'CONF':>5}  {'ACTION':24} ROUTE")
    rows = zip(summary["exceptions"], summary["findings"], summary["decisions"], strict=True)
    for exc, f, d in rows:
        flag = "  [SUSPICIOUS]" if f.suspicious_content else ""
        print(f"{f.exception_id:9} {exc.key[:14]:14} {f.category.value:22} "
              f"{f.confidence:5.2f}  {f.proposed_action.type.value:24} {d.route.value}{flag}")
    print(f"\nTotal cost: ${summary['cost_usd']:.4f}  (live API calls: {summary['live_calls']})")
    queued = len(summary["ledger"].queued)
    print(f"Wrote {summary['run_dir']}/ (findings, decisions, ledger, audit). "
          f"{queued} action(s) waiting for approval.")


def _latest_run(out_root: str) -> Path:
    runs = sorted(p for p in Path(out_root).glob("run-*") if (p / "ledger.json").exists())
    if not runs:
        raise FileNotFoundError(f"no runs with a ledger found in {out_root}/")
    return runs[-1]


def _open_ledger(run: str | None, out_root: str) -> tuple[Ledger, Path]:
    run_dir = Path(run) if run else _latest_run(out_root)
    audit = AuditLog(run_dir / "audit.jsonl", run_dir.name)
    return Ledger(run_dir / "ledger.json", audit), run_dir


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

    def add_llm_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", default="data/sample")
        p.add_argument("--mode", choices=["replay", "record", "live"],
                       default=os.getenv("RECO_MODE", "replay"))
        p.add_argument("--model", default=None)
        p.add_argument("--cassettes", default="cassettes")

    run = sub.add_parser("run", help="match, investigate, apply policy")
    add_llm_args(run)
    run.add_argument("--limit", type=int, default=None, help="only the first N exceptions")
    run.add_argument("--out", default="runs")

    ev = sub.add_parser("eval", help="score a run against ground truth, write reports/")
    add_llm_args(ev)
    ev.add_argument("--label", default=None, help='e.g. "Seed 42 (dev)"')
    ev.add_argument("--reports", default="reports")

    q = sub.add_parser("queue", help="list actions waiting for approval")
    q.add_argument("--run", default=None)
    q.add_argument("--out", default="runs")

    ap = sub.add_parser("approve", help="approve or reject a queued action")
    ap.add_argument("--exception", required=True, help="for example EXC-0004")
    ap.add_argument("--by", required=True, help="your name (recorded in the audit log)")
    ap.add_argument("--confirm-amount", type=int, default=None,
                    help="re-type the amount in paise (required above Rs 10,000)")
    ap.add_argument("--reject", action="store_true")
    ap.add_argument("--reason", default="")
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default="runs")

    sm = sub.add_parser("smoke", help="check API key, model and request shape")
    sm.add_argument("--model", default=None)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "generate":
            from data.generate import generate

            summary = generate(args.seed, args.n_orders, args.out)
            print(f"Wrote {summary['orders']} orders to {summary['out']}")
            return 0
        if args.cmd == "smoke":
            return smoke(args.model)
        if args.cmd == "queue":
            ledger, run_dir = _open_ledger(args.run, args.out)
            print(f"Run {run_dir.name}: {len(ledger.queued)} waiting")
            for e in ledger.queued:
                big = "  [DOUBLE CONFIRM: use --confirm-amount]" if e.double_confirm else ""
                print(f"  {e.exception_id}  {e.action_type.value:24} "
                      f"payment={e.payment_id}  amount={e.amount_paise} paise{big}")
            return 0
        if args.cmd == "approve":
            ledger, run_dir = _open_ledger(args.run, args.out)
            if args.reject:
                entry = ledger.reject(args.exception, args.by, args.reason)
            else:
                entry = ledger.approve(args.exception, args.by, args.confirm_amount)
            print(f"{entry.exception_id}: {entry.status} by {entry.approved_by} "
                  f"({entry.action_type.value}, {entry.amount_paise} paise). "
                  "Recorded in the local ledger; no money was moved.")
            return 0

        llm = LLMClient(mode=args.mode, cassette_dir=args.cassettes, model=args.model)
        if args.cmd == "eval":
            label = args.label or Path(args.data).name
            report = evaluate_dataset(args.data, llm, label)
            path = write_reports(report, args.reports)
            for key, value in report["metrics"].items():
                print(f"  {key:26} {value}")
            print(f"Wrote {path} and {Path(args.reports) / 'eval.md'}")
            return 0
        summary = run_pipeline(args.data, llm, args.out, args.limit)
        _print_run(summary)
        return 0
    except (LLMError, LedgerError, FileNotFoundError) as err:
        print(f"ERROR: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
