"""RecoAgent demo and approval UI.

Runs in REPLAY mode: the agent's responses come from recorded cassettes, so this app needs
no API key, costs nothing and gives the same answers every time. Approvals you click here
write to a throwaway local ledger; no external system is called and no money moves.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Streamlit puts app/ on the path, not the repo root
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from recoagent import config  # noqa: E402
from recoagent.agent import PROMPT_VERSION  # noqa: E402
from recoagent.cli import run_pipeline  # noqa: E402
from recoagent.executor import LedgerError  # noqa: E402
from recoagent.llm import LLMClient, LLMConfigError  # noqa: E402
from recoagent.policy import Route  # noqa: E402

DATA_ROOT = Path(os.getenv("RECO_DATA_ROOT", ROOT / "data"))
CASSETTES = Path(os.getenv("RECO_CASSETTE_DIR", ROOT / "cassettes"))
REPORTS = Path(os.getenv("RECO_REPORTS_DIR", ROOT / "reports"))

ROUTE_LABEL = {
    Route.AUTO_APPLY: "Auto-applied (safe)",
    Route.APPROVAL_QUEUE: "Needs approval",
    Route.NEEDS_HUMAN: "Blocked: needs human",
    Route.NO_ACTION: "No action",
}


def rupees(paise: int | None) -> str:
    return "n/a" if paise is None else f"₹{paise / 100:,.2f}"


def datasets() -> list[str]:
    return sorted(p.parent.name for p in DATA_ROOT.glob("*/orders.csv")
                  if (p.parent / "ground_truth.json").exists())


st.set_page_config(page_title="RecoAgent", page_icon="🧾", layout="wide")
st.title("RecoAgent")
st.caption("AI reconciliation agent for Razorpay merchants. Plain code clears the clean "
           "majority; a Claude agent investigates exceptions; nothing that touches money "
           "happens without a human.")

# ---- sidebar ------------------------------------------------------------------------
available = datasets()
if not available:
    st.error(f"No datasets found in {DATA_ROOT}. Run `python -m data.generate` first.")
    st.stop()
with st.sidebar:
    st.header("Run")
    dataset = st.selectbox("Dataset", available, key="dataset",
                           help="sample = seed 42 (development). heldout = seed 7 (never tuned on).")
    model = st.selectbox("Recorded model", list(config.PRICES_PER_MTOK), key="model")
    st.info("Replay mode: responses come from recorded cassettes. No API key, no cost, "
            "same answers every time.")
    st.caption(f"Prompt version {PROMPT_VERSION}. Synthetic data only. Test mode only.")

run_key = f"run::{dataset}::{model}"
if run_key not in st.session_state:
    try:
        with st.spinner("Matching and replaying investigations..."):
            workdir = tempfile.mkdtemp(prefix="recoagent-")
            llm = LLMClient(mode="replay", cassette_dir=CASSETTES, model=model)
            st.session_state[run_key] = run_pipeline(DATA_ROOT / dataset, llm, workdir,
                                                     run_id="demo")
    except LLMConfigError as err:
        st.error(f"No recording for {dataset} with {model}: {err}")
        st.stop()
run = st.session_state[run_key]
ledger, events = run["ledger"], run["events"]
rows = list(zip(run["exceptions"], run["findings"], run["decisions"], strict=True))

tab_over, tab_inv, tab_queue, tab_audit, tab_eval = st.tabs(
    ["Overview", "Investigations", "Approval queue", "Audit log", "Evaluation"])

# ---- overview -----------------------------------------------------------------------
with tab_over:
    stats = run["stats"]
    counts = {r: sum(1 for *_, d in rows if d.route is r) for r in Route}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Orders matched by rules", stats["orders_matched"])
    c2.metric("Exceptions", stats["exceptions"])
    c3.metric("Waiting for approval", len(ledger.queued))
    c4.metric("Blocked for a human", counts[Route.NEEDS_HUMAN])
    c5.metric("Recorded cost of this run", f"${run['cost_usd']:.2f}")
    st.markdown(
        "**How it works**\n\n"
        "1. **Deterministic matcher** closes orders whose payment, refund and settlement agree.\n"
        "2. **Claude agent** investigates only the exceptions, with read-only tools, and returns "
        "a validated finding (category, evidence, proposed action).\n"
        "3. **Policy** (plain code) decides: only safe data fixes auto-apply; every money action "
        "waits for a human; amounts over ₹10,000 need the amount typed back; suspicious or "
        "low-confidence findings are blocked.\n"
        "4. **Ledger and audit log** record everything. No refund or payout API is ever called.")
    st.dataframe(pd.DataFrame(
        [{"Route": ROUTE_LABEL[r], "Findings": n} for r, n in counts.items()]),
        hide_index=True, use_container_width=True)

# ---- investigations -----------------------------------------------------------------
with tab_inv:
    table = pd.DataFrame([{
        "ID": f.exception_id, "Key": e.key, "Category": f.category.value,
        "Confidence": f.confidence, "Proposed action": f.proposed_action.type.value,
        "Amount": rupees(f.proposed_action.amount_paise), "Route": ROUTE_LABEL[d.route],
        "Suspicious": "YES" if f.suspicious_content else "",
    } for e, f, d in rows])
    st.dataframe(table, hide_index=True, use_container_width=True)

    ids = [e.exception_id for e, *_ in rows]
    chosen = st.selectbox("Inspect an exception", ids, key="inspect")
    exc, finding, decision = rows[ids.index(chosen)]
    if finding.suspicious_content:
        st.warning("Instruction-like text was found in the data for this record. The agent "
                   "treated it as data, and the policy blocked every action on it.")
    left, right = st.columns(2)
    with left:
        st.subheader("What the rules saw")
        st.write(f"`{exc.kind}` **{exc.key}**, reference amount {rupees(exc.amount_paise)}")
        for signal in exc.signals:
            st.code(signal, language=None)
        st.subheader("Agent finding")
        st.write(f"**{finding.category.value}**, confidence {finding.confidence:.2f}, "
                 f"risk {finding.risk_level.value}")
        st.write(finding.root_cause)
        for line in finding.evidence:
            st.markdown(f"- {line}")
    with right:
        st.subheader("Policy decision")
        st.write(f"**{ROUTE_LABEL[decision.route]}**")
        for reason in decision.reasons:
            st.markdown(f"- {reason}")
        st.subheader("Tool calls the agent made")
        calls = [ev for ev in events if ev["exception_id"] == chosen and ev["event"] == "tool_call"]
        if not calls:
            st.caption("No lookups: the agent answered from the case and signals.")
        for i, ev in enumerate(calls, start=1):
            with st.expander(f"{i}. {ev['payload']['tool']}"):
                st.json({"input": ev["payload"]["input"], "output": ev["payload"]["output"]})

# ---- approval queue -----------------------------------------------------------------
with tab_queue:
    st.write("Money actions never run on their own. Approving here only records the "
             "decision in a local ledger; no refund or payout is sent anywhere.")
    approver = st.text_input("Your name (recorded in the audit log)", key="approver")
    queued = ledger.queued
    if not queued:
        st.success("Nothing is waiting for approval.")
    for entry in queued:
        with st.container(border=True):
            st.markdown(f"**{entry.exception_id}**: `{entry.action_type.value}`, payment "
                        f"`{entry.payment_id}`, amount **{rupees(entry.amount_paise)}**")
            confirm = None
            if entry.double_confirm:
                st.warning("Above ₹10,000: type the exact amount in paise to confirm "
                           f"({entry.amount_paise}).")
                typed = st.text_input("Confirm amount (paise)", key=f"confirm_{entry.exception_id}")
                confirm = int(typed) if typed.strip().isdigit() else None
            a, b, _ = st.columns([1, 1, 4])
            approve = a.button("Approve", key=f"approve_{entry.exception_id}", type="primary")
            reject = b.button("Reject", key=f"reject_{entry.exception_id}")
            try:
                if approve:
                    ledger.approve(entry.exception_id, approver, confirm)
                    st.rerun()
                if reject:
                    ledger.reject(entry.exception_id, approver, "rejected in demo UI")
                    st.rerun()
            except LedgerError as err:
                st.error(str(err))
    st.subheader("Ledger")
    st.dataframe(pd.DataFrame([{
        "Exception": e.exception_id, "Action": e.action_type.value,
        "Amount": rupees(e.amount_paise), "Status": e.status,
        "By": e.approved_by or "", "Large": "double confirm" if e.double_confirm else "",
    } for e in ledger.entries.values()]), hide_index=True, use_container_width=True)

# ---- audit log ----------------------------------------------------------------------
with tab_audit:
    kinds = sorted({ev["event"] for ev in events + ledger.audit.events})
    picked = st.multiselect("Event types", kinds, default=kinds, key="audit_filter")
    all_events = sorted({json.dumps(ev, sort_keys=True): ev for ev in
                         events + ledger.audit.events}.values(), key=lambda ev: ev["ts"])
    st.dataframe(pd.DataFrame([{
        "Time": ev["ts"], "Event": ev["event"], "Actor": ev["actor"],
        "Exception": ev["exception_id"] or "", "Tokens in/out":
            f"{ev['input_tokens']}/{ev['output_tokens']}" if ev["input_tokens"] else "",
        "Cost": f"${ev['cost_usd']:.4f}" if ev["cost_usd"] else "",
        "Detail": json.dumps(ev["payload"], ensure_ascii=False)[:300],
    } for ev in all_events if ev["event"] in picked]), hide_index=True, use_container_width=True)

# ---- evaluation ---------------------------------------------------------------------
with tab_eval:
    report = REPORTS / "eval.md"
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("No evaluation report yet. Run `python -m recoagent.cli eval`.")
