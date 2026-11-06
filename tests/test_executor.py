"""Tests for the idempotent local ledger and human approvals."""

import pytest

from recoagent.audit import AuditLog
from recoagent.cli import main
from recoagent.executor import Ledger, LedgerError, action_key
from recoagent.policy import Decision, Route
from recoagent.schemas import Finding


def finding(exception_id="EXC-0001", category="DUPLICATE_PAYMENT", action="REFUND_DUPLICATE",
            payment_id="pay_A", amount=50_000):
    return Finding.model_validate({
        "exception_id": exception_id, "category": category, "confidence": 0.95,
        "evidence": ["e"], "root_cause": "r", "risk_level": "medium",
        "proposed_action": {"type": action, "payment_id": payment_id, "amount_paise": amount},
    })


def decision(route=Route.APPROVAL_QUEUE, amount=50_000, double=False, exception_id="EXC-0001"):
    return Decision(exception_id=exception_id, route=route, reasons=["r"], double_confirm=double,
                    amount_paise=amount)


@pytest.fixture
def ledger(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl", "run-t")
    return Ledger(tmp_path / "ledger.json", audit)


def events(ledger, kind):
    return [e for e in ledger.audit.events if e["event"] == kind]


def test_money_action_is_queued_not_executed(ledger):
    entry = ledger.submit(finding(), decision())
    assert entry.status == "QUEUED" and entry.approved_by is None
    assert events(ledger, "execution") == []


def test_submit_is_idempotent(ledger):
    first = ledger.submit(finding(), decision())
    second = ledger.submit(finding(), decision())
    assert first.action_key == second.action_key and len(ledger.entries) == 1


def test_action_key_is_deterministic_and_sensitive():
    base = action_key("EXC-1", "REFUND_DUPLICATE", "pay_A", 100)
    assert base == action_key("EXC-1", "REFUND_DUPLICATE", "pay_A", 100)
    assert base != action_key("EXC-1", "REFUND_DUPLICATE", "pay_A", 101)
    assert base != action_key("EXC-2", "REFUND_DUPLICATE", "pay_A", 100)


def test_needs_human_and_no_action_never_reach_the_ledger(ledger):
    assert ledger.submit(finding(), decision(Route.NEEDS_HUMAN)) is None
    assert ledger.submit(finding(), decision(Route.NO_ACTION)) is None
    assert ledger.entries == {}


def test_safe_action_auto_applies_and_is_audited(ledger):
    f = finding(category="UNIT_ERROR", action="FIX_STORE_AMOUNT", payment_id=None)
    entry = ledger.submit(f, decision(Route.AUTO_APPLY))
    assert entry.status == "AUTO_APPLIED"
    (ex,) = events(ledger, "execution")
    assert ex["actor"] == "policy" and "no external system" in ex["payload"]["result"]


def test_approval_records_the_human_and_executes_once(ledger):
    ledger.submit(finding(), decision())
    entry = ledger.approve("EXC-0001", "anudeep")
    assert entry.status == "APPROVED" and entry.approved_by == "anudeep"
    (approval,) = events(ledger, "approval")
    assert approval["actor"] == "human:anudeep"
    again = ledger.approve("EXC-0001", "anudeep")  # idempotent
    assert again.status == "APPROVED"
    assert len(events(ledger, "approval")) == 1 and len(events(ledger, "execution")) == 1


def test_large_amounts_need_the_amount_typed_back(ledger):
    ledger.submit(finding(amount=1_500_000), decision(amount=1_500_000, double=True))
    with pytest.raises(LedgerError, match="re-type"):
        ledger.approve("EXC-0001", "anudeep")
    with pytest.raises(LedgerError, match="re-type"):
        ledger.approve("EXC-0001", "anudeep", confirm_amount=1_500_00)
    assert ledger.get("EXC-0001").status == "QUEUED"
    assert ledger.approve("EXC-0001", "anudeep", confirm_amount=1_500_000).status == "APPROVED"


def test_approval_needs_a_named_approver_and_a_real_entry(ledger):
    ledger.submit(finding(), decision())
    with pytest.raises(LedgerError, match="approver name"):
        ledger.approve("EXC-0001", "  ")
    with pytest.raises(LedgerError, match="no ledger entry"):
        ledger.approve("EXC-9999", "anudeep")
    assert ledger.get("EXC-0001").status == "QUEUED"


def test_cannot_approve_something_that_was_not_queued(ledger):
    safe = finding(category="UNIT_ERROR", action="FIX_STORE_AMOUNT", payment_id=None)
    ledger.submit(safe, decision(Route.AUTO_APPLY))
    with pytest.raises(LedgerError, match="not awaiting approval"):
        ledger.approve("EXC-0001", "anudeep")


def test_reject_flow_and_no_later_approval(ledger):
    ledger.submit(finding(), decision())
    entry = ledger.reject("EXC-0001", "anudeep", "customer already refunded")
    assert entry.status == "REJECTED" and "already refunded" in entry.note
    with pytest.raises(LedgerError):
        ledger.approve("EXC-0001", "anudeep")
    assert events(ledger, "execution") == []


def test_ledger_persists_to_disk_and_reloads(tmp_path):
    path = tmp_path / "ledger.json"
    first = Ledger(path)
    first.submit(finding(), decision())
    first.approve("EXC-0001", "anudeep")
    reloaded = Ledger(path)
    assert reloaded.get("EXC-0001").status == "APPROVED"
    assert reloaded.queued == []


# ---- CLI: queue and approve -------------------------------------------------------
def _seed_run(tmp_path, amount=50_000, double=False):
    run_dir = tmp_path / "run-20261106-000000"
    audit = AuditLog(run_dir / "audit.jsonl", run_dir.name)
    Ledger(run_dir / "ledger.json", audit).submit(
        finding(amount=amount), decision(amount=amount, double=double))
    return run_dir


def test_cli_queue_and_approve(tmp_path, capsys):
    _seed_run(tmp_path)
    assert main(["queue", "--out", str(tmp_path)]) == 0
    assert "EXC-0001" in capsys.readouterr().out
    assert main(["approve", "--exception", "EXC-0001", "--by", "anudeep",
                 "--out", str(tmp_path)]) == 0
    assert "no money was moved" in capsys.readouterr().out
    assert Ledger(tmp_path / "run-20261106-000000" / "ledger.json").get("EXC-0001").status == "APPROVED"


def test_cli_large_amount_requires_confirm_amount(tmp_path, capsys):
    _seed_run(tmp_path, amount=1_500_000, double=True)
    args = ["approve", "--exception", "EXC-0001", "--by", "anudeep", "--out", str(tmp_path)]
    assert main(args) == 1
    assert "re-type" in capsys.readouterr().out
    assert main([*args, "--confirm-amount", "1500000"]) == 0


def test_cli_reject_and_missing_runs(tmp_path, capsys):
    _seed_run(tmp_path)
    assert main(["approve", "--exception", "EXC-0001", "--by", "anudeep", "--reject",
                 "--reason", "dup", "--out", str(tmp_path)]) == 0
    assert "REJECTED" in capsys.readouterr().out
    assert main(["queue", "--out", str(tmp_path / "nowhere")]) == 1
