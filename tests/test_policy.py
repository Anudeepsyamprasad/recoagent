"""Tests for the policy guardrails."""

import pytest

from recoagent import config
from recoagent.policy import Route, decide, decide_all, seen_amounts
from recoagent.schemas import Finding, ReconException

KNOWN = {"pay_A", "pay_B"}


def exc(amount=50_000, signals=None, exception_id="EXC-0001"):
    return ReconException(exception_id=exception_id, kind="order", key="TMC-1",
                          signals=signals or ["amount_differs:order=50000,paid=40000"],
                          payment_ids=["pay_A"], amount_paise=amount)


def finding(category="DUPLICATE_PAYMENT", action="REFUND_DUPLICATE", payment_id="pay_A",
            amount=50_000, confidence=0.95, suspicious=False):
    return Finding.model_validate({
        "exception_id": "EXC-0001", "category": category, "confidence": confidence,
        "evidence": ["e"], "root_cause": "r", "risk_level": "medium",
        "suspicious_content": suspicious,
        "proposed_action": {"type": action, "payment_id": payment_id, "amount_paise": amount},
    })


def run(f, e=None, seen=None, known=KNOWN, on=True):
    e = e or exc()
    return decide(f, e, seen=seen if seen is not None else {50_000, 10_000, 40_000},
                  known_payment_ids=known, automation_on=on)


def test_money_action_always_goes_to_the_approval_queue():
    d = run(finding())
    assert d.route is Route.APPROVAL_QUEUE and not d.double_confirm
    assert "always needs human approval" in d.reasons[0]


def test_amount_over_ten_thousand_rupees_needs_double_confirmation():
    big = 1_000_001
    d = run(finding(amount=big), seen={big})
    assert d.route is Route.APPROVAL_QUEUE and d.double_confirm and d.amount_paise == big
    exactly = config.LARGE_AMOUNT_PAISE
    assert not run(finding(amount=exactly), seen={exactly}).double_confirm  # strictly above


def test_even_maximum_confidence_cannot_auto_apply_a_money_action():
    assert run(finding(confidence=1.0)).route is Route.APPROVAL_QUEUE


def test_safe_action_auto_applies_only_at_high_confidence():
    safe = dict(category="UNIT_ERROR", action="FIX_STORE_AMOUNT", payment_id=None)
    assert run(finding(**safe, confidence=0.90)).route is Route.AUTO_APPLY
    assert run(finding(**safe, confidence=0.85)).route is Route.AUTO_APPLY
    low = run(finding(**safe, confidence=0.80))
    assert low.route is Route.APPROVAL_QUEUE and "below auto-apply" in low.reasons[0]


def test_kill_switch_disables_all_auto_apply():
    safe = dict(category="UNIT_ERROR", action="FIX_STORE_AMOUNT", payment_id=None)
    d = run(finding(**safe), on=False)
    assert d.route is Route.APPROVAL_QUEUE and "kill switch" in d.reasons[0]


def test_low_confidence_goes_to_a_human():
    d = run(finding(confidence=0.69))
    assert d.route is Route.NEEDS_HUMAN and "below floor" in d.reasons[0]
    assert run(finding(confidence=0.70)).route is Route.APPROVAL_QUEUE


def test_suspicious_content_blocks_everything_even_with_perfect_confidence():
    for kwargs in (dict(), dict(category="UNIT_ERROR", action="FIX_STORE_AMOUNT",
                                payment_id=None)):
        d = run(finding(**kwargs, confidence=1.0, suspicious=True))
        assert d.route is Route.NEEDS_HUMAN and "suspicious" in d.reasons[0]


def test_agent_error_and_needs_human_categories_are_not_actionable():
    assert run(Finding.agent_error("EXC-0001", "boom")).route is Route.NEEDS_HUMAN
    assert run(Finding.needs_human("EXC-0001", "unclear")).route is Route.NEEDS_HUMAN


def test_no_action_proposed():
    d = run(finding(action="NO_ACTION", payment_id=None, amount=None))
    assert d.route is Route.NO_ACTION


def test_action_must_match_the_playbook_for_its_category():
    d = run(finding(category="DUPLICATE_PAYMENT", action="CAPTURE_OR_VOID"))
    assert d.route is Route.NEEDS_HUMAN and "does not match playbook" in d.reasons[0]


def test_invented_amounts_are_rejected():
    d = run(finding(amount=123_457))
    assert d.route is Route.NEEDS_HUMAN and "no invented numbers" in d.reasons[0]


def test_hallucinated_or_missing_payment_ids_are_rejected():
    ghost = run(finding(payment_id="pay_ghost"))
    assert ghost.route is Route.NEEDS_HUMAN and "does not exist" in ghost.reasons[0]
    missing = run(finding(payment_id=None))
    assert missing.route is Route.NEEDS_HUMAN and "requires a payment_id" in missing.reasons[0]


def test_actions_that_need_no_payment_id_are_fine_without_one():
    d = run(finding(category="MISSING_PAYMENT", action="MARK_ORDER_UNPAID", payment_id=None))
    assert d.route is Route.APPROVAL_QUEUE


def test_seen_amounts_come_from_signals_reference_and_differences():
    e = exc(amount=40_000, signals=["amount_differs:order=50000,paid=40000", "multiple:2"])
    seen = seen_amounts(e, [])
    assert {40_000, 50_000, 10_000} <= seen  # 10_000 is the difference
    assert 7 not in seen


def test_seen_amounts_use_this_exceptions_tool_amounts_but_not_timestamps_or_counts():
    e = exc()
    events = [
        {"exception_id": "EXC-0001", "event": "tool_call", "payload": {
            "output": {"payments": [{"amount": 123_400, "created_at": 1_790_000_000,
                                     "fee": 2_468}], "count": 3}}},
        {"exception_id": "EXC-0002", "event": "tool_call", "payload": {
            "output": {"payments": [{"amount": 999_900}]}}},
        {"exception_id": "EXC-0001", "event": "llm_call", "payload": {"amount": 777_700}},
    ]
    seen = seen_amounts(e, events)
    assert 123_400 in seen and 2_468 in seen
    assert 1_790_000_000 not in seen and 3 not in seen
    assert 999_900 not in seen and 777_700 not in seen


def test_decide_all_pairs_findings_with_exceptions(tmp_path):
    from data.generate import generate
    from recoagent.ingest import load_dataset
    from recoagent.matcher import run_matcher

    generate(seed=42, n_orders=200, out=tmp_path)
    ds = load_dataset(tmp_path)
    excs = run_matcher(ds).exceptions[:2]
    findings = [Finding.needs_human(e.exception_id, "x") for e in excs]
    decisions = decide_all(findings, excs, ds, [], automation_on=True)
    assert [d.exception_id for d in decisions] == [e.exception_id for e in excs]
    with pytest.raises(ValueError):
        decide_all(findings[:1], excs, ds, [])  # length mismatch must not pass silently
