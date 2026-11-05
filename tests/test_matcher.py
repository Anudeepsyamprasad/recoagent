"""Tests for ingest and the deterministic matcher."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from data.generate import SETTLEMENT_DELAY, generate
from recoagent import config
from recoagent.ingest import Dataset, load_dataset
from recoagent.matcher import run_matcher
from recoagent.schemas import (
    Category,
    Order,
    Payment,
    PaymentMethod,
    PaymentStatus,
    Refund,
    RefundStatus,
    Settlement,
    StoreStatus,
)

# Each planted category must produce at least this signal (a FACT, not a category name).
EXPECTED_SIGNAL = {
    "AMOUNT_MISMATCH": "amount_differs",
    "MISSING_PAYMENT": "no_captured_payment",
    "ORPHAN_PAYMENT": "payment_without_order",
    "DUPLICATE_PAYMENT": "multiple_captured_payments",
    "UNRECORDED_REFUND": "refund_mismatch",
    "AUTH_NOT_CAPTURED": "payment_authorized_not_captured",
    "UNIT_ERROR": "order_amount_is_payment_amount_div_100",
    "SETTLEMENT_SHORTFALL": "settlement_differs",
}


@pytest.fixture(scope="module", params=[42, 7])
def run(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"seed{request.param}")
    generate(seed=request.param, n_orders=200, out=out)
    truth = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))["records"]
    ds = load_dataset(out)
    return ds, truth, run_matcher(ds)


def test_matcher_flags_exactly_the_planted_exceptions(run):
    _, truth, result = run
    planted = {r["key"] for r in truth if r["label"] != "MATCHED"}
    flagged = {e.key for e in result.exceptions}
    assert flagged == planted  # no misses, no false alarms on this synthetic data


def test_near_misses_are_auto_matched(run):
    _, truth, result = run
    matched = set(result.matched_order_refs)
    near = [r["key"] for r in truth if r["near_miss"]]
    assert len(near) == 10
    assert set(near) <= matched


def test_each_category_shows_its_expected_signal(run):
    _, truth, result = run
    by_key = {e.key: e for e in result.exceptions}
    for r in truth:
        if r["label"] != "MATCHED":
            signals = " ".join(by_key[r["key"]].signals)
            assert EXPECTED_SIGNAL[r["label"]] in signals, r


def test_signals_never_name_a_category(run):
    _, _, result = run
    names = [c.value.lower() for c in Category if c is not Category.MATCHED]
    for e in result.exceptions:
        for s in e.signals:
            assert not any(n in s.lower() for n in names), s


def test_matched_and_exceptions_partition_the_orders(run):
    ds, _, result = run
    order_exceptions = [e.key for e in result.exceptions if e.kind == "order"]
    assert len(result.matched_order_refs) + len(order_exceptions) == len(ds.orders)
    assert not set(result.matched_order_refs) & set(order_exceptions)
    assert len(result.matched_order_refs) >= 0.85 * len(ds.orders)


def test_exception_ids_are_sequential_and_runs_repeat(run):
    ds, _, result = run
    assert [e.exception_id for e in result.exceptions] == [
        f"EXC-{i:04d}" for i in range(1, len(result.exceptions) + 1)
    ]
    again = run_matcher(ds)
    assert [e.model_dump() for e in again.exceptions] == [e.model_dump() for e in result.exceptions]


def test_injected_order_is_flagged_for_the_agent(run):
    _, truth, result = run
    injected = next(r["key"] for r in truth if r["injection"])
    assert injected in {e.key for e in result.exceptions}


def test_settlement_delay_matches_generator():
    assert timedelta(days=config.SETTLEMENT_DELAY_DAYS) == SETTLEMENT_DELAY


def test_ingest_round_trip_counts(tmp_path):
    summary = generate(seed=42, n_orders=200, out=tmp_path)
    ds = load_dataset(tmp_path)
    assert len(ds.orders) == summary["orders"]
    assert len(ds.payments) == summary["payments"]
    assert len(ds.refunds) == summary["refunds"]
    assert len(ds.settlements) == summary["settlements"]


def test_ingest_reports_bad_rows_with_line_number(tmp_path):
    generate(seed=42, n_orders=200, out=tmp_path)
    text = (tmp_path / "orders.csv").read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    bad = text[1].split(",")
    bad[header.index("amount_paise")] = "not-a-number"
    text[1] = ",".join(bad)
    (tmp_path / "orders.csv").write_text("\n".join(text) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"orders.csv line 2"):
        load_dataset(tmp_path)


def test_ingest_missing_folder_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope")


# ---- hand-built mini cases: tolerance edges and rules the generator never plants ----
T0 = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def _order(amount=50_000, status=StoreStatus.PAID, refunded=0):
    return Order(order_ref="TMC-1", razorpay_order_id="order_A", customer_email="a@example.com",
                 amount_paise=amount, store_status=status, created_at=T0,
                 refunded_amount_paise=refunded)


def _pay(amount=50_000, captured=True, status=PaymentStatus.CAPTURED, pid="pay_A"):
    fee = amount * 2 // 100
    tax = (fee * 18 + 50) // 100
    return Payment(id=pid, order_id="order_A", amount=amount, status=status,
                   method=PaymentMethod.UPI, captured=captured, fee=fee, tax=tax,
                   email="a@example.com", created_at=int(T0.timestamp()) + 60)


def _settle(pay, delta=0):
    return Settlement(settlement_id="setl_A", payment_id=pay.id,
                      amount_settled=pay.amount - pay.fee - pay.tax + delta, fee=pay.fee,
                      tax=pay.tax, utr="UTR1", settled_at=pay.created_at + 2 * 86_400)


def _match(order, pays, settlements=(), refunds=(), as_of=None):
    ds = Dataset([order], list(pays), list(refunds), list(settlements))
    return run_matcher(ds, as_of=as_of or T0 + timedelta(days=30))


def test_one_paisa_difference_is_tolerated_two_is_not():
    pay = _pay()
    assert not _match(_order(), [pay], [_settle(pay, delta=1)]).exceptions
    assert not _match(_order(amount=50_001), [pay], [_settle(pay)]).exceptions
    assert _match(_order(), [pay], [_settle(pay, delta=2)]).exceptions
    assert _match(_order(amount=50_002), [pay], [_settle(pay)]).exceptions


def test_settlement_not_flagged_before_it_is_due_but_flagged_after():
    pay = _pay()
    soon = T0 + timedelta(days=1)
    assert not _match(_order(), [pay], as_of=soon).exceptions
    late = _match(_order(), [pay])
    assert late.exceptions[0].signals == ["settlement_missing_past_due"]


def test_matching_ignores_timestamps_so_timezone_edges_are_safe():
    late_evening = datetime(2026, 10, 10, 23, 30, tzinfo=timezone.utc)  # next day in IST
    order = _order().model_copy(update={"created_at": late_evening})
    pay = _pay().model_copy(update={"created_at": int(late_evening.timestamp()) + 30})
    assert not _match(order, [pay], [_settle(pay)]).exceptions


def test_recorded_partial_refund_is_clean_unrecorded_is_not():
    pay = _pay().model_copy(update={"amount_refunded": 10_000})
    refund = Refund(id="rfnd_A", payment_id="pay_A", amount=10_000,
                    status=RefundStatus.PROCESSED, created_at=pay.created_at + 3_600)
    assert not _match(_order(refunded=10_000), [pay], [_settle(pay)], [refund]).exceptions
    flagged = _match(_order(refunded=0), [pay], [_settle(pay)], [refund])
    assert flagged.exceptions[0].signals == ["refund_mismatch:razorpay=10000,store=0"]


def test_pending_refund_is_not_counted_as_money_returned():
    pay = _pay()
    refund = Refund(id="rfnd_A", payment_id="pay_A", amount=5_000,
                    status=RefundStatus.PENDING, created_at=pay.created_at + 60)
    assert not _match(_order(), [pay], [_settle(pay)], [refund]).exceptions


def test_captured_payment_but_store_not_paid_is_flagged():
    pay = _pay()
    result = _match(_order(status=StoreStatus.PENDING), [pay], [_settle(pay)])
    assert result.exceptions[0].signals == ["captured_payment_but_store_status_is_pending"]
    # A cancelled order whose payment was fully refunded is fine.
    refunded = pay.model_copy(update={"status": PaymentStatus.REFUNDED})
    assert not _match(_order(status=StoreStatus.CANCELLED), [refunded], [_settle(pay)]).exceptions


def test_failed_payment_leaves_paid_order_flagged():
    failed = _pay(captured=False, status=PaymentStatus.FAILED)
    result = _match(_order(), [failed])
    assert result.exceptions[0].signals == ["no_captured_payment"]
