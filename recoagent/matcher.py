"""Deterministic matcher: plain rules that close the clean records.

Everything the rules cannot explain becomes a ReconException with factual `signals`.
The matcher never names a root cause. It joins on Razorpay order id (never on dates),
which is why timezone edge cases do not trip it.

Rules, in order, for each store order:
  1. Look at all payments linked to the order's Razorpay order id.
  2. Store says paid but no captured payment          -> signal
  3. More than one captured payment                   -> signal
  4. One captured payment: compare amount, refunds and settlement.
Then: every captured payment whose order is not in the store                -> signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from recoagent import config
from recoagent.ingest import Dataset
from recoagent.schemas import (
    Order,
    Payment,
    PaymentStatus,
    ReconException,
    RefundStatus,
    StoreStatus,
)


@dataclass
class MatchResult:
    matched_order_refs: list[str]
    exceptions: list[ReconException]

    @property
    def stats(self) -> dict[str, int]:
        return {
            "orders_matched": len(self.matched_order_refs),
            "exceptions": len(self.exceptions),
        }


def _order_signals(ds: Dataset, order: Order, as_of_ts: int) -> tuple[list[str], list[Payment]]:
    """Return (signals, payments involved) for one order. No signals means matched."""
    payments = ds.payments_by_order_id.get(order.razorpay_order_id, [])
    captured = [p for p in payments if p.captured]
    signals: list[str] = []

    if order.store_status is not StoreStatus.PAID:
        # Store does not say paid, but money was captured (and not fully refunded).
        live = [p for p in captured if p.status is not PaymentStatus.REFUNDED]
        if live:
            signals.append(f"captured_payment_but_store_status_is_{order.store_status.value}")
        return signals, payments

    if not captured:
        signals.append("no_captured_payment")
        if any(p.status is PaymentStatus.AUTHORIZED for p in payments):
            signals.append("payment_authorized_not_captured")
        return signals, payments

    if len(captured) > 1:
        signals.append(f"multiple_captured_payments:{len(captured)}")
        return signals, payments

    pay = captured[0]

    # Amount check (with rounding tolerance), plus a hint for 100x unit slips.
    diff = pay.amount - order.amount_paise
    if abs(diff) > config.AMOUNT_TOLERANCE_PAISE:
        if order.amount_paise * 100 == pay.amount:
            signals.append(
                f"order_amount_is_payment_amount_div_100:order={order.amount_paise},paid={pay.amount}"
            )
        else:
            signals.append(f"amount_differs:order={order.amount_paise},paid={pay.amount}")

    # Refund check: what Razorpay processed versus what the store recorded.
    refunded = sum(
        r.amount
        for r in ds.refunds_by_payment.get(pay.id, [])
        if r.status is RefundStatus.PROCESSED
    )
    if refunded != order.refunded_amount_paise:
        signals.append(f"refund_mismatch:razorpay={refunded},store={order.refunded_amount_paise}")

    # Settlement check, only once the payout is due.
    due_ts = pay.created_at + int(timedelta(days=config.SETTLEMENT_DELAY_DAYS).total_seconds())
    if due_ts <= as_of_ts:
        row = ds.settlement_by_payment.get(pay.id)
        expected = pay.amount - pay.fee - pay.tax
        if row is None:
            signals.append("settlement_missing_past_due")
        elif abs(row.amount_settled - expected) > config.SETTLEMENT_TOLERANCE_PAISE:
            signals.append(f"settlement_differs:expected={expected},settled={row.amount_settled}")

    return signals, payments


def run_matcher(ds: Dataset, as_of: datetime | None = None) -> MatchResult:
    """Split a dataset into auto-matched orders and exceptions (deterministic, stable order)."""
    as_of_ts = int((as_of or ds.as_of).timestamp())
    matched: list[str] = []
    found: list[tuple[str, str, list[str], list[str], int]] = []  # kind, key, signals, pay ids, amt

    for order in sorted(ds.orders, key=lambda o: o.order_ref):
        signals, payments = _order_signals(ds, order, as_of_ts)
        if not signals:
            matched.append(order.order_ref)
            continue
        captured = [p for p in payments if p.captured]
        # Reference amount: the money actually captured if any, else what the store expects.
        amount = captured[0].amount if captured else order.amount_paise
        found.append(("order", order.order_ref, signals, [p.id for p in payments], amount))

    for pay in sorted(ds.payments, key=lambda p: p.id):
        if pay.captured and pay.order_id not in ds.order_by_razorpay_id:
            found.append(("payment", pay.id, ["payment_without_order"], [pay.id], pay.amount))

    exceptions = [
        ReconException(
            exception_id=f"EXC-{i:04d}", kind=kind, key=key, signals=signals,
            payment_ids=pay_ids, amount_paise=amount,
        )
        for i, (kind, key, signals, pay_ids, amount) in enumerate(found, start=1)
    ]
    return MatchResult(matched_order_refs=matched, exceptions=exceptions)
