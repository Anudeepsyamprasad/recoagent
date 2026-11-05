"""Load the four CSVs, validate every row with pydantic, and index them for lookups."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from recoagent.schemas import Order, Payment, Refund, Settlement


@dataclass
class Dataset:
    """All records for one reconciliation run, plus indexes for fast lookups."""

    orders: list[Order]
    payments: list[Payment]
    refunds: list[Refund]
    settlements: list[Settlement]
    order_by_ref: dict[str, Order] = field(init=False, repr=False)
    order_by_razorpay_id: dict[str, Order] = field(init=False, repr=False)
    payments_by_order_id: dict[str, list[Payment]] = field(init=False, repr=False)
    refunds_by_payment: dict[str, list[Refund]] = field(init=False, repr=False)
    settlement_by_payment: dict[str, Settlement] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.order_by_ref = {o.order_ref: o for o in self.orders}
        self.order_by_razorpay_id = {o.razorpay_order_id: o for o in self.orders}
        self.payments_by_order_id = {}
        for p in self.payments:
            self.payments_by_order_id.setdefault(p.order_id, []).append(p)
        self.refunds_by_payment = {}
        for r in self.refunds:
            self.refunds_by_payment.setdefault(r.payment_id, []).append(r)
        self.settlement_by_payment = {s.payment_id: s for s in self.settlements}

    @property
    def as_of(self) -> datetime:
        """'Now' for settlement-due checks: the latest payment in the data.

        Using the data's own latest timestamp (not the real clock) keeps runs
        reproducible. Pass an explicit as_of to the matcher to override.
        """
        latest = max((p.created_at for p in self.payments), default=0)
        return datetime.fromtimestamp(latest, timezone.utc)


def _read(path: Path, model):
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for line_no, raw in enumerate(csv.DictReader(fh), start=2):  # line 1 = header
            try:
                rows.append(model(**raw))
            except ValidationError as exc:
                raise ValueError(f"{path.name} line {line_no}: {exc}") from exc
    return rows


def load_dataset(directory: Path | str) -> Dataset:
    """Load orders.csv, payments.csv, refunds.csv and settlements.csv from a folder."""
    d = Path(directory)
    if not (d / "orders.csv").exists():
        raise FileNotFoundError(f"orders.csv not found in {d}")
    return Dataset(
        orders=_read(d / "orders.csv", Order),
        payments=_read(d / "payments.csv", Payment),
        refunds=_read(d / "refunds.csv", Refund),
        settlements=_read(d / "settlements.csv", Settlement),
    )
