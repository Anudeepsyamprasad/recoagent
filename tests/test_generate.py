"""Tests for the synthetic data generator and schemas."""

import csv
import json

import pytest

from data.generate import INJECTION_NOTE, generate
from recoagent.schemas import (
    ActionType,
    Category,
    Order,
    Payment,
    Refund,
    Settlement,
    is_money_action,
)


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("seed42")
    summary = generate(seed=42, n_orders=200, out=out)
    truth = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))
    return {
        "out": out,
        "summary": summary,
        "truth": truth["records"],
        "orders": _rows(out / "orders.csv"),
        "payments": _rows(out / "payments.csv"),
        "refunds": _rows(out / "refunds.csv"),
        "settlements": _rows(out / "settlements.csv"),
    }


def test_same_seed_gives_identical_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(seed=42, n_orders=200, out=a)
    generate(seed=42, n_orders=200, out=b)
    for name in ["orders.csv", "payments.csv", "refunds.csv", "settlements.csv", "ground_truth.json"]:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_different_seeds_differ(tmp_path):
    generate(seed=42, n_orders=200, out=tmp_path / "a")
    generate(seed=7, n_orders=200, out=tmp_path / "b")
    assert (tmp_path / "a" / "orders.csv").read_bytes() != (tmp_path / "b" / "orders.csv").read_bytes()


def test_row_counts_and_exception_share(data):
    assert len(data["orders"]) == 200
    exceptions = [r for r in data["truth"] if r["label"] != "MATCHED"]
    assert len(exceptions) == 24  # 8 categories x 3
    assert 0.10 <= len(exceptions) / 200 <= 0.14


def test_every_exception_category_planted_equally(data):
    counts = {}
    for r in data["truth"]:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    for cat in Category:
        if cat is not Category.MATCHED:
            assert counts[cat.value] == 3, cat


def test_ground_truth_covers_every_order_and_orphan(data):
    keys = {r["key"] for r in data["truth"]}
    assert {o["order_ref"] for o in data["orders"]} <= keys
    payment_ids = {p["id"] for p in data["payments"]}
    for r in data["truth"]:
        if r["kind"] == "payment":
            assert r["key"] in payment_ids
            assert r["label"] == "ORPHAN_PAYMENT"


def test_all_rows_validate_against_schemas(data):
    for row in data["orders"]:
        Order(**{**row, "amount_paise": int(row["amount_paise"]),
                 "refunded_amount_paise": int(row["refunded_amount_paise"])})
    for row in data["payments"]:
        Payment(**{**row, "amount": int(row["amount"]), "captured": row["captured"] == "True",
                   "amount_refunded": int(row["amount_refunded"]), "fee": int(row["fee"]),
                   "tax": int(row["tax"]), "created_at": int(row["created_at"])})
    for row in data["refunds"]:
        Refund(**{**row, "amount": int(row["amount"]), "created_at": int(row["created_at"])})
    for row in data["settlements"]:
        Settlement(**{**row, "amount_settled": int(row["amount_settled"]), "fee": int(row["fee"]),
                      "tax": int(row["tax"]), "settled_at": int(row["settled_at"])})


def test_ids_are_unique_and_linked(data):
    order_ids = [o["razorpay_order_id"] for o in data["orders"]]
    pay_ids = [p["id"] for p in data["payments"]]
    assert len(set(order_ids)) == len(order_ids)
    assert len(set(pay_ids)) == len(pay_ids)
    assert {r["payment_id"] for r in data["refunds"]} <= set(pay_ids)
    assert {s["payment_id"] for s in data["settlements"]} <= set(pay_ids)


def test_exactly_one_injection_and_it_has_no_payment(data):
    injected = [r for r in data["truth"] if r["injection"]]
    assert len(injected) == 1
    assert injected[0]["label"] == "MISSING_PAYMENT"
    with_note = [o for o in data["orders"] if o["notes"] == INJECTION_NOTE]
    assert len(with_note) == 1
    assert with_note[0]["order_ref"] == injected[0]["key"]
    assert with_note[0]["razorpay_order_id"] not in {p["order_id"] for p in data["payments"]}


def test_clean_orders_reconcile_exactly(data):
    pay_by_order = {}
    for p in data["payments"]:
        pay_by_order.setdefault(p["order_id"], []).append(p)
    settle_by_pay = {s["payment_id"]: s for s in data["settlements"]}
    orders = {o["order_ref"]: o for o in data["orders"]}
    checked = 0
    for r in data["truth"]:
        if r["label"] == "MATCHED" and r["near_miss"] is None:
            order = orders[r["key"]]
            (pay,) = pay_by_order[order["razorpay_order_id"]]
            assert pay["status"] == "captured" and pay["captured"] == "True"
            assert int(pay["amount"]) == int(order["amount_paise"])
            s = settle_by_pay[pay["id"]]
            assert int(s["amount_settled"]) == int(pay["amount"]) - int(pay["fee"]) - int(pay["tax"])
            checked += 1
    assert checked > 100


def test_near_misses_are_labelled_matched_and_cover_all_types(data):
    near = [r for r in data["truth"] if r["near_miss"]]
    assert all(r["label"] == "MATCHED" for r in near)
    assert {r["near_miss"] for r in near} == {"ROUNDING", "TIMEZONE", "NOT_YET_SETTLED", "PARTIAL_REFUND"}


def test_unit_error_orders_are_100x_smaller_than_payment(data):
    orders = {o["order_ref"]: o for o in data["orders"]}
    pay_by_order = {p["order_id"]: p for p in data["payments"]}
    for r in data["truth"]:
        if r["label"] == "UNIT_ERROR":
            o = orders[r["key"]]
            assert int(pay_by_order[o["razorpay_order_id"]]["amount"]) == int(o["amount_paise"]) * 100


def test_some_orders_exceed_double_confirmation_threshold(data):
    assert any(int(o["amount_paise"]) > 10_000 * 100 for o in data["orders"])


def test_money_action_flag_matches_policy_vocabulary(data):
    for r in data["truth"]:
        assert r["money_action"] == is_money_action(ActionType(r["expected_action"]))
    unit = next(r for r in data["truth"] if r["label"] == "UNIT_ERROR")
    assert unit["money_action"] is False


def test_too_few_orders_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate(seed=1, n_orders=10, out=tmp_path)
