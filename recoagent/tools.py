"""Read-only lookup tools the agent may call, plus their API definitions.

Nothing here can change data. The only way the agent "acts" is by calling submit_finding,
which just returns a structured conclusion for the policy layer and a human to review.
Tool outputs contain untrusted text (for example order notes): callers must treat them
as data, never as instructions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from recoagent import config
from recoagent.ingest import Dataset
from recoagent.schemas import Finding, FindingCategory, Order, Payment, PaymentStatus

MAX_RESULTS = 10


def _inline_refs(node, defs):
    """Replace $ref pointers with the definitions they point to (simpler schema for the API)."""
    if isinstance(node, dict):
        if "$ref" in node:
            target = defs[node["$ref"].split("/")[-1]]
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return {**_inline_refs(target, defs), **extra}
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _finding_schema() -> dict:
    raw = Finding.model_json_schema()
    schema = _inline_refs(raw, raw.get("$defs", {}))
    # AGENT_ERROR is reserved for our own code; the model must not be offered it.
    category = schema["properties"]["category"]
    category["enum"] = [v for v in category["enum"] if v != FindingCategory.AGENT_ERROR.value]
    return schema


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties, "required": required}}


_STR = {"type": "string"}
_INT = {"type": "integer"}

TOOL_DEFINITIONS: list[dict] = [
    _tool("get_order", "Fetch one store order by its order_ref (for example TMC-10023).",
          {"order_ref": _STR}, ["order_ref"]),
    _tool("get_payment", "Fetch one Razorpay payment by its id (pay_...).",
          {"payment_id": _STR}, ["payment_id"]),
    _tool("get_payments_for_order",
          "List every Razorpay payment attempt linked to a Razorpay order id (order_...).",
          {"razorpay_order_id": _STR}, ["razorpay_order_id"]),
    _tool("search_payments",
          "Find payments by exact amount (paise) and/or email and/or creation time window "
          "(unix seconds). Give at least one filter. Returns at most 10.",
          {"amount_paise": _INT, "email": _STR, "created_after": _INT, "created_before": _INT},
          []),
    _tool("search_orders",
          "Find store orders by exact amount (paise) and/or customer email. Each result says "
          "whether the order already has a captured payment. Give at least one filter. "
          "Returns at most 10.",
          {"amount_paise": _INT, "customer_email": _STR}, []),
    _tool("get_refunds", "List Razorpay refunds for a payment id.",
          {"payment_id": _STR}, ["payment_id"]),
    _tool("get_settlement",
          "Get the settlement (payout) row for a payment, or say why there is none "
          "(not yet due versus missing past due).",
          {"payment_id": _STR}, ["payment_id"]),
    _tool("compute_expected_settlement",
          "Compute what should have been settled for a payment: captured amount minus fee "
          "minus tax, in paise.",
          {"payment_id": _STR}, ["payment_id"]),
    {"name": "submit_finding",
     "description": "Finish the investigation by submitting your structured finding. Call "
                    "this exactly once, when you have enough evidence.",
     "input_schema": _finding_schema()},
]

READ_ONLY_TOOLS = frozenset(t["name"] for t in TOOL_DEFINITIONS if t["name"] != "submit_finding")


def _order_view(o: Order) -> dict:
    return o.model_dump(mode="json")


def _payment_view(p: Payment) -> dict:
    return p.model_dump(mode="json")


class ToolBox:
    """Runs the read-only tools against one dataset. run() never raises."""

    def __init__(self, ds: Dataset, as_of: datetime | None = None):
        self.ds = ds
        self.as_of = as_of or ds.as_of
        self._payment_by_id = {p.id: p for p in ds.payments}

    def run(self, name: str, args: dict) -> dict:
        if name not in READ_ONLY_TOOLS:
            return {"error": f"unknown tool {name!r}"}
        if not isinstance(args, dict):
            return {"error": "tool input must be a JSON object"}
        try:
            return getattr(self, f"_{name}")(**args)
        except TypeError as exc:  # missing or unexpected argument
            return {"error": f"bad arguments for {name}: {exc}"}

    # ---- tools -------------------------------------------------------------
    def _get_order(self, order_ref: str) -> dict:
        order = self.ds.order_by_ref.get(order_ref)
        return {"found": True, "order": _order_view(order)} if order else {"found": False}

    def _get_payment(self, payment_id: str) -> dict:
        pay = self._payment_by_id.get(payment_id)
        if not pay:
            return {"found": False}
        known_order = pay.order_id in self.ds.order_by_razorpay_id
        return {"found": True, "payment": _payment_view(pay), "order_exists_in_store": known_order}

    def _get_payments_for_order(self, razorpay_order_id: str) -> dict:
        pays = self.ds.payments_by_order_id.get(razorpay_order_id, [])
        return {"count": len(pays), "payments": [_payment_view(p) for p in pays]}

    def _search_payments(self, amount_paise: int | None = None, email: str | None = None,
                         created_after: int | None = None,
                         created_before: int | None = None) -> dict:
        if all(v is None for v in (amount_paise, email, created_after, created_before)):
            return {"error": "give at least one filter"}
        hits = [
            p for p in self.ds.payments
            if (amount_paise is None or p.amount == amount_paise)
            and (email is None or p.email == email)
            and (created_after is None or p.created_at >= created_after)
            and (created_before is None or p.created_at <= created_before)
        ]
        hits.sort(key=lambda p: p.created_at)
        return {"count": len(hits), "payments": [_payment_view(p) for p in hits[:MAX_RESULTS]]}

    def _search_orders(self, amount_paise: int | None = None,
                       customer_email: str | None = None) -> dict:
        if amount_paise is None and customer_email is None:
            return {"error": "give at least one filter"}
        hits = [
            o for o in self.ds.orders
            if (amount_paise is None or o.amount_paise == amount_paise)
            and (customer_email is None or o.customer_email == customer_email)
        ]
        hits.sort(key=lambda o: o.order_ref)
        out = []
        for o in hits[:MAX_RESULTS]:
            pays = self.ds.payments_by_order_id.get(o.razorpay_order_id, [])
            out.append({**_order_view(o), "has_captured_payment": any(p.captured for p in pays)})
        return {"count": len(hits), "orders": out}

    def _get_refunds(self, payment_id: str) -> dict:
        refunds = self.ds.refunds_by_payment.get(payment_id, [])
        return {"count": len(refunds), "refunds": [r.model_dump(mode="json") for r in refunds]}

    def _get_settlement(self, payment_id: str) -> dict:
        row = self.ds.settlement_by_payment.get(payment_id)
        if row:
            return {"settled": True, "settlement": row.model_dump(mode="json")}
        pay = self._payment_by_id.get(payment_id)
        if not pay:
            return {"settled": False, "reason": "payment not found"}
        if not pay.captured:
            return {"settled": False, "reason": "payment was never captured, so nothing to settle"}
        due = datetime.fromtimestamp(pay.created_at, UTC) + timedelta(
            days=config.SETTLEMENT_DELAY_DAYS)
        if due > self.as_of:
            return {"settled": False, "reason": "not yet due", "due_at": due.isoformat()}
        return {"settled": False, "reason": "missing past due", "due_at": due.isoformat()}

    def _compute_expected_settlement(self, payment_id: str) -> dict:
        pay = self._payment_by_id.get(payment_id)
        if not pay:
            return {"found": False}
        if not pay.captured or pay.status is PaymentStatus.FAILED:
            return {"found": True, "expected_settled_paise": 0,
                    "reason": "payment was never captured"}
        return {"found": True, "captured_paise": pay.amount, "fee_paise": pay.fee,
                "tax_paise": pay.tax,
                "expected_settled_paise": pay.amount - pay.fee - pay.tax}
