"""Core data models and vocabularies for RecoAgent.

Money is ALWAYS integer paise (1 rupee = 100 paise). Never use floats for money.
Field names mirror Razorpay's entities (payment, refund, settlement) so a test-mode
adapter can map onto them later.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ----------------------------------------------------------------------------
# Vocabularies
# ----------------------------------------------------------------------------
class Category(StrEnum):
    """Root-cause labels. MATCHED means 'no exception'."""

    MATCHED = "MATCHED"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    MISSING_PAYMENT = "MISSING_PAYMENT"
    ORPHAN_PAYMENT = "ORPHAN_PAYMENT"
    DUPLICATE_PAYMENT = "DUPLICATE_PAYMENT"
    UNRECORDED_REFUND = "UNRECORDED_REFUND"
    AUTH_NOT_CAPTURED = "AUTH_NOT_CAPTURED"
    UNIT_ERROR = "UNIT_ERROR"
    SETTLEMENT_SHORTFALL = "SETTLEMENT_SHORTFALL"


class ActionType(StrEnum):
    """What the agent may propose. Nothing here ever calls a real money API."""

    NO_ACTION = "NO_ACTION"
    CONFIRM_AMOUNT = "CONFIRM_AMOUNT"
    MARK_ORDER_UNPAID = "MARK_ORDER_UNPAID"
    LINK_OR_REFUND_ORPHAN = "LINK_OR_REFUND_ORPHAN"
    REFUND_DUPLICATE = "REFUND_DUPLICATE"
    UPDATE_STORE_REFUNDED = "UPDATE_STORE_REFUNDED"
    CAPTURE_OR_VOID = "CAPTURE_OR_VOID"
    FIX_STORE_AMOUNT = "FIX_STORE_AMOUNT"
    RAISE_SETTLEMENT_QUERY = "RAISE_SETTLEMENT_QUERY"


# The correct action for each category (used by the generator's ground truth
# and later by the evaluation harness).
EXPECTED_ACTION: dict[Category, ActionType] = {
    Category.MATCHED: ActionType.NO_ACTION,
    Category.AMOUNT_MISMATCH: ActionType.CONFIRM_AMOUNT,
    Category.MISSING_PAYMENT: ActionType.MARK_ORDER_UNPAID,
    Category.ORPHAN_PAYMENT: ActionType.LINK_OR_REFUND_ORPHAN,
    Category.DUPLICATE_PAYMENT: ActionType.REFUND_DUPLICATE,
    Category.UNRECORDED_REFUND: ActionType.UPDATE_STORE_REFUNDED,
    Category.AUTH_NOT_CAPTURED: ActionType.CAPTURE_OR_VOID,
    Category.UNIT_ERROR: ActionType.FIX_STORE_AMOUNT,
    Category.SETTLEMENT_SHORTFALL: ActionType.RAISE_SETTLEMENT_QUERY,
}

# Actions that touch money or customer money state: these ALWAYS need a human.
# FIX_STORE_AMOUNT is a pure data fix in the store, so it is the one non-money
# action (per the data design doc). NO_ACTION is trivially safe.
SAFE_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.NO_ACTION, ActionType.FIX_STORE_AMOUNT}
)


def is_money_action(action: ActionType) -> bool:
    return action not in SAFE_ACTIONS


class StoreStatus(StrEnum):
    PAID = "paid"
    PENDING = "pending"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class PaymentStatus(StrEnum):
    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    REFUNDED = "refunded"
    FAILED = "failed"


class PaymentMethod(StrEnum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class RefundStatus(StrEnum):
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


# ----------------------------------------------------------------------------
# Entities
# ----------------------------------------------------------------------------
class _Row(BaseModel):
    # Reject unknown columns so a schema typo fails loudly instead of silently.
    model_config = ConfigDict(extra="forbid")


class Order(_Row):
    """Store side (the merchant's own system)."""

    order_ref: str  # e.g. TMC-10023 (the store's receipt number)
    razorpay_order_id: str  # order_xxxxxxxxxxxxxx
    customer_email: str
    amount_paise: int = Field(gt=0)
    currency: str = "INR"
    store_status: StoreStatus
    created_at: datetime  # UTC
    notes: str = ""  # UNTRUSTED free text: may contain prompt injection
    refunded_amount_paise: int = Field(default=0, ge=0)  # refunds the store recorded


class Payment(_Row):
    """Razorpay payment entity."""

    id: str  # pay_xxxxxxxxxxxxxx
    order_id: str  # order_xxxxxxxxxxxxxx
    amount: int = Field(gt=0)
    currency: str = "INR"
    status: PaymentStatus
    method: PaymentMethod
    captured: bool
    amount_refunded: int = Field(default=0, ge=0)
    fee: int = Field(default=0, ge=0)
    tax: int = Field(default=0, ge=0)
    email: str
    created_at: int  # Unix seconds (UTC), like Razorpay


class Refund(_Row):
    """Razorpay refund entity."""

    id: str  # rfnd_xxxxxxxxxxxxxx
    payment_id: str
    amount: int = Field(gt=0)
    status: RefundStatus
    created_at: int  # Unix seconds (UTC)


class Settlement(_Row):
    """Simplified recon view: one row per payment inside a settlement."""

    settlement_id: str  # setl_xxxxxxxxxxxxxx
    payment_id: str
    amount_settled: int
    fee: int = Field(ge=0)
    tax: int = Field(ge=0)
    utr: str
    settled_at: int  # Unix seconds (UTC)


class ReconException(_Row):
    """A record the deterministic matcher could not close.

    `signals` are plain-language FACTS the rules observed (for example
    'amount_differs:paid=1000,order=900'). The matcher deliberately does NOT assign a
    category: choosing the root cause and the action is the agent's job.
    """

    exception_id: str  # EXC-0001, assigned in a stable order
    kind: Literal["order", "payment"]
    key: str  # order_ref for orders, payment id for orphan payments
    signals: list[str]
    payment_ids: list[str] = Field(default_factory=list)
    amount_paise: int = Field(ge=0)  # reference amount, used for thresholds


# ----------------------------------------------------------------------------
# Agent output
# ----------------------------------------------------------------------------
class FindingCategory(StrEnum):
    """What the agent may conclude. AGENT_ERROR is set by our code, never by the model."""

    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    MISSING_PAYMENT = "MISSING_PAYMENT"
    ORPHAN_PAYMENT = "ORPHAN_PAYMENT"
    DUPLICATE_PAYMENT = "DUPLICATE_PAYMENT"
    UNRECORDED_REFUND = "UNRECORDED_REFUND"
    AUTH_NOT_CAPTURED = "AUTH_NOT_CAPTURED"
    UNIT_ERROR = "UNIT_ERROR"
    SETTLEMENT_SHORTFALL = "SETTLEMENT_SHORTFALL"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    AGENT_ERROR = "AGENT_ERROR"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProposedAction(_Row):
    type: ActionType
    payment_id: str | None = None
    amount_paise: int | None = Field(default=None, ge=0)
    note: str = ""


class Finding(_Row):
    """The agent's structured conclusion for one exception (validated, never free text)."""

    exception_id: str
    category: FindingCategory
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(min_length=1)
    root_cause: str
    proposed_action: ProposedAction
    risk_level: RiskLevel
    suspicious_content: bool = False

    @classmethod
    def needs_human(cls, exception_id: str, reason: str, *, suspicious: bool = False) -> Finding:
        return cls(
            exception_id=exception_id, category=FindingCategory.NEEDS_HUMAN, confidence=0.0,
            evidence=[reason], root_cause=reason,
            proposed_action=ProposedAction(type=ActionType.NO_ACTION),
            risk_level=RiskLevel.HIGH, suspicious_content=suspicious,
        )

    @classmethod
    def agent_error(cls, exception_id: str, reason: str) -> Finding:
        return cls(
            exception_id=exception_id, category=FindingCategory.AGENT_ERROR, confidence=0.0,
            evidence=[reason], root_cause=reason,
            proposed_action=ProposedAction(type=ActionType.NO_ACTION),
            risk_level=RiskLevel.HIGH,
        )
