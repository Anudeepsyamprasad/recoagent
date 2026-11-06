"""Guardrails in plain code. A model cannot talk its way past these.

Given a Finding, the policy decides what happens next:
  AUTO_APPLY      safe (non-money) action, high confidence, automation on
  APPROVAL_QUEUE  any money action (always), or a safe action that is not auto-eligible
  NEEDS_HUMAN     the finding cannot be trusted as proposed (reasons are recorded)
  NO_ACTION       nothing to do

Rules, checked in this order (the first that fires decides):
  1. AGENT_ERROR, suspicious content, NEEDS_HUMAN category      -> NEEDS_HUMAN
  2. confidence below the floor                                 -> NEEDS_HUMAN
  3. NO_ACTION proposed                                         -> NO_ACTION
  4. action does not match the playbook for its category        -> NEEDS_HUMAN
  5. required payment_id missing, or not a real payment         -> NEEDS_HUMAN
  6. proposed amount not traceable to evidence                  -> NEEDS_HUMAN
  7. money action                                               -> APPROVAL_QUEUE
     (amount above Rs 10,000 additionally needs double confirmation)
  8. safe action: AUTO_APPLY if confidence >= 0.85 and the kill switch is off,
     otherwise APPROVAL_QUEUE
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from recoagent import config
from recoagent.ingest import Dataset
from recoagent.schemas import (
    EXPECTED_ACTION,
    ActionType,
    Category,
    Finding,
    FindingCategory,
    ReconException,
    is_money_action,
)

# Actions that must name the payment they act on.
PAYMENT_ID_REQUIRED = frozenset({
    ActionType.CONFIRM_AMOUNT, ActionType.LINK_OR_REFUND_ORPHAN, ActionType.REFUND_DUPLICATE,
    ActionType.UPDATE_STORE_REFUNDED, ActionType.CAPTURE_OR_VOID,
    ActionType.RAISE_SETTLEMENT_QUERY,
})

# Only integer fields with these names count as "amounts seen in tool results".
AMOUNT_KEYS = frozenset({"amount", "amount_settled", "amount_refunded", "fee", "tax"})


class Route(StrEnum):
    AUTO_APPLY = "AUTO_APPLY"
    APPROVAL_QUEUE = "APPROVAL_QUEUE"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    NO_ACTION = "NO_ACTION"


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exception_id: str
    route: Route
    reasons: list[str]
    double_confirm: bool = False
    amount_paise: int | None = None


def _collect_amounts(node: object, out: set[int]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                _collect_amounts(value, out)
            elif isinstance(value, int) and not isinstance(value, bool) and (
                key in AMOUNT_KEYS or key.endswith("_paise")
            ):
                out.add(value)
    elif isinstance(node, list):
        for item in node:
            _collect_amounts(item, out)


def seen_amounts(exc: ReconException, events: list[dict]) -> set[int]:
    """Every amount the agent could legitimately cite for this exception.

    Sources: the matcher's reference amount and signals, plus amount fields in this
    exception's tool outputs. Differences between two seen amounts are also allowed
    (a mismatch or shortfall is a difference, which no tool returns directly).
    """
    seen = {exc.amount_paise}
    for signal in exc.signals:
        seen.update(int(n) for n in re.findall(r"=(\d+)", signal))
    for event in events:
        if event.get("exception_id") == exc.exception_id and event.get("event") == "tool_call":
            _collect_amounts(event["payload"].get("output", {}), seen)
    return seen | {abs(a - b) for a in seen for b in seen if a != b}


def decide(finding: Finding, exc: ReconException, *, seen: set[int],
           known_payment_ids: set[str], automation_on: bool = True) -> Decision:
    action = finding.proposed_action

    def out(route: Route, *reasons: str, double: bool = False,
            amount: int | None = None) -> Decision:
        return Decision(exception_id=finding.exception_id, route=route, reasons=list(reasons),
                        double_confirm=double, amount_paise=amount)

    if finding.category is FindingCategory.AGENT_ERROR:
        return out(Route.NEEDS_HUMAN, "agent error: investigation did not complete")
    if finding.suspicious_content:
        return out(Route.NEEDS_HUMAN, "suspicious content in data: every action is blocked")
    if finding.category is FindingCategory.NEEDS_HUMAN:
        return out(Route.NEEDS_HUMAN, "agent asked for a human")
    if finding.confidence < config.CONFIDENCE_FLOOR:
        return out(Route.NEEDS_HUMAN,
                   f"confidence {finding.confidence:.2f} below floor {config.CONFIDENCE_FLOOR}")
    if action.type is ActionType.NO_ACTION:
        return out(Route.NO_ACTION, "no action proposed")

    expected = EXPECTED_ACTION[Category(finding.category.value)]
    if action.type is not expected:
        return out(Route.NEEDS_HUMAN,
                   f"action {action.type.value} does not match playbook {expected.value} "
                   f"for {finding.category.value}")
    if action.type in PAYMENT_ID_REQUIRED and not action.payment_id:
        return out(Route.NEEDS_HUMAN, f"{action.type.value} requires a payment_id")
    if action.payment_id and action.payment_id not in known_payment_ids:
        return out(Route.NEEDS_HUMAN, f"payment_id {action.payment_id} does not exist")
    if action.amount_paise is not None and action.amount_paise not in seen:
        return out(Route.NEEDS_HUMAN,
                   f"amount {action.amount_paise} not found in evidence (no invented numbers)")

    amount = action.amount_paise if action.amount_paise is not None else exc.amount_paise
    if is_money_action(action.type):
        large = amount > config.LARGE_AMOUNT_PAISE
        reasons = ["money action: always needs human approval"]
        if large:
            reasons.append("amount above Rs 10,000: double confirmation required")
        return out(Route.APPROVAL_QUEUE, *reasons, double=large, amount=amount)

    if not automation_on:
        return out(Route.APPROVAL_QUEUE, "kill switch is on: no auto-apply", amount=amount)
    if finding.confidence < config.AUTO_APPLY_CONFIDENCE:
        return out(Route.APPROVAL_QUEUE,
                   f"confidence {finding.confidence:.2f} below auto-apply "
                   f"{config.AUTO_APPLY_CONFIDENCE}", amount=amount)
    return out(Route.AUTO_APPLY, "safe action with high confidence", amount=amount)


def decide_all(findings: list[Finding], exceptions: list[ReconException], ds: Dataset,
               events: list[dict], automation_on: bool | None = None) -> list[Decision]:
    on = config.automation_enabled() if automation_on is None else automation_on
    known = {p.id for p in ds.payments}
    return [
        decide(f, e, seen=seen_amounts(e, events), known_payment_ids=known, automation_on=on)
        for f, e in zip(findings, exceptions, strict=True)
    ]
