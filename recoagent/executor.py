"""Idempotent executor over a LOCAL ledger. It never calls a refund or payout API.

Safe actions are recorded as AUTO_APPLIED. Money actions are recorded as QUEUED and only
become APPROVED when a named human approves them (large amounts need the amount typed
back). "Approved" here means "recorded as approved in the ledger": no money moves.

Idempotent: the action key is a hash of (exception, action type, payment, amount), so
submitting or approving twice never creates a second entry or a second execution.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from recoagent.audit import AuditLog
from recoagent.policy import Decision, Route
from recoagent.schemas import ActionType, Finding


class LedgerError(ValueError):
    """An approval or rejection that must not go through."""


Status = Literal["QUEUED", "AUTO_APPLIED", "APPROVED", "REJECTED"]


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_key: str
    exception_id: str
    action_type: ActionType
    payment_id: str | None
    amount_paise: int | None
    status: Status
    double_confirm: bool = False
    approved_by: str | None = None
    note: str = ""
    created_at: str
    updated_at: str


def action_key(exception_id: str, action_type: str, payment_id: str | None,
               amount: int | None) -> str:
    blob = f"{exception_id}|{action_type}|{payment_id}|{amount}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path | str | None = None, audit: AuditLog | None = None):
        self.path = Path(path) if path else None
        self.audit = audit
        self.entries: dict[str, LedgerEntry] = {}
        if self.path and self.path.exists():
            for raw in json.loads(self.path.read_text(encoding="utf-8")):
                entry = LedgerEntry.model_validate(raw)
                self.entries[entry.exception_id] = entry

    # ---- helpers -----------------------------------------------------------
    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = [e.model_dump(mode="json") for e in self.entries.values()]
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _log(self, event: str, entry: LedgerEntry, actor: str, **payload) -> None:
        if self.audit:
            self.audit.log(event, exception_id=entry.exception_id, actor=actor,
                           payload={"action_key": entry.action_key,
                                    "action_type": entry.action_type.value,
                                    "amount_paise": entry.amount_paise, **payload})

    def get(self, exception_id: str) -> LedgerEntry | None:
        return self.entries.get(exception_id)

    @property
    def queued(self) -> list[LedgerEntry]:
        return [e for e in self.entries.values() if e.status == "QUEUED"]

    # ---- submit (called by the pipeline after the policy decides) ----------
    def submit(self, finding: Finding, decision: Decision) -> LedgerEntry | None:
        if decision.route not in (Route.AUTO_APPLY, Route.APPROVAL_QUEUE):
            return None  # NEEDS_HUMAN / NO_ACTION never reach the ledger
        action = finding.proposed_action
        key = action_key(finding.exception_id, action.type.value, action.payment_id,
                         decision.amount_paise)
        existing = self.entries.get(finding.exception_id)
        if existing and existing.action_key == key:
            return existing  # idempotent: same action, nothing new happens
        status: Status = "AUTO_APPLIED" if decision.route is Route.AUTO_APPLY else "QUEUED"
        now = _now()
        entry = LedgerEntry(
            action_key=key, exception_id=finding.exception_id, action_type=action.type,
            payment_id=action.payment_id, amount_paise=decision.amount_paise, status=status,
            double_confirm=decision.double_confirm, note=action.note,
            created_at=now, updated_at=now,
        )
        self.entries[finding.exception_id] = entry
        self._save()
        if status == "AUTO_APPLIED":
            self._log("execution", entry, "policy", result="recorded in local ledger; "
                      "no external system was called")
        return entry

    # ---- human decisions ---------------------------------------------------
    def approve(self, exception_id: str, approver: str,
                confirm_amount: int | None = None) -> LedgerEntry:
        entry = self._require(exception_id, approver)
        if entry.status == "APPROVED":
            return entry  # idempotent: no second approval, no second execution
        if entry.status != "QUEUED":
            raise LedgerError(f"{exception_id} is {entry.status}, not awaiting approval")
        if entry.double_confirm and confirm_amount != entry.amount_paise:
            raise LedgerError(
                f"{exception_id} is above Rs 10,000: re-type the exact amount "
                f"({entry.amount_paise} paise) with --confirm-amount to approve")
        entry = entry.model_copy(update={"status": "APPROVED", "approved_by": approver,
                                         "updated_at": _now()})
        self.entries[exception_id] = entry
        self._save()
        self._log("approval", entry, f"human:{approver}")
        self._log("execution", entry, "executor", result="recorded as approved in local "
                  "ledger; no external system was called")
        return entry

    def reject(self, exception_id: str, approver: str, reason: str = "") -> LedgerEntry:
        entry = self._require(exception_id, approver)
        if entry.status == "REJECTED":
            return entry
        if entry.status != "QUEUED":
            raise LedgerError(f"{exception_id} is {entry.status}, not awaiting approval")
        entry = entry.model_copy(update={"status": "REJECTED", "approved_by": approver,
                                         "note": reason or entry.note, "updated_at": _now()})
        self.entries[exception_id] = entry
        self._save()
        self._log("approval", entry, f"human:{approver}", decision="rejected", reason=reason)
        return entry

    def _require(self, exception_id: str, approver: str) -> LedgerEntry:
        if not approver or not approver.strip():
            raise LedgerError("an approver name is required")
        entry = self.entries.get(exception_id)
        if entry is None:
            raise LedgerError(f"no ledger entry for {exception_id}")
        return entry
