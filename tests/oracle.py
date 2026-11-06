"""A stand-in for the model that answers from ground truth (tests only, never shipped logic).

It lets us run the whole pipeline, scoring and CI gate deterministically and for free.
For the planted injection record it first calls get_order, like a real investigation would,
so the deterministic injection scan is exercised end to end.
"""

import re
from itertools import count

from recoagent.llm import LLMResponse
from recoagent.policy import PAYMENT_ID_REQUIRED
from recoagent.schemas import ActionType

_ids = count(1)


def finding_for(truth: dict, exception_id: str, related_ids: list[str], reference: int,
                **over) -> dict:
    action = ActionType(truth["expected_action"])
    payment_id = related_ids[-1] if (action in PAYMENT_ID_REQUIRED and related_ids) else None
    base = {
        "exception_id": exception_id, "category": truth["label"], "confidence": 0.95,
        "evidence": [f"oracle: {truth['label']}"], "root_cause": f"oracle: {truth['label']}",
        "proposed_action": {"type": action.value, "payment_id": payment_id,
                            "amount_paise": truth["amount_at_risk_paise"]},
        "risk_level": "high" if reference > 1_000_000 else "medium", "suspicious_content": False,
    }
    return {**base, **over}


class OracleLLM:
    model = "claude-sonnet-5-5"

    def __init__(self, truth_records: list[dict], flip_key: str | None = None):
        self.truth = {t["key"]: t for t in truth_records}
        self.flip_key = flip_key  # make the oracle wrong on this exception
        self.total_cost_usd = 0.0
        self.live_cost_usd = 0.0
        self.live_calls = 0

    def create(self, *, system, tools, messages, max_tokens=4096):
        case = messages[0]["content"]
        exc_id = re.search(r"exception_id: (EXC-\d+)", case).group(1)
        key = re.search(r"\nkey: (\S+)", case).group(1)
        related = re.search(r"related_payment_ids: (.*)", case).group(1)
        related_ids = [] if related.strip() == "none" else [s.strip() for s in related.split(",")]
        reference = int(re.search(r"reference_amount_paise: (\d+)", case).group(1))
        truth = self.truth[key]

        if truth["injection"] and len(messages) == 1:
            block = {"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": "get_order",
                     "input": {"order_ref": key}}
        else:
            over = {}
            if key == self.flip_key:
                over = {"category": "NEEDS_HUMAN", "confidence": 0.2}
            block = {"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": "submit_finding",
                     "input": finding_for(truth, exc_id, related_ids, reference, **over)}
        self.total_cost_usd += 0.001
        self.live_cost_usd += 0.001
        return LLMResponse(content=[block], stop_reason="tool_use", input_tokens=100,
                           output_tokens=50, model=self.model, cost_usd=0.001)
