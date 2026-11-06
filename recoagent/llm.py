"""Anthropic client wrapper: record / replay / live modes, retries, cost tracking.

Modes
- replay: read recorded responses from cassettes/. No API key, no network, no cost.
          (CI and the public demo use this.)
- record: use a cassette if it exists, otherwise call the API once and save the response.
          Re-running never re-spends money on a request you already recorded.
- live:   always call the API; save nothing.

A cassette is keyed by a hash of the exact request (model, system prompt, tools, messages).
Change the prompt or the model and the key changes, so stale recordings are never reused.

Model-API notes baked in (Sonnet 5.5): we send no temperature/top_p and no forced
tool_choice, and responses are kept as raw content blocks so any thinking blocks can be
replayed unchanged on the next turn.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from recoagent import config

MODES = ("replay", "record", "live")


class LLMError(RuntimeError):
    """The model call failed for good (after retries) or is misconfigured."""


class LLMConfigError(LLMError):
    """Setup problem (no key, bad key, unknown model, rejected request). Fatal: retrying or
    carrying on would only repeat the same failure for every exception."""


class CassetteMiss(LLMConfigError):
    """Replay mode could not find a recording for this exact request."""


@dataclass
class LLMResponse:
    content: list[dict]  # raw content blocks (text, tool_use, thinking, ...)
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float
    from_cassette: bool = False


def request_key(request: dict) -> str:
    blob = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


FATAL_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
)


def is_retryable(exc: Exception) -> bool:
    if isinstance(
        exc, (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)
    ):
        return True  # APITimeoutError is a subclass of APIConnectionError
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return False


class AnthropicTransport:
    """The one place that talks to the real API. Swapped for a fake in tests."""

    def __init__(self) -> None:
        self._client: anthropic.Anthropic | None = None

    def __call__(self, request: dict) -> dict:
        if self._client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMConfigError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key "
                    "(or use --mode replay, which needs no key)."
                )
            self._client = anthropic.Anthropic(
                api_key=api_key, timeout=config.REQUEST_TIMEOUT_S, max_retries=0
            )
        msg = self._client.messages.create(**request)
        return {
            "content": [block.model_dump(exclude_none=True) for block in msg.content],
            "stop_reason": msg.stop_reason,
            "usage": {"input_tokens": msg.usage.input_tokens,
                      "output_tokens": msg.usage.output_tokens},
            "model": msg.model,
        }


class LLMClient:
    def __init__(self, mode: str = "replay", cassette_dir: Path | str = "cassettes",
                 model: str | None = None, transport: Callable[[dict], dict] | None = None,
                 sleep: Callable[[float], None] = time.sleep, force: bool = False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.cassette_dir = Path(cassette_dir)
        self.model = model or config.get_model()
        self.transport = transport or AnthropicTransport()
        self.sleep = sleep
        self.force = force  # record mode: re-call even if a cassette exists
        self.total_cost_usd = 0.0  # everything this run used, including replayed recordings
        self.live_cost_usd = 0.0  # money actually spent on API calls right now (cap uses this)
        self.live_calls = 0

    # ---- public ------------------------------------------------------------
    def create(self, *, system: str, tools: list[dict], messages: list[dict],
               max_tokens: int = config.MAX_TOKENS) -> LLMResponse:
        request = {"model": self.model, "max_tokens": max_tokens, "system": system,
                   "tools": tools, "messages": messages}
        key = request_key(request)
        path = self.cassette_dir / f"{key}.json"

        if self.mode == "replay" or (self.mode == "record" and path.exists() and not self.force):
            if not path.exists():
                raise CassetteMiss(
                    f"No cassette for this request (key {key}). Record it with "
                    "--mode record, or check that the prompt, tools and model match the "
                    "recording."
                )
            raw = json.loads(path.read_text(encoding="utf-8"))["response"]
            return self._finish(raw, from_cassette=True)

        raw = self._call_with_retries(request)
        self.live_calls += 1
        if self.mode == "record":
            self.cassette_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "key": key, "model": self.model,
                "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "response": raw,
            }, indent=1, ensure_ascii=False), encoding="utf-8")
        return self._finish(raw, from_cassette=False)

    # ---- internals ---------------------------------------------------------
    def _call_with_retries(self, request: dict) -> dict:
        for attempt in range(1, config.MAX_ATTEMPTS + 1):
            try:
                return self.transport(request)
            except LLMError:
                raise
            except FATAL_ERRORS as exc:
                raise LLMConfigError(
                    f"the API rejected the request ({type(exc).__name__}): {exc}. Check your "
                    "key, the model name and billing, then run `smoke` again."
                ) from exc
            except Exception as exc:  # classified just below
                if is_retryable(exc) and attempt < config.MAX_ATTEMPTS:
                    self.sleep(2 ** (attempt - 1))  # 1s, then 2s
                    continue
                raise LLMError(f"model call failed ({type(exc).__name__}): {exc}") from exc
        raise AssertionError("unreachable")  # pragma: no cover

    def _finish(self, raw: dict, *, from_cassette: bool) -> LLMResponse:
        usage = raw.get("usage", {})
        in_tok, out_tok = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        cost = config.cost_usd(self.model, in_tok, out_tok)
        self.total_cost_usd += cost
        if not from_cassette:
            self.live_cost_usd += cost
        return LLMResponse(
            content=list(raw["content"]), stop_reason=raw.get("stop_reason"),
            input_tokens=in_tok, output_tokens=out_tok, model=self.model, cost_usd=cost,
            from_cassette=from_cassette,
        )
