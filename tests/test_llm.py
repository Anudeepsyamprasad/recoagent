"""Tests for the LLM client: cassettes, retries, cost. No network, no API key."""

import json

import anthropic
import pytest

try:  # anthropic 2.x depends on httpx2; older SDKs used httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from recoagent import config
from recoagent.llm import CassetteMiss, LLMClient, LLMConfigError, LLMError, request_key

SYSTEM = "sys"
TOOLS = [{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
MSGS = [{"role": "user", "content": "hello"}]
RAW = {
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1_000, "output_tokens": 500},
    "model": "claude-sonnet-5-5-20261001",
}
REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class FakeTransport:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(mode, tmp_path, transport=None, sleeps=None, **kw):
    sleeps = sleeps if sleeps is not None else []
    return LLMClient(mode=mode, cassette_dir=tmp_path, model="claude-sonnet-5-5",
                     transport=transport, sleep=sleeps.append, **kw)


def _call(client):
    return client.create(system=SYSTEM, tools=TOOLS, messages=MSGS)


def test_record_then_replay_returns_same_response(tmp_path):
    transport = FakeTransport([RAW])
    recorded = _call(_client("record", tmp_path, transport))
    assert not recorded.from_cassette and transport.calls == 1
    replayed = _call(_client("replay", tmp_path))
    assert replayed.from_cassette
    assert replayed.content == recorded.content
    assert replayed.cost_usd == pytest.approx(recorded.cost_usd)


def test_replay_without_cassette_raises_clear_error(tmp_path):
    with pytest.raises(CassetteMiss, match="--mode record"):
        _call(_client("replay", tmp_path))


def test_record_mode_does_not_respend_on_an_existing_cassette(tmp_path):
    transport = FakeTransport([RAW])
    client = _client("record", tmp_path, transport)
    _call(client)
    _call(client)
    assert transport.calls == 1
    forced = _client("record", tmp_path, transport, force=True)
    _call(forced)
    assert transport.calls == 2


def test_live_mode_calls_every_time_and_saves_nothing(tmp_path):
    transport = FakeTransport([RAW])
    client = _client("live", tmp_path, transport)
    _call(client)
    _call(client)
    assert transport.calls == 2
    assert list(tmp_path.glob("*.json")) == []


def test_request_key_is_stable_and_sensitive_to_every_input():
    base = {"model": "m", "max_tokens": 10, "system": "s", "tools": [], "messages": []}
    assert request_key(base) == request_key(dict(reversed(list(base.items()))))
    for field, value in [("model", "x"), ("system", "y"), ("max_tokens", 11),
                         ("messages", [{"role": "user", "content": "z"}])]:
        assert request_key({**base, field: value}) != request_key(base)


def test_cassette_file_is_plain_json(tmp_path):
    _call(_client("record", tmp_path, FakeTransport([RAW])))
    (path,) = tmp_path.glob("*.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["response"]["content"] == RAW["content"]
    assert "api_key" not in path.read_text(encoding="utf-8").lower()


def test_cost_uses_per_million_token_prices(tmp_path):
    client = _client("live", tmp_path, FakeTransport([RAW]))
    resp = _call(client)
    # 1000 in at $2/M + 500 out at $10/M
    assert resp.cost_usd == pytest.approx(0.002 + 0.005)
    assert client.total_cost_usd == pytest.approx(resp.cost_usd)
    assert config.cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)


def test_unknown_model_price_fails_loudly():
    with pytest.raises(ValueError, match="No price"):
        config.cost_usd("claude-mystery", 1, 1)


def test_retries_transient_errors_with_exponential_backoff(tmp_path):
    flaky = [anthropic.APIConnectionError(request=REQ), anthropic.APIConnectionError(request=REQ),
             RAW]
    transport, sleeps = FakeTransport(flaky), []
    resp = _call(_client("live", tmp_path, transport, sleeps))
    assert resp.content == RAW["content"]
    assert transport.calls == 3 and sleeps == [1, 2]


def test_gives_up_after_max_attempts(tmp_path):
    err = anthropic.RateLimitError("slow down", response=httpx.Response(429, request=REQ),
                                   body=None)
    transport, sleeps = FakeTransport([err]), []
    with pytest.raises(LLMError, match="RateLimitError"):
        _call(_client("live", tmp_path, transport, sleeps))
    assert transport.calls == config.MAX_ATTEMPTS and sleeps == [1, 2]


def test_bad_request_is_not_retried(tmp_path):
    err = anthropic.BadRequestError("nope", response=httpx.Response(400, request=REQ), body=None)
    transport, sleeps = FakeTransport([err]), []
    with pytest.raises(LLMConfigError, match="BadRequestError"):
        _call(_client("live", tmp_path, transport, sleeps))
    assert transport.calls == 1 and sleeps == []


def test_missing_api_key_gives_actionable_message(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match=r"\.env"):
        _call(_client("live", tmp_path))


def test_invalid_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        LLMClient(mode="yolo", cassette_dir=tmp_path)


def test_bad_key_is_fatal_and_says_what_to_check(tmp_path):
    err = anthropic.AuthenticationError("invalid x-api-key",
                                        response=httpx.Response(401, request=REQ), body=None)
    transport = FakeTransport([err])
    with pytest.raises(LLMConfigError, match="smoke"):
        _call(_client("live", tmp_path, transport))
    assert transport.calls == 1


def test_cassette_miss_counts_as_a_setup_error():
    assert issubclass(CassetteMiss, LLMConfigError) and issubclass(LLMConfigError, LLMError)


def test_cap_counts_live_spend_only_not_replayed_cost(tmp_path):
    _call(_client("record", tmp_path, FakeTransport([RAW])))
    again = _client("record", tmp_path, FakeTransport([RAW]))
    _call(again)  # served from the cassette
    assert again.total_cost_usd > 0 and again.live_cost_usd == 0
