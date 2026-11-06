"""Tests for scoring, reports and the CI gate, driven end to end by the oracle stand-in."""

import json

import pytest

from data.generate import generate
from recoagent.cli import evaluate_dataset, run_pipeline
from recoagent.eval import build_markdown, evaluate, slug, write_reports
from recoagent.eval_gate import check, main as gate_main
from recoagent.executor import LedgerEntry
from recoagent.llm import LLMClient
from recoagent.policy import Route
from tests.oracle import OracleLLM


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval42")
    generate(seed=42, n_orders=200, out=out)
    return out


@pytest.fixture(scope="module")
def truth(data_dir):
    return json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))["records"]


@pytest.fixture(scope="module")
def perfect(data_dir, truth):
    return evaluate_dataset(data_dir, OracleLLM(truth), "Seed 42 (dev)")


def test_a_perfect_agent_scores_perfectly_and_safely(perfect):
    m, c = perfect["metrics"], perfect["counts"]
    assert m["auto_match_rate"] == m["detection_recall"] == m["detection_precision"] == 1.0
    assert m["category_accuracy"] == m["action_accuracy"] == m["amount_accuracy"] == 1.0
    assert m["unsafe_actions"] == 0 and m["injection_handled"] is True
    assert perfect["failures"] == []
    assert c["planted_exceptions"] == 24 and c["judged_by_agent"] == 24


def test_routing_counts_follow_the_policy(perfect):
    c = perfect["counts"]
    assert c["auto_applied"] == 3  # the three unit errors are the only safe action
    assert c["needs_human"] == 1  # the injection record
    assert c["approval_queue"] == 20
    assert c["auto_applied"] + c["needs_human"] + c["approval_queue"] + c["no_action"] == 24


def test_injection_is_caught_by_the_scan_even_though_the_oracle_never_flags_it(data_dir, truth):
    summary = run_pipeline(data_dir, OracleLLM(truth), data_dir / "runs", run_id="inj")
    key = next(t["key"] for t in truth if t["injection"])
    idx = next(i for i, e in enumerate(summary["exceptions"]) if e.key == key)
    assert summary["findings"][idx].suspicious_content is True
    assert summary["decisions"][idx].route is Route.NEEDS_HUMAN
    assert summary["ledger"].get(summary["exceptions"][idx].exception_id) is None


def test_ledger_holds_no_executed_money_action_without_a_human(data_dir, truth):
    summary = run_pipeline(data_dir, OracleLLM(truth), data_dir / "runs", run_id="ledger")
    statuses = {e.status for e in summary["ledger"].entries.values()}
    assert statuses == {"QUEUED", "AUTO_APPLIED"}


def test_a_wrong_answer_lowers_the_scores_and_is_listed_as_a_failure(data_dir, truth):
    victim = next(t["key"] for t in truth if t["label"] == "DUPLICATE_PAYMENT")
    report = evaluate_dataset(data_dir, OracleLLM(truth, flip_key=victim), "flipped")
    assert report["metrics"]["category_accuracy"] == pytest.approx(23 / 24)
    (failure,) = report["failures"]
    assert failure["key"] == victim and failure["got"] == "NEEDS_HUMAN"
    assert failure["route"] == "NEEDS_HUMAN"
    assert report["amount_at_risk_unresolved_paise"] > 0
    assert report["metrics"]["unsafe_actions"] == 0  # a wrong answer is not an unsafe one


def test_auto_applied_money_action_is_counted_as_unsafe(data_dir, truth):
    summary = run_pipeline(data_dir, OracleLLM(truth), data_dir / "runs", run_id="unsafe")
    bad = [e.model_copy(update={"status": "AUTO_APPLIED"}) if e.action_type.value ==
           "REFUND_DUPLICATE" else e for e in summary["ledger"].entries.values()]
    report = evaluate(truth, summary["exceptions"], summary["findings"], summary["decisions"],
                      bad, cost_usd=0.0, label="x", model="m", prompt_version="v", n_orders=200)
    assert report["metrics"]["unsafe_actions"] == 3
    assert any("unsafe" in f for f in check(report))


def test_approved_without_a_named_human_is_unsafe(data_dir, truth):
    summary = run_pipeline(data_dir, OracleLLM(truth), data_dir / "runs", run_id="noname")
    entries = list(summary["ledger"].entries.values())
    forged = LedgerEntry.model_validate({**entries[0].model_dump(), "status": "APPROVED",
                                         "approved_by": None})
    report = evaluate(truth, summary["exceptions"], summary["findings"], summary["decisions"],
                      [forged], cost_usd=0.0, label="x", model="m", prompt_version="v",
                      n_orders=200)
    assert report["metrics"]["unsafe_actions"] == 1


def test_cost_is_reported_per_run_and_per_exception(perfect):
    assert perfect["metrics"]["cost_per_run_usd"] == pytest.approx(0.024 + 0.001, abs=0.002)
    assert perfect["metrics"]["cost_per_exception_usd"] > 0


def test_gate_passes_a_perfect_run_and_fails_a_degraded_one(perfect):
    assert check(perfect) == []
    degraded = json.loads(json.dumps(perfect))
    degraded["metrics"]["category_accuracy"] = 0.5
    degraded["metrics"]["detection_recall"] = 0.5
    degraded["metrics"]["injection_handled"] = False
    failures = check(degraded)
    assert len(failures) == 3
    assert any("category accuracy" in f for f in failures)


def test_markdown_and_json_reports_merge_every_dataset(perfect, tmp_path):
    other = json.loads(json.dumps(perfect))
    other["label"] = "Seed 7 (held-out)"
    write_reports(perfect, tmp_path)
    write_reports(other, tmp_path)
    assert sorted(p.name for p in tmp_path.glob("eval_*.json")) == [
        "eval_seed_42_dev.json", "eval_seed_7_held_out.json"]
    md = (tmp_path / "eval.md").read_text(encoding="utf-8")
    assert "Seed 42 (dev)" in md and "Seed 7 (held-out)" in md
    assert "| Unsafe actions | 0 | 0 | 0 |" in md
    assert "None on these runs." in md and "Small samples" in md
    assert slug("Seed 7 (held-out)") == "seed_7_held_out"


def test_markdown_lists_failures(data_dir, truth):
    victim = next(t["key"] for t in truth if t["label"] == "UNIT_ERROR")
    report = evaluate_dataset(data_dir, OracleLLM(truth, flip_key=victim), "flipped")
    md = build_markdown([report])
    assert victim in md and "Known failures" in md and "None on these runs." not in md


def test_gate_main_fails_clearly_when_cassettes_are_missing(data_dir, tmp_path, capsys):
    code = gate_main(["--data", str(data_dir), "--cassettes", str(tmp_path / "none")])
    out = capsys.readouterr().out
    assert code == 1 and "GATE FAILED" in out and "re-record" in out


def test_gate_main_passes_when_every_request_has_a_recording(data_dir, truth, tmp_path, capsys):
    """Record the oracle's responses as cassettes, then replay them through the real gate."""
    cassettes = tmp_path / "cassettes"
    oracle = OracleLLM(truth)
    recorder = LLMClient(mode="record", cassette_dir=cassettes, model="claude-sonnet-5-5",
                         transport=lambda req: _as_raw(oracle, req))
    run_pipeline(data_dir, recorder, tmp_path / "runs", run_id="rec")
    assert len(list(cassettes.glob("*.json"))) >= 24
    code = gate_main(["--data", str(data_dir), "--cassettes", str(cassettes)])
    assert code == 0 and "[PASS]" in capsys.readouterr().out


def _as_raw(oracle, request):
    resp = oracle.create(system=request["system"], tools=request["tools"],
                         messages=request["messages"])
    return {"content": resp.content, "stop_reason": resp.stop_reason,
            "usage": {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
            "model": resp.model}
