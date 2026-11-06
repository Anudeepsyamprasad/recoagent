"""Smoke tests for the Streamlit demo, using Streamlit's headless AppTest runner.

Cassettes are recorded here from the ground-truth oracle, so the test needs no API key.
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from data.generate import generate
from recoagent.cli import run_pipeline
from recoagent.llm import LLMClient
from tests.oracle import OracleLLM

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def oracle_transport(oracle):
    def call(request):
        resp = oracle.create(system=request["system"], tools=request["tools"],
                             messages=request["messages"])
        return {"content": resp.content, "stop_reason": resp.stop_reason,
                "usage": {"input_tokens": 100, "output_tokens": 50}, "model": request["model"]}
    return call


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    generate(seed=42, n_orders=200, out=data / "sample")
    truth = json.loads((data / "sample" / "ground_truth.json").read_text())["records"]
    cassettes = tmp_path / "cassettes"
    llm = LLMClient(mode="record", cassette_dir=cassettes, model="claude-sonnet-5-5",
                    transport=oracle_transport(OracleLLM(truth)))
    run_pipeline(data / "sample", llm, tmp_path / "rec", run_id="rec")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "eval.md").write_text("# Eval table\n", encoding="utf-8")
    monkeypatch.setenv("RECO_DATA_ROOT", str(data))
    monkeypatch.setenv("RECO_CASSETTE_DIR", str(cassettes))
    monkeypatch.setenv("RECO_REPORTS_DIR", str(tmp_path / "reports"))
    return tmp_path


def load():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_renders_overview_and_all_tabs(env):
    at = load()
    assert len(at.tabs) == 5
    labels = {m.label for m in at.metric}
    assert {"Exceptions", "Waiting for approval"} <= labels
    assert next(m for m in at.metric if m.label == "Exceptions").value == "24"


def test_investigation_inspector_works_for_every_exception(env):
    at = load()
    for exc_id in at.selectbox(key="inspect").options:
        at.selectbox(key="inspect").select(exc_id).run()
        assert not at.exception


def test_approve_records_the_human_and_moves_the_item_out_of_the_queue(env):
    at = load()
    waiting = int(next(m for m in at.metric if m.label == "Waiting for approval").value)
    assert waiting > 0
    first = next(b for b in at.button if b.key.startswith("approve_"))
    exc_id = first.key.removeprefix("approve_")
    at.sidebar.text_input(key="approver").input("anudeep").run()
    at.button(key=f"approve_{exc_id}").click().run()
    assert not at.exception
    run = at.session_state["run::sample::claude-sonnet-5-5"]
    assert run["ledger"].get(exc_id).status == "APPROVED"
    assert run["ledger"].get(exc_id).approved_by == "anudeep"


def test_approve_without_a_name_shows_an_error_and_changes_nothing(env):
    at = load()
    first = next(b for b in at.button if b.key.startswith("approve_"))
    exc_id = first.key.removeprefix("approve_")
    at.button(key=f"approve_{exc_id}").click().run()
    assert [e.value for e in at.error]
    run = at.session_state["run::sample::claude-sonnet-5-5"]
    assert run["ledger"].get(exc_id).status == "QUEUED"


def test_reject_works(env):
    at = load()
    first = next(b for b in at.button if b.key.startswith("reject_"))
    exc_id = first.key.removeprefix("reject_")
    at.sidebar.text_input(key="approver").input("anudeep").run()
    at.button(key=f"reject_{exc_id}").click().run()
    run = at.session_state["run::sample::claude-sonnet-5-5"]
    assert run["ledger"].get(exc_id).status == "REJECTED"


def test_missing_cassettes_show_a_friendly_error_not_a_crash(env, monkeypatch):
    monkeypatch.setenv("RECO_CASSETTE_DIR", str(env / "empty"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("No recording" in e.value for e in at.error)
