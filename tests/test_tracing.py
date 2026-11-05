"""Tracing and replay: record a run (clean, retried, fallback, LLM outage), replay it from the recording alone, and get the same result."""
import json

import pytest

from papertrail.agent.llm import LLMError
from papertrail.agent.prompts import PROMPT_VERSION
from papertrail.agent.tracing import NullTracer, Tracer, load_run, replay
from tests.test_agent import FakeClient, _good


def _run(toolbox, replies, tmp_path):
    from papertrail.agent.graph import investigate

    acct = min(toolbox.accounts)
    tracer = Tracer(tmp_path)
    result = investigate(acct, toolbox, FakeClient(replies(acct)), tracer=tracer)
    return acct, tracer, result


def test_trace_records_steps_tools_llm_calls_and_prompt_version(toolbox, tmp_path):
    acct, tracer, result = _run(toolbox, lambda a: [_good(toolbox, a)], tmp_path)
    events = load_run(tracer.run_id, tmp_path)
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end" and result["trace_run_id"] == tracer.run_id
    tools = [e for e in events if e["type"] == "tool_call"]
    assert [t["tool"] for t in tools] == ["get_alert", "get_subgraph", "get_shap", "get_txn_history"]
    assert all(t["inputs"] == {"account": acct} and t["latency_s"] >= 0 and t["output"]["source_id"] for t in tools)
    llm = [e for e in events if e["type"] == "llm_call"]
    assert len(llm) == 1 and llm[0]["prompt_version"] == PROMPT_VERSION and llm[0]["tokens"] == {"prompt": 100, "completion": 50}
    assert [e["node"] for e in events if e["type"] == "step"] == ["gather", "draft", "verify", "done"]
    assert not [e for e in events if e["type"] == "guardrail_rejection"]


def test_guardrail_rejection_of_an_uncited_number_is_traced(toolbox, tmp_path):
    def replies(a):
        bad = _good(toolbox, a)
        bad["evidence"][0]["claim"] = "Ranked 4 of 200 by the blend model, moving 987654 dollars"
        return [bad, _good(toolbox, a)]
    _, tracer, result = _run(toolbox, replies, tmp_path)
    rej = [e for e in load_run(tracer.run_id, tmp_path) if e["type"] == "guardrail_rejection"]
    assert len(rej) == 1 and rej[0]["draft"] == 1 and "987654" in rej[0]["errors"][0] and result["retried"]


@pytest.mark.parametrize("scenario", ["clean", "retry", "fallback", "outage"])
def test_a_recorded_run_replays_to_the_same_result_without_data_or_llm(toolbox, tmp_path, scenario):
    def replies(a):
        bad = {**_good(toolbox, a), "narrative": "It moved 987654 dollars."}
        return {"clean": [_good(toolbox, a)], "retry": [bad, _good(toolbox, a)], "fallback": [bad, bad],
                "outage": [LLMError("all providers down")]}[scenario]
    _, tracer, original = _run(toolbox, replies, tmp_path)
    out = replay(tracer.run_id, tmp_path)               # uses only the JSONL: the toolbox and the fake client are not passed in
    assert out["diffs"] == []
    assert out["result"]["summary"] == original["summary"] and out["result"]["verified"] == original["verified"]


def test_replay_notices_when_the_code_changes_behaviour(toolbox, tmp_path, monkeypatch):
    """If the guardrail changed so that it no longer rejects the recorded draft, replay reports the difference."""
    def replies(a):
        return [{**_good(toolbox, a), "narrative": "It moved 987654 dollars."}, _good(toolbox, a)]
    _, tracer, _ = _run(toolbox, replies, tmp_path)
    monkeypatch.setattr("papertrail.agent.graph.check", lambda *a, **k: [])
    out = replay(tracer.run_id, tmp_path)
    assert out["diffs"] and "drafts" in out["diffs"]


def test_tracing_is_off_by_default_and_writes_nothing(toolbox, tmp_path, monkeypatch):
    from papertrail.agent.graph import investigate

    monkeypatch.delenv("PAPERTRAIL_TRACING", raising=False)
    monkeypatch.setenv("PAPERTRAIL_TRACE_DIR", str(tmp_path))
    assert isinstance(Tracer.from_env(), NullTracer)
    r = investigate(min(toolbox.accounts), toolbox, FakeClient([_good(toolbox, min(toolbox.accounts))]))
    assert "trace_run_id" not in r and list(tmp_path.iterdir()) == []


def test_env_flag_turns_jsonl_tracing_on(toolbox, tmp_path, monkeypatch):
    from papertrail.agent.graph import investigate

    monkeypatch.setenv("PAPERTRAIL_TRACING", "jsonl")
    monkeypatch.setenv("PAPERTRAIL_TRACE_DIR", str(tmp_path))
    r = investigate(min(toolbox.accounts), toolbox, FakeClient([_good(toolbox, min(toolbox.accounts))]))
    lines = (tmp_path / f"{r['trace_run_id']}.jsonl").read_text().splitlines()
    assert all(json.loads(x)["run_id"] == r["trace_run_id"] for x in lines) and len(lines) > 5
