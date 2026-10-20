"""The investigator agent: tools (on the real demo bundle), the number-checking guardrail, the LangGraph flow with a
fake LLM, and the Groq -> OpenRouter failover with mocked HTTP. No API key or network is needed."""
import json
from pathlib import Path

import httpx
import pytest

from papertrail.agent.llm import (
    CURATED,
    ChatClient,
    LLMError,
    ModelChoice,
    Provider,
    available,
    discover_free_models,
    parse_json,
    provider_of,
    providers_from_env,
)
from papertrail.agent.schemas import CaseSummary
from papertrail.agent.verify import check, numbers_in_output, numbers_in_text, supported

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"
needs_bundle = pytest.mark.skipif(not (DEMO / "alerts.parquet").exists(), reason="demo bundle not installed")


# ---------------------------------------------------------------- guardrail
def test_numbers_are_read_like_a_person_writes_them():
    assert numbers_in_text("17,050.43 sent in 3 payments") == [(17050.43, 2), (3.0, 0)]
    assert [v for v, _ in numbers_in_text("2022-09-09 14:20")] == [2022.0, 9.0, 9.0, 14.0, 20.0]
    assert [v for v, _ in numbers_in_text("account 222697_8087F4940")] == [222697.0]      # digits glued to a word are part of the id


def test_rounding_is_allowed_computing_is_not():
    pool = numbers_in_output({"amount": 17050.43, "n": 3, "id": "acct 12_ab"})
    assert supported(17050, 0, pool) and supported(17050.4, 1, pool) and supported(3, 0, pool) and supported(12, 0, pool)
    assert not supported(17051, 0, pool) and not supported(51151.29, 2, pool)


def _outputs(account="A_1"):
    return {f"alert:{account}": {"source_id": f"alert:{account}", "accounts_in_queue": 200,
                                 "models": {"blend": {"rank": 4, "risk_score": 5.2, "flagged_payments": 3}}},
            f"subgraph:{account}": {"source_id": f"subgraph:{account}", "fan_out_distinct_receivers": 9, "max_receivers_paying_one_account": 8}}


def _summary(**change):
    s = {"account_id": "A_1", "risk_score": 5.2, "suspected_typology": "SCATTER-GATHER", "recommended_action": "escalate_SAR",
         "evidence": [{"claim": "Ranked 4 of 200 by the blend model", "source_id": "alert:A_1"},
                      {"claim": "Pays 9 receivers, 8 of whom pay the same next account", "source_id": "subgraph:A_1"}],
         "narrative": "The account scatters money to 9 receivers that gather it again."}
    s.update(change)
    return CaseSummary.model_validate(s)


def test_a_faithful_summary_passes():
    assert check(_summary(), _outputs(), "A_1") == []


def test_the_checker_catches_each_kind_of_error():
    errs = check(_summary(account_id="B_2", risk_score=9.9, narrative="It moved 12,345 in total.",
                          evidence=[{"claim": "Ranked 4 of 200", "source_id": "alert:A_1"},
                                    {"claim": "Pays 11 receivers", "source_id": "subgraph:A_1"},
                                    {"claim": "Something", "source_id": "history:A_1"}]), _outputs(), "A_1")
    text = " | ".join(errs)
    assert "account_id" in text and "risk_score" in text and "evidence 2 states 11" in text
    assert "'history:A_1', which is not a tool output" in text and "narrative states 12345" in text


def test_a_number_must_come_from_the_cited_source_not_another_one():
    s = _summary(evidence=[{"claim": "Ranked 4 of 200", "source_id": "alert:A_1"}, {"claim": "Ranked 4 of 200", "source_id": "subgraph:A_1"}])
    assert any("evidence 2 states 4, 200" in e for e in check(s, _outputs(), "A_1"))


# ---------------------------------------------------------------- tools
@needs_bundle
def test_tools_cover_the_queue_cite_their_source_and_never_leak_the_truth():
    from papertrail.agent.tools import HIDDEN, Toolbox

    tb = Toolbox(DEMO)
    account = min(tb.accounts)
    out = tb.gather(account)
    assert set(out) == {f"{t}:{account}" for t in ("alert", "subgraph", "shap", "history")}
    blob = json.dumps(out)
    assert not any(f'"{h}"' in blob for h in HIDDEN) and "true positive" not in blob and "false positive" not in blob
    assert out[f"history:{account}"]["count"] <= 20
    with pytest.raises(KeyError):
        tb.get_alert("not-an-account")


# ---------------------------------------------------------------- graph (fake LLM)
class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def chat_json(self, messages, **_):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r, {"provider": "fake", "model": "m", "latency_s": 0.1, "prompt_tokens": 100, "completion_tokens": 50}


@pytest.fixture
def toolbox():
    pytest.importorskip("langgraph")
    from papertrail.agent.tools import Toolbox

    if not (DEMO / "alerts.parquet").exists():
        pytest.skip("demo bundle not installed")
    return Toolbox(DEMO)


def _good(tb, account):
    a = tb.get_alert(account)
    return {"account_id": account, "risk_score": a["models"]["blend"]["risk_score"], "suspected_typology": "UNCLEAR",
            "recommended_action": "enhanced_review", "narrative": "The account is high in the queue; review the network.",
            "evidence": [{"claim": f"Ranked {a['models']['blend']['rank']} of {a['accounts_in_queue']} by the blend model", "source_id": a["source_id"]},
                         {"claim": f"{a['models']['blend']['flagged_payments']} payments flagged by the blend", "source_id": a["source_id"]}]}


def test_graph_accepts_a_faithful_first_draft(toolbox):
    from papertrail.agent.graph import investigate

    acct = min(toolbox.accounts)
    fake = FakeClient([_good(toolbox, acct)])
    r = investigate(acct, toolbox, fake)
    assert r["verified"] and r["drafts"] == 1 and r["calls"] == 5 and fake.calls == 1 and r["tokens"] == 150


def test_graph_retries_once_with_the_errors_then_accepts(toolbox):
    from papertrail.agent.graph import investigate

    acct = min(toolbox.accounts)
    bad = {**_good(toolbox, acct), "narrative": "It moved 98,765,432 through the network."}
    r = investigate(acct, toolbox, FakeClient([bad, _good(toolbox, acct)]))
    assert r["verified"] and r["retried"] and r["calls"] == 6 and "98765432" in " ".join(r["rejections"][0])


def test_graph_falls_back_to_unclear_and_keeps_only_verified_claims(toolbox):
    from papertrail.agent.graph import investigate

    acct = min(toolbox.accounts)
    bad = {**_good(toolbox, acct), "suspected_typology": "CYCLE", "narrative": "It moved 98,765,432."}
    bad["evidence"] = bad["evidence"] + [{"claim": "Sent 123,456,789 to one account", "source_id": f"history:{acct}"}]
    r = investigate(acct, toolbox, FakeClient([bad, bad]))
    s = r["summary"]
    assert not r["verified"] and s["suspected_typology"] == "UNCLEAR" and r["calls"] <= 6
    assert len(s["evidence"]) == 2 and all("123,456,789" not in e["claim"] for e in s["evidence"])


def test_graph_survives_an_llm_outage(toolbox):
    from papertrail.agent.graph import investigate

    acct = min(toolbox.accounts)
    r = investigate(acct, toolbox, FakeClient([LLMError("all providers down")]))
    assert not r["verified"] and r["drafts"] == 1 and r["summary"]["suspected_typology"] == "UNCLEAR"


def test_graph_treats_schema_violations_as_errors_to_fix(toolbox):
    from papertrail.agent.graph import investigate

    acct = min(toolbox.accounts)
    r = investigate(acct, toolbox, FakeClient([{**_good(toolbox, acct), "suspected_typology": "PONZI"}, _good(toolbox, acct)]))
    assert r["verified"] and r["retried"] and "suspected_typology" in r["rejections"][0][0]


# ---------------------------------------------------------------- LLM client (mocked HTTP)
def test_keys_go_to_the_provider_their_prefix_names(monkeypatch):
    assert (provider_of("gsk_x"), provider_of("sk-or-v1-x"), provider_of("xai-x"), provider_of("abc")) == ("groq", "openrouter", "xai", None)
    for v in ("LLM_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-abc")              # an OpenRouter key under the Groq name
    assert [p.name for p in providers_from_env()] == ["openrouter"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abc")
    monkeypatch.setenv("LLM_API_KEY", "gsk_abc")
    assert [p.name for p in providers_from_env()] == ["groq", "openrouter"]


def test_replies_in_fences_or_after_thinking_are_parsed():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('<think>hmm {not json}</think>{"a": 2}') == {"a": 2}
    assert parse_json("Sure! here it is: {\"a\": 3} hope it helps") == {"a": 3}
    assert parse_json("no json here") is None


CATALOGUE = (ModelChoice("groq", "g1", "G1"), ModelChoice("groq", "g2", "G2"),
             ModelChoice("openrouter", "a/model:free", "A"), ModelChoice("openrouter", "b/model:free", "B"))


def _client(handler, model=None):
    return ChatClient([Provider("groq", "gsk_x"), Provider("openrouter", "sk-or-x")], http=httpx.Client(transport=httpx.MockTransport(handler)),
                      model=model, catalogue=CATALOGUE)


def _ok(content):
    return httpx.Response(200, json={"model": "served", "choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def test_groq_limit_fails_over_to_openrouter_free_models():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append((req.url.host, body["model"]))
        if req.url.host == "api.groq.com":
            return httpx.Response(429, json={"error": {"message": "Rate limit reached for model"}})
        if body["model"] == "a/model:free":
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return _ok('{"ok": true}')

    c = _client(handler)
    parsed, meta = c.chat_json([{"role": "user", "content": "json please"}])
    assert parsed == {"ok": True} and meta["provider"] == "openrouter" and meta["attempts"] == 4
    assert [h for h, _ in seen] == ["api.groq.com", "api.groq.com", "openrouter.ai", "openrouter.ai"]


def test_a_rejected_key_skips_its_provider_and_a_non_json_reply_moves_on():
    def handler(req):
        if req.url.host == "api.groq.com":
            return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})
        return _ok("I cannot comply") if json.loads(req.content)["model"] == "a/model:free" else _ok('{"x": 1}')

    c = _client(handler)
    parsed, meta = c.chat_json([{"role": "user", "content": "json"}])
    assert parsed == {"x": 1} and meta["attempts"] == 3                # 1 groq (key dead -> skip), 2 openrouter
    assert [e["status"] for e in c.log] == [401, 200, 200] and c.log[1]["info"].startswith("reply was not JSON")


def test_everything_down_raises_a_readable_error():
    c = _client(lambda req: httpx.Response(500, json={"error": {"message": "boom"}}))
    with pytest.raises(LLMError, match="no model answered"):
        c.chat_json([{"role": "user", "content": "json"}])
    with pytest.raises(LLMError, match="no API key"):
        ChatClient([], http=httpx.Client(transport=httpx.MockTransport(lambda r: _ok("{}")))).chat_json([])


def test_a_model_that_thinks_past_its_budget_is_skipped_and_the_last_good_model_goes_first():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body["model"])
        assert body.get("reasoning") == {"effort": "low", "exclude": True} or req.url.host != "openrouter.ai"
        if body["model"] == "a/model:free":                      # all budget spent on reasoning, empty answer
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
        if req.url.host == "api.groq.com":
            return httpx.Response(429, json={"error": {"message": "limit"}})
        return _ok('{"ok": 1}')

    c = _client(handler)
    c.chat_json([{"role": "user", "content": "json"}])
    assert ("openrouter", "a/model:free") in c.skip and c.last_good == ("openrouter", "b/model:free")
    calls.clear()
    c.chat_json([{"role": "user", "content": "json"}])
    assert calls == ["b/model:free"]                           # straight to the model that worked, no wasted attempts


# ---------------------------------------------------------------- chat
def test_chat_answers_are_checked_and_retried(toolbox):
    from papertrail.agent.chat import ask

    acct = min(toolbox.accounts)
    a = toolbox.get_alert(acct)
    good = {"answer": f"It ranks {a['models']['blend']['rank']} of {a['accounts_in_queue']} under the blend model.", "sources": [a["source_id"]]}
    r = ask("Why is it flagged?", acct, toolbox, FakeClient([good]))
    assert r["verified"] and not r["retried"] and r["sources"] == [a["source_id"]]
    bad = {"answer": "It moved 98,765,432 in total.", "sources": [a["source_id"]]}
    r = ask("How much?", acct, toolbox, FakeClient([bad, good]))
    assert r["verified"] and r["retried"]
    r = ask("How much?", acct, toolbox, FakeClient([bad, bad]))
    assert not r["verified"] and "could not answer" in r["answer"] and "98765432" not in r["answer"]


def test_chat_rejects_unknown_sources_and_survives_an_outage(toolbox):
    from papertrail.agent.chat import ask, check_answer

    acct = min(toolbox.accounts)
    out = toolbox.gather(acct)
    assert "unknown sources" in check_answer({"answer": "x", "sources": ["web:google"]}, out)[0]
    assert check_answer({"answer": "I can only answer questions about this account.", "sources": []}, out) == []
    r = ask("hi", acct, toolbox, FakeClient([LLMError("down")]))
    assert not r["verified"] and "unavailable" in r["answer"]



def test_a_picked_model_goes_first_and_the_curated_list_backs_it_up():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content)["model"])
        return httpx.Response(429, json={"error": {"message": "busy"}}) if len(seen) == 1 else _ok('{"ok": 1}')

    pick = ModelChoice("openrouter", "x/new:free", "New")
    _, meta = _client(handler, model=pick).chat_json([{"role": "user", "content": "json"}])
    assert seen == ["x/new:free", "g1"] and meta["model"] == "served"


def test_only_models_whose_provider_has_a_key_are_offered():
    assert {c.provider for c in available(CURATED, [Provider("groq", "gsk_x")])} == {"groq"}
    assert available(CURATED, []) == []
    assert all(c.provider in ("groq", "openrouter") for c in CURATED) and len({c.key for c in CURATED}) == len(CURATED)


def test_discovery_keeps_free_json_capable_text_models_only():
    data = {"data": [
        {"id": "big/model:free", "name": "Big (free)", "context_length": 900, "supported_parameters": ["response_format"], "architecture": {"output_modalities": ["text"]}},
        {"id": "small/model:free", "name": "Small (free)", "context_length": 100, "supported_parameters": ["structured_outputs"]},
        {"id": "nojson/model:free", "supported_parameters": ["tools"]},
        {"id": "paid/model", "supported_parameters": ["response_format"]},
        {"id": "img/model:free", "supported_parameters": ["response_format"], "architecture": {"output_modalities": ["image"]}},
        {"id": "x/content-safety:free", "supported_parameters": ["response_format"]},
        {"id": CURATED[-1].model, "supported_parameters": ["response_format"]},
    ]}
    found = discover_free_models(httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=data))))
    assert [c.model for c in found] == ["big/model:free", "small/model:free"] and found[0].label == "Big" and "untested" in found[0].note
