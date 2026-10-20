"""The investigator as a LangGraph state machine:

    gather (4 tool calls) -> draft (LLM) -> verify --ok--> done
                                ^             |
                                +-- retry ----+ (once, with the checker's error list)
                                              +-- still failing --> fallback (typology UNCLEAR, only verified evidence)

At most 6 tool + LLM calls per case (4 tools, 2 drafts). The result is a plain dict, ready for the app and for the cache.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from papertrail.agent.llm import ChatClient, LLMError
from papertrail.agent.prompts import draft_messages
from papertrail.agent.schemas import CaseSummary
from papertrail.agent.tools import Toolbox
from papertrail.agent.verify import check, verified_evidence

MAX_CALLS, MAX_DRAFTS = 6, 2


class State(TypedDict, total=False):
    account: str
    outputs: dict[str, dict]
    raw: dict | None
    summary: CaseSummary | None
    errors: list[str]
    drafts: int
    calls: int
    llm: list[dict]
    history: list[dict]
    result: dict


def build(toolbox: Toolbox, client: ChatClient):
    def gather(s: State) -> State:
        out = toolbox.gather(s["account"])
        return {"outputs": out, "calls": s.get("calls", 0) + len(out), "drafts": 0, "errors": [], "llm": [], "history": []}

    def draft(s: State) -> State:
        if s["calls"] >= MAX_CALLS:
            return {"errors": s["errors"] + ["call budget used up"], "summary": None}
        msgs = draft_messages(s["account"], s["outputs"], s["errors"] or None, s.get("raw"))
        calls, drafts = s["calls"] + 1, s["drafts"] + 1
        try:
            raw, meta = client.chat_json(msgs)
        except LLMError as e:
            return {"calls": calls, "drafts": drafts, "raw": None, "summary": None, "errors": [f"LLM unavailable: {e}"], "llm": s["llm"]}
        hist = s["history"] + [{"draft": drafts, "raw": raw}]
        try:
            summary = CaseSummary.model_validate(raw)
        except ValidationError as e:
            errs = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()][:8]
            hist[-1]["errors"] = errs
            return {"calls": calls, "drafts": drafts, "raw": raw, "summary": None, "errors": errs, "llm": s["llm"] + [meta], "history": hist}
        return {"calls": calls, "drafts": drafts, "raw": raw, "summary": summary, "errors": [], "llm": s["llm"] + [meta], "history": hist}

    def verify(s: State) -> State:
        if s.get("summary") is None:
            return {}
        errs = check(s["summary"], s["outputs"], s["account"])
        return {"errors": errs, "history": s["history"][:-1] + [{**s["history"][-1], "errors": errs}]}

    def route(s: State) -> str:
        if s.get("summary") is not None and not s["errors"]:
            return "done"
        if s["drafts"] < MAX_DRAFTS and s["calls"] < MAX_CALLS and not any(e.startswith("LLM unavailable") for e in s["errors"]):
            return "retry"
        return "fallback"

    def done(s: State) -> State:
        return {"result": _result(s, s["summary"], verified=True)}

    def fallback(s: State) -> State:
        """Keep only what passes the checker; never ship an unverified claim."""
        alert = s["outputs"][f"alert:{s['account']}"]
        sm = s.get("summary")
        evidence = verified_evidence(sm, s["outputs"]) if sm else []
        safe = {"account_id": s["account"], "risk_score": alert["models"]["blend"]["risk_score"], "suspected_typology": "UNCLEAR",
                "evidence": [e.model_dump() for e in evidence] or [{"claim": f"Ranked {alert['models']['blend']['rank']} of {alert['accounts_in_queue']} by the blend model",
                                                                   "source_id": alert["source_id"]}],
                "narrative": "The automatic summary could not be fully verified against the tool outputs, so only checked facts are shown. "
                             "Review the network and payment history by hand.",
                "recommended_action": "enhanced_review"}
        return {"result": _result(s, CaseSummary.model_validate(safe), verified=False)}

    g = StateGraph(State)
    for name, fn in (("gather", gather), ("draft", draft), ("verify", verify), ("done", done), ("fallback", fallback)):
        g.add_node(name, fn)
    g.set_entry_point("gather")
    g.add_edge("gather", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route, {"done": "done", "retry": "draft", "fallback": "fallback"})
    g.add_edge("done", END)
    g.add_edge("fallback", END)
    return g.compile()


def _result(s: State, summary: CaseSummary, verified: bool) -> dict[str, Any]:
    last = s["llm"][-1] if s["llm"] else {}
    tokens = sum((m.get("prompt_tokens") or 0) + (m.get("completion_tokens") or 0) for m in s["llm"])
    return {"summary": summary.model_dump(), "verified": verified, "drafts": s["drafts"], "calls": s["calls"],
            "retried": s["drafts"] > 1, "rejections": [h.get("errors") for h in s["history"] if h.get("errors")],
            "drafts_log": s["history"],                       # each draft as the model wrote it, with the checker's verdict
            "errors": s["errors"], "provider": last.get("provider"), "model": last.get("model"),
            "llm_latency_s": round(sum(m.get("latency_s") or 0 for m in s["llm"]), 2), "tokens": tokens,
            "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}


def investigate(account: str, toolbox: Toolbox, client: ChatClient | None = None) -> dict:
    """Run the whole case and return the result dict (summary, verified flag, provider, model, tokens, timings)."""
    t0 = time.time()
    out = build(toolbox, client or ChatClient()).invoke({"account": account}, {"recursion_limit": 20})["result"]
    out["total_latency_s"] = round(time.time() - t0, 2)
    assert out["calls"] <= MAX_CALLS
    return out
