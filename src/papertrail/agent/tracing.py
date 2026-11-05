"""Tracing and replay for the investigator.

Off by default (`PAPERTRAIL_TRACING` unset or `off`), so the app behaves exactly as before. With `PAPERTRAIL_TRACING=jsonl` every investigation
appends one JSON object per event to `$PAPERTRAIL_TRACE_DIR/<run_id>.jsonl` (default data/local/traces):

    run_start        account, prompt_version
    step             graph node name, duration, and the guardrail errors / draft count after it
    tool_call        tool, inputs, output, latency_s
    llm_call         messages, response, meta (provider, model, tokens, latency), prompt_version, or `error`
    guardrail_rejection   draft number and the checker's error list (an uncited or unsupported number, a bad source_id, ...)
    run_end          the final result dict

`replay()` re-runs the same graph from a recorded run using the RECORDED tool outputs and the RECORDED LLM replies, so a failure can be
reproduced with neither the demo bundle's live data nor an LLM key. The sink is plain JSONL on purpose: it needs no account or dependency, and
`Tracer` takes any callable as a sink if you want to forward events to Langfuse or OpenTelemetry (not implemented here).
"""
from __future__ import annotations

import copy
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from papertrail.agent.llm import LLMError
from papertrail.agent.prompts import PROMPT_VERSION

DEFAULT_DIR = "data/local/traces"
TOOLS = ("get_alert", "get_subgraph", "get_shap", "get_txn_history")
COMPARED = ("summary", "verified", "drafts", "calls", "retried", "rejections", "errors")     # result fields replay must reproduce


class Tracer:
    """Collects events for one run. `sink(event)` receives each event dict; the default sink appends to the run's JSONL file."""

    def __init__(self, trace_dir: Path | str = DEFAULT_DIR, run_id: str | None = None, sink: Callable[[dict], None] | None = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.events: list[dict] = []
        self.path = Path(trace_dir) / f"{self.run_id}.jsonl"
        self._sink = sink or self._write
        self.enabled = True

    def event(self, type_: str, **data: Any) -> None:
        ev = {"run_id": self.run_id, "seq": len(self.events), "t": round(time.time(), 3), "type": type_, **data}
        self.events.append(ev)
        self._sink(ev)

    def _write(self, ev: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(ev, default=str) + "\n")

    @classmethod
    def from_env(cls) -> Tracer | NullTracer:
        mode = os.environ.get("PAPERTRAIL_TRACING", "off").lower()
        return cls(os.environ.get("PAPERTRAIL_TRACE_DIR", DEFAULT_DIR)) if mode == "jsonl" else NullTracer()


class NullTracer:
    enabled = False
    run_id = None

    def event(self, *_a, **_k) -> None:
        pass


class TracedToolbox:
    """Wraps a Toolbox: `gather` calls the four tools one by one and records inputs, output and latency of each."""

    def __init__(self, inner, tracer: Tracer):
        self.inner, self.tracer = inner, tracer

    def gather(self, account: str) -> dict[str, dict]:
        outs = {}
        for name in TOOLS:
            t0 = time.perf_counter()
            out = getattr(self.inner, name)(account)
            self.tracer.event("tool_call", tool=name, inputs={"account": account}, output=out, latency_s=round(time.perf_counter() - t0, 4))
            outs[out["source_id"]] = out
        return outs

    def __getattr__(self, item):
        return getattr(self.inner, item)


class TracedClient:
    """Wraps a chat client: records each LLM call's messages, reply, model, tokens, latency and prompt version (or the error)."""

    def __init__(self, inner, tracer: Tracer):
        self.inner, self.tracer = inner, tracer

    def chat_json(self, messages, *a, **k):
        t0 = time.perf_counter()
        try:
            raw, meta = self.inner.chat_json(messages, *a, **k)
        except LLMError as e:
            self.tracer.event("llm_call", messages=messages, error=str(e), latency_s=round(time.perf_counter() - t0, 3), prompt_version=PROMPT_VERSION)
            raise
        self.tracer.event("llm_call", messages=messages, response=raw, meta=meta, latency_s=round(time.perf_counter() - t0, 3),
                          prompt_version=PROMPT_VERSION,
                          tokens={"prompt": meta.get("prompt_tokens"), "completion": meta.get("completion_tokens")})
        return raw, meta

    def __getattr__(self, item):
        return getattr(self.inner, item)


# ---------------------------------------------------------------------------------------------------- replay
class ReplayToolbox:
    def __init__(self, outputs: dict[str, dict]):
        self.outputs = outputs

    def gather(self, account: str) -> dict[str, dict]:
        return copy.deepcopy(self.outputs)


class ReplayClient:
    """Returns the recorded LLM replies in order (and re-raises recorded LLM failures)."""

    def __init__(self, calls: list[dict]):
        self.calls = list(calls)
        self.i = 0

    def chat_json(self, messages, *a, **k):
        if self.i >= len(self.calls):
            raise LLMError("replay ran out of recorded LLM calls: the code now makes more calls than the recorded run did")
        c = self.calls[self.i]
        self.i += 1
        if "error" in c:
            raise LLMError(c["error"].removeprefix("LLM unavailable: "))
        return copy.deepcopy(c["response"]), c["meta"]


def load_run(run_id: str, trace_dir: Path | str = DEFAULT_DIR) -> list[dict]:
    path = Path(trace_dir) / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no trace for run {run_id!r} in {trace_dir}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replay(run_id: str, trace_dir: Path | str = DEFAULT_DIR) -> dict:
    """Re-execute a recorded investigation from its recorded tool outputs and LLM replies.
    Returns {"result", "recorded", "diffs"}: `diffs` lists the compared fields that differ from the recorded run (empty = reproduced)."""
    from papertrail.agent.graph import investigate

    events = load_run(run_id, trace_dir)
    start = next(e for e in events if e["type"] == "run_start")
    end = next((e for e in events if e["type"] == "run_end"), None)
    outputs = {e["output"]["source_id"]: e["output"] for e in events if e["type"] == "tool_call"}
    result = investigate(start["account"], ReplayToolbox(outputs), ReplayClient([e for e in events if e["type"] == "llm_call"]), tracer=NullTracer())
    recorded = end["result"] if end else None
    diffs = [k for k in COMPARED if recorded is not None and result.get(k) != recorded.get(k)]
    return {"result": result, "recorded": recorded, "diffs": diffs}
