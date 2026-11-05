# Investigator tracing and replay

Off by default. `PAPERTRAIL_TRACING=jsonl` makes every `investigate()` call append events to `data/local/traces/<run_id>.jsonl`
(override with `PAPERTRAIL_TRACE_DIR`). Code: `src/papertrail/agent/tracing.py`; hooks in `agent/graph.py`.

| event | what it holds |
|---|---|
| `run_start` / `run_end` | account, prompt version / the final result dict (and `trace_run_id`) |
| `step` | each graph node (`gather`, `draft`, `verify`, `done` or `fallback`): duration, draft count, guardrail errors |
| `tool_call` | one per data tool: inputs, full output, latency |
| `llm_call` | messages, reply, provider, model, prompt/completion tokens, latency, `prompt_version` (hash of the system prompt), or the error |
| `guardrail_rejection` | draft number and the checker's reasons (a number not found in the cited tool output, a bad `source_id`, ...) |

## Replay

```bash
python scripts/replay.py --list
python scripts/replay.py <run_id>      # exit 0 if the result matches the recording, 1 and a diff if not
```

Replay feeds the graph the recorded tool outputs and the recorded LLM replies (including recorded LLM failures), then re-runs the real
graph and guardrail. It needs neither the demo bundle's data nor an API key. It compares the summary, the verified flag, draft and call
counts, the rejections and the errors. Because the guardrail and graph logic really execute, replay also detects a code change that alters
behaviour on a recorded case (`tests/test_tracing.py::test_replay_notices_when_the_code_changes_behaviour`).

What replay does not do: it does not re-ask the LLM, so it reproduces failures in *our* code (graph, guardrail, schema), not in the model.

## Design notes and gaps

- JSONL rather than Langfuse or OpenTelemetry: it needs no account, key or dependency and fits the replay format. `Tracer(sink=callable)`
  takes any sink, but a Langfuse/OTel exporter is **not implemented** and nothing here was tested against either.
- Traces contain full prompts and tool outputs (synthetic data only). Keep `data/local/` out of git (it is ignored).
- All recorded runs in the tests use a scripted LLM. No live-LLM trace has been recorded in this environment (no network or keys used),
  so token counts and latencies in traces from real runs have not been checked here.
