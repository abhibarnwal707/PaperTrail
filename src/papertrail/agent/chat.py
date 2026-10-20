"""Ask the investigator: free-form questions about one account, answered from the same four tool outputs as the case
summary and held to the same rule (every number in the answer must be in a cited output; one retry, then a safe reply).
Questions outside the account's data are declined, not answered from the model's general knowledge."""
from __future__ import annotations

import json
import time

from papertrail.agent.llm import ChatClient, LLMError
from papertrail.agent.tools import Toolbox
from papertrail.agent.verify import numbers_in_output, numbers_in_text, supported

EXAMPLE_QUESTIONS = (
    "Why was this account flagged?",
    "Who does it pay, and who pays it?",
    "Does any money come back to it in a loop?",
    "Which laundering pattern does it look like?",
    "What are its largest recent payments?",
    "Do the three models agree about it?",
)
CHIP_LABELS = {q: label for q, label in zip(EXAMPLE_QUESTIONS, ("Why flagged?", "Who pays it, who does it pay?", "Any money loops?",
                                                                  "Which pattern?", "Largest payments?", "Do the models agree?"))}
MAX_TURNS = 6                                         # earlier turns sent along with a new question

SYSTEM = """You answer an investigator's questions about ONE bank account, using only the tool outputs below (JSON, each with a source_id).
Answer with ONE JSON object and nothing else: {"answer": str, "sources": [source_id, ...]}.
Rules (a checker rejects the answer otherwise):
1. Use only the tool outputs. If they do not contain the answer, say so plainly. If the question is not about this account's data,
   answer "I can only answer questions about this account's alert, network, score drivers and payments." with sources [].
2. Every number you write must be copied from a cited output (you may drop decimals). Never add up, average or compute percentages.
   Do not write dates. The drivers output explains "the riskiest payment"; say "the most recent payment" only for the first payment
   in the history output, described exactly as listed there.
3. Write 1 to 4 short sentences in plain English for a compliance officer. Do not use the outputs' field names
   (say "5 different senders", not "fan_in_distinct_senders").
4. "sources" lists every source_id you used. The data is synthetic (IBM AML dataset)."""


def _messages(account: str, outputs: dict, question: str, turns: list[dict], errors: list[str] | None, previous: dict | None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Account: {account}\n\nTool outputs:\n{json.dumps(outputs, indent=1)}"},
            {"role": "assistant", "content": '{"answer": "Ready. Ask me about this account.", "sources": []}'}]
    for t in turns[-MAX_TURNS:]:
        msgs += [{"role": "user", "content": t["question"]}, {"role": "assistant", "content": json.dumps({"answer": t["answer"], "sources": t["sources"]})}]
    msgs.append({"role": "user", "content": question})
    if errors:
        msgs += [{"role": "assistant", "content": json.dumps(previous or {})},
                 {"role": "user", "content": "The checker rejected that answer:\n- " + "\n- ".join(errors) + "\nAnswer again as JSON, fixing every point."}]
    return msgs


def check_answer(raw: dict, outputs: dict) -> list[str]:
    errors = []
    answer, sources = raw.get("answer"), raw.get("sources")
    if not isinstance(answer, str) or not answer.strip():
        return ['"answer" must be a non-empty string']
    if not isinstance(sources, list):
        return ['"sources" must be a list of source_ids']
    unknown = [s for s in sources if s not in outputs]
    if unknown:
        errors.append(f"unknown sources {unknown}; use only {sorted(outputs)}")
    pool = set().union(*(numbers_in_output(outputs[s]) for s in sources if s in outputs)) if sources else set()
    bad = [v for v, d in numbers_in_text(answer) if not supported(v, d, pool)]
    if bad:
        errors.append(f"the answer states {', '.join(f'{v:g}' for v in bad)}, not found in the cited sources: cite the source or remove the number")
    return errors


def ask(question: str, account: str, toolbox: Toolbox, client: ChatClient | None = None, turns: list[dict] | None = None) -> dict:
    """One verified answer: {question, answer, sources, verified, retried, model, latency_s}. At most 2 LLM calls."""
    client = client or ChatClient()
    turns = turns or []
    outputs = toolbox.gather(account)
    t0, raw, errors, model = time.time(), None, None, None
    for attempt in (1, 2):
        try:
            raw, meta = client.chat_json(_messages(account, outputs, question, turns, errors, raw), max_tokens=2000)
        except LLMError as e:
            return {"question": question, "answer": f"The language model is unavailable right now ({str(e)[:120]}).", "sources": [],
                    "verified": False, "retried": attempt > 1, "model": None, "latency_s": round(time.time() - t0, 1)}
        model = meta.get("model")
        errors = check_answer(raw, outputs)
        if not errors:
            return {"question": question, "answer": raw["answer"].strip(), "sources": raw["sources"], "verified": True,
                    "retried": attempt > 1, "model": model, "latency_s": round(time.time() - t0, 1)}
    return {"question": question, "answer": "I could not answer that with figures I can verify from this account's data. Try asking it differently, "
            "or look at the network and payment history above.", "sources": [], "verified": False, "retried": True, "model": model,
            "latency_s": round(time.time() - t0, 1), "rejected": errors}


def preview_answer(account: str, toolbox: Toolbox) -> dict:
    """A sample exchange written straight from the tool outputs (no LLM, no quota): shows a visitor what answers look like."""
    out = toolbox.gather(account)
    a, g, d = out[f"alert:{account}"], out[f"subgraph:{account}"], out[f"shap:{account}"]
    blend = a["models"]["blend"]
    ups = [x for x in d["drivers"] if (x["push"] or 0) > 0][:2]
    why = " and ".join(f'"{x["factor"]}" ({x["push"]:+.2f})' for x in ups) or "no single strong factor"
    text = (f"It ranks {blend['rank']} of {a['accounts_in_queue']} under the blend model, with {blend['flagged_payments']} flagged "
            f"payment{'s' if blend['flagged_payments'] != 1 else ''}. The strongest pushes on its score come from {why}. "
            f"Around it, {g['fan_in_distinct_senders']} different accounts pay it and it pays {g['fan_out_distinct_receivers']}.")
    return {"question": EXAMPLE_QUESTIONS[0], "answer": text, "sources": [a["source_id"], d["source_id"], g["source_id"]]}
