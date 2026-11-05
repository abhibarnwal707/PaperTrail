"""A deterministic, no-LLM baseline for the investigator: what you get from the tool outputs with fixed rules and a template.

It exists to answer "what does the LLM add?". Its pieces:
- `rule_typology`: a few thresholds on the subgraph counts. Thresholds were fixed up front (not tuned on the evaluation cases) and it can only
  name CYCLE, FAN-IN, FAN-OUT, GATHER-SCATTER; it never names SCATTER-GATHER, STACK, BIPARTITE or RANDOM, so it cannot get those right.
- `policy_action`: the escalation rule written in the agent's prompt (prompts.py, rule 6), applied in code. Where the prompt is vague
  ("looks ordinary", "rests on a single factor") the code uses the nearest checkable condition, noted below.
- `template_summary`: a CaseSummary built from the alert and SHAP outputs; every number is copied from a tool output, so it passes the guardrail
  by construction.
"""
from __future__ import annotations

from papertrail.agent.schemas import CaseSummary

FAN_MIN, FAN_MAX_OTHER = 5, 2          # a-priori thresholds: "many" = 5 or more distinct counterparties, "few" = 2 or fewer
TOP_RANK = 50


def rule_typology(sub: dict) -> str:
    fan_out, fan_in = sub["fan_out_distinct_receivers"], sub["fan_in_distinct_senders"]
    if sub["two_step_cycles"] > 0 or sub["three_step_cycles"] > 0:
        return "CYCLE"
    if fan_in >= FAN_MIN and fan_out >= FAN_MIN:
        return "GATHER-SCATTER"
    if fan_in >= FAN_MIN and fan_out <= FAN_MAX_OTHER:
        return "FAN-IN"
    if fan_out >= FAN_MIN and fan_in <= FAN_MAX_OTHER:
        return "FAN-OUT"
    return "UNCLEAR"


def policy_action(alert: dict, typology: str) -> str:
    """Prompt rule 6 in code. escalate: >= 2 models flag a payment, blend rank in the top 50, and a named pattern.
    close: exactly one model flags and no pattern is named (stands in for "network looks ordinary, score rests on one factor")."""
    flags = sum(m["flagged_payments"] > 0 for m in alert["models"].values())
    if flags >= 2 and alert["models"]["blend"]["rank"] <= TOP_RANK and typology != "UNCLEAR":
        return "escalate_SAR"
    if flags == 1 and typology == "UNCLEAR":
        return "close_false_positive"
    return "enhanced_review"


def template_summary(account: str, outputs: dict[str, dict]) -> dict:
    alert, shap = outputs[f"alert:{account}"], outputs[f"shap:{account}"]
    blend = alert["models"]["blend"]
    typ = rule_typology(outputs[f"subgraph:{account}"])
    evidence = [{"claim": f"Ranked {blend['rank']} of {alert['accounts_in_queue']} by the blend model", "source_id": alert["source_id"]},
                {"claim": f"{blend['flagged_payments']} payments flagged by the blend model", "source_id": alert["source_id"]}]
    for d in shap["drivers"][:3]:
        evidence.append({"claim": f"{d['factor']} {d['direction']} (push {d['push']})", "source_id": shap["source_id"]})
    drivers = ", ".join(d["factor"] for d in shap["drivers"][:3]) or "no recorded drivers"
    narrative = f"The account ranks {blend['rank']} in the alert queue. The largest score drivers are: {drivers}. Pattern by simple rules: {typ}."
    return CaseSummary(account_id=account, risk_score=blend["risk_score"], suspected_typology=typ, evidence=evidence, narrative=narrative,
                       recommended_action=policy_action(alert, typ)).model_dump()
