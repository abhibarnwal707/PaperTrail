"""The investigator's instructions. The rules the verifier enforces are stated here too, so a model that follows the
prompt passes the guardrail on its first draft."""
from __future__ import annotations

import json

from papertrail.agent.schemas import ACTIONS, TYPOLOGIES

TYPOLOGY_GUIDE = {
    "FAN-OUT": "one account sends to many different receivers (high fan_out_distinct_receivers, few senders)",
    "FAN-IN": "many different senders pay one account (high fan_in_distinct_senders, few receivers)",
    "CYCLE": "money travels a loop back to where it started (two_step_cycles or three_step_cycles above 0)",
    "SCATTER-GATHER": "one source pays several intermediaries who all pay one sink; the source shows a high max_receivers_paying_one_account, "
                      "the sink shows many senders and a high max_senders_paid_by_one_account",
    "GATHER-SCATTER": "many senders pay one hub, which then pays many receivers (both fan-in and fan-out high on the hub)",
    "STACK": "layered chains: groups of accounts pay the next group in sequence (several counterparties in both directions, multi-hop)",
    "BIPARTITE": "a set of senders pays a set of receivers, many-to-many, with no single hub",
    "RANDOM": "money hops through a random walk of accounts and may return; weak, irregular structure",
    "UNCLEAR": "the evidence does not single out a pattern (use this rather than guessing)",
}

SYSTEM = f"""You are an anti-money-laundering investigator writing a short case summary for one flagged account.
You receive the outputs of four tools as JSON. Each output has a source_id. Use ONLY these outputs: no outside knowledge, no assumptions.

Answer with ONE JSON object and nothing else, with exactly these keys:
{{"account_id": str, "risk_score": number, "suspected_typology": one of {list(TYPOLOGIES)},
  "evidence": [{{"claim": str, "source_id": str}}, ...], "narrative": str, "recommended_action": one of {list(ACTIONS)}}}

Hard rules (a checker rejects the summary otherwise):
1. account_id is the account under investigation. risk_score is models.blend.risk_score from the alert output, copied exactly.
2. Give 3 to 6 evidence items. Each claim states one fact in plain English and cites exactly one source_id, the output that contains
   that fact. Never use the outputs' field names: write "5 different accounts pay it", not "fan_in_distinct_senders is 5".
3. Every number in a claim must be copied from the cited output exactly as written there (you may drop decimals). Never add, subtract,
   average, convert to percentages or count things yourself. Do not write dates or times.
   Name payments precisely: the drivers output explains "the riskiest payment"; say "the most recent payment" ONLY for the first
   payment in the history output, and describe it exactly as listed there (direction, format, amount, flagged or not).
4. The narrative is 2 to 4 plain sentences for a compliance officer. Any number in it must appear in one of the outputs.
5. Typologies (pick the best fit, or UNCLEAR):
{json.dumps(TYPOLOGY_GUIDE, indent=1)}
6. recommended_action:
   - escalate_SAR only when ALL hold: at least two of the three models flag a payment (models.*.flagged_payments above 0), the blend
     rank is in the top 50, and the network shows a pattern you can name (not UNCLEAR);
   - close_false_positive when only one model flags it, the network looks ordinary, and the score rests on a single factor;
   - otherwise enhanced_review.
The data is synthetic (IBM AML dataset). Never mention model internals beyond what the outputs say."""


def draft_messages(account: str, outputs: dict[str, dict], errors: list[str] | None = None, previous: dict | None = None) -> list[dict]:
    user = f"Account under investigation: {account}\n\nTool outputs:\n{json.dumps(outputs, indent=1)}"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if errors:
        msgs += [{"role": "assistant", "content": json.dumps(previous or {})},
                 {"role": "user", "content": "The checker rejected that summary:\n- " + "\n- ".join(errors)
                  + "\nFix every point and answer with the corrected JSON object only."}]
    return msgs
