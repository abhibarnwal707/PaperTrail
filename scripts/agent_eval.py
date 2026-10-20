"""Phase 9.4: run the investigator on 30 showcase accounts, cache the summaries for the app, and write the evaluation.

    python scripts/agent_eval.py              # resumes: accounts already in the cache are skipped
    python scripts/agent_eval.py --force      # re-run everything

Needs network and a key in .env or the environment (LLM_API_KEY for Groq, OPENROUTER_API_KEY as the fallback).
Writes data/demo/cached_cases.json (read by the app), reports/agent_eval.md + .json (the table), and
reports/agent_cases_review.md (every claim next to the tool value it cites, for the hand check).
The ground truth (alerts.typologies_truth) is read here only to score the agent; the tools never show it to the agent.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from papertrail.agent.graph import investigate
from papertrail.agent.llm import ChatClient, load_env
from papertrail.agent.tools import Toolbox
from papertrail.agent.verify import numbers_in_output, numbers_in_text, supported

DEMO, REPORTS = ROOT / "data" / "demo", ROOT / "reports"
CACHE = DEMO / "cached_cases.json"
PER_TYPOLOGY, FALSE_POSITIVES = 3, 6


def pick_accounts(alerts: pl.DataFrame) -> list[str]:
    """3 laundering accounts per known pattern (the highest ranked), plus 6 clean accounts the models flag: 30 in all."""
    a = alerts.sort("blend_rank")
    chosen: list[str] = []
    for typ in sorted({t for ts in a["typologies_truth"].drop_nulls().to_list() for t in ts.split("; ") if t}):
        hit = a.filter(pl.col("typologies_truth").str.contains(typ, literal=True) & ~pl.col("account").is_in(chosen))
        chosen += hit.head(PER_TYPOLOGY)["account"].to_list()
    chosen += a.filter((pl.col("n_laundering") == 0) & ~pl.col("account").is_in(chosen)).head(FALSE_POSITIVES)["account"].to_list()
    return chosen


def claim_stats(result: dict, outputs: dict) -> tuple[int, int]:
    """(claims whose numbers all check out, all claims) over the FIRST draft: the model's raw faithfulness, before the guardrail."""
    first = next((h["raw"] for h in result.get("drafts_log", []) if isinstance(h.get("raw"), dict)), None)
    ev = (first or {}).get("evidence") or []
    pools = {sid: numbers_in_output(o) for sid, o in outputs.items()}
    ok = sum(1 for e in ev if isinstance(e, dict) and e.get("source_id") in pools
             and all(supported(v, d, pools[e["source_id"]]) for v, d in numbers_in_text(str(e.get("claim", "")))))
    return ok, len(ev)


def recent_payment_errors(summary: dict, history: dict) -> list[str]:
    """Meaning check the number guardrail cannot do: every sentence about "the most recent payment" must match the first
    payment in the history output (direction, format, flagged, amount)."""
    if not history["payments"]:
        return []
    first, errs = history["payments"][0], []
    texts = [e["claim"] for e in summary["evidence"]] + re.split(r"(?<=[.!?])\s+", summary["narrative"])
    for t in texts:
        low = t.lower()
        if "most recent payment" not in low:
            continue
        probs = []
        if first["format"] not in t and any(f in t for f in ("ACH", "Wire", "Cheque", "Cash", "Credit Card", "Bitcoin", "Reinvestment")):
            probs.append(f"format is {first['format']}")
        if "flagged" in low and "not flagged" not in low and "unflagged" not in low and not first["flagged_by_blend"]:
            probs.append("it was not flagged")
        if ("incoming" in low or "received" in low) and first["direction"] == "sent":
            probs.append("it was sent")
        if ("outgoing" in low or " sent " in f" {low} ") and first["direction"] == "received":
            probs.append("it was received")
        amounts = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.\d{2}\b", t)]
        if amounts and first["amount"] is not None and all(abs(v - first["amount"]) > 0.5 for v in amounts):
            probs.append(f"its amount is {first['amount']}")
        if probs:
            errs.append(f"{t[:120]} -> {', '.join(probs)}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between cases (free-tier rate limits)")
    args = ap.parse_args()
    load_env(ROOT / ".env")
    tb = Toolbox(DEMO)
    client = ChatClient()
    print("providers:", [p.name for p in client.providers] or "NONE (set LLM_API_KEY / OPENROUTER_API_KEY)")
    cache = {} if args.force or not CACHE.exists() else json.loads(CACHE.read_text())
    truth = {r["account"]: r for r in tb.alerts.select("account", "typologies_truth", "n_laundering", "tag").to_dicts()}
    accounts = pick_accounts(tb.alerts)
    print(f"{len(accounts)} accounts; {sum(a in cache for a in accounts)} already cached")

    for i, acct in enumerate(accounts, 1):
        if acct in cache:
            continue
        t0 = time.time()
        res = investigate(acct, tb, client)
        cache[acct] = res
        CACHE.write_text(json.dumps(cache, indent=1))
        s = res["summary"]
        print(f"[{i:2d}/{len(accounts)}] {acct:18s} {s['suspected_typology']:15s} {s['recommended_action']:21s} verified={res['verified']} "
              f"drafts={res['drafts']} {res['provider']}/{res['model']} {time.time() - t0:.1f}s | truth: {truth[acct]['typologies_truth'] or 'clean'}")
        time.sleep(args.pause)

    rows = []
    for acct in accounts:
        res, t = cache[acct], truth[acct]
        s = res["summary"]
        true = [x for x in (t["typologies_truth"] or "").split("; ") if x]
        outs = tb.gather(acct)
        ok, n = claim_stats(res, outs)
        meaning = recent_payment_errors(s, outs[f"history:{acct}"])
        rows.append({"account": acct, "truth": "; ".join(true) or "clean", "laundering": bool(t["n_laundering"]),
                     "predicted": s["suspected_typology"], "typology_correct": s["suspected_typology"] in true if true else None,
                     "action": s["recommended_action"], "verified": res["verified"], "retried": res["retried"], "drafts": res["drafts"],
                     "first_draft_claims_ok": ok, "first_draft_claims": n, "final_claims": len(s["evidence"]), "meaning_errors": meaning,
                     "latency_s": res.get("total_latency_s"), "tokens": res.get("tokens"), "provider": res.get("provider"), "model": res.get("model")})
    df = pl.DataFrame(rows)
    pos, neg = df.filter(pl.col("laundering")), df.filter(~pl.col("laundering"))
    claims_ok, claims = int(df["first_draft_claims_ok"].sum()), int(df["first_draft_claims"].sum())
    lat = [x for x in df["latency_s"].to_list() if x is not None]
    toks = [x for x in df["tokens"].to_list() if x]
    summary = {
        "cases": df.height, "verified_share": float(df["verified"].mean()),
        "first_draft_factual_accuracy": claims_ok / claims if claims else None, "first_draft_claims": claims,
        "meaning_errors": int(df["meaning_errors"].list.len().sum()), "cases_with_meaning_errors": int((df["meaning_errors"].list.len() > 0).sum()),
        "retries": int(df["retried"].sum()), "fallbacks_to_unclear": int((~df["verified"]).sum()),
        "typology_accuracy": float(pos["typology_correct"].mean()) if pos.height else None,
        "typology_abstained": int((pos["predicted"] == "UNCLEAR").sum()), "laundering_cases": pos.height,
        "clean_cases": neg.height, "clean_escalated": int((neg["action"] == "escalate_SAR").sum()),
        "laundering_escalated": int((pos["action"] == "escalate_SAR").sum()),
        "latency_mean_s": statistics.mean(lat) if lat else None, "latency_p95_s": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None,
        "tokens_mean": statistics.mean(toks) if toks else None,
        "models_used": df.group_by("provider", "model").len().sort("len", descending=True).to_dicts(),
    }
    REPORTS.joinpath("agent_eval.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=1, default=str))
    write_markdown(summary, df)
    write_review(accounts, cache, tb, truth)
    print(json.dumps(summary, indent=1, default=str))


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def write_markdown(s: dict, df: pl.DataFrame) -> None:
    used = ", ".join(f"{m['provider']}/{m['model']} ({m['len']})" for m in s["models_used"])
    lines = [
        "# Investigator agent evaluation (Phase 9.4)", "",
        (f"{s['cases']} showcase accounts: {s['laundering_cases']} laundering (3 per known pattern), {s['clean_cases']} clean accounts the models flagged. "
        "Generated by `scripts/agent_eval.py`; the ground truth is used only for scoring."), "",
        "| Metric | Value |", "|---|---|",
        f"| Summaries that passed the guardrail | {_pct(s['verified_share'])} |",
        f"| Factual accuracy of first drafts (claims whose numbers match the cited tool output) | {_pct(s['first_draft_factual_accuracy'])} of {s['first_draft_claims']} claims |",
        f"| Meaning errors about \"the most recent payment\" (automatic check; numbers right, wrong payment) | {s['meaning_errors']} in {s['cases_with_meaning_errors']} cases |",
        f"| Drafts rejected and retried (hallucinations caught) | {s['retries']} of {s['cases']} |",
        f"| Fell back to UNCLEAR after the retry | {s['fallbacks_to_unclear']} |",
        f"| Typology accuracy (laundering accounts) | {_pct(s['typology_accuracy'])} ({s['typology_abstained']} answered UNCLEAR) |",
        f"| Laundering accounts escalated to SAR | {s['laundering_escalated']} of {s['laundering_cases']} |",
        f"| Clean accounts escalated to SAR (false escalations) | {s['clean_escalated']} of {s['clean_cases']} |",
        f"| Latency per case, mean / p95 | {s['latency_mean_s'] or 0:.1f} s / {s['latency_p95_s'] or 0:.1f} s |",
        f"| Tokens per case, mean | {s['tokens_mean'] or 0:,.0f} |",
        f"| Models used | {used} |", "",
        ("Factual accuracy here is automatic (numbers checked against the cited source). The hand check of meaning, not just numbers, "
        "is recorded in `reports/agent_cases_review.md`."), "",
        "| Account | Truth | Predicted | Action | Verified | Drafts | Claims ok (1st draft) | Model |", "|---|---|---|---|---|---|---|---|",
    ]
    for r in df.to_dicts():
        lines.append(f"| `{r['account']}` | {r['truth']} | {r['predicted']} | {r['action']} | {'yes' if r['verified'] else 'no'} | {r['drafts']} | "
                     f"{r['first_draft_claims_ok']}/{r['first_draft_claims']} | {r['model']} |")
    REPORTS.joinpath("agent_eval.md").write_text("\n".join(lines) + "\n")


def write_review(accounts: list[str], cache: dict, tb: Toolbox, truth: dict) -> None:
    out = ["# Hand check of the 30 case summaries", "", "For each claim: is it true, and does the cited output support it? Mark the box.", ""]
    for acct in accounts:
        r, s = cache[acct], cache[acct]["summary"]
        out += [f"## `{acct}`  truth: {truth[acct]['typologies_truth'] or 'clean'}", "",
                f"Predicted **{s['suspected_typology']}**, action **{s['recommended_action']}**, verified {r['verified']}, drafts {r['drafts']}.", "",
                f"> {s['narrative']}", ""]
        out += [f"- [ ] {e['claim']}  (`{e['source_id']}`)" for e in s["evidence"]]
        out.append("")
    REPORTS.joinpath("agent_cases_review.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
