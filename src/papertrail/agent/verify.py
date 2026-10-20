"""The guardrail: a summary is accepted only if every cited source exists and every number it states is in the cited
tool output (the narrative may use numbers from any tool output). Rounding is allowed (17,050.43 may be written 17,050),
computing is not (a sum or a percentage that no tool returned is an error)."""
from __future__ import annotations

import re

from papertrail.agent.schemas import CaseSummary

# a number not glued to a word or another number on its left: "2022-09-09" gives 2022, 9, 9; "8087F4940" gives 8087 only
_NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")


def numbers_in_text(text: str) -> list[tuple[float, int]]:
    """(value, decimals as written) for each number in a string."""
    out = []
    for m in _NUM.finditer(text):
        s = m.group().replace(",", "")
        out.append((float(s), len(s.split(".")[1]) if "." in s else 0))
    return out


def numbers_in_output(obj) -> set[float]:
    """Every number in a tool output: numeric values, and numbers written inside its strings (ids, times, labels)."""
    found: set[float] = set()

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            found.add(float(x))
        elif isinstance(x, str):
            found.update(v for v, _ in numbers_in_text(x))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)

    walk(obj)
    return found


def supported(value: float, decimals: int, pool: set[float]) -> bool:
    """True when `value`, written with `decimals` places, is a rounding of some number in `pool` (sign-insensitive)."""
    tol = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(abs(value) - abs(p)) <= tol for p in pool)


def check(summary: CaseSummary, outputs: dict[str, dict], account: str) -> list[str]:
    """All the reasons to reject the summary (empty list = accepted)."""
    errors = []
    if summary.account_id != account:
        errors.append(f"account_id is {summary.account_id!r}, the case is {account!r}")
    alert = outputs.get(f"alert:{account}")
    if alert and abs(summary.risk_score - alert["models"]["blend"]["risk_score"]) > 0.01:
        errors.append(f"risk_score {summary.risk_score} does not match the blend risk_score {alert['models']['blend']['risk_score']} in alert:{account}")
    if len(summary.evidence) < 2:
        errors.append("give at least 2 evidence items")
    pools = {sid: numbers_in_output(o) for sid, o in outputs.items()}
    everything = set().union(*pools.values()) if pools else set()
    for i, ev in enumerate(summary.evidence, 1):
        if ev.source_id not in outputs:
            errors.append(f"evidence {i} cites {ev.source_id!r}, which is not a tool output (use one of {sorted(outputs)})")
            continue
        bad = [v for v, d in numbers_in_text(ev.claim) if not supported(v, d, pools[ev.source_id])]
        if bad:
            errors.append(f"evidence {i} states {', '.join(_fmt(v) for v in bad)}, not found in {ev.source_id}: copy numbers exactly from that source")
    bad = [v for v, d in numbers_in_text(summary.narrative) if not supported(v, d, everything)]
    if bad:
        errors.append(f"the narrative states {', '.join(_fmt(v) for v in bad)}, which no tool returned: remove or copy exactly")
    return errors


def verified_evidence(summary: CaseSummary, outputs: dict[str, dict]) -> list:
    """The evidence items that pass on their own (used for the fallback summary)."""
    pools = {sid: numbers_in_output(o) for sid, o in outputs.items()}
    return [ev for ev in summary.evidence if ev.source_id in pools and all(supported(v, d, pools[ev.source_id]) for v, d in numbers_in_text(ev.claim))]


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v}"
