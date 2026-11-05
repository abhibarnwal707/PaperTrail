from pathlib import Path

import pytest

from papertrail.agent.baseline import policy_action, rule_typology, template_summary
from papertrail.agent.schemas import CaseSummary
from papertrail.agent.verify import check


def _sub(out=0, inn=0, c2=0, c3=0):
    return {"fan_out_distinct_receivers": out, "fan_in_distinct_senders": inn, "two_step_cycles": c2, "three_step_cycles": c3}


@pytest.mark.parametrize("sub,expected", [(_sub(c2=1), "CYCLE"), (_sub(inn=6, out=6), "GATHER-SCATTER"), (_sub(inn=6, out=1), "FAN-IN"),
                                          (_sub(out=7, inn=0), "FAN-OUT"), (_sub(out=3, inn=3), "UNCLEAR"), (_sub(out=5, inn=3), "UNCLEAR")])
def test_rule_typology(sub, expected):
    assert rule_typology(sub) == expected


def _alert(flags=(1, 1, 0), rank=10):
    return {"models": {m: {"flagged_payments": f, "rank": rank, "risk_score": 3.0} for m, f in zip(("blend", "xgb", "gnn"), flags)}}


@pytest.mark.parametrize("flags,rank,typ,expected", [((1, 1, 0), 10, "CYCLE", "escalate_SAR"), ((1, 1, 0), 60, "CYCLE", "enhanced_review"),
                                                     ((1, 1, 0), 10, "UNCLEAR", "enhanced_review"), ((1, 0, 0), 10, "UNCLEAR", "close_false_positive"),
                                                     ((1, 0, 0), 10, "CYCLE", "enhanced_review")])
def test_policy_action(flags, rank, typ, expected):
    assert policy_action(_alert(flags, rank), typ) == expected


def test_template_summary_passes_the_guardrail_on_every_demo_account():
    demo = Path(__file__).resolve().parents[1] / "data" / "demo"
    if not (demo / "alerts.parquet").exists():
        pytest.skip("demo bundle not installed")
    from papertrail.agent.tools import Toolbox
    tb = Toolbox(demo)
    for acct in sorted(tb.accounts):
        outs = tb.gather(acct)
        assert check(CaseSummary.model_validate(template_summary(acct, outs)), outs, acct) == [], acct
