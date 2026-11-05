"""Train a candidate, register it, gate it against production, then promote or reject (used by scripts/promote.py and the drift loop)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from papertrail.lifecycle import data as D
from papertrail.lifecycle.config import LifecycleConfig
from papertrail.lifecycle.gate import check_point_in_time, evaluate_gate
from papertrail.lifecycle.model_card import render
from papertrail.lifecycle.registry import PRODUCTION, Registry

SLICE_NOTE = "Trained on the ~8k-row demo slice (laundering-enriched sample), not the 5.08M-row dataset; metrics here show the lifecycle, not model quality."


def run(seed: int = 0, txns=None, cfg: LifecycleConfig | None = None, label: str = "manual") -> dict:
    cfg = cfg or LifecycleConfig()
    reg = Registry(cfg)
    txns = txns if txns is not None else D.load_slice()
    frame = D.feature_frame(txns)
    _, _, test = D.temporal_thirds(frame)
    work = Path(tempfile.mkdtemp())
    res = D.train_model(frame, work / "cand", seed=seed, name=f"xgb_slice_seed{seed}")
    tags = {"data_hash": D.data_hash(txns), "git_sha": D.git_sha(), "feature_list_hash": D.feature_list_hash(D.feature_names()),
            "n_features": str(len(D.feature_names())), "trained_on": f"demo slice, {txns.height} rows", "eval_on": f"last 20% in time, {test.height} rows",
            "limitation": SLICE_NOTE, "trigger": label}
    version = reg.register(work / "cand", res["metrics"], tags, run_name=f"train_v_seed{seed}")

    cur_v = reg.version_of(PRODUCTION)
    current = D.evaluate_dir(reg.materialise(cur_v, work / "cur"), test) if cur_v else None
    leak_ok, leak_detail = check_point_in_time(txns, D.feature_names())
    decision = evaluate_gate(res["metrics"], current, cfg.tolerances, leak_ok, leak_detail)
    if decision.promote:
        reg.promote(version)
    else:
        reg.reject(version, decision.reasons)
    reg.log_decision("gate_decision", {"candidate_version": version, "current_version": cur_v or "none", "promoted": decision.promote, "trigger": label},
                     {f"cand_{k}": v for k, v in res["metrics"].items()}, decision.to_dict())
    print(render(reg.info(version), cfg.model_name))
    print(f"gate: {'PROMOTED' if decision.promote else 'REJECTED'} v{version} (current production: v{cur_v or 'none'})")
    for c in decision.checks:
        print(f"  [{'ok' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    return {"version": version, "promoted": decision.promote, "decision": decision.to_dict(), "candidate_metrics": res["metrics"], "current_metrics": current}


