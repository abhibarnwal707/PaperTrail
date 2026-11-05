"""One-page model card for a registered version, rendered from registry metadata only (nothing hand-typed)."""
from __future__ import annotations

import json


def render(info: dict, name: str) -> str:
    t = info["tags"]
    m = json.loads(t.get("metrics_json", "{}"))
    rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in m.items())
    return f"""# Model card: {name} v{info['version']}

Status: {', '.join(info['aliases']) or 'unassigned'}{' (rejected: ' + t['rejected_reasons'] + ')' if 'rejected_reasons' in t else ''}

## Data
- Trained on: {t.get('trained_on', 'n/a')}
- Data fingerprint: `{t.get('data_hash', 'n/a')}`; git commit `{t.get('git_sha', 'n/a')}`
- Features: {t.get('n_features', 'n/a')} (list fingerprint `{t.get('feature_list_hash', 'n/a')}`), point-in-time (each uses only earlier transactions)

## Held-out metrics ({t.get('eval_on', 'n/a')})
| metric | value |
|---|---|
{rows}

## Intended use
Ranking payments for analyst review in an anti-money-laundering queue. A score is a prioritisation signal, not a verdict.

## Limitations
- Trained on synthetic data (IBM AML); real-world laundering patterns and base rates differ.
- {t.get('limitation', 'See the training-scope note in docs/MODEL_LIFECYCLE.md.')}
- Calendar-dependent features (7-day windows, all-time totals, PageRank) drift by construction; see `drift.py`.
- Threshold is chosen on validation only; it must be re-chosen on any retrain.
"""
