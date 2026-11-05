"""Re-execute a recorded investigation from its recorded tool outputs and LLM replies (no live data, no API key).

    PAPERTRAIL_TRACING=jsonl python -c "..."      # record: any investigate() call writes data/local/traces/<run_id>.jsonl
    python scripts/replay.py <run_id>             # replay; exit 0 if the result matches the recording, 1 if it differs
    python scripts/replay.py --list               # recorded runs
"""
import argparse
import json
import os
import sys
from pathlib import Path

from papertrail.agent.tracing import DEFAULT_DIR, replay

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    d = Path(os.environ.get("PAPERTRAIL_TRACE_DIR", DEFAULT_DIR))
    if a.list or not a.run_id:
        print("\n".join(sorted(p.stem for p in d.glob("*.jsonl"))) or f"no traces in {d}")
        sys.exit(0)
    out = replay(a.run_id, d)
    s = out["result"]["summary"]
    print(f"replayed {a.run_id}: typology={s['suspected_typology']} verified={out['result']['verified']} drafts={out['result']['drafts']}")
    print("reproduced the recording" if not out["diffs"] else f"DIFFERS from the recording in: {out['diffs']}")
    if out["diffs"]:
        print(json.dumps({k: {"recorded": out["recorded"].get(k), "replayed": out["result"].get(k)} for k in out["diffs"]}, indent=1, default=str)[:3000])
    sys.exit(1 if out["diffs"] else 0)
