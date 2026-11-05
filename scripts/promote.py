"""Train a candidate on the demo slice, register it, and run the evaluation gate against the current production model.

    python scripts/promote.py            # train + register + gate (+ promote or reject)
    python scripts/promote.py --seed 1

Candidate and current model are both scored on the SAME held-out slice (last 20% in time), so the comparison is like for like.
The first model registered has nothing to be compared with, so only the leakage check applies. Every decision is logged to MLflow.
"""
import argparse
import sys

from papertrail.lifecycle.promotion import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(0 if run(ap.parse_args().seed)["promoted"] else 1)
