#!/usr/bin/env bash
# Run a notebook-style .py file on the Mac with the project venv.
# - points XGBoost at the OpenMP library bundled inside scikit-learn (no Homebrew needed)
# - caffeinate keeps the Mac awake until the script ends
# - output goes to the terminal and to logs/<name>.log
# Usage: scripts/run_local.sh notebooks/03c_stability.py      (PAPERTRAIL_QUICK=1 for a one-seed first look)
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${PAPERTRAIL_VENV:-$(cd .. && pwd)/.venv}"
export DYLD_FALLBACK_LIBRARY_PATH="$(ls -d "$VENV"/lib/python*/site-packages/sklearn/.dylibs | head -1)"
export PYTHONUNBUFFERED=1
mkdir -p logs
log="logs/$(basename "${1%.py}").log"
echo "running $1 -> $log"
# caffeinate is a protected system binary (it would strip DYLD_*), so it watches this script instead of wrapping python
caffeinate -i -w $$ &
"$VENV/bin/python" -u "$1" 2>&1 | tee "$log"
