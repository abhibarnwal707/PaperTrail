"""Restore the previous production model version in the registry (and log the decision to MLflow).

    python scripts/rollback.py
"""
from papertrail.lifecycle.registry import Registry

if __name__ == "__main__":
    reg = Registry()
    retired, restored = reg.rollback()
    reg.log_decision("rollback", {"retired_version": retired, "restored_version": restored})
    print(f"rolled back: production is now v{restored} (v{retired} retired)")
