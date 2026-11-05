"""Write the registry's production model into a directory the API can load (meta.json + boosters).

    python scripts/export_production.py data/local/production_model
    PAPERTRAIL_MODEL_DIR=data/local/production_model uvicorn api.main:app

The API itself never imports mlflow; if the directory is missing or broken it serves the packaged model in data/demo/model.
"""
import sys
from pathlib import Path

from papertrail.lifecycle.registry import PRODUCTION, Registry

if __name__ == "__main__":
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "data/local/production_model")
    reg = Registry()
    v = reg.version_of(PRODUCTION)
    if v is None:
        sys.exit("no production version registered; run scripts/promote.py first")
    print(f"exported production v{v} to {reg.materialise(v, dest)}")
