"""The landing page fills every placeholder from the demo bundle and inlines its images."""
import json
from pathlib import Path

from app import landing

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"


def test_build_has_no_placeholders_left():
    metrics = json.loads((DEMO / "metrics.json").read_text())
    page = landing.build(metrics, {"blend": "1_A", "xgb": "2_B", "gnn": "3_C"})
    assert "__" not in page.replace("__proto__", "")
    assert page.count("data:image/jpeg;base64,") == 4 and "data-img=" in page
    assert f"{metrics['models']['blend']['pr_auc']:.3f}" in page
    assert "1_A" in page
