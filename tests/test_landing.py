"""The landing page fills every placeholder from the demo bundle, links into the console and inlines each image once."""
import json
from pathlib import Path

import pytest

from app import landing

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"
HREFS = {k: f"?page={k}" for k in ("home", "queue", "top", "try", "models", "about", "theme")}


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_build_fills_everything(theme):
    metrics = json.loads((DEMO / "metrics.json").read_text())
    top, foot = landing.build(metrics, HREFS, theme, ("<svg></svg>", "Switch"))
    page = top + foot
    assert "__" not in page
    assert page.count("data:image/jpeg;base64,") == 4
    assert f"{metrics['models']['blend']['pr_auc']:.3f}" in top
    assert all(f'href="{h}"' in page for h in HREFS.values())
    assert ('class="lp light"' in top) == (theme == "light")
    assert "<iframe" not in page and "top.location" not in page          # plain links: an iframe cannot navigate the app
