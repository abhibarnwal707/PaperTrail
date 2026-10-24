"""The home page: a designed, full-bleed landing page (app/landing/index.html) filled with the demo bundle's real numbers.

Pure string work (no streamlit import, so it is unit-tested). The HTML is embedded in an iframe that resizes itself to its
content; its images are inlined as data URIs so the page needs no static file serving on Streamlit Cloud.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "landing"
HOME_CSS = """
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"], .mobile-nav { display: none !important; }
.block-container { max-width: none !important; padding: 0 !important; }
.stApp, [data-testid="stAppViewContainer"] { background: #000 !important; }
iframe[title="st.iframe"], [data-testid="stIFrame"], iframe { border: 0 !important; display: block; width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
"""


def _data_uri(name: str) -> str:
    raw = (HERE / "img" / f"{name}.jpg").read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def build(metrics: dict, top_accounts: dict[str, str]) -> str:
    """The page's HTML. `metrics` is data/demo/metrics.json; `top_accounts` maps model -> its rank-1 account."""
    page = (HERE / "index.html").read_text(encoding="utf-8")
    bl, overlap = metrics["models"]["blend"], metrics.get("overlap_at_1pct_fpr") or {}
    data = {"models": {k: metrics["models"][k] for k in ("blend", "xgb", "gnn")}, "top": top_accounts["blend"], "tops": top_accounts}
    ntest, npos = metrics["n_test_transactions"], metrics["n_test_positives"]
    fills = {
        "__RATE__": f"{ntest / npos:,.0f}",
        "__NPOS__": f"{npos:,}",
        "__NTEST__": f"{ntest:,}",
        "__BL_PR__": f"{bl['pr_auc']:.3f}",
        "__RC_PCT__": f"{bl['recall_at_1pct_fpr'] * 100:.0f}",
        "__GNN_ONLY__": str(overlap.get("gnn_only", 0)),
        "__TOP_ACCOUNT__": top_accounts["blend"],
        "__DATA__": json.dumps(data).replace("</", "<\\/"),
    }
    for k, v in fills.items():
        page = page.replace(k, v)
    imgs = {img.stem: _data_uri(img.stem) for img in sorted((HERE / "img").glob("*.jpg"))}      # each image once, however often the page shows it
    return page.replace("__IMGS__", json.dumps(imgs))
