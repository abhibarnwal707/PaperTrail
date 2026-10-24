"""The home page: a designed, full-bleed landing page (app/landing/home.html) filled with the demo bundle's real numbers.

Pure string work (no streamlit import, so it is unit-tested). The page is rendered straight into the app's DOM with
st.html (not an iframe), so its links are ordinary ?page=... links like the rest of the console and keep working; its
switches are CSS (checkbox / radio + :has), with a small optional script for entrances and counters. The images are
inlined once each as CSS data URIs, so nothing needs static file serving on Streamlit Cloud.

The template has two parts split at <!--SPLIT-->: everything above the "open a case" form (which is native Streamlit
widgets, rendered by the app between the two) and the footer.
"""
from __future__ import annotations

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent / "landing"
MODEL_ORDER = ("blend", "xgb", "gnn")
HOME_CSS = """
.block-container { max-width: none !important; padding: 0 !important; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
.mobile-nav { padding: 12px 16px 0 16px; }
.st-key-lp-cta { background: #93f1f7; color: #000; padding: clamp(56px, 7vw, 120px) clamp(18px, 4vw, 56px) !important; }
.st-key-lp-dlg { background: var(--panel); border-radius: 28px; padding: 28px !important; box-shadow: 0 30px 70px rgba(0,0,0,.25); }
.st-key-lp-dlg button[kind="primary"] { border-radius: 999px !important; min-height: 52px; background: var(--lime) !important; border-color: var(--lime) !important; color: var(--onbrand) !important; }
.st-key-lp-dlg button[kind="primary"] p { font-weight: 700 !important; font-size: 16px !important; color: var(--onbrand) !important; }
.st-key-lp-dlg button[kind="primary"]:hover { filter: brightness(1.08); transform: translateY(-1px); }
"""


def _data_uri(name: str) -> str:
    raw = (HERE / "img" / f"{name}.jpg").read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def _vals(models: dict) -> str:
    out = []
    for m in MODEL_ORDER:
        x = models[m]
        out.append(f'<div class="lp-vals lp-v-{m}"><div class="lp-val big"><span>pr-auc</span><b>{x["pr_auc"]:.3f}</b></div>'
                   f'<div class="lp-val"><span>found at 1% false alarms</span><b>{x["recall_at_1pct_fpr"] * 100:.1f}%</b></div>'
                   f'<div class="lp-val"><span>precision, top 1,000</span><b>{x["precision_at_1000"] * 100:.1f}%</b></div></div>')
    return "".join(out)


def _blocks(word: str = "papertrail") -> str:
    heights = (78, 100, 70, 92, 64, 86, 100, 74, 94, 68)
    return "".join(f'<span style="height:{heights[i % len(heights)]}%">{c}</span>' for i, c in enumerate(word))


def build(metrics: dict, hrefs: dict[str, str], theme: str = "dark", theme_switch: tuple[str, str] = ("", "")) -> tuple[str, str]:
    """(top, footer) HTML. `metrics` is data/demo/metrics.json; `hrefs` maps home/queue/top/try/models/about/theme to links;
    `theme_switch` is (icon svg, title) for the light/dark link."""
    page = (HERE / "home.html").read_text(encoding="utf-8")
    bl, overlap = metrics["models"]["blend"], metrics.get("overlap_at_1pct_fpr") or {}
    ntest, npos = metrics["n_test_transactions"], metrics["n_test_positives"]
    fills = {
        "__THEME__": "light" if theme == "light" else "",
        "__THEME_ICON__": theme_switch[0],
        "__THEME_TITLE__": theme_switch[1],
        "__RATE__": f"{ntest / npos:,.0f}",
        "__NPOS__": f"{npos:,}",
        "__NTEST__": f"{ntest:,}",
        "__BL_PR__": f"{bl['pr_auc']:.3f}",
        "__RC_PCT__": f"{bl['recall_at_1pct_fpr'] * 100:.0f}",
        "__GNN_ONLY__": str(overlap.get("gnn_only", 0)),
        "__VALS__": _vals(metrics["models"]),
        "__BLOCKS__": _blocks(),
    }
    fills.update({f"__H_{k.upper()}__": v for k, v in hrefs.items()})
    for k, v in fills.items():
        page = page.replace(k, v)
    for img in sorted((HERE / "img").glob("*.jpg")):                 # each image once, in the stylesheet
        page = page.replace(f"__IMG_{img.stem}__", _data_uri(img.stem))
    top, foot = page.split("<!--SPLIT-->")
    return top, foot
