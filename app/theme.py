"""Look and feel of the console: one stylesheet and the small HTML builders (no streamlit import, so it is unit-tested).

Style: the home page's "Candy on Black" system carried into every screen. Each page opens on a candy-coloured band with a
big lowercase title and its own artwork cropped in a circle; figures sit on flat candy tiles with one chamfered corner;
panels are flat and rounded (tone steps, not rules); buttons and switches are pills. Short copy, details in tooltips.
Two palettes, switched by ?theme= in the URL: "Candy on Black" (dark: true black, bright pastel candies) and
"Candy on Paper" (light: white stock, deeper candies; light pages need darker, more saturated accents than dark ones).
The token names are the colours' jobs in the dark palette, and every palette keeps the jobs:
- lime is the brand and the selection (and the blend's series);
- sky is XGBoost, violet is the GNN;
- coral means risk: flagged payments, risk-raising reasons, critical alerts.
The rules below use tokens only (tints via color-mix), so a palette is just the values on :root.
State lives in the URL (page, model, account, theme), so every view can be linked and the tables stay plain HTML.
"""
from __future__ import annotations

import base64
import functools
import html
from pathlib import Path
from urllib.parse import urlencode

PALETTES = {
    # Candy on Black: true black, bright pastel candies (light, vivid accents keep contrast on dark surfaces)
    "dark": {"name": "Candy on Black", "bg": "#000000", "side": "#050507", "panel": "#0d0d11", "panel2": "#17171d", "text": "#f2f0fa", "text2": "#cfcbdc",
             "muted": "#9a97a8", "faint": "#6a677a", "lime": "#93f1f7", "sky": "#b9c8ff", "violet": "#fbb65c", "coral": "#ff5c8a",
             "amber": "#f3ea7a", "mint": "#9ae187", "onbrand": "#000000", "watch": "#3b3948", "neutral": "#e9e6f5", "grid": "rgba(255,255,255,0.06)",
             "f1": "#9ae187", "f2": "#93f1f7", "f3": "#f3ea7a", "f4": "#fbb65c", "f5": "#b9c8ff", "onfill": "#000000", "edge": "transparent"},
    # Candy on Paper: white stock, near-black ink, and deeper versions of the dark palette's candies (light pages need darker,
    # more saturated accents than dark ones, or the series wash out)
    "light": {"name": "Candy on Paper", "bg": "#f7f7f4", "side": "#efefea", "panel": "#ffffff", "panel2": "#f3f3ef", "text": "#0b0b0f", "text2": "#2c2c33",
              "muted": "#5c5c66", "faint": "#8a8a94", "lime": "#0a8c96", "sky": "#3d55d9", "violet": "#d26a00", "coral": "#d6205c",
              "amber": "#a87c00", "mint": "#2b8a3e", "onbrand": "#ffffff", "watch": "#dcdce3", "neutral": "#2c2c33", "grid": "rgba(11,11,15,0.07)",
              "f1": "#9ae187", "f2": "#93f1f7", "f3": "#f3ea7a", "f4": "#fbb65c", "f5": "#b9c8ff", "onfill": "#000000", "edge": "rgba(11,11,15,0.07)"},
}
THEMES = tuple(PALETTES)
MODEL_KEY = {"blend": "lime", "xgb": "sky", "gnn": "violet"}
MODEL_VAR = {m: f"var(--{k})" for m, k in MODEL_KEY.items()}
FONT_LINKS = "@import url('https://api.fontshare.com/v2/css?f[]=pilcrow-rounded@400,500,700&display=swap');@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');"


def model_color(p: dict, model: str) -> str:
    return p[MODEL_KEY[model]]


def css(theme_name: str = "dark") -> str:
    """The stylesheet for one palette: the tokens on :root, then the rules (which use only tokens)."""
    p = PALETTES.get(theme_name, PALETTES["dark"])
    tokens = "".join(f"--{k}: {v}; " for k, v in p.items() if k not in ("name", "grid"))
    root = (f":root {{ {tokens}--line: color-mix(in srgb, var(--text) 8%, transparent); --line2: color-mix(in srgb, var(--text) 14%, transparent); "
            "--font: 'Pilcrow Rounded', 'Geist', Inter, system-ui, sans-serif; --mono: 'Geist Mono', ui-monospace, 'SF Mono', Menlo, monospace; --r: 22px; "
            f"color-scheme: {'light' if theme_name == 'light' else 'dark'}; }}")
    return FONT_LINKS + root + _RULES


_RULES = """
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg) !important; color: var(--text); font-family: var(--font); }
.stApp [data-testid="stMarkdownContainer"], [data-testid="stSidebar"], .brand b, [data-testid="stExpander"] summary { color: var(--text) !important; }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu, [data-testid="stStatusWidget"] { display: none !important; }
.block-container { max-width: 1440px; padding: 28px 40px 80px 40px !important; }
[data-testid="stHeadingWithActionElements"] a { display: none !important; }
p, li, label, span, div { font-size: 14px; }
.stApp, .stApp * { font-variant-numeric: tabular-nums; }
.stApp a, .stApp a:visited, .stApp a:hover { color: inherit !important; text-decoration: none !important; }
.stApp, .stApp p, .stApp div, .stApp span, .stApp label, .stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp button, .stApp a, .stApp summary { font-family: var(--font) !important; }
.stApp .mono, .stApp .mono *, .stApp td.num, .stApp td.id, .stApp td.id a, .stApp input, .stApp [data-baseweb="select"] div { font-family: var(--mono) !important; }
.stApp h1 span { font-size: inherit !important; font-weight: inherit !important; line-height: inherit !important; font-family: inherit !important; }
.stApp [data-testid="stIconMaterial"], .stApp [data-testid="stExpanderIcon"], .stApp span[class*="material"] { font-family: 'Material Symbols Rounded' !important; }
svg.i { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; flex: none; }
@keyframes pt-rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
@keyframes pt-open { from { clip-path: circle(0% at 50% 50%); transform: translateY(-50%) scale(1.08) rotate(-8deg); } to { clip-path: circle(75% at 50% 50%); transform: translateY(-50%) scale(1) rotate(0); } }
@keyframes pt-ping { 0% { transform: scale(.8); opacity: .9; } 100% { transform: scale(1.8); opacity: 0; } }
@media (prefers-reduced-motion: reduce) { .stApp *, .stApp *::before, .stApp *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--side) !important; border-right: 0 !important; min-width: 248px !important; max-width: 248px !important; width: 248px !important; }
[data-testid="stSidebar"] > div:first-child { padding: 0; }
[data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] { padding: 26px 0 0 0 !important; }
.brand { display: flex; align-items: center; gap: 12px; padding: 0 24px 30px 24px; }
.brand .mark { width: 26px; height: 26px; border-radius: 50%; background: var(--text); box-shadow: inset 0 0 0 7px var(--text), inset 0 0 0 20px var(--side); display: block; flex: none; }
.stApp .brand b { font-size: 26px; font-weight: 700; letter-spacing: -.05em; text-transform: lowercase; }
.nav { padding: 0 14px 14px 14px; }
.nav a { display: flex; align-items: center; gap: 12px; padding: 11px 16px; border-radius: 999px; font-size: 15px; line-height: 18px; margin-bottom: 3px; font-weight: 500; transition: background .2s, transform .2s; }
.stApp .nav a { color: var(--muted) !important; }
.stApp .nav a:hover { color: var(--text) !important; background: var(--panel2); transform: translateX(3px); }
.stApp .nav a.on { color: var(--onfill) !important; background: var(--f2); font-weight: 700; }
.nav a.on svg { color: var(--onfill); }
.nav.theme-sw { padding-top: 6px; }
.side-card { margin: 10px 16px 16px 16px; padding: 18px; border-radius: 24px; background: var(--panel); box-shadow: 0 0 0 1px var(--edge); }
.side-card .t { font-size: 13px; color: var(--muted); margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
.side-card .row { display: flex; justify-content: space-between; font-size: 13px; padding: 4px 0; color: var(--muted); }
.side-card .row b { color: var(--text); font-weight: 500; font-family: var(--mono); font-size: 12.5px; }
.live { width: 8px; height: 8px; border-radius: 50%; background: var(--lime); display: inline-block; position: relative; }
.live::after { content: ''; position: absolute; inset: 0; border-radius: 50%; border: 2px solid var(--lime); animation: pt-ping 2.2s ease-out infinite; }

/* ---------- header band: a candy field, a big lowercase title, and the page's artwork cropped in a circle ---------- */
.hdr { position: relative; overflow: hidden; display: flex; align-items: flex-end; min-height: 270px; margin: 0 0 22px 0; padding: 38px 44px 34px 44px; border-radius: 34px; background: var(--fill, var(--f2)); color: #000; animation: pt-rise .6s cubic-bezier(.2,.8,.2,1) both; }
.hdr .hl { position: relative; z-index: 2; max-width: calc(100% - 330px); display: flex; flex-direction: column; gap: 10px; }
.hdr .art { position: absolute; z-index: 1; right: -70px; top: 50%; width: 400px; height: 400px; border-radius: 50%; background-size: cover; background-position: center; background-color: #06060a;
  border: 14px solid #000; transform: translateY(-50%); animation: pt-open 1.2s .1s cubic-bezier(.2,.8,.2,1) both; transition: transform .6s cubic-bezier(.2,.8,.2,1); }
.hdr:hover .art { transform: translateY(-50%) scale(1.04) rotate(-3deg); }
.crumb { color: rgba(0,0,0,.65); font-size: 14px; display: flex; gap: 8px; align-items: center; font-weight: 500; }
.stApp .crumb a { color: #000 !important; text-decoration: underline !important; text-underline-offset: 3px; }
.stApp .hdr h1 { font-size: clamp(40px, 5vw, 76px) !important; font-weight: 700 !important; line-height: .95 !important; letter-spacing: -.05em; margin: 0 !important; padding: 0 !important; color: #000 !important; text-transform: lowercase; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.stApp .hdr h1.mono { text-transform: none; font-size: clamp(32px, 3.6vw, 54px) !important; letter-spacing: -.03em; font-family: var(--font) !important; }
.stApp .hdr h1.mono span { font-family: var(--font) !important; }
.stApp .hdr h1 .pill, .stApp .hdr h1 .pill * { font-family: var(--font) !important; font-size: 14px !important; font-weight: 700 !important; line-height: 1 !important; letter-spacing: 0 !important; }
.hdr .sub { color: rgba(0,0,0,.72); font-size: 17px; font-weight: 500; max-width: 60ch; }
.hdr .right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.seg { display: inline-flex; padding: 4px; background: #000; border-radius: 999px; gap: 2px; }
.seg a { display: inline-flex; align-items: center; gap: 8px; padding: 9px 16px; border-radius: 999px; font-size: 14px; font-weight: 700; transition: background .2s, color .2s; }
.stApp .seg a { color: #a9a9b3 !important; }
.stApp .seg a:hover { color: #fff !important; }
.stApp .seg a.on { color: #000 !important; background: #fff; }
.seg .sw { width: 9px; height: 9px; border-radius: 50%; }
.btn { display: inline-flex; align-items: center; gap: 8px; padding: 0 20px; height: 44px; border-radius: 999px; font-size: 14px; font-weight: 700; white-space: nowrap; transition: transform .2s, filter .2s; }
.btn:hover { transform: translateY(-2px); filter: brightness(1.08); }
.btn:active { transform: scale(.97); }
.stApp .btn.primary { background: #000; color: var(--fill, var(--f2)) !important; }
.stApp .btn.ghost { background: transparent; box-shadow: inset 0 0 0 2px rgba(0,0,0,.55); color: #000 !important; }
.btn.icon { width: 44px; padding: 0; justify-content: center; }
.btn svg.i { width: 17px; height: 17px; }

/* ---------- panels ---------- */
.card { background: var(--panel); box-shadow: 0 0 0 1px var(--edge); border-radius: var(--r); padding: 22px 24px; animation: pt-rise .6s cubic-bezier(.2,.8,.2,1) both; }
/* st.container(border=True) is a bordered stVerticalBlock in Streamlit 1.66; ours always open with a card head (.ch) */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ch) { background: var(--panel) !important; border: 0 !important; box-shadow: 0 0 0 1px var(--edge);
  border-radius: var(--r) !important; padding: 22px 24px 24px 24px !important; animation: pt-rise .6s cubic-bezier(.2,.8,.2,1) both; }
[data-testid="stHorizontalBlock"] { align-items: stretch !important; gap: 18px !important; }
[data-testid="stColumn"] > [data-testid="stVerticalBlock"], [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"],
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] { height: 100%; }
.ch { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; }
.stApp .ch .t { font-size: 22px; font-weight: 700; letter-spacing: -.035em; text-transform: lowercase; display: flex; align-items: center; gap: 10px; line-height: 1.1; }
.ch .t svg.i { width: 30px; height: 30px; padding: 7px; border-radius: 50%; background: var(--f2); color: #000; stroke-width: 2.2; }
.ch .s { color: var(--muted); font-size: 13.5px; margin-top: 4px; }
.ch .r { color: var(--muted); font-size: 13px; display: flex; gap: 10px; align-items: center; }
.tip { display: inline-grid; place-items: center; width: 18px; height: 18px; border-radius: 50%; background: var(--panel2); color: var(--muted); font-size: 11px; cursor: help; font-weight: 700; letter-spacing: 0; text-transform: none; }
.gap { height: 20px; }

/* ---------- figure tiles: flat candy fields with one chamfered corner, like the home page ---------- */
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 20px; }
.kpis.k3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.kpi { position: relative; overflow: hidden; background: var(--k) !important; color: #000; box-shadow: none !important; border-radius: 26px; padding: 22px 24px 20px 24px; min-height: 172px;
  clip-path: polygon(0 0, calc(100% - 44px) 0, 100% 44px, 100% 100%, 0 100%); transition: transform .3s cubic-bezier(.2,.8,.2,1); }
.kpi:hover { transform: translateY(-4px); }
.kpis .kpi:nth-child(4n+1) { --k: var(--f2); } .kpis .kpi:nth-child(4n+2) { --k: var(--f1); } .kpis .kpi:nth-child(4n+3) { --k: var(--f3); } .kpis .kpi:nth-child(4n+4) { --k: var(--f4); }
.kpis .kpi:nth-child(2) { animation-delay: .06s; } .kpis .kpi:nth-child(3) { animation-delay: .12s; } .kpis .kpi:nth-child(4) { animation-delay: .18s; }
.kpis.k3 .kpi:not(.on) { --k: var(--panel); color: var(--text); box-shadow: 0 0 0 1px var(--edge) !important; }
.kpis.k3 .kpi.on { --k: var(--f2); }
.kpi .top { display: flex; align-items: center; justify-content: space-between; color: inherit; font-size: 15px; font-weight: 500; }
.kpi .top > span:first-child { opacity: .78; }
.kpi .ic { width: 38px; height: 38px; border-radius: 50%; display: grid; place-items: center; background: #000; color: var(--k); opacity: 1; }
.kpis.k3 .kpi:not(.on) .ic { background: var(--panel2); color: var(--text); }
.kpi .ic svg.i { width: 17px; height: 17px; }
.kpi .v { font-size: 52px; line-height: 1; font-weight: 700; letter-spacing: -.055em; margin-top: 18px; }
.kpi .v small { font-size: 17px; opacity: .65; font-weight: 500; margin-left: 4px; letter-spacing: 0; }
.kpi .f { display: flex; align-items: center; gap: 8px; white-space: nowrap; margin-top: 10px; color: inherit; font-size: 13px; }
.kpi .f > span:not(.pill):not(.meter) { opacity: .72; }
.kpi .pill { background: #000 !important; border-color: #000 !important; }
.kpi .pill.grey { color: #fff; }
.kpi .bar { height: 8px; background: rgba(0,0,0,.14); border-radius: 8px; margin-top: 16px; overflow: hidden; }
.kpi .bar i { display: block; height: 100%; border-radius: 8px; background: #000 !important; }
.kpis.k3 .kpi:not(.on) .bar { background: var(--panel2); }
.kpis.k3 .kpi:not(.on) .bar i { background: var(--text) !important; }
.kpi .meter .m { background: rgba(0,0,0,.14); }
.kpi .meter .m i { background: #000; }
.kpi .meter span, .kpi .mono { color: inherit !important; opacity: .75; }
.kpis.k3 .kpi:not(.on) .meter .m { background: var(--panel2); } .kpis.k3 .kpi:not(.on) .meter .m i { background: var(--coral); }

/* ---------- pills and chips ---------- */
.pill { display: inline-flex; align-items: center; gap: 6px; height: 26px; padding: 0 11px; border-radius: 999px; font-size: 12.5px; font-weight: 700; white-space: nowrap; border: 0; }
.pill .d { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.pill.coral { color: var(--coral); background: color-mix(in srgb, var(--coral) 14%, transparent); }
.pill.amber { color: var(--amber); background: color-mix(in srgb, var(--amber) 14%, transparent); }
.pill.mint { color: var(--mint); background: color-mix(in srgb, var(--mint) 14%, transparent); }
.pill.lime { color: var(--lime); background: color-mix(in srgb, var(--lime) 14%, transparent); }
.pill.grey { color: var(--muted); background: var(--panel2); }
.pill.big { height: 34px; padding: 0 15px; font-size: 14px; }
.hdr .pill { background: #000 !important; }
.stApp .hdr h1 .pill { margin-left: 4px; }
.hdr .pill.grey { color: #fff; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { display: inline-flex; align-items: center; gap: 6px; height: 28px; padding: 0 12px; border-radius: 999px; font-size: 12.5px; background: var(--panel2); color: var(--text2); white-space: nowrap; cursor: default; font-weight: 500; }
.chip .a { font-size: 10px; color: var(--muted); }
.chip.up { background: color-mix(in srgb, var(--coral) 12%, var(--panel2)); }
.chip.up .a { color: var(--coral); }
.dots { display: inline-flex; gap: 5px; }
.dots i { width: 11px; height: 11px; border-radius: 50%; background: var(--panel2); display: inline-block; }
.meter { display: flex; align-items: center; gap: 10px; }
.meter .m { width: 70px; height: 7px; background: var(--panel2); border-radius: 7px; overflow: hidden; }
.meter .m i { display: block; height: 100%; background: var(--coral); border-radius: 7px; }
.meter span { font-family: var(--mono); font-size: 12.5px; color: var(--muted); }

/* ---------- tables: tone steps instead of rules ---------- */
.tbl-wrap { overflow: auto; max-height: 680px; margin: 0 -24px -24px -24px; }
table.tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px; }
.stApp .tbl th, .stApp .tbl td { border: 0 !important; }
.tbl th { position: sticky; top: 0; background: var(--panel); text-align: left; font-weight: 700; font-size: 13px; color: var(--muted); padding: 0 24px; height: 44px; white-space: nowrap; z-index: 1; }
.tbl td { padding: 0 24px; height: 58px; vertical-align: middle; white-space: nowrap; transition: background .15s; }
.tbl tbody tr:nth-child(odd) td { background: color-mix(in srgb, var(--text) 2.4%, transparent); }
.tbl th.num, .tbl td.num { text-align: right; }
.tbl tbody tr:hover td { background: color-mix(in srgb, var(--f2) 12%, transparent); }
.tbl td.id a { font-family: var(--mono); color: var(--text) !important; font-weight: 500; border-bottom: 2px solid transparent; transition: border-color .2s; }
.tbl td.id a:hover { color: var(--text) !important; border-bottom-color: var(--lime); }
.tbl td.dim { color: var(--muted); }
.tbl td.wrap { white-space: normal; line-height: 20px; padding-top: 12px; padding-bottom: 12px; color: var(--text2); }
.tbl .best { color: var(--lime); font-weight: 700; }
.tbl .rest { color: var(--text2); }
.dir { display: inline-grid; place-items: center; width: 30px; height: 30px; border-radius: 50%; background: var(--panel2); }
.dir.out { color: var(--muted); } .dir.in { color: #000; background: var(--f1); }
.amt-in { color: var(--mint); font-weight: 700; }
.rk { display: inline-grid; place-items: center; min-width: 34px; height: 28px; border-radius: 999px; background: var(--panel2); color: var(--text2); font-family: var(--mono); font-size: 12.5px; }
.pager { display: flex; justify-content: space-between; align-items: center; color: var(--muted); font-size: 14px; padding: 18px 4px 0 4px; }
.pager .pg { display: flex; gap: 8px; }
.pager a { padding: 10px 18px; border-radius: 999px; background: var(--f2); color: #000 !important; font-weight: 700; transition: transform .2s; }
.pager a:hover { transform: translateY(-2px); }
.toggle { display: inline-flex; align-items: center; gap: 10px; color: var(--text) !important; font-size: 14px; font-weight: 500; }
.toggle .sw { width: 44px; height: 24px; border-radius: 999px; background: var(--watch); position: relative; transition: background .2s; }
.toggle .sw::after { content: ''; position: absolute; top: 3px; left: 3px; width: 18px; height: 18px; border-radius: 50%; background: #fff; transition: left .25s cubic-bezier(.2,.8,.2,1); }
.toggle.on .sw { background: var(--f1); } .toggle.on .sw::after { left: 23px; background: #000; }

/* ---------- reasons: diverging bars ---------- */
.rsn { display: grid; grid-template-columns: minmax(0, 1fr) 120px 50px; align-items: center; gap: 14px; padding: 11px 12px; border-radius: 16px; transition: background .15s; }
.rsn:nth-child(odd) { background: color-mix(in srgb, var(--text) 2.4%, transparent); }
.rsn:hover { background: color-mix(in srgb, var(--coral) 8%, transparent); }
.rsn .l { font-size: 14px; line-height: 18px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; }
.rsn .l small { display: block; color: var(--muted); font-size: 12.5px; font-family: var(--mono); font-weight: 400; }
.rsn .b { position: relative; height: 10px; background: var(--panel2); border-radius: 10px; }
.rsn .b::before { content: ''; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 2px; border-radius: 2px; background: var(--watch); }
.rsn .b i { position: absolute; top: 0; bottom: 0; border-radius: 10px; }
.rsn .b i.up { left: 50%; background: var(--coral); }
.rsn .b i.dn { right: 50%; background: var(--muted); }
.rsn .c { text-align: right; font-family: var(--mono); font-size: 13px; color: var(--muted); }
.rsn .c.up { color: var(--coral); font-weight: 500; }

/* ---------- verdict, legend, agent, misc ---------- */
.verdict { display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: -6px; }
.stack { display: flex; height: 12px; gap: 3px; margin: 6px 0 14px 0; }
.stack i { display: block; border-radius: 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 22px; color: var(--muted); font-size: 13.5px; }
.legend .d { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
.legend b { color: var(--text); font-family: var(--mono); font-weight: 500; margin-left: 6px; }
.note { color: var(--muted); font-size: 13.5px; line-height: 20px; }
.note b { color: var(--text); font-weight: 700; }
.ai { display: flex; gap: 16px; align-items: center; padding: 20px 22px; border-radius: 24px; background: var(--panel2); }
.ai .ic { width: 42px; height: 42px; border-radius: 50%; display: grid; place-items: center; background: var(--f5); color: #000; flex: none; }
.ai b { display: block; font-size: 17px; font-weight: 700; letter-spacing: -.02em; }
.ai span { color: var(--muted); font-size: 13.5px; }
.case { padding: 22px 24px; border-radius: 24px; background: var(--panel2); position: relative; overflow: hidden; }
.case::before { content: ''; position: absolute; right: -60px; top: -60px; width: 160px; height: 160px; border-radius: 50%; background: var(--f5); opacity: .9; }
.case > * { position: relative; }
.case-top { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding-right: 90px; }
.case-top .ic { width: 38px; height: 38px; border-radius: 50%; display: grid; place-items: center; background: #000; color: var(--f5); }
.stApp .case-top b { font-size: 20px; font-weight: 700; letter-spacing: -.035em; margin-right: 4px; text-transform: lowercase; }
.case-n { font-size: 16px; line-height: 24px; margin: 16px 0 12px 0; color: var(--text); max-width: 900px; }
.case-ev { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 6px; }
.case-ev li { display: flex; gap: 12px; align-items: baseline; padding: 10px 14px; border-radius: 14px; background: color-mix(in srgb, var(--text) 3.5%, transparent); font-size: 14px; line-height: 20px; }
.case-ev .src { flex: none; width: 70px; font-size: 12px; color: var(--muted); font-family: var(--mono); }
.case-meta { margin-top: 12px; color: var(--faint); font-size: 12px; font-family: var(--mono); }
.agent-steps { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 0 0 18px 0; }
.agent-steps .st { display: inline-flex; align-items: center; gap: 8px; height: 34px; padding: 0 14px 0 6px; border-radius: 999px; font-size: 13px; color: #000; font-weight: 500; }
.agent-steps .st:nth-child(1) { background: var(--f1); } .agent-steps .st:nth-child(3) { background: var(--f2); } .agent-steps .st:nth-child(5) { background: var(--f3); } .agent-steps .st:nth-child(7) { background: var(--f4); }
.agent-steps .st b { display: inline-grid; place-items: center; width: 24px; height: 24px; border-radius: 50%; background: #000; color: #fff; font-size: 12px; font-weight: 700; }
.agent-steps .ar { color: var(--faint); font-size: 14px; }
.pane-t { font-size: 18px; font-weight: 700; margin: 0 0 10px 0; display: flex; align-items: center; gap: 10px; letter-spacing: -.03em; text-transform: lowercase; }
.pane-t span { color: var(--muted); font-weight: 500; font-size: 13px; letter-spacing: 0; text-transform: none; }
.chat { display: flex; flex-direction: column; gap: 10px; margin: 4px 0 12px 0; }
.msg { max-width: 92%; padding: 12px 15px; font-size: 14px; line-height: 20px; animation: pt-rise .4s cubic-bezier(.2,.8,.2,1) both; }
.msg.user { align-self: flex-end; background: var(--f2); color: #000; border-radius: 20px 20px 6px 20px; font-weight: 500; }
.msg.bot { align-self: flex-start; background: var(--panel2); border-radius: 20px 20px 20px 6px; }
.msg.preview { box-shadow: inset 0 0 0 2px var(--watch); background: transparent; }
.msg .who { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); margin-bottom: 4px; font-weight: 700; }
.msg .who svg { width: 14px; height: 14px; color: var(--f5); }
.msg .srcs { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; align-items: center; }
.msg .srcs .s { font-family: var(--mono); font-size: 11px; padding: 2px 9px; border-radius: 999px; background: var(--f5); color: #000; }
.msg .srcs .ok { color: var(--mint); font-size: 12px; font-weight: 700; } .msg .srcs .no { color: var(--amber); font-size: 12px; font-weight: 700; }
.msg .srcs .m { color: var(--faint); font-size: 11px; font-family: var(--mono); }
.try-l { color: var(--muted); font-size: 13px; margin: 4px 0 8px 0; font-weight: 500; }
.stApp [class*="st-key-chip-"] button { border-radius: 999px !important; min-height: 36px !important; padding: 2px 14px !important; background: var(--panel2) !important; border: 0 !important; }
.stApp [class*="st-key-chip-"] button p { font-size: 13px !important; font-weight: 500 !important; }
.stApp [class*="st-key-chip-"] button:hover:not(:disabled) { background: var(--f5) !important; color: #000 !important; }
.stApp [class*="st-key-chip-"] button:hover:not(:disabled) p { color: #000 !important; }
[data-testid="stChatInput"] { background: var(--panel2) !important; border: 0 !important; border-radius: 999px !important; padding-left: 8px; }
[data-testid="stChatInput"] div, [data-testid="stChatInput"] [data-baseweb] { background: transparent !important; border-color: transparent !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-family: var(--font) !important; font-size: 14px !important; background: transparent !important; }
[data-testid="stChatInput"] button { background: var(--f2) !important; color: #000 !important; border-radius: 50% !important; }
.empty { border-radius: 20px; padding: 22px; color: var(--muted); font-size: 14px; background: var(--panel2); }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }
.step { display: flex; flex-direction: column; gap: 10px; min-height: 210px; }
.step .n { display: inline-grid; place-items: center; width: 44px; height: 44px; border-radius: 50%; background: var(--sc, var(--f2)); color: #000; font-family: var(--font); font-weight: 700; font-size: 16px; margin-bottom: 8px; }
.facts .step:nth-child(1) { --sc: var(--f1); } .facts .step:nth-child(2) { --sc: var(--f2); } .facts .step:nth-child(3) { --sc: var(--f3); } .facts .step:nth-child(4) { --sc: var(--f4); } .facts .step:nth-child(5) { --sc: var(--f5); }
.stApp .step b { font-size: 22px; font-weight: 700; letter-spacing: -.035em; text-transform: lowercase; }
.step span { color: var(--muted); font-size: 14px; line-height: 20px; }

/* ---------- Streamlit widgets ---------- */
[data-testid="stWidgetLabel"] p, label p { color: var(--muted) !important; font-size: 13.5px !important; font-weight: 500; }
[data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] [role="group"], [data-baseweb="select"] > div { background: var(--panel2) !important; border-color: transparent !important; border-radius: 14px !important; }
.stApp input { background: transparent !important; color: var(--text) !important; font-size: 14px !important; caret-color: var(--lime); }
[data-testid="stNumberInput"] button { background: transparent !important; color: var(--muted) !important; border-color: transparent !important; }
[data-testid="stSelectbox"] svg { color: var(--muted) !important; }
[data-testid="stCheckbox"] p { font-size: 14px !important; color: var(--text) !important; }
[data-testid="stCheckbox"] label > div:first-of-type { background: var(--watch) !important; }
[data-testid="stCheckbox"] label > div:first-of-type > div { background: #fff !important; }
[data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type { background: var(--f1) !important; }
[data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type > div { background: #000 !important; }
/* the slider fill is an inline gradient in the config's primary colour: keep a neutral track, put the brand on the thumb */
[data-testid="stSlider"] div[style*="relative"] > div:first-child:empty { background: var(--watch) !important; }
[data-testid="stSlider"] div[style*="translate(-50%"] { background: var(--f2) !important; box-shadow: 0 0 0 5px color-mix(in srgb, var(--f2) 25%, transparent) !important; width: 20px !important; height: 20px !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] p { font-family: var(--mono) !important; font-size: 12.5px; color: var(--text) !important; }
[data-testid="stSliderTickBar"] p { color: var(--faint) !important; }
[data-testid="stExpander"] details { background: var(--panel); border: 0 !important; box-shadow: 0 0 0 1px var(--edge); border-radius: 22px; }
[data-testid="stExpander"] summary p { font-size: 15px !important; color: var(--text) !important; font-weight: 700; }
[data-testid="stPlotlyChart"] { background: transparent; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"], [role="listbox"], [role="option"] { background: var(--panel) !important; color: var(--text) !important; }
[data-baseweb="popover"] li:hover, [role="option"]:hover, [role="option"][aria-selected="true"] { background: var(--panel2) !important; }
.stApp [data-testid^="stBaseButton"] { background: var(--panel2) !important; color: var(--text) !important; border: 0 !important; border-radius: 999px !important; min-height: 42px; padding: 0 20px !important; transition: transform .2s, background .2s; }
.stApp [data-testid^="stBaseButton"]:hover:not(:disabled) { background: var(--f2) !important; color: #000 !important; transform: translateY(-2px); }
.stApp [data-testid^="stBaseButton"]:hover:not(:disabled) p { color: #000 !important; }
.stApp [data-testid^="stBaseButton"]:disabled { opacity: .45; }
.stApp [data-testid^="stBaseButton"] p { font-size: 14px !important; font-weight: 700; }
.stApp [data-testid="stBaseButton-segmented_control"], .stApp [data-testid="stBaseButton-segmented_controlActive"] { min-height: 36px; padding: 0 14px !important; }
.stApp [data-testid="stBaseButton-segmented_controlActive"] { background: var(--f2) !important; color: #000 !important; }
.stApp [data-testid="stBaseButton-segmented_controlActive"] p { color: #000 !important; }

/* ---------- phone ---------- */
.mobile-nav { display: none; }
.mobile-nav .row { display: flex; flex-wrap: wrap; gap: 6px; padding: 2px 0 14px 0; }
.mobile-nav .row a { color: var(--muted) !important; font-size: 14px; padding: 8px 14px; border-radius: 999px; background: var(--panel2); font-weight: 500; }
.mobile-nav .row a.on { color: #000 !important; background: var(--f2); font-weight: 700; }
@media (max-width: 1100px) {
  .hdr .hl { max-width: calc(100% - 200px); }
  .hdr .art { width: 300px; height: 300px; right: -110px; }
}
@media (max-width: 760px) {
  .mobile-nav { display: block; margin-bottom: 8px; }
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display: none !important; }
  .block-container { padding: 16px 16px 56px 16px !important; }
  .hdr { min-height: 0; padding: 150px 22px 24px 22px; border-radius: 26px; }
  .hdr .hl { max-width: 100%; }
  .hdr .art { width: 190px; height: 190px; right: -30px; top: 100px; border-width: 10px; }
  .stApp .hdr h1 { font-size: 40px !important; }
  .kpis, .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .kpis.k3 { grid-template-columns: 1fr; gap: 10px; }
  .kpi { min-height: 140px; padding: 18px; } .kpi .v { font-size: 36px; }
  .rsn { grid-template-columns: minmax(0, 1fr) 70px 44px; }
}
"""

_ICONS = {
    "home": '<path d="M3 11 12 3l9 8"/><path d="M5 10v10h14V10"/>',
    "queue": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
    "account": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "try": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "models": '<path d="M3 3v18h18"/><path d="M7 16v-4M12 16V8M17 16v-7"/>',
    "about": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
    "flag": '<path d="M4 22V4c4-2 8 2 16 0v10c-8 2-12-2-16 0"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    "graph": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8.6 13.5 6.8 4M15.4 6.5l-6.8 4"/>',
    "out": '<path d="M7 17 17 7M8 7h9v9"/>',
    "in": '<path d="M17 7 7 17M16 17H7V8"/>',
    "spark": '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 17v4M17 19h4"/>',
    "back": '<path d="M15 18l-6-6 6-6"/>',
    "next": '<path d="M9 18l6-6-6-6"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
    "db": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "trail": '<path d="M5 3h11l3 3v15H5z"/><path d="M9 9h6M9 13h6M9 17h3"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "moon": '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
}


def icon(name: str) -> str:
    return f'<svg class="i" viewBox="0 0 24 24" aria-hidden="true">{_ICONS[name]}</svg>'


def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def href(state: dict, **change) -> str:
    """A link that keeps the current view state (page, model, reveal...) and changes only what is given. None removes a key."""
    merged = {**state, **change}
    return "?" + urlencode({k: v for k, v in merged.items() if v not in (None, "")})


def link(url: str, inner: str, cls: str = "", title: str = "") -> str:
    t = f' title="{esc(title)}"' if title else ""
    return f'<a class="{cls}" target="_self" href="{esc(url)}"{t}>{inner}</a>'


def sidebar_html(state: dict, pages: dict, facts: list[tuple[str, str]]) -> str:
    nav = "".join(link(href(state, page=k, reveal=None, p=None), icon(k) + esc(label), "on" if state["page"] == k else "") for k, label in pages.items())
    rows = "".join(f'<div class="row"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in facts)
    return (f'<div class="brand"><span class="mark"></span><b>papertrail</b></div>'
            f'<div class="nav">{nav}</div>'
            f'<div class="nav theme-sw">{theme_link(state)}</div>'
            f'<div class="side-card"><div class="t"><span class="live"></span>Test window</div>{rows}</div>')


def theme_link(state: dict) -> str:
    """The sidebar's light/dark switch, labelled with where the click goes."""
    light = state.get("theme") == "light"
    return link(href(state, theme=None if light else "light"), icon("moon" if light else "sun") + ("Dark theme" if light else "Light theme"))


def mobile_nav_html(state: dict, pages: dict) -> str:
    nav = "".join(link(href(state, page=k, reveal=None, p=None), esc(v), "on" if state["page"] == k else "") for k, v in pages.items())
    nav += link(href(state, theme=None if state.get("theme") == "light" else "light"), "Dark" if state.get("theme") == "light" else "Light")
    return f'<div class="mobile-nav"><div class="row">{nav}</div></div>'


def seg(state: dict, models: dict, **keep) -> str:
    """The model switch: three linked segments, each with its series colour."""
    return '<div class="seg">' + "".join(
        link(href(state, model=k, **keep), f'<span class="sw" style="background:{MODEL_VAR[k]}"></span>{esc(v)}', "on" if state["model"] == k else "")
        for k, v in models.items()) + "</div>"


def header(title: str, sub: str = "", right: str = "", crumb: str = "", mono: bool = False, badge: str = "", art: str = "", tone: str = "f2") -> str:
    """The page's band: a candy field (`tone` is f1-f5), a big lowercase title and the page's artwork cropped in a circle."""
    pic = f'<div class="art" role="img" aria-hidden="true" style="background-image:url({art_uri(art)})"></div>' if art else ""
    return (f'<div class="hdr" style="--fill:var(--{tone})"><div class="hl">{f"<div class=crumb>{crumb}</div>" if crumb else ""}'
            f'<h1 class="{"mono" if mono else ""}"><span>{esc(title)}</span>{badge}</h1>'
            f'{f"<div class=sub>{sub}</div>" if sub else ""}<div class="right">{right}</div></div>{pic}</div>')


@functools.cache
def art_uri(name: str) -> str:
    """A page artwork (app/landing/img/<name>.jpg, drawn by scripts/make_landing_art.py) as a data URI, read once."""
    path = Path(__file__).resolve().parent / "landing" / "img" / f"{name}.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def button(text: str, url: str, kind: str = "primary", ic: str = "", external: bool = False) -> str:
    if external:                                   # another site: open in a new tab, keep the console where it is
        return f'<a class="btn {kind}" target="_blank" rel="noopener noreferrer" href="{esc(url)}">{icon(ic) if ic else ""}{esc(text)}</a>'
    return link(url, (icon(ic) if ic else "") + esc(text), f"btn {kind}")


def theme_button(state: dict) -> str:
    """Sun/moon switch between the two palettes; the icon shows where the click goes."""
    light = state.get("theme") == "light"
    nxt = None if light else "light"
    title = f'Switch to {PALETTES["dark" if light else "light"]["name"]} ({"dark" if light else "light"} theme)'
    return link(href(state, theme=nxt), icon("moon" if light else "sun"), "btn ghost icon", title)


def tip(text: str) -> str:
    return f'<span class="tip" title="{esc(text)}">?</span>'


def card_head(title: str, sub: str = "", right: str = "", ic: str = "", hint: str = "") -> str:
    t = (icon(ic) if ic else "") + esc(title) + (" " + tip(hint) if hint else "")
    return f'<div class="ch"><div><div class="t">{t}</div>{f"<div class=s>{esc(sub)}</div>" if sub else ""}</div><div class="r">{right}</div></div>'


def kpi_cards(cards: list[dict], cols: int = 4) -> str:
    """Each card: label, value (HTML), ic, and optional foot (HTML), bar (0-1), color, on, hint."""
    out = []
    for c in cards:
        bar = f'<div class="bar"><i style="width:{max(0, min(c["bar"], 1)) * 100:.1f}%;background:{c.get("color", "var(--lime)")}"></i></div>' if "bar" in c else ""
        hint = f' title="{esc(c["hint"])}"' if c.get("hint") else ""
        foot = f'<div class="f">{c["foot"]}</div>' if c.get("foot") else ""
        out.append(f'<div class="card kpi{" on" if c.get("on") else ""}"{hint}><div class="top"><span>{esc(c["label"])}</span><span class="ic">{icon(c["ic"])}</span></div>'
                   f'<div class="v">{c["value"]}</div>{foot}{bar}</div>')
    return f'<div class="kpis{" k3" if cols == 3 else ""}">{"".join(out)}</div>'


def pill(text: str, tone: str = "grey", dot: bool = True, big: bool = False, title: str = "") -> str:
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="pill {tone}{" big" if big else ""}"{t}>{"<span class=d></span>" if dot else ""}{esc(text)}</span>'


TIER_TONE = {"Critical": "coral", "High": "amber", "Watch": "grey"}


def tier_pill(tier: str, big: bool = False, title: str = "") -> str:
    return pill(tier, TIER_TONE.get(tier, "grey"), big=big, title=title)


def chips(items: list[tuple[str, bool, str]]) -> str:
    """(label, raises, tooltip) chips; an arrow and its colour show the direction."""
    return '<div class="chips">' + "".join(
        f'<span class="chip{" up" if up else ""}" title="{esc(t)}"><span class="a">{"▲" if up else "▼"}</span>{esc(lab)}</span>' for lab, up, t in items) + "</div>"


def model_dots(flags: dict) -> str:
    """One square per model, lit in the model's colour when it flags the account."""
    return '<span class="dots">' + "".join(
        f'<i title="{esc(k)}: {"flags" if v else "does not flag"}" style="{f"background:{MODEL_VAR[k]}" if v else ""}"></i>' for k, v in flags.items()) + "</span>"


def meter(n: int, d: int) -> str:
    w = n / d * 100 if d else 0
    return f'<span class="meter"><span class="m"><i style="width:{w:.0f}%"></i></span><span>{n}/{d}</span></span>'


def toggle(label: str, on: bool, url: str) -> str:
    return link(url, f'<span class="sw"></span>{esc(label)}', f"toggle{' on' if on else ''}")


def table(columns: list[tuple], rows: list[dict], wrap: bool = True) -> str:
    """columns: (key, header, kind) with kind in {'id','num','text','wrap','dim'}; rows hold already-escaped HTML in the cells."""
    head = "".join(f'<th class="{"num" if "num" in k else ""}">{esc(h)}</th>' for _, h, k in columns) if any(h for _, h, _ in columns) else ""
    body = "".join("<tr>" + "".join(f'<td class="{k}">{r.get(key, "")}</td>' for key, _, k in columns) + "</tr>" for r in rows)
    t = f'<table class="tbl">{f"<thead><tr>{head}</tr></thead>" if head else ""}<tbody>{body}</tbody></table>'
    return f'<div class="tbl-wrap">{t}</div>' if wrap else t


def reasons_list(reasons, top: int = 8) -> str:
    """The strongest drivers as diverging bars around a centre line: coral pushes the score up, grey pulls it down."""
    rows = reasons.head(top).to_dicts()
    if not rows:
        return '<div class="empty">No explanation stored for this account.</div>'
    peak = max(abs(r["contribution"]) for r in rows) or 1.0
    out = []
    for r in rows:
        w = abs(r["contribution"]) / peak * 50
        up = r["contribution"] > 0
        out.append(f'<div class="rsn" title="{esc(r["label"])}: {esc(r["value_text"])}"><div class="l">{esc(r["label"])}<small>{esc(r["value_text"])}</small></div>'
                   f'<div class="b"><i class="{"up" if up else "dn"}" style="width:{w:.1f}%"></i></div><div class="c{" up" if up else ""}">{r["contribution"]:+.2f}</div></div>')
    return "".join(out)


def legend(parts: list[tuple]) -> str:
    """(label, value, colour) entries: a coloured mark, the label and the figure."""
    return '<div class="legend">' + "".join(f'<span><span class="d" style="background:{c}"></span>{esc(lab)}<b>{v}</b></span>' for lab, v, c in parts) + "</div>"


def ai_placeholder(text: str = "No summary has been written for this account yet.") -> str:
    return f'<div class="ai"><span class="ic">{icon("spark")}</span><div><b>AI case summary</b><span>{esc(text)}</span></div></div>'


SOURCE_NAME = {"alert": "Alert", "subgraph": "Network", "shap": "Drivers", "history": "History"}
ACTION_PILL = {"escalate_SAR": ("Escalate (SAR)", "coral"), "enhanced_review": ("Enhanced review", "amber"), "close_false_positive": ("Close: false positive", "mint")}


def agent_steps() -> str:
    steps = [("1", "Gather evidence: alert · network · drivers · history"), ("2", "Draft with an LLM"), ("3", "Check every number"), ("4", "Cite sources")]
    return '<div class="agent-steps">' + '<span class="ar">→</span>'.join(f'<span class="st"><b>{n}</b>{esc(t)}</span>' for n, t in steps) + "</div>"


def chat_html(turns: list[dict], preview: dict | None = None) -> str:
    """The conversation as bubbles; an agent answer shows the sources it used and whether its numbers checked out."""
    def bot(text: str, foot: str, cls: str = "") -> str:
        return f'<div class="msg bot {cls}"><div class="who">{icon("spark")}Investigator</div>{esc(text)}<div class="srcs">{foot}</div></div>'

    out = []
    if preview and not turns:
        out.append(f'<div class="msg user">{esc(preview["question"])}</div>')
        out.append(bot(preview["answer"], '<span class="m">preview · written straight from the tool outputs, no AI</span>', "preview"))
    for t in turns:
        srcs = "".join(f'<span class="s">{esc(SOURCE_NAME.get(x.split(":")[0], x))}</span>' for x in t.get("sources", []))
        mark = '<span class="ok">✓ numbers verified</span>' if t.get("verified") else '<span class="no">not verified</span>'
        model = f'<span class="m">{esc((t.get("model") or "").split("/")[-1])} · {t.get("latency_s", 0):.0f} s</span>' if t.get("model") else ""
        out.append(f'<div class="msg user">{esc(t["question"])}</div>')
        out.append(bot(t["answer"], srcs + mark + model))
    return f'<div class="chat">{"".join(out)}</div>'


def case_card(result: dict, live: bool) -> str:
    """The investigator's summary: verdict pills, the narrative, each claim with the tool it came from, and how it was made."""
    s = result["summary"]
    act, tone = ACTION_PILL.get(s["recommended_action"], (s["recommended_action"], "grey"))
    check = (pill("Verified against tool outputs", "mint") if result.get("verified")
             else pill("Partly unverified: only checked facts shown", "amber"))
    ev = "".join(f'<li><span class="src">{esc(SOURCE_NAME.get(e["source_id"].split(":")[0], e["source_id"]))}</span>{esc(e["claim"])}</li>' for e in s["evidence"])
    how = " · ".join(x for x in (
        "live" if live else "cached", f'{result.get("provider") or "?"}/{result.get("model") or "?"}',
        f'{result["drafts"]} draft{"s" if result.get("drafts", 1) != 1 else ""}' if result.get("drafts") else "",
        f'{result["total_latency_s"]:.1f} s' if result.get("total_latency_s") else "", result.get("generated_at", "")) if x)
    return (f'<div class="case"><div class="case-top"><span class="ic">{icon("spark")}</span><b>AI case summary</b>'
            f'{pill(s["suspected_typology"].title() if s["suspected_typology"] != "UNCLEAR" else "Pattern unclear", "lime" if s["suspected_typology"] != "UNCLEAR" else "grey")}'
            f'{pill(act, tone)}{check}</div><p class="case-n">{esc(s["narrative"])}</p><ul class="case-ev">{ev}</ul>'
            f'<div class="case-meta">{esc(how)}</div></div>')
