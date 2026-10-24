"""Look and feel of the console: one stylesheet and the small HTML builders (no streamlit import, so it is unit-tested).

Style: a "control room" dashboard. Rounded cards with a hairline border and a faint top sheen, big figures with small
chips under them, icon navigation, a segmented model switch, and short copy (the details live in hover tooltips).
Two palettes, switched by ?theme= in the URL: "Night Arcade" (dark: purple-tinted black, bright pastel-neon accents) and
"Riso Paper" (light: cream stock, deep risograph inks; light pages need darker, more saturated accents than dark ones).
The token names are the colours' jobs in the dark palette, and every palette keeps the jobs:
- lime is the brand and the selection (and the blend's series);
- sky is XGBoost, violet is the GNN;
- coral means risk: flagged payments, risk-raising reasons, critical alerts.
The rules below use tokens only (tints via color-mix), so a palette is just the values on :root.
State lives in the URL (page, model, account, theme), so every view can be linked and the tables stay plain HTML.
"""
from __future__ import annotations

import html
from urllib.parse import urlencode

PALETTES = {
    # Night Arcade: purple-tinted black, bright pastel-neons (light, vivid accents keep contrast on dark surfaces)
    "dark": {"name": "Candy on Black", "bg": "#000000", "side": "#050507", "panel": "#0d0d11", "panel2": "#17171d", "text": "#f2f0fa", "text2": "#cfcbdc",
             "muted": "#9a97a8", "faint": "#6a677a", "lime": "#93f1f7", "sky": "#b9c8ff", "violet": "#fbb65c", "coral": "#ff5c8a",
             "amber": "#f3ea7a", "mint": "#9ae187", "onbrand": "#000000", "watch": "#3b3948", "neutral": "#e9e6f5", "grid": "rgba(255,255,255,0.06)"},
    # Riso Paper: warm cream stock and deep risograph inks (darker, saturated accents so nothing washes out on a light page)
    "light": {"name": "Riso Paper", "bg": "#f4efe6", "side": "#efe9dd", "panel": "#fffcf6", "panel2": "#f6f0e5", "text": "#1f1c17", "text2": "#3d3930",
              "muted": "#6d675d", "faint": "#8f887b", "lime": "#2f43d0", "sky": "#0b8a84", "violet": "#8a3ffc", "coral": "#c20e6e",
              "amber": "#b86a00", "mint": "#18885a", "onbrand": "#ffffff", "watch": "#d9d1c3", "neutral": "#3d3930", "grid": "rgba(31,28,23,0.08)"},
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
.block-container { max-width: 1360px; padding: 26px 34px 64px 34px !important; }
[data-testid="stHeadingWithActionElements"] a { display: none !important; }
p, li, label, span, div { font-size: 13px; }
.stApp, .stApp * { font-variant-numeric: tabular-nums; }
.stApp a, .stApp a:visited, .stApp a:hover { color: inherit !important; text-decoration: none !important; }
.stApp, .stApp p, .stApp div, .stApp span, .stApp label, .stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp button, .stApp a, .stApp summary { font-family: var(--font) !important; }
.stApp .mono, .stApp .mono *, .stApp td.num, .stApp td.id, .stApp td.id a, .stApp input, .stApp [data-baseweb="select"] div { font-family: var(--mono) !important; }
.stApp h1 span { font-size: inherit !important; font-weight: inherit !important; line-height: inherit !important; font-family: inherit !important; }
.stApp .hdr h1 .pill, .stApp .hdr h1 .pill * { font-family: var(--font) !important; font-size: 12.5px !important; font-weight: 500 !important; line-height: 1 !important; letter-spacing: 0 !important; }
.stApp .hdr h1 .pill { margin-left: 12px !important; vertical-align: middle; }
.stApp [data-testid="stIconMaterial"], .stApp [data-testid="stExpanderIcon"], .stApp span[class*="material"] { font-family: 'Material Symbols Rounded' !important; }
svg.i { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; flex: none; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--side) !important; border-right: 1px solid var(--line); min-width: 236px !important; max-width: 236px !important; width: 236px !important; }
[data-testid="stSidebar"] > div:first-child { padding: 0; }
[data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] { padding: 22px 0 0 0 !important; }
.brand { display: flex; align-items: center; gap: 10px; padding: 0 20px 26px 20px; }
.brand .logo { width: 30px; height: 30px; border-radius: 9px; background: var(--lime); display: grid; place-items: center; color: var(--onbrand); }
.brand .logo svg { width: 17px; height: 17px; stroke-width: 2.2; }
.brand b { font-size: 16px; font-weight: 600; letter-spacing: -.01em; }
.nav-group { padding: 0 22px 8px 22px; color: var(--faint); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
.nav { padding: 0 12px 18px 12px; }
.nav a { display: flex; align-items: center; gap: 11px; padding: 9px 12px; border-radius: 10px; font-size: 13.5px; line-height: 18px; margin-bottom: 2px; }
.stApp .nav a { color: var(--muted) !important; }
.stApp .nav a:hover { color: var(--text) !important; background: color-mix(in srgb, var(--text) 3.5%, transparent); }
.stApp .nav a.on { color: var(--text) !important; background: color-mix(in srgb, var(--text) 6%, transparent); box-shadow: inset 0 0 0 1px var(--line); }
.nav a.on svg { color: var(--lime); }
.side-card { margin: 8px 16px 16px 16px; padding: 14px; border-radius: 12px; background: var(--panel); border: 1px solid var(--line); }
.side-card .t { font-size: 12px; color: var(--muted); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.side-card .row { display: flex; justify-content: space-between; font-size: 12px; padding: 3px 0; color: var(--muted); }
.side-card .row b { color: var(--text); font-weight: 500; font-family: var(--mono); font-size: 12px; }
.live { width: 7px; height: 7px; border-radius: 50%; background: var(--lime); box-shadow: 0 0 0 3px color-mix(in srgb, var(--lime) 15%, transparent); display: inline-block; }

/* ---------- header ---------- */
.hdr { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; margin: 0 0 22px 0; flex-wrap: wrap; }
.crumb { color: var(--muted); font-size: 12px; margin-bottom: 6px; display: flex; gap: 6px; align-items: center; }
.crumb a { color: var(--muted) !important; }
.stApp .hdr h1 { font-size: 28px !important; font-weight: 600 !important; line-height: 34px !important; letter-spacing: -.02em; margin: 0 !important; padding: 0 !important; color: var(--text) !important; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.stApp .hdr h1.mono { font-size: 24px !important; font-weight: 500 !important; letter-spacing: 0; }
.hdr .sub { color: var(--muted); font-size: 14px; margin-top: 4px; }
.hdr .right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.seg { display: inline-flex; padding: 3px; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; gap: 2px; }
.seg a { display: inline-flex; align-items: center; gap: 7px; padding: 6px 13px; border-radius: 9px; font-size: 13px; }
.stApp .seg a { color: var(--muted) !important; }
.stApp .seg a.on { color: var(--text) !important; background: var(--panel2); box-shadow: inset 0 0 0 1px var(--line2); }
.seg .sw { width: 8px; height: 8px; border-radius: 50%; }
.btn { display: inline-flex; align-items: center; gap: 8px; padding: 0 15px; height: 36px; border-radius: 10px; font-size: 13px; font-weight: 500; white-space: nowrap; }
.stApp .btn.primary { background: var(--lime); color: var(--onbrand) !important; }
.stApp .btn.ghost { background: var(--panel); border: 1px solid var(--line2); color: var(--text) !important; }
.btn:hover { filter: brightness(1.08); }

/* ---------- cards ---------- */
.card { background: linear-gradient(180deg, color-mix(in srgb, var(--text) 2.8%, transparent), color-mix(in srgb, var(--text) 0%, transparent) 120px), var(--panel); border: 1px solid var(--line); border-radius: var(--r); padding: 18px 20px; }
/* st.container(border=True) is a bordered stVerticalBlock in Streamlit 1.66; ours always open with a card head (.ch) */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ch) { background: linear-gradient(180deg, color-mix(in srgb, var(--text) 2.8%, transparent), transparent 120px), var(--panel) !important;
  border: 1px solid var(--line) !important; border-radius: var(--r) !important; padding: 16px 18px 18px 18px !important; }
[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
[data-testid="stColumn"] > [data-testid="stVerticalBlock"], [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"],
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] { height: 100%; }
.ch { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; }
.ch .t { font-size: 15px; font-weight: 600; letter-spacing: -.01em; display: flex; align-items: center; gap: 8px; }
.ch .s { color: var(--muted); font-size: 12.5px; margin-top: 2px; }
.ch .r { color: var(--muted); font-size: 12px; display: flex; gap: 10px; align-items: center; }
.tip { display: inline-grid; place-items: center; width: 16px; height: 16px; border-radius: 50%; border: 1px solid var(--line2); color: var(--muted); font-size: 10px; cursor: help; font-weight: 600; }
.gap { height: 18px; }

/* ---------- KPI cards ---------- */
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
.kpis.k3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.kpi { position: relative; overflow: hidden; }
.kpi.on { border-color: color-mix(in srgb, var(--lime) 35%, transparent); box-shadow: 0 0 0 1px color-mix(in srgb, var(--lime) 12%, transparent), 0 10px 40px -20px color-mix(in srgb, var(--lime) 35%, transparent); }
.kpi .top { display: flex; align-items: center; justify-content: space-between; color: var(--muted); font-size: 13px; }
.kpi .ic { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; background: color-mix(in srgb, var(--text) 4%, transparent); border: 1px solid var(--line); color: var(--text); }
.kpi .v { font-size: 32px; line-height: 38px; font-weight: 600; letter-spacing: -.03em; margin-top: 14px; }
.kpi .v small { font-size: 14px; color: var(--muted); font-weight: 400; margin-left: 4px; letter-spacing: 0; }
.kpi .f { display: flex; align-items: center; gap: 8px; white-space: nowrap; margin-top: 8px; color: var(--muted); font-size: 12px; }
.kpi .bar { height: 4px; background: color-mix(in srgb, var(--text) 6%, transparent); border-radius: 4px; margin-top: 14px; overflow: hidden; }
.kpi .bar i { display: block; height: 100%; border-radius: 4px; }

/* ---------- pills and chips ---------- */
.pill { display: inline-flex; align-items: center; gap: 6px; height: 22px; padding: 0 9px; border-radius: 999px; font-size: 11.5px; font-weight: 500; white-space: nowrap; border: 1px solid transparent; }
.pill .d { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.pill.coral { color: var(--coral); background: color-mix(in srgb, var(--coral) 10%, transparent); border-color: color-mix(in srgb, var(--coral) 22%, transparent); }
.pill.amber { color: var(--amber); background: color-mix(in srgb, var(--amber) 10%, transparent); border-color: color-mix(in srgb, var(--amber) 22%, transparent); }
.pill.mint { color: var(--mint); background: color-mix(in srgb, var(--mint) 10%, transparent); border-color: color-mix(in srgb, var(--mint) 22%, transparent); }
.pill.lime { color: var(--lime); background: color-mix(in srgb, var(--lime) 10%, transparent); border-color: color-mix(in srgb, var(--lime) 24%, transparent); }
.pill.grey { color: var(--muted); background: color-mix(in srgb, var(--text) 4%, transparent); border-color: var(--line); }
.pill.big { height: 28px; padding: 0 12px; font-size: 12.5px; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { display: inline-flex; align-items: center; gap: 6px; height: 24px; padding: 0 9px; border-radius: 8px; font-size: 12px; background: color-mix(in srgb, var(--text) 4%, transparent); border: 1px solid var(--line); color: var(--text2); white-space: nowrap; cursor: default; }
.chip .a { font-size: 11px; color: var(--muted); }
.chip.up .a { color: var(--coral); }
.dots { display: inline-flex; gap: 5px; }
.dots i { width: 9px; height: 9px; border-radius: 3px; background: color-mix(in srgb, var(--text) 8%, transparent); display: inline-block; }
.meter { display: flex; align-items: center; gap: 10px; }
.meter .m { width: 64px; height: 5px; background: color-mix(in srgb, var(--text) 7%, transparent); border-radius: 4px; overflow: hidden; }
.meter .m i { display: block; height: 100%; background: var(--coral); border-radius: 4px; }
.meter span { font-family: var(--mono); font-size: 12px; color: var(--muted); }

/* ---------- tables ---------- */
.tbl-wrap { overflow: auto; max-height: 640px; margin: 0 -18px -18px -18px; border-top: 1px solid var(--line); }
table.tbl { width: 100%; border-collapse: collapse; font-size: 13px; }
.stApp .tbl th, .stApp .tbl td { border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; }
.tbl th { position: sticky; top: 0; background: var(--panel); text-align: left; font-weight: 500; font-size: 12px; color: var(--muted); padding: 0 18px; height: 40px; border-bottom: 1px solid var(--line); white-space: nowrap; z-index: 1; }
.tbl td { padding: 0 18px; height: 52px; border-bottom: 1px solid color-mix(in srgb, var(--text) 4.5%, transparent); vertical-align: middle; white-space: nowrap; }
.tbl tr:last-child td { border-bottom: 0; }
.tbl th.num, .tbl td.num { text-align: right; }
.tbl tbody tr:hover td { background: color-mix(in srgb, var(--text) 2.5%, transparent); }
.tbl td.id a { font-family: var(--mono); color: var(--text) !important; }
.tbl td.id a:hover { color: var(--lime) !important; }
.tbl td.dim { color: var(--muted); }
.tbl td.wrap { white-space: normal; line-height: 19px; padding-top: 10px; padding-bottom: 10px; color: var(--text2); }
.tbl .best { color: var(--lime); font-weight: 600; }
.tbl .rest { color: var(--text2); }
.dir { display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 8px; background: color-mix(in srgb, var(--text) 4%, transparent); border: 1px solid var(--line); }
.dir.out { color: var(--muted); } .dir.in { color: var(--mint); }
.amt-in { color: var(--mint); }
.rk { color: var(--faint); font-family: var(--mono); font-size: 12px; }
.pager { display: flex; justify-content: space-between; align-items: center; color: var(--muted); font-size: 12px; padding: 14px 0 0 0; }
.pager .pg { display: flex; gap: 8px; }
.pager a { padding: 6px 12px; border-radius: 9px; border: 1px solid var(--line2); color: var(--text) !important; }
.toggle { display: inline-flex; align-items: center; gap: 8px; color: var(--muted) !important; font-size: 12.5px; }
.toggle .sw { width: 30px; height: 18px; border-radius: 999px; background: color-mix(in srgb, var(--text) 10%, transparent); position: relative; }
.toggle .sw::after { content: ''; position: absolute; top: 3px; left: 3px; width: 12px; height: 12px; border-radius: 50%; background: var(--muted); transition: left .15s; }
.toggle.on .sw { background: color-mix(in srgb, var(--lime) 25%, transparent); } .toggle.on .sw::after { left: 15px; background: var(--lime); }

/* ---------- reasons: diverging bars ---------- */
.rsn { display: grid; grid-template-columns: minmax(0, 1fr) 110px 46px; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid color-mix(in srgb, var(--text) 4.5%, transparent); }
.rsn:last-child { border-bottom: 0; }
.rsn .l { font-size: 13px; line-height: 17px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rsn .l small { display: block; color: var(--muted); font-size: 12px; font-family: var(--mono); }
.rsn .b { position: relative; height: 8px; background: color-mix(in srgb, var(--text) 4%, transparent); border-radius: 4px; }
.rsn .b::before { content: ''; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--line2); }
.rsn .b i { position: absolute; top: 0; bottom: 0; border-radius: 4px; }
.rsn .b i.up { left: 50%; background: linear-gradient(90deg, color-mix(in srgb, var(--coral) 50%, transparent), var(--coral)); }
.rsn .b i.dn { right: 50%; background: linear-gradient(270deg, color-mix(in srgb, var(--muted) 35%, transparent), var(--muted)); }
.rsn .c { text-align: right; font-family: var(--mono); font-size: 12px; color: var(--muted); }
.rsn .c.up { color: var(--coral); }

/* ---------- verdict, stack, legend, misc ---------- */
.verdict { display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: -6px; }
.stack { display: flex; height: 12px; gap: 3px; margin: 6px 0 14px 0; }
.stack i { display: block; border-radius: 4px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 22px; color: var(--muted); font-size: 12.5px; }
.legend .d { display: inline-block; width: 8px; height: 8px; border-radius: 3px; margin-right: 7px; }
.legend b { color: var(--text); font-family: var(--mono); font-weight: 500; margin-left: 6px; }
.note { color: var(--muted); font-size: 12.5px; line-height: 18px; }
.note b { color: var(--text); font-weight: 500; }
.ai { display: flex; gap: 14px; align-items: center; padding: 16px 18px; border-radius: var(--r); border: 1px dashed var(--line2);
      background: linear-gradient(90deg, color-mix(in srgb, var(--violet) 7%, transparent), color-mix(in srgb, var(--lime) 4%, transparent)); }
.ai .ic { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; background: color-mix(in srgb, var(--violet) 14%, transparent); color: var(--violet); }
.ai b { display: block; font-size: 14px; font-weight: 600; }
.ai span { color: var(--muted); font-size: 12.5px; }
.case { padding: 18px 20px; border-radius: var(--r); border: 1px solid var(--line); background: linear-gradient(90deg, color-mix(in srgb, var(--violet) 6%, transparent), transparent 60%), var(--panel); }
.case-top { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.case-top .ic { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; background: color-mix(in srgb, var(--violet) 14%, transparent); color: var(--violet); }
.case-top b { font-size: 15px; font-weight: 600; margin-right: 4px; }
.case-n { font-size: 14px; line-height: 21px; margin: 14px 0 10px 0; color: var(--text); max-width: 900px; }
.case-ev { list-style: none; padding: 0; margin: 0; }
.case-ev li { display: flex; gap: 10px; align-items: baseline; padding: 7px 0; border-top: 1px solid color-mix(in srgb, var(--text) 4.5%, transparent); font-size: 13px; line-height: 19px; }
.case-ev .src { flex: none; width: 64px; font-size: 11.5px; color: var(--muted); font-family: var(--mono); }
.case-meta { margin-top: 10px; color: var(--faint); font-size: 11.5px; font-family: var(--mono); }
.agent-steps { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: -2px 0 14px 0; }
.agent-steps .st { display: inline-flex; align-items: center; gap: 7px; height: 26px; padding: 0 10px; border-radius: 999px; background: color-mix(in srgb, var(--text) 4%, transparent); border: 1px solid var(--line); font-size: 12px; color: var(--text2); }
.agent-steps .st b { font-family: var(--mono); font-size: 11px; color: var(--violet); font-weight: 500; }
.agent-steps .ar { color: var(--faint); font-size: 12px; }
.pane-t { font-size: 13px; font-weight: 600; margin: 0 0 8px 0; display: flex; align-items: center; gap: 8px; }
.pane-t span { color: var(--muted); font-weight: 400; font-size: 12px; }
.chat { display: flex; flex-direction: column; gap: 10px; margin: 4px 0 10px 0; }
.msg { max-width: 92%; padding: 10px 13px; font-size: 13px; line-height: 19px; }
.msg.user { align-self: flex-end; background: color-mix(in srgb, var(--lime) 14%, transparent); border: 1px solid color-mix(in srgb, var(--lime) 30%, transparent); border-radius: 14px 14px 4px 14px; }
.msg.bot { align-self: flex-start; background: color-mix(in srgb, var(--text) 4%, transparent); border: 1px solid var(--line); border-radius: 14px 14px 14px 4px; }
.msg.preview { border-style: dashed; }
.msg .who { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
.msg .who svg { width: 13px; height: 13px; color: var(--violet); }
.msg .srcs { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 7px; align-items: center; }
.msg .srcs .s { font-family: var(--mono); font-size: 10.5px; padding: 1px 7px; border-radius: 6px; background: color-mix(in srgb, var(--violet) 12%, transparent); color: var(--violet); }
.msg .srcs .ok { color: var(--mint); font-size: 11px; } .msg .srcs .no { color: var(--amber); font-size: 11px; }
.msg .srcs .m { color: var(--faint); font-size: 10.5px; font-family: var(--mono); }
.try-l { color: var(--muted); font-size: 12px; margin: 2px 0 6px 0; }
.stApp [class*="st-key-chip-"] button { border-radius: 999px !important; min-height: 30px !important; padding: 2px 12px !important; background: color-mix(in srgb, var(--violet) 8%, transparent) !important; border-color: color-mix(in srgb, var(--violet) 28%, transparent) !important; }
.stApp [class*="st-key-chip-"] button p { font-size: 12px !important; }
.stApp [class*="st-key-chip-"] button:hover:not(:disabled) { border-color: var(--violet) !important; color: var(--violet) !important; }
[data-testid="stChatInput"] { background: var(--panel2) !important; border: 1px solid var(--line2) !important; border-radius: 12px !important; }
[data-testid="stChatInput"] div, [data-testid="stChatInput"] [data-baseweb] { background: transparent !important; border-color: transparent !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-family: var(--font) !important; font-size: 13px !important; background: transparent !important; }
[data-testid="stChatInput"] button { background: var(--lime) !important; color: var(--onbrand) !important; border-radius: 9px !important; }
.empty { border: 1px dashed var(--line2); border-radius: 12px; padding: 18px; color: var(--muted); font-size: 13px; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
.step { display: flex; flex-direction: column; gap: 8px; }
.step .n { font-family: var(--mono); font-size: 12px; color: var(--lime); }
.step b { font-size: 15px; font-weight: 600; }
.step span { color: var(--muted); font-size: 13px; line-height: 19px; }
.hero { padding: 30px 30px 28px 30px; background: radial-gradient(600px 220px at 85% 0%, color-mix(in srgb, var(--lime) 10%, transparent), transparent 70%), var(--panel); }
.stApp .hero h1 { font-size: 34px !important; line-height: 40px !important; font-weight: 600 !important; letter-spacing: -.025em; margin: 14px 0 8px 0 !important; padding: 0 !important; max-width: 720px; }
.hero p { color: var(--muted); font-size: 15px; max-width: 620px; line-height: 22px; margin: 0; }

/* ---------- Streamlit widgets ---------- */
[data-testid="stWidgetLabel"] p, label p { color: var(--muted) !important; font-size: 12.5px !important; }
[data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] [role="group"] { background: var(--panel2) !important; border-color: var(--line2) !important; border-radius: 10px !important; }
.stApp input { background: transparent !important; color: var(--text) !important; font-size: 13px !important; caret-color: var(--lime); }
[data-testid="stNumberInput"] button { background: transparent !important; color: var(--muted) !important; border-color: var(--line2) !important; }
[data-testid="stSelectbox"] svg { color: var(--muted) !important; }
[data-testid="stCheckbox"] p { font-size: 13px !important; color: var(--text) !important; }
[data-testid="stCheckbox"] label > div:first-of-type { background: var(--watch) !important; }
[data-testid="stCheckbox"] label > div:first-of-type > div { background: var(--panel) !important; }
[data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type { background: var(--lime) !important; }
/* the slider fill is an inline gradient in the config's primary colour: keep a neutral track, put the brand on the thumb */
[data-testid="stSlider"] div[style*="relative"] > div:first-child:empty { background: var(--watch) !important; }
[data-testid="stSlider"] div[style*="translate(-50%"] { background: var(--lime) !important; box-shadow: 0 0 0 4px color-mix(in srgb, var(--lime) 18%, transparent) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] p { font-family: var(--mono) !important; font-size: 12px; color: var(--text) !important; }
[data-testid="stSliderTickBar"] p { color: var(--faint) !important; }
[data-testid="stExpander"] details { background: var(--panel); border: 1px solid var(--line) !important; border-radius: 12px; }
[data-testid="stExpander"] summary p { font-size: 13px !important; color: var(--text) !important; }
[data-testid="stPlotlyChart"] { background: transparent; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"], [role="listbox"], [role="option"] { background: var(--panel) !important; color: var(--text) !important; }
[data-baseweb="popover"] li:hover, [role="option"]:hover, [role="option"][aria-selected="true"] { background: var(--panel2) !important; }
.btn.icon { width: 36px; padding: 0; justify-content: center; }
.stApp [data-testid^="stBaseButton"] { background: var(--panel) !important; color: var(--text) !important; border: 1px solid var(--line2) !important; border-radius: 10px !important; min-height: 36px; }
.stApp [data-testid^="stBaseButton"]:hover:not(:disabled) { border-color: var(--lime) !important; color: var(--lime) !important; }
.stApp [data-testid^="stBaseButton"]:disabled { opacity: .45; }
.stApp [data-testid^="stBaseButton"] p { font-size: 13px !important; font-weight: 500; }

/* ---------- phone ---------- */
.mobile-nav { display: none; }
.mobile-nav .row { display: flex; flex-wrap: wrap; gap: 6px; padding: 2px 0 12px 0; }
.mobile-nav .row a { color: var(--muted) !important; font-size: 13px; padding: 6px 11px; border-radius: 9px; border: 1px solid var(--line); }
.mobile-nav .row a.on { color: var(--text) !important; background: color-mix(in srgb, var(--text) 6%, transparent); border-color: var(--line2); }
@media (max-width: 760px) {
  .mobile-nav { display: block; margin-bottom: 8px; }
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display: none !important; }
  .block-container { padding: 16px 16px 56px 16px !important; }
  .kpis, .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .kpis.k3 { grid-template-columns: 1fr; gap: 10px; }
  .kpi .v { font-size: 26px; }
  .hdr { align-items: flex-start; }
  .hero { padding: 22px; } .stApp .hero h1 { font-size: 26px !important; line-height: 32px !important; }
  .rsn { grid-template-columns: minmax(0, 1fr) 70px 42px; }
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
    return (f'<div class="brand"><span class="logo">{icon("trail")}</span><b>PaperTrail</b></div>'
            f'<div class="nav">{nav}</div>'
            f'<div class="side-card"><div class="t"><span class="live"></span>Test window</div>{rows}</div>')


def mobile_nav_html(state: dict, pages: dict) -> str:
    nav = "".join(link(href(state, page=k, reveal=None, p=None), esc(v), "on" if state["page"] == k else "") for k, v in pages.items())
    return f'<div class="mobile-nav"><div class="row">{nav}</div></div>'


def seg(state: dict, models: dict, **keep) -> str:
    """The model switch: three linked segments, each with its series colour."""
    return '<div class="seg">' + "".join(
        link(href(state, model=k, **keep), f'<span class="sw" style="background:{MODEL_VAR[k]}"></span>{esc(v)}', "on" if state["model"] == k else "")
        for k, v in models.items()) + "</div>"


def header(title: str, sub: str = "", right: str = "", crumb: str = "", mono: bool = False, badge: str = "") -> str:
    return (f'<div class="hdr"><div>{f"<div class=crumb>{crumb}</div>" if crumb else ""}'
            f'<h1 class="{"mono" if mono else ""}"><span>{esc(title)}</span>{badge}</h1>'
            f'{f"<div class=sub>{sub}</div>" if sub else ""}</div><div class="right">{right}</div></div>')


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
