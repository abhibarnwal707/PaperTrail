"""Look and feel of the console: one stylesheet and the small HTML builders (no streamlit import, so it is unit-tested).

Style: the "data terminal" of the taste-skill app library, dark, dense, flat. Decisions that make it this product's:
- the figures are set in a monospace, like a ledger tape (the product is called PaperTrail);
- the one accent is a sodium amber, used as a 1px selection line and on the data that matters (flagged payments, risk-raising reasons);
- cards are DARKER than the page, separated by 1px lines, no shadows, no gradients;
- state lives in the URL (page, model, account), so every view can be linked and the HTML tables stay plain HTML.
"""
from __future__ import annotations

import html
from urllib.parse import urlencode

ACCENT = "#e2a03a"
FONT_LINKS = ("@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&"
              "family=IBM+Plex+Mono:wght@400;500;600&display=swap');")

CSS = FONT_LINKS + """
:root {
  --ground: #171719; --surface: #131217; --raised: #212023; --border: #222224; --rule: #1b1a1f; --control: #3a3a3e;
  --text: #ffffff; --muted: #8a8c92; --accent: __ACCENT__; --positive: #5fc39c; --negative: #e5566f;
  --font: 'Public Sans', Inter, system-ui, sans-serif; --mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--ground) !important; color: var(--text); font-family: var(--font); }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu, [data-testid="stStatusWidget"] { display: none !important; }
.block-container { max-width: 1320px; padding: 28px 32px 72px 32px !important; }
h1, h2, h3, h4 { font-family: var(--font); letter-spacing: 0; }
[data-testid="stHeadingWithActionElements"] a { display: none !important; }
p, li, label, span, div { font-size: 13px; }
.stApp, .stApp * { font-variant-numeric: tabular-nums; }
.stApp a, .stApp a:visited, .stApp a:hover { color: inherit !important; text-decoration: none !important; }
/* Streamlit sets its own face on most elements: win it back, then give figures the mono face with a more specific rule */
.stApp, .stApp p, .stApp div, .stApp span, .stApp label, .stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button, .stApp a, .stApp summary { font-family: var(--font) !important; }
.stApp .num, .stApp .rtop .rc, .stApp td.num, .stApp .kpi-v, .stApp .kpi-v small, .stApp .kpi-s, .stApp .tape td, .stApp .scoreaxis, .stApp .scoreaxis span, .stApp .score-fig, .stApp .pager,
.stApp .tbl td.id, .stApp .tbl td.id a, .stApp .reason .v, .stApp .titlerow h1.mono, .stApp .titlerow .sub b, .stApp .tools .num, .stApp input, .stApp [data-baseweb="select"] div { font-family: var(--mono) !important; }
.stApp .tbl th, .stApp .tape th, .stApp th.num { font-family: var(--font) !important; }
.stApp .titlerow h1 { font-size: 26px !important; font-weight: 600 !important; line-height: 32px !important; margin: 0 !important; padding: 0 !important; color: var(--text) !important; }
.stApp .titlerow h1.mono { font-size: 24px !important; font-weight: 500 !important; }
.stApp .titlerow h1 span { font-size: inherit !important; font-weight: inherit !important; line-height: inherit !important; font-family: inherit !important; }

/* ---------- sidebar: on the page ground, one 1px edge, a 1px accent line marks the selection ---------- */
[data-testid="stSidebar"] { background: var(--ground) !important; border-right: 1px solid var(--border); min-width: 224px !important; max-width: 224px !important; width: 224px !important; }
[data-testid="stSidebar"] > div:first-child { padding: 0; }
[data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] { padding: 22px 0 0 0 !important; }
.brand { display: flex; align-items: center; gap: 10px; padding: 0 20px 4px 20px; }
.brand svg { flex: none; }
.brand b { font-size: 16px; font-weight: 600; }
.scope { padding: 0 20px 22px 20px; color: var(--muted); font-size: 13px; line-height: 1.5; border-bottom: 1px solid var(--border); margin-bottom: 14px; }
.nav-group { padding: 14px 20px 6px 20px; color: var(--muted); font-size: 12px; }
.nav a { display: block; position: relative; padding: 7px 20px; font-size: 13px; line-height: 20px; }
.stApp .nav a { color: var(--muted) !important; }
.stApp .nav a:hover, .stApp .nav a.on { color: var(--text) !important; }
.stApp .nav a.on { font-weight: 500; }
.nav a.on::after { content: ''; position: absolute; right: -1px; top: 4px; bottom: 4px; width: 1px; background: var(--accent); }
.nav .note { display: block; color: var(--muted); font-size: 11px; line-height: 1.45; padding: 0 20px 6px 20px; }
.foot { margin-top: 28px; padding: 14px 20px 18px 20px; color: var(--muted); font-size: 11px; line-height: 1.5; border-top: 1px solid var(--border); }

/* ---------- title row ---------- */
.titlerow { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin: 0 0 24px 0; }
.titlerow h1 { font-size: 26px; font-weight: 600; line-height: 32px; margin: 0; padding: 0; }
.titlerow h1.mono { font-family: var(--mono); font-size: 24px; font-weight: 500; }
.titlerow .sub { color: var(--muted); font-size: 14px; line-height: 20px; margin-top: 4px; max-width: 760px; }
.titlerow .sub b { color: var(--text); font-weight: 500; font-family: var(--mono); font-size: 13px; }
.btn { display: inline-block; padding: 0 16px; height: 34px; line-height: 34px; border-radius: 6px; font-size: 13px; font-weight: 500; white-space: nowrap; }
.btn.primary { background: #fff; color: #000 !important; }
.btn.ghost { border: 1px solid var(--control); color: var(--text); }
.btn:hover { opacity: .88; }

/* ---------- figures, cards, KPI strip ---------- */
.num, td.num, th.num, .kpi-v, .tape td { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.kpis { display: flex; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 24px; }
.kpi { flex: 1 1 0; padding: 16px 20px 16px 20px; border-left: 1px solid var(--border); min-width: 0; }
.kpi:first-child { border-left: 0; }
.kpi.on { box-shadow: inset 0 -1px 0 var(--accent); }
.kpi-l { color: var(--muted); font-size: 13px; line-height: 16px; }
.kpi-v { font-size: 20px; line-height: 28px; margin-top: 4px; font-weight: 500; }
.kpi-v small { font-size: 12px; color: var(--muted); font-weight: 400; margin-left: 4px; }
.kpi-s { color: var(--muted); font-size: 12px; line-height: 16px; margin-top: 2px; font-family: var(--mono); }
[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
[data-testid="stColumn"] { align-self: stretch; }
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height: 100%; }
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has([data-testid="stVerticalBlockBorderWrapper"]) { height: 100%; }
[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] { height: 100%; }
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; padding: 14px 16px 20px 16px !important; }
.card-title { font-size: 16px; font-weight: 600; line-height: 22px; margin: 0 0 4px 0; }
.card-sub { color: var(--muted); font-size: 13px; line-height: 17px; margin: 0 0 12px 0; }

/* ---------- tables: rules and alignment, no zebra, no header band ---------- */
.tbl-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: auto; max-height: 620px; }
table.tbl, table.tape { width: 100%; border-collapse: collapse; font-size: 13px; }
.stApp .tbl th, .stApp .tbl td, .stApp .tape th, .stApp .tape td { border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; }
.tbl th, .tape th { position: sticky; top: 0; background: var(--surface); text-align: left; font-weight: 400; font-size: 12px; color: var(--muted); padding: 0 16px; height: 38px; border-bottom: 1px solid var(--border); white-space: nowrap; z-index: 1; }
.tbl td, .tape td { padding: 0 16px; height: 42px; border-bottom: 1px solid var(--rule); vertical-align: middle; white-space: nowrap; }
.tbl tr:last-child td, .tape tr:last-child td { border-bottom: 0; }
.tbl td.wrap { white-space: normal; min-width: 340px; line-height: 19px; padding-top: 8px; padding-bottom: 8px; color: #b4b4b4; font-size: 13px; }
.tbl th.num, .tbl td.num, .tape th.num, .tape td.num { text-align: right; }
.tbl tbody tr:hover td, .tape tbody tr:hover td { background: var(--raised); }
.tbl td.id a { font-family: var(--mono); }
.tbl td.id a:hover { text-decoration: underline; text-underline-offset: 3px; }
.tbl td.dim, .tape td.dim { color: var(--muted); }
.tbl td.best, .tape td.best { color: #fff; font-weight: 600; }
.tbl td.rest, .tape td.rest { color: #b4b4b4; }
.flagmark { display: inline-block; width: 1px; height: 14px; background: var(--accent); margin-right: 8px; vertical-align: middle; }
.pager { display: flex; justify-content: space-between; align-items: center; color: var(--muted); font-size: 12px; margin-top: 12px; font-family: var(--mono); }
.pager a { color: var(--text); border-bottom: 1px solid var(--accent); padding-bottom: 1px; }
.tools { display: flex; gap: 24px; align-items: center; margin: 0 0 12px 0; color: var(--muted); font-size: 13px; }
.tools a { color: var(--text); border-bottom: 1px solid var(--accent); padding-bottom: 1px; }
.tools a.off { color: var(--muted); border-bottom-color: transparent; }

/* ---------- reasons: a ranked list, each with a thin proportional bar ---------- */
.rsn { padding: 10px 0 12px 0; border-bottom: 1px solid var(--rule); }
.rsn:last-child { border-bottom: 0; }
.rtop { display: flex; justify-content: space-between; gap: 16px; font-size: 13px; line-height: 18px; }
.rtop .rc { white-space: nowrap; }
.rval { color: var(--muted); font-size: 13px; line-height: 18px; }
.rbar { height: 2px; margin-top: 7px; }
.rbar i { display: block; height: 2px; background: #5d5f66; }
.rbar i.up { background: var(--accent); }
.pane-r { border-left: 1px solid var(--border); padding-left: 28px; }

/* ---------- score bar and strips ---------- */
.verdict { font-size: 26px; font-weight: 600; line-height: 32px; margin: 0; }
.verdict.on::before { content: ''; display: inline-block; width: 1px; height: 22px; background: var(--accent); margin-right: 12px; vertical-align: -3px; }
.scorebar { position: relative; height: 8px; background: var(--raised); border-radius: 4px; margin: 14px 0 6px 0; }
.scorebar i { position: absolute; left: 0; top: 0; bottom: 0; background: #f2f2f3; border-radius: 4px 0 0 4px; }
.scorebar i.hot { background: var(--accent); }
.scorebar b { position: absolute; top: -5px; bottom: -5px; width: 1px; background: var(--text); }
.scoreaxis { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; font-family: var(--mono); }
.score-fig { font-family: var(--mono); font-size: 36px; line-height: 44px; font-weight: 500; }
.stack { display: flex; height: 10px; gap: 2px; margin: 10px 0 8px 0; }
.stack i { display: block; background: #3a3a3e; }
.stack i.a { background: var(--accent); } .stack i.w { background: #f2f2f3; } .stack i.g { background: #8a8c92; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 20px; color: var(--muted); font-size: 13px; }
.legend .d { display: inline-block; width: 8px; height: 8px; border-radius: 4px; margin-right: 6px; background: #3a3a3e; }
.legend .d.a { background: var(--accent); } .legend .d.w { background: #f2f2f3; } .legend .d.g { background: #8a8c92; }
.reason { display: flex; justify-content: space-between; gap: 16px; padding: 9px 0; border-bottom: 1px solid var(--rule); font-size: 13px; line-height: 18px; }
.reason:last-child { border-bottom: 0; }
.reason .v { font-family: var(--mono); white-space: nowrap; }
.note { color: var(--muted); font-size: 13px; line-height: 19px; }
.note b { color: var(--text); font-weight: 500; }
.empty { border: 1px dashed var(--control); border-radius: 6px; padding: 20px; color: var(--muted); font-size: 13px; }

/* ---------- Streamlit widgets: dark, flat, 1px controls ---------- */
[data-testid="stWidgetLabel"] p, label p { color: var(--muted) !important; font-size: 12px !important; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] { background: var(--surface) !important; border-color: var(--control) !important; border-radius: 6px !important; }
input, [data-baseweb="select"] div { font-family: var(--mono) !important; font-size: 13px !important; color: var(--text) !important; }
[data-testid="stNumberInput"] button { background: var(--surface) !important; border-color: var(--control) !important; }
[data-testid="stCheckbox"] span { font-size: 13px !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--text) !important; }
[data-testid="stSlider"] div[data-testid="stSliderThumbValue"] { font-family: var(--mono); font-size: 12px; color: var(--text); }
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--border) !important; border-radius: 6px; }
[data-testid="stExpander"] summary p { font-size: 13px !important; color: var(--text) !important; }
[data-testid="stPlotlyChart"] { background: transparent; }
/* ---------- phone: the sidebar is replaced by a plain top row of links ---------- */
.mobile-nav { display: none; }
.mobile-nav .row { display: flex; gap: 18px; overflow-x: auto; white-space: nowrap; padding: 2px 0 10px 0; border-bottom: 1px solid var(--border); margin-bottom: 4px; }
.mobile-nav a { color: var(--muted) !important; font-size: 13px; padding-bottom: 6px; border-bottom: 1px solid transparent; }
.mobile-nav a.on { color: var(--text) !important; border-bottom-color: var(--accent); }
.mobile-nav .brandline { display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 15px; padding: 2px 0 10px 0; }
@media (max-width: 760px) { .mobile-nav { display: block; margin-bottom: 16px; } [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display: none !important; } }
@media (max-width: 760px) { .block-container { padding: 16px 16px 56px 16px !important; } .kpis { flex-wrap: wrap; } .kpi { flex: 1 1 45%; border-top: 1px solid var(--border); } .titlerow { flex-direction: column; align-items: flex-start; } }
""".replace("__ACCENT__", ACCENT)

MARK = ('<svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true"><path d="M5 2.5h12v17l-2-1.4-2 1.4-2-1.4-2 1.4-2-1.4-2 1.4z" '
        f'stroke="{ACCENT}" stroke-width="1.4" stroke-linejoin="round"/><path d="M8 7h6M8 10.5h6M8 14h3.5" stroke="{ACCENT}" stroke-width="1.2" stroke-linecap="round"/></svg>')


def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def href(state: dict, **change) -> str:
    """A link that keeps the current view state (page, model, reveal...) and changes only what is given. None removes a key."""
    merged = {**state, **change}
    return "?" + urlencode({k: v for k, v in merged.items() if v not in (None, "")})


def sidebar_html(state: dict, pages: dict, models: dict, blurbs: dict, scope: str) -> str:
    nav = "".join(f'<a class="{"on" if state["page"] == k else ""}" target="_self" href="{esc(href(state, page=k))}">{esc(label)}</a>' for k, label in pages.items())
    mods = "".join(f'<a class="{"on" if state["model"] == k else ""}" target="_self" href="{esc(href(state, model=k))}">{esc(label)}</a>' for k, label in models.items())
    return (f'<div class="brand">{MARK}<b>PaperTrail</b></div><div class="scope">{esc(scope)}</div>'
            f'<div class="nav">{nav}</div><div class="nav-group">Model</div><div class="nav">{mods}'
            f'<span class="note">{esc(blurbs[state["model"]])}</span></div>'
            '<div class="foot">Synthetic data. A demonstration of the method, not a product for real banks.</div>')


def mobile_nav_html(state: dict, pages: dict, models: dict) -> str:
    nav = "".join(f'<a class="{"on" if state["page"] == k else ""}" target="_self" href="{esc(href(state, page=k))}">{esc(v)}</a>' for k, v in pages.items())
    mods = "".join(f'<a class="{"on" if state["model"] == k else ""}" target="_self" href="{esc(href(state, model=k))}">{esc(v.split(" (")[0])}</a>' for k, v in models.items())
    return (f'<div class="mobile-nav"><div class="brandline">{MARK}PaperTrail</div><div class="row">{nav}</div>'
            f'<div class="row" style="border-bottom:0;padding-top:8px"><span style="color:var(--muted);font-size:13px">Model</span>{mods}</div></div>')


def title_row(title: str, sub: str = "", action: str = "", mono: bool = False) -> str:
    return (f'<div class="titlerow"><div><h1 class="{"mono" if mono else ""}">{esc(title)}</h1>'
            f'{f"<div class=sub>{sub}</div>" if sub else ""}</div><div>{action}</div></div>')


def button(text: str, url: str, kind: str = "primary") -> str:
    return f'<a class="btn {kind}" target="_self" href="{esc(url)}">{esc(text)}</a>'


def kpi_strip(cells: list[tuple]) -> str:
    """One container split by 1px dividers: (label, value, sub[, selected]) per cell. A selected cell gets a 1px accent line on its bottom edge."""
    body = "".join(f'<div class="kpi{" on" if len(c) > 3 and c[3] else ""}"><div class="kpi-l">{esc(c[0])}</div><div class="kpi-v">{c[1]}</div>'
                   f'{f"<div class=kpi-s>{esc(c[2])}</div>" if c[2] else ""}</div>' for c in cells)
    return f'<div class="kpis">{body}</div>'


def table(columns: list[tuple], rows: list[dict], cls: str = "tbl", wrap: bool = True) -> str:
    """columns: (key, header, kind) with kind in {'id','num','text','wrap','dim'}; rows hold already-escaped HTML in the cells."""
    head = "".join(f'<th class="{"num" if k == "num" else ""}">{esc(h)}</th>' for _, h, k in columns) if any(h for _, h, _ in columns) else ""
    body = "".join("<tr>" + "".join(f'<td class="{k}">{r.get(key, "")}</td>' for key, _, k in columns) + "</tr>" for r in rows)
    t = f'<table class="{cls}">{f"<thead><tr>{head}</tr></thead>" if head else ""}<tbody>{body}</tbody></table>'
    return f'<div class="tbl-wrap">{t}</div>' if wrap else t


def card_title(title: str, sub: str = "") -> str:
    return f'<div class="card-title">{esc(title)}</div>' + (f'<div class="card-sub">{esc(sub)}</div>' if sub else "")


def reasons_list(reasons, top: int = 8) -> str:
    """The strongest drivers as a ranked list: label and signed push on one line, the value under it, a thin bar for the size.
    Amber raises the score, grey lowers it; the sign is in the text, so colour is never the only signal."""
    rows = reasons.head(top).to_dicts()
    if not rows:
        return '<div class="empty">No explanation is stored for this account.</div>'
    peak = max(abs(r["contribution"]) for r in rows) or 1.0
    out = []
    for r in rows:
        w = abs(r["contribution"]) / peak * 100
        up = "up" if r["contribution"] > 0 else ""
        out.append(f'<div class="rsn"><div class="rtop"><span>{esc(r["label"])}</span><span class="rc">{r["contribution"]:+.2f}</span></div>'
                   f'<div class="rval">{esc(r["value_text"])}</div><div class="rbar"><i class="{up}" style="width:{w:.1f}%"></i></div></div>')
    return "".join(out)


def score_bar(score: float, threshold: float) -> str:
    s, t = max(0.0, min(score, 1.0)) * 100, max(0.0, min(threshold, 1.0)) * 100
    hot = "hot" if score >= threshold else ""
    digits = 3 if threshold < 0.1 else 2
    return (f'<div class="scorebar"><i class="{hot}" style="width:{s:.1f}%"></i><b style="left:{t:.1f}%"></b></div>'
            f'<div class="scoreaxis"><span>0</span><span>alerts start at {threshold:.{digits}f}</span><span>1</span></div>')


def stack_bar(parts: list[tuple]) -> str:
    """(label, value, css class) parts as a 100% bar with 2px seams and a legend of the same marks."""
    total = sum(v for _, v, _ in parts) or 1
    bar = "".join(f'<i class="{c}" style="width:{v / total * 100:.2f}%"></i>' for _, v, c in parts if v)
    leg = "".join(f'<span><span class="d {c}"></span>{esc(lab)} <b class="num">{v}</b></span>' for lab, v, c in parts)
    return f'<div class="stack">{bar}</div><div class="legend">{leg}</div>'
