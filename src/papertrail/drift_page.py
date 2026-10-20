"""The public drift page (docs/index.html on GitHub Pages): verdict, score drift, the PSI table and the daily trend,
with links to the full Evidently reports. One self-contained HTML file, no scripts, in the console's two palettes
(Night Arcade when the reader's system is dark, Riso Paper when it is light)."""
from __future__ import annotations

import html

import polars as pl

from papertrail.drift import MODERATE, RETRAIN_PSI, STABLE, band

CSS = """
:root { --bg:#f4efe6; --panel:#fffcf6; --line:rgba(31,28,23,.10); --text:#1f1c17; --muted:#6d675d; --brand:#2f43d0; --risk:#c20e6e; --amber:#9a5a00; --ok:#18885a; --onbrand:#fff; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d0c13; --panel:#15141d; --line:rgba(242,240,250,.08); --text:#f2f0fa; --muted:#9a97a8; --brand:#d7ff3a; --risk:#ff5c8a; --amber:#ffc24b; --ok:#5cf2b2; --onbrand:#0d0c13; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 Geist, Inter, system-ui, sans-serif; }
main { max-width:1080px; margin:0 auto; padding:40px 20px 64px; }
h1 { font-size:32px; line-height:38px; margin:8px 0 6px; letter-spacing:-.02em; }
h2 { font-size:18px; margin:0 0 4px; }
p.sub, .s { color:var(--muted); margin:0; }
.mono, td.num { font-family: 'Geist Mono', ui-monospace, Menlo, monospace; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:20px; margin-top:16px; }
.grid { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:14px; margin-top:16px; }
.grid .card { margin:0; }
.k { color:var(--muted); font-size:13px; } .v { font-size:28px; font-weight:600; letter-spacing:-.02em; margin-top:6px; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12.5px; font-weight:500; border:1px solid currentColor; }
.stable { color:var(--ok); } .moderate { color:var(--amber); } .major { color:var(--risk); } .cal { color:var(--muted); }
.verdict { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
.verdict b { font-size:22px; }
table { width:100%; border-collapse:collapse; font-size:13.5px; margin-top:12px; }
th { text-align:left; color:var(--muted); font-weight:500; padding:8px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
td { padding:8px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
td:first-child { white-space:normal; min-width:200px; }
body { font-variant-numeric: tabular-nums; }
td.num, th.num { text-align:right; }
.bar { display:inline-block; height:6px; border-radius:3px; vertical-align:middle; margin-right:8px; background:currentColor; }
.heat td.c { text-align:center; padding:8px 4px; font-family:'Geist Mono', ui-monospace, monospace; font-size:12px; }
.links a { color:var(--brand); margin-right:18px; }
.btn { display:inline-block; background:var(--brand); color:var(--onbrand); padding:8px 14px; border-radius:10px; text-decoration:none; font-weight:500; }
.wrap { overflow-x:auto; }
@media (max-width:760px) { .grid { grid-template-columns:repeat(2, minmax(0,1fr)); } h1 { font-size:26px; line-height:32px; } }
"""


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _num(x) -> str:
    """Compact figure: 0.299, 11.5, 8.1k, 1.4M (never scientific notation)."""
    if x is None:
        return ""
    a = abs(x)
    return f"{x / 1e6:.1f}M" if a >= 1e6 else f"{x / 1e3:.1f}k" if a >= 1e4 else f"{x:,.0f}" if a >= 100 else f"{x:.3g}"


def _pill(b: str, text: str | None = None) -> str:
    return f'<span class="pill {b}">{_e(text or b)}</span>'


def _cell_style(value: float) -> str:
    b = band(value)
    alpha = 0.10 if b == "stable" else 0.25 if b == "moderate" else 0.45
    var = {"stable": "--ok", "moderate": "--amber", "major": "--risk"}[b]
    return f"background: color-mix(in srgb, var({var}) {alpha * 100:.0f}%, transparent)"


def render(summary: dict, table: pl.DataFrame, daily: pl.DataFrame, labels: dict | None = None, top: int = 20) -> str:
    """summary: generated, reference, current, decision (from drift.retrain_decision), score_psi {model: psi},
    alert_rate [{window, xgb, gnn, blend}], perf_by_day [{day, n, positives, alert_rate, pr_auc}], links {name: url}."""
    labels = labels or {}
    name = labels.get
    d = summary["decision"]
    stable_rows = table.filter(~pl.col("calendar_dependent"))
    counts = {b: int((stable_rows["band"] == b).sum()) for b in ("stable", "moderate", "major")}
    verdict = ("Retrain recommended", "major") if d["retrain"] else ("No retraining needed", "stable")

    kpis = "".join(f'<div class="card"><div class="k">{_e(k)}</div><div class="v">{v}</div><div class="s">{s}</div></div>' for k, v, s in (
        ("Features checked", table.height, f"{table.height - stable_rows.height} calendar-dependent"),
        ("Stable features", counts["stable"], f"PSI &lt; {STABLE} (calendar-stable only)"),
        ("Shifted features", counts["moderate"] + counts["major"], f"{counts['major']} with PSI &ge; {MODERATE}"),
        ("Blend score PSI", f'{summary["score_psi"]["blend"]:.3f}', "validation &rarr; test"),
    ))

    rows = []
    peak = max(table.head(top)["psi"].max() or 0.0, 0.3)
    for r in table.head(top).to_dicts():
        b = "cal" if r["calendar_dependent"] else r["band"]
        tag = _pill("cal", "calendar") if r["calendar_dependent"] else _pill(r["band"])
        rows.append(f'<tr><td>{_e(name(r["feature"], r["feature"]))}<div class="s mono" style="font-size:12px">{_e(r["feature"])}</div></td>'
                    f'<td class="num">{r["importance"] * 100:.1f}%</td>'
                    f'<td class="num {b}"><span class="bar" style="width:{min(r["psi"] / peak, 1) * 90:.0f}px"></span>{r["psi"]:.3f}</td><td>{tag}</td>'
                    f'<td class="num">{_num(r["ref_mean"])}</td><td class="num">{_num(r["cur_mean"])}</td></tr>')

    days = sorted(daily["day"].unique().to_list())
    feats = [f for f in table.head(top)["feature"].to_list() if f in set(daily["feature"].to_list())][:12]
    lookup = {(r["feature"], r["day"]): r["psi"] for r in daily.to_dicts()}
    heat = "".join(f'<tr><td>{_e(name(f, f))}</td>' + "".join(
        f'<td class="c" style="{_cell_style(lookup[(f, dd)])}">{lookup[(f, dd)]:.2f}</td>' if (f, dd) in lookup else "<td></td>" for dd in days) + "</tr>" for f in feats)

    alert = "".join(f'<tr><td>{_e(r["window"])}</td>' + "".join(f'<td class="num">{r[m] * 100:.2f}%</td>' for m in ("xgb", "gnn", "blend")) + "</tr>"
                    for r in summary["alert_rate"])
    perf = "".join(f'<tr><td>{_e(r["day"])}</td><td class="num">{r["n"]:,}</td><td class="num">{r["positives"]:,}</td>'
                   f'<td class="num">{r["alert_rate"] * 100:.2f}%</td><td class="num">{r["pr_auc"]:.3f}</td></tr>' for r in summary["perf_by_day"])
    links = "".join(f'<a href="{_e(u)}">{_e(k)}</a>' for k, u in summary.get("links", {}).items())
    drifted = ", ".join(name(f, f) for f in d["drifted"]) or "none"
    n_expected = len(d["expected_calendar_drift"])

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PaperTrail drift report</title><meta name="description" content="Data and prediction drift for the PaperTrail AML models">
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><main>
<span class="pill cal">PaperTrail · monitoring</span>
<h1>Drift report</h1>
<p class="sub">Reference: {_e(summary["reference"])}. Current: {_e(summary["current"])}. Generated {_e(summary["generated"])}.</p>
<div class="card verdict">{_pill(verdict[1], "action needed" if d["retrain"] else "healthy")}<div><b>{_e(verdict[0])}</b><p class="s">Rule: {_e(d["rule"])}. Drifted: {_e(drifted)}.</p></div></div>
<div class="grid">{kpis}</div>
<div class="card"><h2>Feature drift</h2><p class="s">The {top} most important model inputs (share of XGBoost gain). PSI bands: stable &lt; {STABLE}, moderate {STABLE}&ndash;{MODERATE}, major &ge; {MODERATE}.
Calendar features (7-day windows, all-time totals, distinct-so-far counts, PageRank, reverse-edge flag) drift by construction because the data starts on Sep 1, so they never trigger retraining.
Here {n_expected} calendar features moved past PSI {RETRAIN_PSI}, as expected.</p>
<div class="wrap"><table><thead><tr><th>Feature</th><th class="num">Importance</th><th class="num">PSI</th><th>Band</th><th class="num">Mean, reference</th><th class="num">Mean, current</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>
<div class="card"><h2>Day by day</h2><p class="s">PSI of each day against the reference, for the most important features. Drift that grows day after day is the calendar effect; a jump on one day is an event.</p>
<div class="wrap"><table class="heat"><thead><tr><th>Feature</th>{"".join(f'<th class="num">{_e(dd)}</th>' for dd in days)}</tr></thead><tbody>{heat}</tbody></table></div></div>
<div class="card"><h2>Prediction drift</h2><p class="s">Share of payments above each model's alert line (set on validation), and the blend's quality per test day. Labels are known here only because the data is synthetic; in production this table arrives weeks late, which is why the input drift above is watched daily.</p>
<div class="wrap"><table><thead><tr><th>Window</th><th class="num">XGBoost</th><th class="num">GNN</th><th class="num">Blend</th></tr></thead><tbody>{alert}</tbody></table></div>
<div class="wrap"><table><thead><tr><th>Test day</th><th class="num">Payments</th><th class="num">Laundering</th><th class="num">Blend alert rate</th><th class="num">Blend PR-AUC</th></tr></thead><tbody>{perf}</tbody></table></div>
<p class="s" style="margin-top:10px">Score PSI, validation &rarr; test: {" · ".join(f"{m} {v:.3f}" for m, v in summary["score_psi"].items())}.</p></div>
<div class="card links">{links}</div>
</main></body></html>"""
