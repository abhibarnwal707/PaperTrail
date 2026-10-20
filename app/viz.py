"""Plotly figures for the console (no streamlit import). Transparent backgrounds, faint grid, and the active palette from
theme.PALETTES (passed in as `p`): lime / sky / violet are the blend / XGBoost / GNN series, coral is risk.
Render with st.plotly_chart(..., theme=None)."""
from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

from app.theme import model_color

TRANSPARENT = "rgba(0,0,0,0)"
SANS = "Geist, Inter, system-ui, sans-serif"
MONO = "Geist Mono, ui-monospace, Menlo, monospace"
NAMES = {"blend": "Blend", "xgb": "XGBoost", "gnn": "GNN", "xgb_tabular": "XGBoost, payment only", "rules_v1": "Rules"}


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _series(p: dict, m: str) -> str:
    return {"xgb_tabular": p["faint"], "rules_v1": p["watch"]}.get(m) or model_color(p, m)


def _base(p: dict, height: int, **layout) -> dict:
    return dict(height=height, margin=dict(l=0, r=0, t=8, b=0), paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
                font=dict(family=SANS, size=12, color=p["muted"]),
                hoverlabel=dict(bgcolor=p["panel2"], bordercolor=_rgba(p["text"], 0.16), font=dict(family=SANS, color=p["text"])), **layout)


def _axis(p: dict, **kw) -> dict:
    return {"showgrid": True, "gridcolor": p["grid"], "zeroline": False, "showline": False, "tickfont": dict(family=MONO, size=11, color=p["faint"]),
            "automargin": True, **kw}


def donut(labels: list[str], values: list[int], colors: list[str], center: str, center_sub: str, p: dict, height: int = 230) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.74, sort=False, direction="clockwise", marker=dict(colors=colors, line=dict(color=p["panel"], width=3)),
                           textinfo="none", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
    fig.update_layout(**_base(p, height), showlegend=False,
                      annotations=[dict(text=f"<b>{center}</b>", x=0.5, y=0.55, showarrow=False, font=dict(size=30, color=p["text"], family=SANS)),
                                   dict(text=center_sub, x=0.5, y=0.38, showarrow=False, font=dict(size=12, color=p["muted"]))])
    return fig


def activity_area(df: pl.DataFrame, model: str, p: dict) -> go.Figure:
    """Payments per hour in the test window, with the ones the model flags laid over them."""
    c = model_color(p, model)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["hour"], y=df["payments"], mode="lines", name="payments", line=dict(color=p["muted"], width=1.4, shape="spline"),
                             fill="tozeroy", fillcolor=_rgba(p["muted"], 0.08), hovertemplate="%{x|%b %d %H:00}: %{y} payments<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["hour"], y=df["flagged"], mode="lines", name="flagged", line=dict(color=c, width=2.2, shape="spline"),
                             fill="tozeroy", fillcolor=_rgba(c, 0.18), hovertemplate="%{x|%b %d %H:00}: %{y} flagged<extra></extra>"))
    fig.update_layout(**_base(p, 230), showlegend=False, hovermode="x unified",
                      xaxis=_axis(p, showgrid=False, tickformat="%b %d<br>%H:00", nticks=6), yaxis=_axis(p, nticks=4))
    return fig


def network_figure(pos: dict, rows: list[dict], center: str, model: str, p: dict) -> go.Figure:
    flag = f"{model}_flag"
    xs_f, ys_f, xs_n, ys_n = [], [], [], []
    for r in rows:
        (x0, y0, _), (x1, y1, _) = pos[r["src"]], pos[r["dst"]]
        (xs_f if r.get(flag) else xs_n).extend([x0, x1, None])
        (ys_f if r.get(flag) else ys_n).extend([y0, y1, None])
    degree: dict[str, int] = {}
    for r in rows:
        for n in (r["src"], r["dst"]):
            degree[n] = degree.get(n, 0) + 1
    nodes = [n for n in pos if n != center]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_n, y=ys_n, mode="lines", line=dict(width=0.8, color=_rgba(p["text"], 0.15)), hoverinfo="skip", name="payment"))
    fig.add_trace(go.Scatter(x=xs_f, y=ys_f, mode="lines", line=dict(width=1.8, color=p["coral"]), hoverinfo="skip", name="flagged payment"))
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name="counterparty",
                             marker=dict(size=[6 + min(degree.get(n, 1), 6) for n in nodes], color=[p["sky"] if pos[n][2] == 1 else p["faint"] for n in nodes],
                                         line=dict(width=0), opacity=0.9),
                             text=[f"{n} · {degree.get(n, 0)} payments" for n in nodes], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="this account", hoverinfo="text", text=[center],
                             marker=dict(size=34, color=_rgba(p["lime"], 0.2), line=dict(width=0))))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", hoverinfo="skip", showlegend=False, marker=dict(size=16, color=p["lime"], line=dict(width=0))))
    fig.update_layout(**_base(p, 420), showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=12, color=p["muted"]), itemsizing="constant"))
    return fig


def timeline(h: pl.DataFrame, model: str, p: dict) -> go.Figure:
    """The account's payments over time: amount on a log axis, flagged ones in coral, money in as mint, money out as grey."""
    d = h.to_dicts()
    flag = f"{model}_flag"
    fig = go.Figure()
    for name, sel, color, size in (("money out", lambda r: r["direction"] == "sent" and not r.get(flag), p["faint"], 8),
                                   ("money in", lambda r: r["direction"] != "sent" and not r.get(flag), p["mint"], 8),
                                   ("flagged", lambda r: bool(r.get(flag)), p["coral"], 11)):
        pts = [r for r in d if sel(r)]
        fig.add_trace(go.Scatter(x=[r["ts"] for r in pts], y=[r["amt_paid"] for r in pts], mode="markers", name=name,
                                 marker=dict(size=size, color=color, line=dict(width=0), opacity=0.95),
                                 text=[f'{r["direction"]} · {r["counterparty"]} · {r["amt_paid"]:,.2f}' for r in pts], hovertemplate="%{x|%b %d %H:%M}<br>%{text}<extra></extra>"))
    fig.update_layout(**_base(p, 200), xaxis=_axis(p, showgrid=False, tickformat="%b %d"), yaxis=_axis(p, type="log", nticks=4),
                      legend=dict(orientation="h", y=1.14, x=0, font=dict(size=12, color=p["muted"])))
    return fig


def gauge(score: float, threshold: float, p: dict) -> go.Figure:
    c = p["coral"] if score >= threshold else p["lime"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(valueformat=".2f", font=dict(size=44, color=p["text"], family=SANS)),
        gauge=dict(shape="angular", axis=dict(range=[0, 1], tickvals=[0, threshold, 1], ticktext=["0", f"alert {threshold:.2f}", "1"],
                                               tickfont=dict(family=MONO, size=11, color=p["muted"]), tickwidth=0, ticklen=0),
                   bar=dict(color=c, thickness=0.28), bgcolor=_rgba(p["text"], 0.05), borderwidth=0,
                   steps=[dict(range=[threshold, 1], color=_rgba(p["coral"], 0.12))],
                   threshold=dict(line=dict(color=p["text"], width=2), thickness=0.9, value=threshold))))
    fig.update_layout(**_base(p, 230))
    fig.update_layout(margin=dict(l=24, r=24, t=18, b=0))
    return fig


def leaderboard(metrics: dict, key: str, order: list[str], p: dict) -> go.Figure:
    m = metrics["models"]
    keys = [k for k in order if k in m][::-1]
    vals = [m[k][key] for k in keys]
    fig = go.Figure(go.Bar(x=vals, y=[NAMES[k] for k in keys], orientation="h", marker=dict(color=[_series(p, k) for k in keys], line=dict(width=0)),
                           text=[f"{v:.3f}" for v in vals], textposition="outside", textfont=dict(family=MONO, size=12, color=p["text"]), cliponaxis=False,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>", width=0.56))
    fig.update_layout(**_base(p, 250), xaxis=_axis(p, range=[0, max(vals) * 1.22], showticklabels=False, showgrid=False),
                      yaxis=dict(automargin=True, tickfont=dict(size=12.5, color=p["text"], family=SANS), showgrid=False))
    return fig


def pr_curves(metrics: dict, p: dict) -> go.Figure:
    fig = go.Figure()
    for m in ("xgb_tabular", "gnn", "xgb", "blend"):
        c = metrics["pr_curves"].get(m)
        if c:
            col = _series(p, m)
            fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", name=NAMES[m], line=dict(color=col, width=2.6 if m == "blend" else 1.8),
                                     fill="tozeroy" if m == "blend" else None, fillcolor=_rgba(col, 0.08) if m == "blend" else None))
    fig.update_layout(**_base(p, 250), xaxis=_axis(p, range=[0, 1], dtick=0.25, title=dict(text="recall", font=dict(size=11, color=p["faint"]))),
                      yaxis=_axis(p, range=[0, 1], dtick=0.25, title=dict(text="precision", font=dict(size=11, color=p["faint"]))),
                      legend=dict(orientation="h", y=1.14, x=0, font=dict(size=12, color=p["muted"])))
    return fig


def typology_heat(rows: list[dict], p: dict) -> go.Figure:
    """Share of each laundering pattern found, per model, as a heatmap from the card colour to the brand colour."""
    models = ["xgb", "gnn", "blend"]
    ys = [("No known pattern" if r["typology"] == "UNMAPPED" else r["typology"].title()) + f"  ·  {r['n']}" for r in rows]
    z = [[r[f"recall_{m}"] for m in models] for r in rows]
    # cells past the midpoint switch to the brand's own "on" ink so the % stays readable in both palettes
    text_color = [[p["onbrand"] if v > 0.55 else p["text"] for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=[NAMES[m] for m in models], y=ys, zmin=0, zmax=1, xgap=4, ygap=4,
                               colorscale=[[0, p["panel2"]], [0.5, _rgba(p["lime"], 0.45)], [1, p["lime"]]], showscale=False,
                               hovertemplate="%{y} · %{x}: %{z:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[NAMES[m] for r in rows for m in models], y=[y for y in ys for _ in models], mode="text", hoverinfo="skip", showlegend=False,
                             text=[f"{v * 100:.0f}%" for row in z for v in row], textfont=dict(family=MONO, size=12, color=[c for row in text_color for c in row])))
    fig.update_layout(**_base(p, 34 * len(rows) + 40), xaxis=dict(side="top", tickfont=dict(size=12, color=p["muted"]), showgrid=False, zeroline=False),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=12, color=p["text"]), showgrid=False, zeroline=False, automargin=True))
    return fig
