"""Plotly figures for the console (no streamlit import). Flat and quiet: no chart chrome, faint grid, one hue (amber) only on the
data that matters, signs in the labels instead of colour on the numbers. Render with st.plotly_chart(..., theme=None)."""
from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

ACCENT, INK, MUTED, FAINT, GRID, CONTROL = "#e2a03a", "#f2f2f3", "#8a8c92", "#5d5f66", "#2b2b2e", "#3a3a3e"
TRANSPARENT = "rgba(0,0,0,0)"
SANS = "Public Sans, Inter, system-ui, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, Menlo, monospace"
MODEL_STYLE = {"blend": (ACCENT, 2.6, None), "xgb": (INK, 1.8, None), "gnn": (MUTED, 1.8, None), "xgb_tabular": (FAINT, 1.4, "dot"), "rules_v1": (FAINT, 1.0, "dot")}


def _base(height: int, **layout) -> dict:
    return dict(height=height, margin=dict(l=0, r=0, t=6, b=6), paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
                font=dict(family=SANS, size=12, color=MUTED), **layout)


def network_figure(pos: dict, rows: list[dict], center: str, model: str) -> go.Figure:
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
    nodes = list(pos)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_n, y=ys_n, mode="lines", line=dict(width=0.8, color=CONTROL), hoverinfo="skip", name="payment"))
    fig.add_trace(go.Scatter(x=xs_f, y=ys_f, mode="lines", line=dict(width=1.8, color=ACCENT), hoverinfo="skip", name="payment the model flags"))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name="account",
        marker=dict(size=[16 if n == center else 5 + min(degree.get(n, 1), 6) for n in nodes],
                    color=["#ffffff" if n == center else MUTED for n in nodes], line=dict(width=0)),
        text=[("This account: " if n == center else "") + n + f" ({degree.get(n, 0)} payments shown)" for n in nodes], hoverinfo="text"))
    fig.update_layout(**_base(500), showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", y=1.04, x=0, font=dict(size=12, color=MUTED)))
    return fig


def reasons_bar(reasons: pl.DataFrame, top: int = 8) -> go.Figure:
    """Signed contribution per reason. Amber raises the score, grey lowers it; the sign is also printed."""
    r = reasons.head(top).reverse()
    contrib = r["contribution"].to_list()
    fig = go.Figure(go.Bar(
        x=contrib, y=[f"{a}: {v}" for a, v in zip(r["label"].to_list(), r["value_text"].to_list())], orientation="h",
        marker=dict(color=[ACCENT if c > 0 else FAINT for c in contrib], line=dict(width=0)),
        text=[f"{c:+.2f}" for c in contrib], textposition="outside", textfont=dict(family=MONO, size=12, color=INK), cliponaxis=False,
        hovertemplate="%{y}<br>%{x:+.2f}<extra></extra>"))
    span = max(abs(c) for c in contrib) if contrib else 1
    fig.update_layout(**_base(max(240, 40 * len(r) + 40)), bargap=0.45,
                      xaxis=dict(range=[min(min(contrib, default=0), 0) - 0.1 * span, max(max(contrib, default=0), 0) + 0.28 * span], showgrid=False,
                                 zeroline=True, zerolinecolor=MUTED, zerolinewidth=1, showticklabels=False),
                      yaxis=dict(automargin=True, tickfont=dict(size=12, color=INK)))
    return fig


def pr_curves(metrics: dict, names: dict) -> go.Figure:
    fig = go.Figure()
    for m, label in names.items():
        c = metrics["pr_curves"].get(m)
        if c:
            color, width, dash = MODEL_STYLE.get(m, (MUTED, 1.5, None))
            fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", name=label, line=dict(color=color, width=width, dash=dash)))
    axis = dict(range=[0, 1.0], showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False, dtick=0.25, automargin=True,
                tickfont=dict(family=MONO, size=11, color=MUTED), title=dict(standoff=10, font=dict(size=12, color=MUTED)))
    fig.update_layout(**_base(340), xaxis=dict(title_text="Recall: share of laundering found", **axis),
                      yaxis=dict(title_text="Precision: share of alerts that are right", **axis),
                      legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12, color=MUTED)))
    return fig
