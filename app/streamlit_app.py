"""PaperTrail investigator console (Streamlit). Run:  streamlit run app/streamlit_app.py

Reads the demo bundle from data/demo/ (override with PAPERTRAIL_DEMO_DIR). Needs no GPU, no PyTorch and no API key.
State lives in the URL (?page=...&model=...&account=...), so every view can be linked and shared.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))       # the investigator agent lives in src/papertrail/agent
from app import landing, logic, theme, viz

st.set_page_config(page_title="PaperTrail", layout="wide", initial_sidebar_state="auto")

DEMO_DIR = Path(os.environ.get("PAPERTRAIL_DEMO_DIR", Path(__file__).resolve().parents[1] / "data" / "demo"))
PAGES = {"home": "Home", "queue": "Alerts", "account": "Investigate", "try": "Simulator", "models": "Models", "about": "About"}
SEG = {"blend": "Blend", "xgb": "XGBoost", "gnn": "GNN"}
PLOT = {"displayModeBar": False}
PER_PAGE = 25
LIVE_LIMIT, CHAT_LIMIT = 5, 12                                        # live summaries / chat questions per visitor session (free-tier quotas)
KEY_NAMES = ("LLM_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")
DRIFT_URL = "https://abhibarnwal707.github.io/PaperTrail/"       # GitHub Pages, built from docs/ (Phase 8)
PROFILE_NAMES = {"new_account": "New account", "typical_customer": "Typical customer", "busy_business": "Busy business", "rapid_sender": "Rapid sender", "dormant_returning": "Dormant, returning"}


@st.cache_resource(show_spinner=False)
def get_bundle(path: str) -> logic.Bundle:
    return logic.load_bundle(path)


@st.cache_resource(show_spinner=False)
def get_toolbox(path: str):
    from papertrail.agent.tools import Toolbox

    return Toolbox(path)


def agent_ready() -> tuple[bool, str]:
    """Live generation needs the agent's packages and at least one key (Streamlit secrets on the Cloud, .env locally)."""
    try:
        from papertrail.agent.llm import load_env
    except ImportError as e:
        return False, f"the agent's packages are not installed here ({e.name})"
    load_env(Path(__file__).resolve().parents[1] / ".env")
    for k in KEY_NAMES:
        try:
            if k in st.secrets and not os.environ.get(k):
                os.environ[k] = str(st.secrets[k])
        except Exception:  # noqa: BLE001  (no secrets file at all)
            break
    return (True, "") if any(os.environ.get(k) for k in KEY_NAMES) else (False, "no API key is configured here")


def view_state() -> dict:
    q = st.query_params
    page = q.get("page", "home")
    model = q.get("model", "blend")
    s = {"page": page if page in PAGES else "home", "model": model if model in logic.MODELS else "blend"}
    if q.get("theme") in theme.THEMES and q.get("theme") != "dark":
        s["theme"] = q.get("theme")
    for k in ("account", "reveal", "p"):
        if q.get(k):
            s[k] = q.get(k)
    return s


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT, theme=None)


def tools(S: dict, *extra: str) -> str:
    """The header's right side: the model switch (where it applies), page actions, then the light/dark switch."""
    return "".join(extra) + theme.theme_button(S)


def crumb(S: dict, *parts: str) -> str:
    return theme.link(theme.href(S, page="queue", account=None, reveal=None, p=None), "Alerts") + "".join(f"<span>/</span><span>{theme.esc(p)}</span>" for p in parts)


# ------------------------------------------------------------------ pages
def page_queue(b: logic.Bundle, S: dict) -> None:
    model = S["model"]
    f = logic.queue_findings(b, model)
    tiers = logic.tier_counts(b, model)
    html(theme.header("Alert queue", f"{f['accounts']} accounts · {b.metrics['test_window'].replace(' to ', ' – ')}",
                      tools(S, theme.seg(S, SEG), theme.button("Open top alert", theme.href(S, page="account", account=f["top"], reveal=None, p=None), ic="next"))))
    flagged = f["flagged"][model]
    html(theme.kpi_cards([
        {"label": "Accounts in queue", "value": f"{f['accounts']}", "ic": "users", "foot": theme.pill(f"{tiers['Critical']} critical", "coral")},
        {"label": "Flagged", "value": f"{flagged}<small>/ {f['accounts']}</small>", "ic": "flag", "bar": flagged / f["accounts"], "color": theme.MODEL_VAR[model],
         "hint": "Accounts with at least one payment above the alert line."},
        {"label": "All models agree", "value": f"{f['every']}", "ic": "check", "foot": "blend, XGBoost and GNN", "hint": "Flagged by all three models."},
        {"label": "Caught only by graph", "value": f"{f['gnn_only']}", "ic": "graph", "foot": theme.pill("GNN only", "lime", dot=False),
         "hint": "Flagged by the GNN, missed by XGBoost."},
    ]))

    c1, c2 = st.columns([4, 8], gap="medium")
    with c1, st.container(border=True):
        html(theme.card_head("Risk mix", hint="Critical: the riskiest payment is far above the alert line. High: just above it. Watch: below it."))
        tier_colors = [P["coral"], P["amber"], P["watch"]]
        plot(viz.donut(list(tiers), list(tiers.values()), tier_colors, str(f["accounts"]), "accounts", P))
        html(theme.legend([(k, v, c) for (k, v), c in zip(tiers.items(), tier_colors)]))
    with c2, st.container(border=True):
        act = logic.hourly_activity(b, model)
        html(theme.card_head("Payment activity", "Per hour, demo slice", f'<span class="legend"><span><span class="d" style="background:var(--muted)"></span>payments<b>{int(act["payments"].sum())}</b></span>'
                             f'<span><span class="d" style="background:{theme.MODEL_VAR[model]}"></span>flagged<b>{int(act["flagged"].sum())}</b></span></span>'))
        plot(viz.activity_area(act, model, P))

    reveal = S.get("reveal") == "1"
    a = b.alerts.sort(f"{model}_rank")
    page = max(1, int(S["p"])) if S.get("p", "").isdigit() else 1
    lo, hi = (page - 1) * PER_PAGE, min(page * PER_PAGE, a.height)
    if lo >= a.height:
        lo, hi, page = 0, min(PER_PAGE, a.height), 1
    rows = []
    for r in a.slice(lo, hi - lo).to_dicts():
        acct = r["account"]
        pct = logic.risk_percent_text(logic.risk_percentile(b, model, r[f"{model}_risk"]))
        reasons = logic.account_reasons(b, acct, top=2).to_dicts()
        row = {"rank": f'<span class="rk">{r[f"{model}_rank"]:02d}</span>',
               "acct": theme.link(theme.href(S, page="account", account=acct, reveal=None, p=None), theme.esc(acct)),
               "tier": theme.tier_pill(logic.risk_tier(b, model, r[f"{model}_risk"]), title=f"riskiest payment: {pct} of the window"),
               "flag": theme.meter(r[f"n_flag_{model}"], r["n_txn"]),
               "sig": theme.chips([(x["label"], x["contribution"] > 0, f'{x["value_text"]} ({x["contribution"]:+.2f})') for x in reasons]),
               "mods": theme.model_dots({m: r[f"n_flag_{m}"] > 0 for m in ("blend", "xgb", "gnn")})}
        if reveal:
            row["truth"] = (theme.pill(r["typologies_truth"] or "Laundering", "coral") if r["n_laundering"] else theme.pill("Clean", "mint"))
        rows.append(row)
    cols = [("rank", "#", "dim"), ("acct", "Account", "id"), ("tier", "Risk", "text"), ("flag", "Flagged", "text"), ("sig", "Top signals", "text"), ("mods", "B · X · G", "text")]
    if reveal:
        cols.append(("truth", "Ground truth", "text"))
    with st.container(border=True):
        html(theme.card_head("Accounts", f"{lo + 1}–{hi} of {a.height}, riskiest first",
                             theme.toggle("Ground truth", reveal, theme.href(S, reveal=None if reveal else 1, p=None))))
        html(theme.table(cols, rows))
    pager = []
    if page > 1:
        pager.append(theme.link(theme.href(S, p=page - 1), "← Previous"))
    if hi < a.height:
        pager.append(theme.link(theme.href(S, p=page + 1), "Next →"))
    html(f'<div class="pager"><span>Page {page} of {(a.height - 1) // PER_PAGE + 1}</span><span class="pg">{"".join(pager)}</span></div>')


def page_account(b: logic.Bundle, S: dict) -> None:
    model = S["model"]
    order = b.alerts.sort(f"{model}_rank")["account"].to_list()
    account = S.get("account") if S.get("account") in order else order[0]
    row = b.alerts.filter(pl.col("account") == account).to_dicts()[0]
    rank = row[f"{model}_rank"]
    nxt = order[rank] if rank < len(order) else None
    tier = logic.risk_tier(b, model, row[f"{model}_risk"])
    reveal = S.get("reveal") == "1"
    right = tools(S, theme.seg(S, SEG), theme.button("Next", theme.href(S, account=nxt, reveal=None), ic="next") if nxt else "")
    html(theme.header(account, "", right, crumb=crumb(S, "Investigate"), mono=True, badge=theme.tier_pill(tier, big=True)))

    cards = []
    for m in ("blend", "xgb", "gnn"):
        pct = logic.risk_percent_text(logic.risk_percentile(b, m, row[f"{m}_risk"]))
        cards.append({"label": SEG[m], "value": f'#{row[f"{m}_rank"]}<small>/ {len(order)}</small>', "ic": {"blend": "layers", "xgb": "target", "gnn": "graph"}[m],
                      "foot": theme.meter(row[f"n_flag_{m}"], row["n_txn"]) + f'<span class="mono">{pct}</span>', "on": m == model})
    html(theme.kpi_cards(cards, cols=3))

    reasons = logic.account_reasons(b, account)
    pos, edges = logic.ring_layout(logic.account_edges(b, account), account)
    c1, c2 = st.columns([7, 5], gap="medium")
    with c1, st.container(border=True):
        html(theme.card_head("Payment network", f"{len(pos) - 1} counterparties", ic="graph", hint="Inner ring: direct counterparties. Outer ring: their counterparties. Coral: payments the selected model flags."))
        plot(viz.network_figure(pos, edges, account, model, P))
    with c2, st.container(border=True):
        html(theme.card_head("Why it's flagged", "Riskiest payment · XGBoost", ic="target", hint="How far each fact pushed the score up (coral) or down (grey)."))
        html(theme.reasons_list(reasons, 8))

    h = logic.account_history(b, account)
    html('<div class="gap"></div>')
    with st.container(border=True):
        html(theme.card_head("Activity", f"Last {h.height} payments", ic="clock",
                             right=theme.toggle("Ground truth", reveal, theme.href(S, reveal=None if reveal else 1))))
        plot(viz.timeline(h, model, P))
        tape = []
        for r in h.to_dicts():
            out = r["direction"] == "sent"
            pct = r.get(f"{model}_pct")
            tape.append({"dir": f'<span class="dir {"out" if out else "in"}">{theme.icon("out" if out else "in")}</span>',
                         "when": f'<span class="mono">{str(r["ts"])[5:16]}</span>', "other": theme.esc(r["counterparty"]), "paid": theme.esc(r["pay_format"]),
                         "amt": f'<span class="{"" if out else "amt-in"}">{"-" if out else "+"}{r["amt_paid"]:,.2f}</span>',
                         "risk": theme.pill("Flagged", "coral") if r.get(f"{model}_flag") else theme.pill("—" if pct is None else logic.risk_percent_text(pct), "grey", dot=False)})
        html(theme.table([("dir", "", "text"), ("when", "When", "dim"), ("other", "Counterparty", "id"), ("paid", "Method", "dim"), ("amt", "Amount", "num"), ("risk", "Risk", "text")], tape))
    if reveal:
        truth = theme.pill(f'Laundering · {row["typologies_truth"]}' if row["n_laundering"] else "Clean account", "coral" if row["n_laundering"] else "mint", big=True)
        html(f'<div class="gap"></div><div class="card" style="display:flex;gap:14px;align-items:center;flex-wrap:wrap">{truth}'
             f'<span class="note"><b>{row["n_laundering"]}</b> laundering payments in the window · picked as <b>{theme.esc(row["tag"])}</b></span></div>')
    html('<div class="gap"></div>')
    investigator_section(b, account)


def investigator_section(b: logic.Bundle, account: str) -> None:
    """The agent's panel: how it works, the case summary (cached or live) and a chat about this account."""
    ok, why = agent_ready()
    with st.container(border=True):
        html(theme.card_head("Investigator agent", "Writes a case summary and answers questions, using only this account's data",
                             ic="spark", hint="An LLM reads the four tool outputs, drafts an answer, and a checker rejects any number "
                             "that is not in the source it cites. One retry, then it falls back to showing only checked facts."))
        html(theme.agent_steps())
        if ok:
            model_picker()
        left, right = st.columns([7, 5], gap="large")
        with left:
            summary_pane(b, account, ok, why)
        with right:
            chat_pane(b, account, ok, why)


@st.cache_data(ttl=3600, show_spinner=False)
def find_free_models() -> list:
    from papertrail.agent.llm import discover_free_models

    return discover_free_models()


def model_picker() -> None:
    """Auto (the curated list in order) or one model; "Find more free models" adds OpenRouter's current free list."""
    from papertrail.agent.llm import available, providers_from_env

    found = st.session_state.setdefault("found_models", [])
    have = {p.name for p in providers_from_env()}
    choices = available() + [c for c in found if c.provider in have]
    options = ["auto"] + [c.key for c in choices]
    by_key = {c.key: c for c in choices}
    label = lambda k: "Auto · best available, with fallback" if k == "auto" else f"{by_key[k].label} · {by_key[k].note.split(' · ')[0]}"
    c1, c2, c3 = st.columns([4, 2, 5], gap="small", vertical_alignment="center")
    pick = c1.selectbox("Model", options, format_func=label, key="model_key", label_visibility="collapsed")
    if c2.button("Find more free models", key="find-models", disabled="openrouter" not in have,
                 help="Fetch OpenRouter's current free models (they change often and are not tested here)"):
        try:
            st.session_state["found_models"] = find_free_models()
        except Exception as e:  # noqa: BLE001
            st.session_state["find_error"] = str(e)[:120]
        st.rerun()
    note = ("Tries the curated models in order and falls back automatically when one is busy." if pick == "auto" else by_key[pick].note)
    if found:
        note += f" · {len(found)} more free models found"
    if st.session_state.get("find_error"):
        note += f" · could not fetch the list: {st.session_state['find_error']}"
    c3.markdown(f'<span class="note">{theme.esc(note)}</span>', unsafe_allow_html=True)


def llm_client():
    """One client per chosen model for this visitor, so it remembers which model worked and which to skip."""
    from papertrail.agent.llm import ChatClient, available

    key = st.session_state.get("model_key", "auto")
    clients = st.session_state.setdefault("clients", {})
    if key not in clients:
        pool = {c.key: c for c in available() + st.session_state.get("found_models", [])}
        clients[key] = ChatClient(model=pool.get(key))
    return clients[key]


def summary_pane(b: logic.Bundle, account: str, ok: bool, why: str) -> None:
    live = st.session_state.setdefault("cases", {})
    result, is_live = (live[account], True) if account in live else (b.cases.get(account), False)
    if result:
        html(theme.case_card(result, is_live))
    else:
        html(theme.ai_placeholder("No summary is cached for this account yet. Generate one live below." if ok
                                  else f"No summary is cached for this account, and live generation is off: {why}."))
    if not ok:
        return
    used = st.session_state.setdefault("live_runs", 0)
    left = LIVE_LIMIT - used
    c1, c2 = st.columns([2, 5], gap="small", vertical_alignment="center")
    clicked = c1.button("Regenerate live" if result else "Generate live", key=f"gen-{account}", disabled=left <= 0)
    c2.markdown(f'<span class="note">{left} live run{"s" if left != 1 else ""} left · about 10-30 s · free models, so wording varies</span>'
                if left > 0 else '<span class="note">Live limit reached for this session; cached summaries stay available.</span>', unsafe_allow_html=True)
    if clicked:
        from papertrail.agent.graph import investigate

        with st.spinner("Gathering evidence, drafting, checking every number..."):
            try:
                live[account] = investigate(account, get_toolbox(str(b.root)), llm_client())
                st.session_state["live_runs"] = used + 1
            except Exception as e:  # noqa: BLE001
                st.error(f"Live generation failed: {str(e)[:200]}")
                return
        st.rerun()


def chat_pane(b: logic.Bundle, account: str, ok: bool, why: str) -> None:
    try:
        from papertrail.agent.chat import (
            CHIP_LABELS,
            EXAMPLE_QUESTIONS,
            ask,
            preview_answer,
        )
    except ImportError:
        html('<div class="note">The chat needs the agent packages, which are not installed here.</div>')
        return
    chats = st.session_state.setdefault("chat", {})
    turns = chats.setdefault(account, [])
    asked = st.session_state.setdefault("chat_runs", 0)
    left = CHAT_LIMIT - asked
    html(f'<div class="pane-t">Ask about this account<span>{left if ok else 0} questions left</span></div>')
    preview = None if turns else preview_answer(account, get_toolbox(str(b.root)))
    html(theme.chat_html(turns, preview))
    question = None
    if len(turns) < 2:
        html('<div class="try-l">Try asking</div>')
        cols = st.columns(2, gap="small")
        for i, q in enumerate(EXAMPLE_QUESTIONS[1:]):
            if cols[i % 2].button(CHIP_LABELS[q], key=f"chip-{account}-{i}", help=q, disabled=not ok or left <= 0, width="stretch"):
                question = q
    typed = st.chat_input("Ask about this account..." if ok else f"Chat is off here: {why}", key=f"ask-{account}", disabled=not ok or left <= 0)
    question = typed or question
    if question:
        with st.spinner("Reading the evidence..."):
            turns.append(ask(question, account, get_toolbox(str(b.root)), llm_client(), turns=turns))
            st.session_state["chat_runs"] = asked + 1
        st.rerun()


def page_try(b: logic.Bundle, S: dict) -> None:
    html(theme.header("Simulator", "Describe a payment, see if XGBoost would alert", tools(S)))
    if not b.boosters:
        html('<div class="empty">The XGBoost models could not be loaded here, so scoring is unavailable.</div>')
        return
    cat = b.categories
    left, right = st.columns([5, 7], gap="medium")
    with left, st.container(border=True):
        html(theme.card_head("Payment", ic="out"))
        amount = st.number_input("Amount", min_value=1.0, max_value=1e9, value=9500.0, step=100.0)
        c1, c2 = st.columns(2)
        fmt = c1.selectbox("Method", cat["pay_format"], index=cat["pay_format"].index("ACH") if "ACH" in cat["pay_format"] else 0)
        cur = c2.selectbox("Currency", list(cat["currencies"]), index=list(cat["currencies"]).index("US Dollar") if "US Dollar" in cat["currencies"] else 0)
        hour = st.slider("Hour of day", 0, 23, 14)
        c3, c4 = st.columns(2)
        diff = c3.toggle("Cross-currency")
        same_bank = c4.toggle("Same bank")
        html('<div class="gap"></div>' + theme.card_head("Sender", ic="users"))
        names = list(b.profiles["profiles"])
        profile = st.selectbox("Profile", names, format_func=lambda n: PROFILE_NAMES.get(n, n.replace("_", " ").capitalize()), help="Illustrative: the median history of training accounts that match.")
        p = b.profiles["profiles"][profile]
        html(f'<div class="note" style="margin:-4px 0 10px 0">{theme.esc(p["description"])} · base rate <b class="mono">{p["laundering_rate"] * 100:.2f}%</b></div>')
        sl, new = b.profiles["sliders"], profile == "new_account"
        n24 = st.slider("Payments sent, last 24h", 0, int(max(sl["s_out_cnt_24h"]["max"], 1)), int(sl["s_out_cnt_24h"]["default"]) if not new else 0, disabled=new)
        uq = st.slider("Distinct recipients", 0, int(max(sl["s_uniq_out"]["max"], 1)), int(sl["s_uniq_out"]["default"]) if not new else 0, disabled=new)
    feats = logic.whatif_features(b, amount=float(amount), pay_format=fmt, pay_currency=cur, different_receive_currency=diff, same_bank=same_bank,
                                  hour=hour, profile=profile, out_cnt_24h=n24, uniq_out=uq)
    res = logic.score_features(b, feats)
    with right, st.container(border=True):
        html(theme.card_head("Model score", "XGBoost · 3-seed average", ic="target", hint="The score ranks payments; it is not a probability. The GNN needs a real payment network, so it can't score made-up payments."))
        plot(viz.gauge(res["probability"], res["threshold"], P))
        html('<div class="verdict">' + (theme.pill("Alert raised", "coral", big=True) if res["alert"] else theme.pill("No alert", "mint", big=True)) + "</div>")
        html('<div class="gap"></div>' + theme.card_head("Drivers"))
        html(theme.reasons_list(res["reasons"], 6))
    with st.expander("All model inputs"):
        rows = [{"f": theme.esc(r["label"]), "v": theme.esc(r["value_text"]), "e": f'{r["contribution"]:+.3f}'} for r in res["reasons"].to_dicts()]
        html(f'<div class="card" style="padding:0 18px">{theme.table([("f", "Feature", "text"), ("v", "Value", "num"), ("e", "Effect", "num")], rows, wrap=False)}</div>')


def page_models(b: logic.Bundle, S: dict) -> None:
    m = b.metrics
    bl = m["models"]["blend"]
    html(theme.header("Model performance", f"{m['n_test_transactions']:,} unseen payments · {m['n_test_positives']:,} laundering · 3-seed means", tools(S, theme.button("Drift report", DRIFT_URL, "ghost", ic="trail", external=True))))

    def delta(k: str, pct: bool = False) -> str:
        d = bl[k] - max(m["models"]["xgb"][k], m["models"]["gnn"][k])
        return theme.pill(f"{d * 100:+.1f} pts" if pct else f"{d:+.3f}", "lime" if d > 0 else "grey", dot=False) + "<span>vs best single</span>"

    html(theme.kpi_cards([
        {"label": "PR-AUC", "value": f"{bl['pr_auc']:.3f}", "ic": "target", "foot": delta("pr_auc"), "hint": "Ranking quality across all alert levels. Random is about 0.001."},
        {"label": "F1", "value": f"{bl['f1']:.3f}", "ic": "layers", "foot": delta("f1"), "hint": "Balance of right alerts and found cases at the chosen alert line."},
        {"label": "Recall at 1% FPR", "value": f"{bl['recall_at_1pct_fpr'] * 100:.1f}<small>%</small>", "ic": "flag", "foot": delta("recall_at_1pct_fpr", True),
         "bar": bl["recall_at_1pct_fpr"], "hint": "Share of laundering found if analysts review 1 in 100 normal payments."},
        {"label": "Precision, top 1,000", "value": f"{bl['precision_at_1000'] * 100:.1f}<small>%</small>", "ic": "check", "foot": delta("precision_at_1000", True),
         "bar": bl["precision_at_1000"], "hint": "Share of the 1,000 highest-scored payments that really are laundering."},
    ]))
    order = ["blend", "xgb", "gnn", "xgb_tabular", "rules_v1"]
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        html(theme.card_head("Leaderboard", "PR-AUC, higher is better", hint="Rules and payment-only XGBoost are the baselines. Accuracy is never reported: only ~1 in 900 payments is laundering."))
        plot(viz.leaderboard(m, "pr_auc", order, P))
    with c2, st.container(border=True):
        html(theme.card_head("Precision vs recall", "Test window"))
        plot(viz.pr_curves(m, P))

    c3, c4 = st.columns([7, 5], gap="medium")
    rows = sorted(m["typology_recall"], key=lambda r: (r["typology"] == "UNMAPPED", -r["n"]))
    with c3, st.container(border=True):
        html(theme.card_head("Recall by pattern", "At each model's alert line", hint="Patterns come from the dataset's ground truth. About 4 in 10 laundering payments match no known pattern."))
        plot(viz.typology_heat(rows, P))
    o = m.get("overlap_at_1pct_fpr") or {}
    with c4, st.container(border=True):
        html(theme.card_head("Why two models", "Laundering caught at 1% FPR"))
        if o:
            parts = [("Both", o.get("both", 0), P["neutral"]), ("GNN only", o.get("gnn_only", 0), P["violet"]), ("XGBoost only", o.get("xgb_only", 0), P["sky"]),
                     ("Neither", o.get("neither", 0), P["watch"])]
            plot(viz.donut([x[0] for x in parts], [x[1] for x in parts], [x[2] for x in parts], f"{(o.get('gnn_only', 0) + o.get('xgb_only', 0))}", "caught by one model", P))
            html(theme.legend(parts))


def page_home(b: logic.Bundle, S: dict) -> None:
    """The landing page (app/landing/home.html, rendered into the app, not an iframe) with a native "open a case" form."""
    order = b.alerts.sort("blend_rank")["account"].to_list()
    go = lambda **c: theme.href(S, **{"reveal": None, "p": None, "account": None, **c})
    hrefs = {"home": go(page="home"), "queue": go(page="queue"), "top": go(page="account", account=order[0]), "try": go(page="try"),
             "models": go(page="models"), "about": go(page="about"), "theme": theme.href(S, theme=None if S.get("theme") == "light" else "light")}
    light = S.get("theme") == "light"
    switch = (theme.icon("moon" if light else "sun") + ("dark" if light else "light"), f'Switch to the {"dark" if light else "light"} theme')
    top, foot = landing.build(b.metrics, hrefs, S.get("theme", "dark"), switch)
    html(f"<style>{landing.HOME_CSS}</style>")
    st.html(top, unsafe_allow_javascript=True)
    home_cta(b, S)
    st.html(foot)


def home_cta(b: logic.Bundle, S: dict) -> None:
    """Pick a model and an account (ordered by that model's rank) and open its case: real widgets, so it works without JS."""
    with st.container(key="lp-cta"):
        left, right = st.columns([6, 5], gap="large", vertical_alignment="center")
        with left:
            st.html('<h2 class="lp-cta-h">open a case.</h2><p class="lp-cta-p">Pick a model, choose an account from its ranking (the riskiest '
                    "first) and read the investigator's case file: network, score drivers, payments and the agent's verified summary.</p>")
        with right, st.container(key="lp-dlg"):
            st.html(f'<h3 class="lp-dlg-h">Open an account</h3><p class="lp-dlg-p">{b.alerts.height} accounts from the test window, riskiest first.</p>')
            model = st.segmented_control("Ranked by", list(SEG), format_func=SEG.get, default=S["model"], key="lp-model") or "blend"
            ranked = b.alerts.sort(f"{model}_rank").select("account", f"{model}_rank").rows()
            rank = dict(ranked)
            account = st.selectbox("Account", [a for a, _ in ranked], format_func=lambda a: f"#{rank[a]} · {a}", key=f"lp-acct-{model}")
            if st.button("Open case →", type="primary", width="stretch", key="lp-open"):
                st.query_params.from_dict({k: v for k, v in {"page": "account", "account": account, "model": model, "theme": S.get("theme")}.items() if v})
                st.rerun()


def page_about(b: logic.Bundle, S: dict) -> None:
    m = b.metrics
    bl = m["models"]["blend"]
    html(f'<div class="card hero"><div style="display:flex;justify-content:space-between;align-items:center">{theme.pill("Portfolio project · synthetic data", "lime")}{tools(S)}</div><h1>Find the laundering. Show why.</h1>'
         '<p>PaperTrail ranks accounts by how suspicious their payments look, explains each score, and puts the riskiest cases first.</p></div><div class="gap"></div>')
    html(theme.kpi_cards([
        {"label": "Payments", "value": "5.08<small>M</small>", "ic": "db", "foot": "IBM AML · HI-Small"},
        {"label": "Laundering rate", "value": "0.1<small>%</small>", "ic": "flag", "foot": "about 1 in 900"},
        {"label": "Blend PR-AUC", "value": f"{bl['pr_auc']:.3f}", "ic": "target", "foot": "test window, scored once"},
        {"label": "Found at 1% FPR", "value": f"{bl['recall_at_1pct_fpr'] * 100:.0f}<small>%</small>", "ic": "check", "foot": "blend of two models"},
    ]))
    steps = [("01", "Features", "Each payment plus the recent history of both accounts, using only earlier payments."),
             ("02", "XGBoost", "Reads the payment and account behaviour: frequency, fan-out, recency."),
             ("03", "Graph network", "A GNN reads the web of payments around each account, from earlier days only."),
             ("04", "Blend", "Scores are standardised and mixed; the blend beats either model alone."),
             ("05", "Investigator agent", "An LLM writes case summaries and answers questions; every number it states is checked against its source.")]
    html('<div class="facts">' + "".join(f'<div class="card step"><span class="n">{n}</span><b>{t}</b><span>{d}</span></div>' for n, t, d in steps) + "</div>")
    html(f'<div class="gap"></div><div class="note">Train Sep 1–6 · validate Sep 7–8 · test {theme.esc(m["test_window"])} ({m["n_test_transactions"]:,} payments). '
         'Synthetic data; a demonstration of the method, not a product for real banks.</div>')


# ------------------------------------------------------------------ run
state = view_state()
P = theme.PALETTES[state.get("theme", "dark")]                 # the active palette, for the plotly figures
st.markdown(f"<style>{theme.css(state.get('theme', 'dark'))}</style>", unsafe_allow_html=True)
if not logic.bundle_present(DEMO_DIR):
    html(theme.header("PaperTrail", "Demo data not installed. Unzip <b>demo_bundle.zip</b> (from notebooks/06-demo-slice.ipynb) into <b>data/demo/</b>."))
    st.stop()
bundle = get_bundle(str(DEMO_DIR))
facts = [("Window", bundle.metrics["test_window"][5:].replace(" to 2022-", " → ")), ("Accounts", f"{bundle.alerts.height}"),
         ("Payments", f"{bundle.metrics['n_test_transactions']:,}"), ("Laundering", f"{bundle.metrics['n_test_positives']:,}")]
st.sidebar.markdown(theme.sidebar_html(state, PAGES, facts), unsafe_allow_html=True)
html(theme.mobile_nav_html(state, PAGES))
{"home": page_home, "queue": page_queue, "account": page_account, "try": page_try, "models": page_models, "about": page_about}[state["page"]](bundle, state)
