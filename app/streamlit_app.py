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
from app import logic, theme, viz

st.set_page_config(page_title="PaperTrail", layout="wide", initial_sidebar_state="auto")
st.markdown(f"<style>{theme.CSS}</style>", unsafe_allow_html=True)

DEMO_DIR = Path(os.environ.get("PAPERTRAIL_DEMO_DIR", Path(__file__).resolve().parents[1] / "data" / "demo"))
PAGES = {"queue": "Alert queue", "account": "Investigate", "try": "Try a transaction", "models": "Models", "about": "About"}
SHORT = {"blend": "the blend", "xgb": "XGBoost", "gnn": "the GNN"}
PLOT = {"displayModeBar": False}
PER_PAGE = 50
PROFILE_NAMES = {"new_account": "New account", "typical_customer": "Typical customer", "busy_business": "Busy business", "rapid_sender": "Rapid sender", "dormant_returning": "Dormant, returning"}


@st.cache_resource(show_spinner=False)
def get_bundle(path: str) -> logic.Bundle:
    return logic.load_bundle(path)


def view_state() -> dict:
    q = st.query_params
    page = q.get("page", "queue")
    model = q.get("model", "blend")
    s = {"page": page if page in PAGES else "queue", "model": model if model in logic.MODELS else "blend"}
    for k in ("account", "reveal", "p"):
        if q.get(k):
            s[k] = q.get(k)
    return s


def fmt_money(x: float, sign: str = "") -> str:
    return f"{sign}{x:,.2f}"


def plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT, theme=None)


# ------------------------------------------------------------------ pages
def page_queue(b: logic.Bundle, S: dict) -> None:
    model = S["model"]
    f = logic.queue_findings(b, model)
    sub = (f"<b>{f['flagged'][model]}</b> of <b>{f['accounts']}</b> accounts carry at least one flagged payment under {SHORT[model]}. "
           f"<b>{f['every']}</b> are flagged by every model; <b>{f['gnn_only']}</b> are caught by the GNN but not by XGBoost.")
    st.markdown(theme.title_row("Alert queue", sub, theme.button("Open the top account", theme.href(S, page="account", account=f["top"]))), unsafe_allow_html=True)
    st.markdown(theme.kpi_strip([
        ("Accounts in the queue", f"{f['accounts']}", "test window Sep 9-10, 2022"),
        (f"Flagged by {SHORT[model]}", f"{f['flagged'][model]}", f"of {f['accounts']} accounts"),
        ("Flagged by every model", f"{f['every']}", "blend, XGBoost and GNN agree"),
        ("GNN only", f"{f['gnn_only']}", "missed by XGBoost"),
    ]), unsafe_allow_html=True)

    reveal = S.get("reveal") == "1"
    t = logic.alert_table(b, model, reveal)
    page = max(1, int(S["p"])) if S.get("p", "").isdigit() else 1
    lo, hi = (page - 1) * PER_PAGE, min(page * PER_PAGE, t.height)
    toggle = (f'<a target="_self" href="{theme.esc(theme.href(S, reveal=None, p=None))}">hide</a>' if reveal
              else f'<a class="off" target="_self" href="{theme.esc(theme.href(S, reveal=1, p=None))}">show</a>')
    st.markdown(f'<div class="tools"><span class="num">{lo + 1}-{hi} of {t.height}</span><span>Answers (which accounts really were laundering): {toggle}</span></div>', unsafe_allow_html=True)

    flags = b.alerts.select("account", f"n_flag_{model}").to_dict(as_series=False)
    flag_of = dict(zip(flags["account"], flags[f"n_flag_{model}"]))
    cols = [("rank", "Rank", "num dim"), ("acct", "Account", "id"), ("risk", "How risky", "dim"), ("flag", "Flagged payments", "num"), ("why", "Why it stands out", "wrap")]
    if reveal:
        cols += [("n_l", "Laundering payments", "num"), ("pat", "True pattern", "dim"), ("tag", "Picked for the demo because", "dim")]
    rows = []
    for r in t.slice(lo, hi - lo).to_dicts():
        a = r["Account"]
        mark = '<span class="flagmark"></span>' if flag_of[a] else ""
        row = {"rank": r["Rank"], "acct": f'{mark}<a target="_self" href="{theme.esc(theme.href(S, page="account", account=a, reveal=None, p=None))}">{theme.esc(a)}</a>',
               "risk": theme.esc(r["How risky"]), "flag": f'{r["Flagged transactions"]} / {r["Transactions in window"]}', "why": theme.esc(r["Why it stands out"])}
        if reveal:
            row.update(n_l=r["Actually laundering (transactions)"], pat=theme.esc(r["True pattern"] or "none"), tag=theme.esc(r["Why it was picked for the demo"]))
        rows.append(row)
    st.markdown(theme.table(cols, rows), unsafe_allow_html=True)
    pager = []
    if page > 1:
        pager.append(f'<a target="_self" href="{theme.esc(theme.href(S, p=page - 1))}">Previous</a>')
    if hi < t.height:
        pager.append(f'<a target="_self" href="{theme.esc(theme.href(S, p=page + 1))}">Next {min(PER_PAGE, t.height - hi)}</a>')
    st.markdown(f'<div class="pager"><span>How risky: where the account\'s riskiest payment ranks among all payments in the window.</span><span>{" · ".join(pager)}</span></div>', unsafe_allow_html=True)


def page_account(b: logic.Bundle, S: dict) -> None:
    model = S["model"]
    order = b.alerts.sort(f"{model}_rank")["account"].to_list()
    account = S.get("account") if S.get("account") in order else order[0]
    row = b.alerts.filter(pl.col("account") == account).to_dicts()[0]
    rank = row[f"{model}_rank"]
    nxt = order[rank] if rank < len(order) else None
    sub = (f"Ranked <b>#{rank}</b> of <b>{len(order)}</b> by {SHORT[model]}. <b>{row[f'n_flag_{model}']}</b> of <b>{row['n_txn']}</b> payments in the window are flagged.")
    actions = theme.button("Back to the queue", theme.href(S, page="queue", account=None), "ghost") + " " + (theme.button("Next account", theme.href(S, account=nxt)) if nxt else "")
    st.markdown(theme.title_row(account, sub, actions, mono=True), unsafe_allow_html=True)
    cells = [(logic.MODELS[m], f'#{row[f"{m}_rank"]}<small>of {len(order)}</small>', f'{row[f"n_flag_{m}"]} flagged of {row["n_txn"]}', m == model) for m in ("blend", "xgb", "gnn")]
    st.markdown(theme.kpi_strip(cells), unsafe_allow_html=True)

    reasons = logic.account_reasons(b, account)
    pos, rows = logic.ring_layout(logic.account_edges(b, account), account)
    with st.container(border=True):
        left, right = st.columns([5, 7], gap="large")
        with left:
            st.markdown(theme.card_title("Why it stands out", "The account's riskiest payment. Each line says how far a fact pushed its score up (amber) or down (grey)."), unsafe_allow_html=True)
            st.markdown(theme.reasons_list(reasons, 8), unsafe_allow_html=True)
            st.markdown('<div class="note" style="margin-top:12px">These reasons come from the XGBoost models. The GNN judges the web of payments around the account instead: see the network.</div>', unsafe_allow_html=True)
        with right:
            st.markdown(theme.card_title("Who it pays and who pays it", f"The account is the white dot; amber lines are payments {SHORT[model]} flags. Up to 150 payments, riskiest first."), unsafe_allow_html=True)
            plot(viz.network_figure(pos, rows, account, model))

    h = logic.account_history(b, account)
    tape_rows = []
    for r in h.to_dicts():
        out = r["direction"] == "sent"
        pct = r.get(f"{model}_pct")
        mark = '<span class="flagmark"></span>' if r.get(f"{model}_flag") else ""
        tape_rows.append({"when": str(r["ts"])[:16], "dir": r["direction"], "other": theme.esc(r["counterparty"]), "paid": theme.esc(r["pay_format"]),
                          "amt": fmt_money(r["amt_paid"], "-" if out else "+"), "risk": "-" if pct is None else logic.risk_percent_text(pct), "flag": mark})
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(theme.card_title("Transaction tape", f"The last {h.height} payments, newest first. A 1px amber mark means {SHORT[model]} flags the payment. Minus is money out, plus is money in."), unsafe_allow_html=True)
    st.markdown(theme.table([("flag", "", "id"), ("when", "When", "dim"), ("dir", "", "dim"), ("other", "Other account", "id"), ("paid", "Paid by", "dim"),
                             ("amt", "Amount", "num"), ("risk", "How risky", "num dim")], tape_rows, cls="tape"), unsafe_allow_html=True)

    reveal = S.get("reveal") == "1"
    ans = (f'<span>Laundering payments in the window: <b>{row["n_laundering"]}</b>. True pattern: <b>{theme.esc(row["typologies_truth"] or "none")}</b>. '
           f'Picked for the demo because: <b>{theme.esc(row["tag"])}</b>. <a target="_self" href="{theme.esc(theme.href(S, reveal=None))}">hide</a></span>' if reveal
           else f'<span>Answer: <a class="off" target="_self" href="{theme.esc(theme.href(S, reveal=1))}">show whether this account really was laundering</a></span>')
    st.markdown(f'<div class="tools" style="margin-top:16px">{ans}</div>', unsafe_allow_html=True)
    st.markdown('<div class="empty" style="margin-top:8px">A written case summary will appear here once the investigator agent is added.</div>', unsafe_allow_html=True)


def page_try(b: logic.Bundle, S: dict) -> None:
    st.markdown(theme.title_row("Try a transaction", "Describe a payment and the sender, and see whether the model would raise an alert. The sender's history comes from a profile you pick."), unsafe_allow_html=True)
    if not b.boosters:
        st.markdown('<div class="empty">The XGBoost models could not be loaded here, so scoring is unavailable.</div>', unsafe_allow_html=True)
        return
    if S["model"] != "xgb":
        st.markdown('<div class="note" style="margin-bottom:16px">Scoring a made-up payment always uses XGBoost. The GNN needs the real payment network around real accounts, so its scores exist only for real transactions in the queue.</div>', unsafe_allow_html=True)
    cat = b.categories
    with st.container(border=True):
        left, right = st.columns([5, 7], gap="large")
        with left:
            st.markdown(theme.card_title("The payment"), unsafe_allow_html=True)
            amount = st.number_input("Amount", min_value=1.0, max_value=1e9, value=9500.0, step=100.0)
            c1, c2 = st.columns(2)
            fmt = c1.selectbox("How it is paid", cat["pay_format"], index=cat["pay_format"].index("ACH") if "ACH" in cat["pay_format"] else 0)
            cur = c2.selectbox("Currency", list(cat["currencies"]), index=list(cat["currencies"]).index("US Dollar") if "US Dollar" in cat["currencies"] else 0)
            diff = st.checkbox("The receiver is paid in a different currency")
            same_bank = st.checkbox("Sender and receiver use the same bank")
            hour = st.slider("Hour of the day", 0, 23, 14)
            st.markdown('<div style="height:12px"></div>' + theme.card_title("The sender"), unsafe_allow_html=True)
            names = list(b.profiles["profiles"])
            profile = st.selectbox("What kind of account is it?", names, format_func=lambda n: PROFILE_NAMES.get(n, n.replace("_", " ").capitalize()))
            p = b.profiles["profiles"][profile]
            st.markdown(f'<div class="note" style="margin:-4px 0 10px 0">{theme.esc(p["description"])}. In the training data, <b class="num">{p["laundering_rate"] * 100:.2f}%</b> of payments from accounts like this were laundering (<span class="num">{p["n_rows"]:,}</span> examples).</div>', unsafe_allow_html=True)
            sl, new = b.profiles["sliders"], profile == "new_account"
            n24 = st.slider("Payments the sender made in the last 24 hours", 0, int(max(sl["s_out_cnt_24h"]["max"], 1)), int(sl["s_out_cnt_24h"]["default"]) if not new else 0, disabled=new)
            uq = st.slider("Different people the sender has paid so far", 0, int(max(sl["s_uniq_out"]["max"], 1)), int(sl["s_uniq_out"]["default"]) if not new else 0, disabled=new)
        feats = logic.whatif_features(b, amount=float(amount), pay_format=fmt, pay_currency=cur, different_receive_currency=diff, same_bank=same_bank,
                                      hour=hour, profile=profile, out_cnt_24h=n24, uniq_out=uq)
        res = logic.score_features(b, feats)
        with right:
            st.markdown(theme.card_title("Result"), unsafe_allow_html=True)
            on = "on" if res["alert"] else ""
            verdict = "Alert" if res["alert"] else "No alert"
            sub = "The model would flag this payment for review." if res["alert"] else "The model would let this payment through."
            st.markdown(f'<div class="verdict {on}">{verdict}</div><div class="note" style="margin-top:2px">{sub}</div>'
                        f'<div style="display:flex;align-items:baseline;gap:12px;margin-top:14px"><span class="score-fig">{res["probability"]:.2f}</span><span class="note">model score</span></div>'
                        f'{theme.score_bar(res["probability"], res["threshold"])}', unsafe_allow_html=True)
            st.markdown('<div class="card-title" style="margin-top:22px">What drove it</div>', unsafe_allow_html=True)
            st.markdown(theme.reasons_list(res["reasons"], 6), unsafe_allow_html=True)
            st.markdown('<div class="note" style="margin-top:12px">An illustration, not a real account: the sender\'s history is the typical history of the chosen kind of account. The score ranks payments; it is not a probability. The data is synthetic.</div>', unsafe_allow_html=True)
    with st.expander("The technical details: every number sent to the model"):
        rows = [{"f": theme.esc(r["label"]), "v": theme.esc(r["value_text"]), "e": f'{r["contribution"]:+.3f}'} for r in res["reasons"].to_dicts()]
        st.markdown(theme.table([("f", "Feature", "text"), ("v", "Value", "num"), ("e", "Effect on the score", "num")], rows), unsafe_allow_html=True)


def page_models(b: logic.Bundle, S: dict) -> None:
    m, f = b.metrics, logic.model_findings(b)
    best = SHORT[f["best_single"]]
    sub = (f"At 1% false alarms the blend finds <b>{f['blend_recall'] * 100:.1f}%</b> of the laundering; the better single model ({best}) finds <b>{f['best_single_recall'] * 100:.1f}%</b>. "
           f"Measured on {m['n_test_transactions']:,} payments the models never saw ({m['test_window']}), <b>{m['n_test_positives']:,}</b> of them laundering.")
    st.markdown(theme.title_row("Models", sub), unsafe_allow_html=True)
    names = {"rules_v1": "Simple rules", "xgb_tabular": "XGBoost, payment details only", "xgb": "XGBoost + account history", "gnn": "GNN", "blend": "Blend of GNN and XGBoost"}
    keys = [("pr_auc", "Ranking quality (PR-AUC)"), ("f1", "Balance (F1)"), ("recall_at_1pct_fpr", "Found at 1% false alarms"), ("precision_at_1000", "Right in the top 1,000")]
    best_of = {k: max(m["models"][n][k] for n in names if n in m["models"]) for k, _ in keys}
    rows = []
    for n, label in names.items():
        if n in m["models"]:
            rows.append({"name": theme.esc(label), **{k: f'<span class="{"best" if m["models"][n][k] == best_of[k] else "rest"}">{m["models"][n][k]:.3f}</span>' for k, _ in keys}})
    st.markdown(theme.table([("name", "Model", "text")] + [(k, h, "num") for k, h in keys], rows), unsafe_allow_html=True)
    st.markdown('<div class="note" style="margin:10px 0 28px 0">Higher is better everywhere; the best in each column is in bold. Only about 1 in 900 payments is laundering, so accuracy would be meaningless and is not reported. '
                '"Found at 1% false alarms": if analysts accept reviewing 1 in 100 normal payments, how much laundering do they catch?</div>', unsafe_allow_html=True)

    with st.container(border=True):
        left, right = st.columns([6, 6], gap="large")
        with left:
            st.markdown(theme.card_title("Precision against recall", "Higher and further right is better. Amber is the blend."), unsafe_allow_html=True)
            plot(viz.pr_curves(m, {k: names[k] for k in ("blend", "xgb", "gnn", "xgb_tabular") if k in m["pr_curves"]}))
        with right:
            st.markdown(theme.card_title("Which kinds of laundering are found", "Share found at each model's own alert level; the best is in bold."), unsafe_allow_html=True)
            trows = []
            for r in sorted(m["typology_recall"], key=lambda r: (r["typology"] == "UNMAPPED", -r["n"])):
                vals = {k: r[f"recall_{k}"] for k in ("xgb", "gnn", "blend")}
                top = max(vals.values())
                trows.append({"t": theme.esc("No known pattern" if r["typology"] == "UNMAPPED" else r["typology"].title()), "n": r["n"],
                              **{k: f'<span class="{"best" if v == top else "rest"}">{v * 100:.0f}%</span>' for k, v in vals.items()}})
            st.markdown(theme.table([("t", "Pattern", "text"), ("n", "Cases", "num dim"), ("xgb", "XGBoost", "num"), ("gnn", "GNN", "num"), ("blend", "Blend", "num")], trows, wrap=False), unsafe_allow_html=True)

    o = m.get("overlap_at_1pct_fpr") or {}
    if o:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(theme.card_title("Why use two models", "Laundering payments caught at 1% false alarms, by which model catches them."), unsafe_allow_html=True)
            st.markdown(theme.stack_bar([("both models", o.get("both", 0), "w"), ("GNN only", o.get("gnn_only", 0), "a"), ("XGBoost only", o.get("xgb_only", 0), "g"), ("neither", o.get("neither", 0), "")]), unsafe_allow_html=True)
    st.markdown('<div class="note" style="margin-top:20px">The data is synthetic (IBM\'s AML dataset) and covers about 10 days. Roughly 4 in 10 laundering payments belong to no known pattern in the dataset; at the strict alert level the models find almost none of those. '
                'Numbers are averages over three training runs, so differences of a few hundredths are noise. Settings such as the alert level were chosen on a separate window; the test window was scored once.</div>', unsafe_allow_html=True)


def page_about(b: logic.Bundle, S: dict) -> None:
    m = b.metrics
    st.markdown(theme.title_row("About", "Find money laundering among millions of ordinary payments, and say why."), unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns(3, gap="large")
        c1.markdown(theme.card_title("The problem") + '<div class="note">Banks must spot laundering in a flood of normal payments, and almost every alert they raise is a false alarm. PaperTrail ranks accounts and payments by how suspicious they look and shows the reasons, so an investigator can start with the most promising cases.</div>', unsafe_allow_html=True)
        c2.markdown(theme.card_title("How it works") + '<div class="note">One model (XGBoost) reads each payment and the recent behaviour of both accounts. A second, a graph neural network, reads the web of payments around them, using only payments from earlier days. The blend of the two finds more laundering than either alone.</div>', unsafe_allow_html=True)
        c3.markdown(theme.card_title("Honest limits") + '<div class="note">Synthetic data, a short time window, and a demonstration of the method. It is not a product for real banks, and the accounts you see here are not real.</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    rows = [("Data", "IBM's synthetic anti-money-laundering dataset (HI-Small): about 5.08 million payments over 18 days, roughly 0.1% of them laundering."),
            ("Training window", "Sep 1-6, 2022. Account histories are built only from payments before each payment."),
            ("Validation window", "Sep 7-8, 2022. Used to choose settings and alert levels."),
            ("Test window", f"Sep 9-10, 2022: {m['n_test_transactions']:,} payments, {m['n_test_positives']:,} of them laundering. Scored once per model; this is what the console shows."),
            ("How it is judged", "Ranking quality (PR-AUC), balance of right alerts and found cases (F1), laundering found at 1% false alarms. Never accuracy.")]
    st.markdown(theme.table([("k", "", "id"), ("v", "", "wrap")], [{"k": theme.esc(k), "v": theme.esc(v)} for k, v in rows], wrap=True), unsafe_allow_html=True)


# ------------------------------------------------------------------ run
if not logic.bundle_present(DEMO_DIR):
    st.markdown(theme.title_row("PaperTrail", "The demo data is not installed yet. Unzip <b>demo_bundle.zip</b> (made by notebooks/06-demo-slice.ipynb) into <b>data/demo/</b>."), unsafe_allow_html=True)
    st.stop()
bundle = get_bundle(str(DEMO_DIR))
state = view_state()
scope = f"Synthetic payments, {bundle.metrics['test_window']}. {bundle.alerts.height} accounts in the queue."
st.sidebar.markdown(theme.sidebar_html(state, PAGES, logic.MODELS, logic.MODEL_BLURB, scope), unsafe_allow_html=True)
st.markdown(theme.mobile_nav_html(state, PAGES, logic.MODELS), unsafe_allow_html=True)
{"queue": page_queue, "account": page_account, "try": page_try, "models": page_models, "about": page_about}[state["page"]](bundle, state)
