"""Click through every view of the Streamlit console against the real demo bundle. Skipped until data/demo/ holds the bundle.
State lives in the URL (?page=...&model=...&account=...), so the views are driven through query parameters."""
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = ("queue", "account", "try", "models", "about")
pytestmark = pytest.mark.skipif(not (DEMO / "alerts.parquet").exists(), reason="demo bundle not installed in data/demo/")


@pytest.fixture
def run(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PAPERTRAIL_DEMO_DIR", str(DEMO))

    def go(**params):
        at = AppTest.from_file(str(APP), default_timeout=120)
        for k, v in params.items():
            at.query_params[k] = v
        return at.run()

    return go


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("model", ["blend", "xgb", "gnn"])
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_for_every_model(run, page, model):
    _ok(run(page=page, model=model))


def test_queue_options_and_unknown_parameters_fall_back_safely(run):
    _ok(run(page="queue", reveal="1", p="2"))
    _ok(run(page="queue", p="not-a-number"))
    _ok(run(page="nonsense", model="zzz"))
    _ok(run(page="account", account="no-such-account"))


def test_account_page_opens_the_account_in_the_link(run):
    import polars as pl

    account = pl.read_parquet(DEMO / "alerts.parquet").sort("blend_rank")["account"][3]
    at = run(page="account", account=account)
    _ok(at)
    assert any(account in m.value for m in at.markdown)


def test_what_if_responds_to_changed_inputs(run):
    at = run(page="try")
    _ok(at)
    at.number_input[0].set_value(100.0).run()
    _ok(at)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_in_the_light_theme(run, page):
    _ok(run(page=page, theme="light"))
    _ok(run(page=page, theme="no-such-theme"))


def _bundle_with_case(tmp_path, account):
    import shutil

    root = tmp_path / "demo"
    shutil.copytree(DEMO, root)
    case = {"summary": {"account_id": account, "risk_score": 5.61, "suspected_typology": "FAN-IN", "recommended_action": "escalate_SAR",
                        "narrative": "Many senders pay this account in a short time.",
                        "evidence": [{"claim": "Ranked 1 of 200 by the blend model", "source_id": f"alert:{account}"},
                                     {"claim": "5 distinct senders", "source_id": f"subgraph:{account}"}]},
            "verified": True, "drafts": 1, "provider": "groq", "model": "llama-3.3-70b-versatile", "total_latency_s": 3.2, "generated_at": "2026-10-20 12:00 UTC"}
    (root / "cached_cases.json").write_text(__import__("json").dumps({account: case}))
    return root


def test_account_page_shows_the_cached_case_summary(tmp_path, monkeypatch):
    import polars as pl
    from streamlit.testing.v1 import AppTest

    account = pl.read_parquet(DEMO / "alerts.parquet").sort("blend_rank")["account"][0]
    monkeypatch.setenv("PAPERTRAIL_DEMO_DIR", str(_bundle_with_case(tmp_path, account)))
    for k in ("LLM_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(k, "")
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params["page"] = "account"
    at.query_params["account"] = account
    at.run()
    _ok(at)
    text = " ".join(m.value for m in at.markdown)
    assert "AI case summary" in text and "Many senders pay this account" in text and "Escalate (SAR)" in text and "Verified against tool outputs" in text


def test_account_without_a_cached_case_says_so(run):
    import polars as pl

    account = pl.read_parquet(DEMO / "alerts.parquet").sort("blend_rank")["account"][-1]
    at = run(page="account", account=account)
    _ok(at)
    assert any("No summary is cached for this account" in m.value for m in at.markdown)
