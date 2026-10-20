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
