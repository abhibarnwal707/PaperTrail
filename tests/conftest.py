from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"


@pytest.fixture
def toolbox():
    """The investigator's four tools over the committed demo bundle (skipped when langgraph or the bundle is missing)."""
    pytest.importorskip("langgraph")
    from papertrail.agent.tools import Toolbox

    if not (DEMO / "alerts.parquet").exists():
        pytest.skip("demo bundle not installed")
    return Toolbox(DEMO)
