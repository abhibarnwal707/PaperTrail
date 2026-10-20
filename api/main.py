"""PaperTrail scoring API.

Self-contained on purpose (no `papertrail` import): it needs only fastapi, pydantic, numpy, polars and xgboost, which keeps the
Docker image small. The model lives in `$PAPERTRAIL_MODEL_DIR` (default data/demo/model) as one or more booster files plus `meta.json`:
    {"feature_names": [...in training order...], "threshold": 0.93, "model": "xgb_v2_final", "files": ["xgb_seed0.json", ...]}
`files` is optional (default ["xgb.json"]); with several files the probabilities are averaged, as in the evaluated 3-seed ensemble.
Account-level SHAP evidence is read from `$PAPERTRAIL_DEMO_DIR/shap.parquet` (columns: account, feature, value, contribution).

The API scores a transaction from FEATURES the caller supplies. History features (rolling counts, counterparty counts, ...) need an
account-history store and are computed upstream; any feature the caller leaves out is treated as missing, which XGBoost handles.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

log = logging.getLogger("papertrail.api")


class Scorer:
    """One or more XGBoost boosters (their probabilities are averaged), the feature order, and the alert threshold."""

    def __init__(self, boosters, feature_names: list[str], threshold: float, name: str = "xgb"):
        self.boosters = list(boosters) if isinstance(boosters, (list, tuple)) else [boosters]
        self.feature_names, self.threshold, self.name = list(feature_names), float(threshold), name

    @property
    def booster(self):                            # first booster, for single-model callers and tests
        return self.boosters[0]

    @classmethod
    def load(cls, model_dir: Path) -> Scorer:
        import xgboost as xgb  # imported lazily so the API module loads without a working xgboost

        meta = json.loads((model_dir / "meta.json").read_text())
        boosters = []
        for fname in meta.get("files", ["xgb.json"]):
            b = xgb.Booster()
            b.load_model(str(model_dir / fname))
            boosters.append(b)
        return cls(boosters, meta["feature_names"], meta["threshold"], meta.get("model", "xgb"))

    def score(self, rows: list[dict]) -> np.ndarray:
        import xgboost as xgb

        x = np.array([[np.nan if r.get(f) is None else r[f] for f in self.feature_names] for r in rows], dtype=np.float32)
        d = xgb.DMatrix(x, feature_names=self.feature_names)
        return np.mean([b.predict(d) for b in self.boosters], axis=0)


class ScoreRequest(BaseModel):
    features: dict[str, float | None] = Field(..., description="feature name -> value; omitted or null means missing")


class ScoreResponse(BaseModel):
    score: float
    flagged: bool
    threshold: float
    model: str


class Driver(BaseModel):
    feature: str
    value: float | None
    contribution: float


class ExplainResponse(BaseModel):
    account: str
    drivers: list[Driver]


def create_app(scorer: Scorer | None = None, demo_dir: Path | None = None) -> FastAPI:
    """App factory: tests inject a scorer and a demo folder; production loads them from the environment at startup."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.scorer is None:
            model_dir = Path(os.environ.get("PAPERTRAIL_MODEL_DIR", app.state.demo_dir / "model"))
            try:
                app.state.scorer = Scorer.load(model_dir)
                log.info("loaded model %s from %s", app.state.scorer.name, model_dir)
            except Exception as e:  # noqa: BLE001
                log.warning("no model loaded (%r); /score will return 503", e)
        yield

    app = FastAPI(title="PaperTrail scoring API", version="0.1.0", lifespan=lifespan)
    app.state.scorer = scorer
    app.state.demo_dir = Path(demo_dir or os.environ.get("PAPERTRAIL_DEMO_DIR", "data/demo"))

    @app.middleware("http")
    async def _latency(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-ms"] = f"{ms:.2f}"
        log.info("%s %s -> %s in %.1f ms", request.method, request.url.path, response.status_code, ms)
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": app.state.scorer is not None}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        s: Scorer | None = app.state.scorer
        if s is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        unknown = sorted(set(req.features) - set(s.feature_names))
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown features: {unknown}")
        p = float(s.score([req.features])[0])
        return ScoreResponse(score=p, flagged=p >= s.threshold, threshold=s.threshold, model=s.name)

    @app.get("/explain/{account_id}", response_model=ExplainResponse)
    def explain(account_id: str, top: int = 10) -> ExplainResponse:
        path = app.state.demo_dir / "shap.parquet"
        if not path.exists():
            raise HTTPException(status_code=404, detail="no explanation bundle installed")
        import polars as pl

        rows = (pl.read_parquet(path).filter(pl.col("account") == account_id)
                  .with_columns(pl.col("contribution").abs().alias("_abs")).sort("_abs", descending=True).head(max(1, min(top, 50))))
        if rows.is_empty():
            raise HTTPException(status_code=404, detail=f"no explanation for account {account_id}")
        return ExplainResponse(account=account_id, drivers=[Driver(**r) for r in rows.select("feature", "value", "contribution").to_dicts()])

    return app


app = create_app()
