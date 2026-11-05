"""MLflow model-registry operations. Stages are aliases (MLflow 3 deprecates stages):
`candidate` (registered, not yet decided), `production` (served), `previous` (what rollback restores).
Every version carries tags: data_hash, git_sha, feature_list_hash, n_features, trained_on, and its metrics are logged on its run."""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from papertrail.lifecycle.config import LifecycleConfig

PRODUCTION, PREVIOUS, CANDIDATE = "production", "previous", "candidate"


class Registry:
    def __init__(self, cfg: LifecycleConfig | None = None):
        self.cfg = cfg or LifecycleConfig()
        artifact_root = None
        if self.cfg.tracking_uri.startswith("sqlite:///"):             # keep artifacts next to the db, not in ./mlruns
            db = Path(self.cfg.tracking_uri.removeprefix("sqlite:///")).resolve()
            db.parent.mkdir(parents=True, exist_ok=True)
            artifact_root = (db.parent / "mlartifacts").as_uri()
        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        mlflow.set_registry_uri(self.cfg.tracking_uri)
        self.client = MlflowClient(self.cfg.tracking_uri, self.cfg.tracking_uri)
        if artifact_root and self.client.get_experiment_by_name(self.cfg.experiment) is None:
            self.client.create_experiment(self.cfg.experiment, artifact_location=artifact_root)
        mlflow.set_experiment(self.cfg.experiment)

    # -- registration ---------------------------------------------------------------------------------------------
    def register(self, model_dir: Path, metrics: dict[str, float], tags: dict[str, str], run_name: str = "train") -> str:
        """Log the model dir + metrics to a run, register it as a new version and mark it `candidate`. Returns the version."""
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
            mlflow.log_artifacts(str(model_dir), "model")
            mlflow.set_tags(tags)
        try:
            self.client.create_registered_model(self.cfg.model_name)
        except MlflowException:
            pass                                                     # already exists
        mv = self.client.create_model_version(self.cfg.model_name, f"{run.info.artifact_uri}/model", run.info.run_id,
                                              tags={**tags, "metrics_json": json.dumps(metrics)})
        self.client.set_registered_model_alias(self.cfg.model_name, CANDIDATE, mv.version)
        return str(mv.version)

    # -- lookups --------------------------------------------------------------------------------------------------
    def version_of(self, alias: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(self.cfg.model_name, alias).version)
        except MlflowException:
            return None

    def info(self, version: str) -> dict:
        mv = self.client.get_model_version(self.cfg.model_name, version)
        return {"version": str(mv.version), "tags": dict(mv.tags), "aliases": list(mv.aliases), "run_id": mv.run_id, "created": mv.creation_timestamp}

    def materialise(self, version: str, dest: Path) -> Path:
        """Download a version's files into `dest` in the layout the API loads (meta.json + boosters)."""
        mv = self.client.get_model_version(self.cfg.model_name, version)
        dest.mkdir(parents=True, exist_ok=True)
        local = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=str(dest))
        return Path(local)

    # -- transitions ----------------------------------------------------------------------------------------------
    def promote(self, version: str) -> None:
        """`version` becomes production; the old production becomes `previous` (the rollback target)."""
        old = self.version_of(PRODUCTION)
        if old is not None and old != version:
            self.client.set_registered_model_alias(self.cfg.model_name, PREVIOUS, old)
        self.client.set_registered_model_alias(self.cfg.model_name, PRODUCTION, version)
        self._drop(CANDIDATE, version)

    def reject(self, version: str, reasons: list[str]) -> None:
        self.client.set_model_version_tag(self.cfg.model_name, version, "rejected_reasons", "; ".join(reasons)[:5000])
        self._drop(CANDIDATE, version)

    def rollback(self) -> tuple[str, str]:
        """Restore the `previous` version as production. Returns (retired_version, restored_version)."""
        prev, cur = self.version_of(PREVIOUS), self.version_of(PRODUCTION)
        if prev is None or cur is None:
            raise RuntimeError("nothing to roll back to: no previous production version")
        self.client.set_registered_model_alias(self.cfg.model_name, PRODUCTION, prev)
        self.client.delete_registered_model_alias(self.cfg.model_name, PREVIOUS)
        self.client.set_model_version_tag(self.cfg.model_name, cur, "retired", time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
        return cur, prev

    def _drop(self, alias: str, version: str) -> None:
        if self.version_of(alias) == version:
            self.client.delete_registered_model_alias(self.cfg.model_name, alias)

    # -- audit ----------------------------------------------------------------------------------------------------
    def log_decision(self, name: str, params: dict, metrics: dict[str, float] | None = None, payload: dict | None = None) -> None:
        """One MLflow run per lifecycle decision (promotion, rejection, rollback, drift check): the audit trail."""
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            if metrics:
                mlflow.log_metrics(metrics)
            if payload is not None:
                mlflow.log_dict(payload, "decision.json")
