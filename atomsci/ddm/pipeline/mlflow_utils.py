
import os
import logging
import time
from typing import Optional
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth


MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
MLFLOW_RUN_NAME = os.getenv("MLFLOW_RUN_NAME")
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
REQUESTS_TIMEOUT = 60


log = logging.getLogger("ATOM")


def _check_mlflow_configured():
    missing_env = []

    if MLFLOW_URL is None:
        missing_env.append("MLFLOW_TRACKING_URI")

    if MLFLOW_EXPERIMENT_NAME is None:
        missing_env.append("MLFLOW_EXPERIMENT_NAME")

    if MLFLOW_RUN_NAME is None:
        missing_env.append("MLFLOW_RUN_NAME")

    if MLFLOW_USERNAME is None:
        missing_env.append("MLFLOW_TRACKING_USERNAME")

    if MLFLOW_PASSWORD is None:
        missing_env.append("MLFLOW_TRACKING_PASSWORD")

    if len(missing_env):
        raise RuntimeError(f"mlflow config env vars not set: {missing_env}")


def get_or_create_experiment():
    log.debug(f"Getting or creating mlflow experiment: {MLFLOW_EXPERIMENT_NAME}")

    _check_mlflow_configured()

    resp = requests.get(
        f"{MLFLOW_URL}/api/2.0/mlflow/experiments/get-by-name",
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
        params={"experiment_name": MLFLOW_EXPERIMENT_NAME},
        timeout=REQUESTS_TIMEOUT,
    )
    log.debug(f"Response from call to experiments/get-by-name: {resp}")

    if resp.status_code == 200:
        experiment_id = resp.json()["experiment"]["experiment_id"]
        log.debug(f"Got existing experiment with id: {experiment_id}")
    else:
        experiment_id = None

    if experiment_id is None:
        log.debug(f"Calling: {MLFLOW_URL}/api/2.0/mlflow/experiments/create")
        log.debug(f"Experiment name is: {MLFLOW_EXPERIMENT_NAME}")
        resp = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/create",
            json={"name": MLFLOW_EXPERIMENT_NAME},
        )
        log.debug(f"Response: {resp}")
        log.debug(f"Status code: {resp.status_code}")
        log.debug(f"Content: {resp.content}")
        experiment_id = resp.json()["experiment_id"]
        log.debug(f"Created new experiment with id: {experiment_id}")

    return experiment_id


def create_run(experiment_id):
    log.debug(f"Creating new mlflow run: {MLFLOW_RUN_NAME}")

    _check_mlflow_configured()

    run_resp = requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/runs/create",
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
        json={
            "experiment_id": experiment_id,
            "start_time": int(time.time() * 1000),
            "run_name": MLFLOW_RUN_NAME,
        },
    )
    run_id = run_resp.json()["run"]["info"]["run_id"]
    log.debug(f"Created new run with id: {run_id}")

    return run_id


def set_tag(run_id, tag_name, tag_value):
    _check_mlflow_configured()

    requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/runs/set-tag",
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
        json={
            "run_id": run_id,
            "key": tag_name,
            "value": tag_value,
        },
    )


def log_metric(run_id, metric_name, metric_value, step=0):
    _check_mlflow_configured()

    requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/runs/log-metric",
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
        json={
            "run_id": run_id,
            "key": metric_name,
            "value": metric_value,
            "timestamp": int(time.time() * 1000),
            "step": step,
        },
    )


def log_param(run_id, param_name, param_value):
    requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/runs/log-parameter",
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
        json={
            "run_id": run_id,
            "key": param_name,
            "value": str(param_value),
        },
    )


def log_artifact(
    run_id: str,
    local_path: str,
    artifact_path: Optional[str] = None
):
    """
    Upload a single artifact file.

    Parameters
    ----------
    run_id : str
        MLflow run id
    local_path : str
        Path to the local file to upload
    artifact_path : str | None
        Destination path within the run's artifact store.
        If None, defaults to the local file's basename.
    """
    log.debug(f"Logging artifact to mlflow (run id: {run_id}; local_path: {local_path}; artifact_path: {artifact_path})")
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    if artifact_path is None:
        # MLflow REST requires a path - default to filename
        artifact_path = local_path.name

    with local_path.open("rb") as f:
        files = {
            "file": (local_path.name, f),
        }
        data = {
            "run_id": run_id,
            "path": artifact_path,
        }

        resp = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/artifacts/upload",
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
            files=files,
            data=data,
            timeout=REQUESTS_TIMEOUT,
        )

    return resp.json() if resp.content else None


def end_run(run_id):
    requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/runs/update",
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
        json={
            "run_id": run_id,
            "status": "FINISHED",
            "end_time": int(time.time() * 1000),
        },
    )
