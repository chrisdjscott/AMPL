
import os
import json
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
# optional: nest this run under a parent workflow run and/or attach extra tags
MLFLOW_PARENT_RUN_ID = os.getenv("MLFLOW_PARENT_RUN_ID")
MLFLOW_TAGS = os.getenv("MLFLOW_TAGS")
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
    elif resp.status_code == 404:
        # Experiment does not exist yet; create it below.
        experiment_id = None
    else:
        resp.raise_for_status()

    if experiment_id is None:
        log.debug(f"Calling: {MLFLOW_URL}/api/2.0/mlflow/experiments/create")
        log.debug(f"Experiment name is: {MLFLOW_EXPERIMENT_NAME}")
        resp = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/create",
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
            json={"name": MLFLOW_EXPERIMENT_NAME},
            timeout=REQUESTS_TIMEOUT,
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
        timeout=REQUESTS_TIMEOUT,
    )
    run_id = run_resp.json()["run"]["info"]["run_id"]
    log.debug(f"Created new run with id: {run_id}")

    # nest under the workflow's parent run, if one was provided
    if MLFLOW_PARENT_RUN_ID:
        set_tag(run_id, "mlflow.parentRunId", MLFLOW_PARENT_RUN_ID)

    # arbitrary categorical tags (e.g. snakemake_experiment, model)
    if MLFLOW_TAGS:
        for key, value in json.loads(MLFLOW_TAGS).items():
            set_tag(run_id, key, str(value))

    return run_id


def set_tag(run_id, tag_name, tag_value):
    try:
        _check_mlflow_configured()

        resp = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/set-tag",
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
            json={
                "run_id": run_id,
                "key": tag_name,
                "value": tag_value,
            },
            timeout=REQUESTS_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Failed to set mlflow tag '{tag_name}', continuing: {e}")


def log_metric(run_id, metric_name, metric_value, step=0):
    try:
        _check_mlflow_configured()

        resp = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/log-metric",
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
            json={
                "run_id": run_id,
                "key": metric_name,
                "value": metric_value,
                "timestamp": int(time.time() * 1000),
                "step": step,
            },
            timeout=REQUESTS_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Failed to log mlflow metric '{metric_name}', continuing: {e}")


def log_param(run_id, param_name, param_value):
    try:
        _check_mlflow_configured()

        resp = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/log-parameter",
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
            json={
                "run_id": run_id,
                "key": param_name,
                "value": str(param_value),
            },
            timeout=REQUESTS_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Failed to log mlflow param '{param_name}', continuing: {e}")


def _get_artifact_uri(run_id):
    resp = requests.get(
        f"{MLFLOW_URL}/api/2.0/mlflow/runs/get",
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
        params={"run_id": run_id},
        timeout=REQUESTS_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["run"]["info"]["artifact_uri"]


def log_artifact(
    run_id: str,
    local_path: str,
    artifact_path: Optional[str] = None
):
    """
    Upload a single artifact file via the MLflow proxied artifact endpoint.

    This requires the tracking server to be started with artifact serving
    enabled (``mlflow server --serve-artifacts``), so that the run's
    artifact_uri is an ``mlflow-artifacts:/`` URI.

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
    _check_mlflow_configured()

    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    if artifact_path is None:
        artifact_path = local_path.name

    artifact_uri = _get_artifact_uri(run_id)
    if not artifact_uri.startswith("mlflow-artifacts:/"):
        raise RuntimeError(
            f"Cannot upload artifact via REST: run artifact_uri is '{artifact_uri}', "
            "not a proxied mlflow-artifacts URI. Start the tracking server with --serve-artifacts."
        )

    prefix = artifact_uri[len("mlflow-artifacts:/"):].strip("/")
    dest = f"{prefix}/{artifact_path}"

    with local_path.open("rb") as f:
        resp = requests.put(
            f"{MLFLOW_URL}/api/2.0/mlflow-artifacts/artifacts/{dest}",
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
            data=f,
            timeout=REQUESTS_TIMEOUT,
        )
    resp.raise_for_status()


def end_run(run_id):
    try:
        _check_mlflow_configured()

        resp = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/update",
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD),
            json={
                "run_id": run_id,
                "status": "FINISHED",
                "end_time": int(time.time() * 1000),
            },
            timeout=REQUESTS_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Failed to end mlflow run, continuing: {e}")
