#!/usr/bin/env python
"""Integration tests for StreamingFileDataset end-to-end training.

Each test trains the same model twice on a small Delaney slice: once with
``streaming=False`` (eager featurisation, the existing FileDataset path) and
once with ``streaming=True`` (per-batch featurisation through
StreamingFileDataset). Same seed, same splitter, same hyperparameters.

The streaming path is expected to land within a loose tolerance of the eager
path on test r2_score, demonstrating that:

* the dataset surface accepted by ``model.fit`` / ``model.predict`` is
  equivalent under streaming, and
* the per-batch featuriser yields semantically the same tensors as the
  one-shot featuriser used by the eager path.

Targets cover both the Keras and PyTorch wrapper branches:

* ``model_type=NN`` + ``featurizer=graphconv`` exercises
  ``GraphConvDCModelWrapper`` (Keras / TensorFlow, ``ConvMolFeaturizer``).
* ``model_type=AttentiveFPModel`` + ``featurizer=MolGraphConvFeaturizer``
  exercises ``PytorchDeepChemModelWrapper`` (DGL-backed AttentiveFP).
"""

import contextlib
import json
import os
import shutil
import sys

import pandas as pd
import pytest

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.utils.curate_data as curate_data
import atomsci.ddm.utils.struct_utils as struct_utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import integrative_utilities  # noqa: E402

TEST_DIR = os.path.dirname(os.path.realpath(__file__))


@contextlib.contextmanager
def _chdir(target):
    prev = os.getcwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(prev)


def _curate_delaney_in(test_dir):
    """Curate the bundled Delaney CSV into the test directory.

    Mirrors the curation step from ``test_delaney_NN.py`` but lives next to
    this test module so the eager and streaming runs share a fixed input.
    """
    csv_path = os.path.join(test_dir, 'delaney-processed_curated_fit.csv')
    if os.path.isfile(csv_path):
        return csv_path

    integrative_utilities.copy_delaney(dest=test_dir)
    raw_df = pd.read_csv(os.path.join(test_dir, 'delaney-processed.csv'))
    raw_df['rdkit_smiles'] = raw_df['smiles'].apply(curate_data.base_smiles_from_smiles)
    raw_df['inchi_key'] = raw_df['smiles'].apply(struct_utils.smiles_to_inchi_key)
    raw_df['compound_id'] = raw_df['inchi_key']

    curated_df = curate_data.average_and_remove_duplicates(
        'measured log solubility in mols per litre',
        10,
        'Yes',
        raw_df,
        100000,
        compound_id='compound_id',
        smiles_col='rdkit_smiles',
    )
    curated_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope='module')
def delaney_fit_csv():
    csv_path = _curate_delaney_in(TEST_DIR)
    yield csv_path


def _clean_result(test_dir):
    result_dir = os.path.join(test_dir, 'result')
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    for fname in os.listdir(test_dir):
        if fname.endswith('_train_valid_test_random.csv'):
            os.remove(os.path.join(test_dir, fname))


def _train_and_score(config_name, *, streaming, run_label):
    """Train one model with the requested streaming mode and return its test r2.

    The config is loaded fresh per call so the ``streaming`` flag flip is the
    only difference between the two runs.
    """
    config_path = os.path.join(TEST_DIR, config_name)
    with open(config_path) as f:
        config = json.loads(f.read())

    config['streaming'] = 'True' if streaming else 'False'
    config['result_dir'] = os.path.join(TEST_DIR, 'result', run_label)

    params = parse.wrapper(config)
    assert params.streaming is streaming, (
        f"parser did not honour streaming={streaming} (got {params.streaming})"
    )

    with _chdir(TEST_DIR):
        pipeline = mp.ModelPipeline(params)
        pipeline.train_model()

    metrics_path = os.path.join(pipeline.output_dir, 'model_metrics.json')
    assert os.path.exists(metrics_path), (
        f"model_metrics.json missing for {run_label} (looked under {pipeline.output_dir})"
    )
    with open(metrics_path) as f:
        metrics = json.loads(f.read())

    best_test = integrative_utilities.find_best_test_metric(metrics)
    assert best_test is not None, f"no best/test metric block for {run_label}"
    return best_test['prediction_results']['r2_score']


def _assert_comparable(eager_r2, streaming_r2, *, model_label, tol=0.25):
    """Both runs must finish with a finite r2 and be within ``tol`` of each other."""
    print(
        f"[{model_label}] eager test r2 = {eager_r2:.4f}, "
        f"streaming test r2 = {streaming_r2:.4f}, "
        f"|delta| = {abs(eager_r2 - streaming_r2):.4f}"
    )
    assert eager_r2 == eager_r2, f"{model_label} eager r2 is NaN"
    assert streaming_r2 == streaming_r2, f"{model_label} streaming r2 is NaN"
    assert abs(eager_r2 - streaming_r2) < tol, (
        f"{model_label}: streaming test r2 ({streaming_r2:.4f}) diverged from "
        f"eager ({eager_r2:.4f}) by more than {tol}"
    )


def test_streaming_graphconv_matches_eager(delaney_fit_csv):
    """GraphConvModel (Keras/TF) eager vs streaming on Delaney.

    Hits the ``GraphConvDCModelWrapper`` branch (model_wrapper.py:189).
    """
    _clean_result(TEST_DIR)
    eager_r2 = _train_and_score(
        'config_streaming_delaney_graphconv.json',
        streaming=False,
        run_label='graphconv_eager',
    )
    streaming_r2 = _train_and_score(
        'config_streaming_delaney_graphconv.json',
        streaming=True,
        run_label='graphconv_streaming',
    )
    _assert_comparable(eager_r2, streaming_r2, model_label='GraphConvModel')


@pytest.mark.dgl_required
def test_streaming_attentivefp_matches_eager(delaney_fit_csv):
    """AttentiveFPModel (PyTorch) eager vs streaming on Delaney.

    Hits the ``PytorchDeepChemModelWrapper`` branch (model_wrapper.py:221).
    """
    _clean_result(TEST_DIR)
    eager_r2 = _train_and_score(
        'config_streaming_delaney_attentivefp.json',
        streaming=False,
        run_label='attentivefp_eager',
    )
    streaming_r2 = _train_and_score(
        'config_streaming_delaney_attentivefp.json',
        streaming=True,
        run_label='attentivefp_streaming',
    )
    _assert_comparable(eager_r2, streaming_r2, model_label='AttentiveFPModel')


if __name__ == '__main__':
    csv_path = _curate_delaney_in(TEST_DIR)
    test_streaming_graphconv_matches_eager(csv_path)
    test_streaming_attentivefp_matches_eager(csv_path)
