import atomsci.ddm.pipeline.perf_data as perf_data
import atomsci.ddm.pipeline.model_pipeline as model_pipeline
import atomsci.ddm.pipeline.parameter_parser as parse
import logging
import os
import tempfile
import deepchem as dc
import numpy as np
import shutil
import pandas as pd
import json
from types import SimpleNamespace

import pytest

def copy_to_temp(dskey, res_dir):
    """
    Copy a dataset to a temporary directory.

    Parameters:
    dskey (str): Path to the original dataset.
    res_dir (str): Path to the temporary directory.

    Returns:
    str: Path to the copied dataset in the temporary directory.
    """
    new_dskey = shutil.copy(dskey, res_dir)
    return new_dskey

def setup_paths():
    """
    Set up the paths for the test, including creating a temporary directory and copying the dataset to it.

    Returns:
    tuple: A tuple containing:
        - res_dir (str): Path to the temporary result directory.
        - tmp_dskey (str): Path to the copied dataset in the temporary directory.
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    res_dir = tempfile.mkdtemp()
    dskey = os.path.join(script_path, '../test_datasets/aurka_chembl_base_smiles_union.csv')
    tmp_dskey = copy_to_temp(dskey, res_dir)

    return res_dir, tmp_dskey

def read_params(json_file, res_dir, tmp_dskey):
    """
    Read parameters from a JSON file and update them with the result directory and dataset key.

    Parameters:
    json_file (str): Path to the JSON file containing parameters.
    res_dir (str): Path to the result directory.
    tmp_dskey (str): Path to the copied dataset in the temporary directory.

    Returns:
    dict: Updated parameters.
    """

    with open(json_file, 'r') as file:
        params = json.load(file)
    params['result_dir'] = res_dir
    params['dataset_key'] = tmp_dskey
    return params

def make_relative_to_file(relative_path):
    """
    Generates the full path relative to the location of this file.

    Parameters:
    relative_path (str): The relative path to convert.

    Returns:
    str: The absolute path corresponding to the relative path.
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    result = os.path.join(script_path, relative_path)

    return result

def test_KFoldRegressionPerfData():
    """
    Test the KFoldRegressionPerfData class to ensure it correctly handles k-fold regression performance data.

    """
    res_dir, tmp_dskey = setup_paths()

    params = read_params(make_relative_to_file('config_perf_data_KFoldRegressoinPerfData.json'),
        res_dir, tmp_dskey)

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, 'train')

    assert isinstance(perf, perf_data.KFoldRegressionPerfData)

    ids = sorted(list(mp.data.combined_training_data().ids))
    weights = perf.get_weights(ids)
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    real_vals = perf.get_real_values(ids)
    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones(len(ids)))

    pred_vals = d.y
    # This should have r2 of 1
    r2 = perf.accumulate_preds(pred_vals, ids)
    assert r2 == 1
    # do a few more folds
    r2 = perf.accumulate_preds(pred_vals, ids)
    r2 = perf.accumulate_preds(pred_vals, ids)

    (res_ids, res_vals, res_std) = perf.get_pred_values()
    (r2_mean, r2_std) = perf.compute_perf_metrics()

    assert np.allclose(res_vals, real_vals)
    assert np.allclose(res_std, np.zeros_like(res_std))

    # perfect score every time
    assert r2_mean==1
    assert r2_std==0

def test_KFoldRegressionPerfDataMulti():
    """
    Test the KFoldRegressionPerfData class for multi-fold regression performance data.
    """
    res_dir, tmp_dskey = setup_paths()

    # duplicate pIC50 column
    df = pd.read_csv(tmp_dskey)
    df['pIC50_dupe'] = df['pIC50']
    df.to_csv(tmp_dskey, index=False)

    params = read_params(make_relative_to_file('config_perf_data_KFoldRegressoinPerfDataMulti.json'),
        res_dir, tmp_dskey)

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, 'train')

    assert isinstance(perf, perf_data.KFoldRegressionPerfData)

    ids = sorted(list(mp.data.combined_training_data().ids))
    weights = perf.get_weights(ids)
    assert weights.shape == (len(ids),2)
    assert np.allclose(weights, np.ones_like(weights))

    real_vals = perf.get_real_values(ids)
    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones_like(weights))

    pred_vals = d.y
    # This should have r2 of 1
    r2 = perf.accumulate_preds(pred_vals, ids)
    assert r2 == 1
    # do a few more folds
    r2 = perf.accumulate_preds(pred_vals, ids)
    r2 = perf.accumulate_preds(pred_vals, ids)

    (res_ids, res_vals, res_std) = perf.get_pred_values()
    (r2_mean, r2_std) = perf.compute_perf_metrics()

    assert np.allclose(res_vals, real_vals)
    assert np.allclose(res_std, np.zeros_like(res_std))

    # perfect score every time
    assert r2_mean==1
    assert r2_std==0

def test_KFoldClassificationPerfData():
    """
    Test the KFoldClassificationPerfData functionality.

    """
    res_dir, tmp_dskey = setup_paths()

    params = read_params(
        make_relative_to_file('config_perf_data_KFoldClassificationPerfData.json'),
        res_dir, tmp_dskey)

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, 'train')

    assert isinstance(perf, perf_data.KFoldClassificationPerfData)

    ids = sorted(list(mp.data.combined_training_data().ids))
    weights = perf.get_weights(ids)
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    real_vals = perf.get_real_values(ids)
    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones(len(ids)))

    num_classes = 2
    # input to to_one_hot needs to have the shape (N,) not (N,1)
    pred_vals = dc.metrics.to_one_hot(d.y.reshape(len(d.y)), num_classes)
    # This should have r2 of 1
    roc_auc_score = perf.accumulate_preds(pred_vals, ids)
    assert roc_auc_score == 1
    # do a few more folds
    roc_auc_score = perf.accumulate_preds(pred_vals, ids)
    roc_auc_score = perf.accumulate_preds(pred_vals, ids)

    (res_ids, res_classes, res_probs, res_std) = perf.get_pred_values()
    (roc_auc_mean, roc_auc_std) = perf.compute_perf_metrics()

    # std should be zero
    assert all((res_std==np.zeros_like(res_std)).flatten())
    # probs should match predictions
    assert all((res_probs==pred_vals.reshape(len(d.y), 1, num_classes)).flatten())
    # all predictions are correct
    assert all(res_classes==real_vals)
    # perfect score every time
    assert roc_auc_mean==1
    assert roc_auc_std==0

def test_SimpleRegressionPerfData():
    """
    Test the SimpleRegressionPerfData class for correct performance data creation and metrics computation.

    """
    res_dir, tmp_dskey = setup_paths()

    params = read_params(
        make_relative_to_file('config_perf_data_SimpleRegressionPerfData.json'),
        res_dir, tmp_dskey)

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, 'train')

    assert isinstance(perf, perf_data.SimpleRegressionPerfData)

    real_vals = perf.get_real_values()
    weights = perf.get_weights()
    ids = np.array(range(len(real_vals))) # these are not used by SimpleRegressionPerfData
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, 
                             ids=ids, w=np.ones(len(ids)))

    pred_vals = d.y
    # This should have r2 of 1 ids are ignored
    r2 = perf.accumulate_preds(pred_vals, ids)
    assert r2 == 1

    (res_ids, res_vals, _) = perf.get_pred_values()
    (r2_mean, _) = perf.compute_perf_metrics()

    # the predicted values should equal the real values
    assert all(real_vals == res_vals)

    # should be a perfect score
    assert r2_mean == 1

def test_SimpleClassificationPerfData():
    """
    Test function for SimpleClassificationPerfData.

    This function sets up a model pipeline, trains a model, and creates performance data
    for a simple classification task. It then verifies the following:

    """
    res_dir, tmp_dskey = setup_paths()

    params = read_params(
        make_relative_to_file('config_perf_data_SimpleClassificationPerfData.json'),
        res_dir, tmp_dskey)

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, 'train')

    assert isinstance(perf, perf_data.SimpleClassificationPerfData)

    ids = sorted(list(mp.data.train_valid_dsets[0][0].ids))
    weights = perf.get_weights()
    real_vals = perf.get_real_values()
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones(len(ids)))

    num_classes = 2
    # input to to_one_hot needs to have the shape (N,) not (N,1)
    pred_vals = dc.metrics.to_one_hot(d.y.reshape(len(d.y)), num_classes)
    # This should have r2 of 1
    roc_auc_score = perf.accumulate_preds(pred_vals, ids=ids)
    assert roc_auc_score == 1

    (res_ids, res_classes, res_probs, _) = perf.get_pred_values()
    (roc_auc_mean, _) = perf.compute_perf_metrics()

    # probs should match predictions
    assert all((res_probs==pred_vals.reshape(len(d.y), 1, num_classes)).flatten())
    # all predictions are correct
    assert all(res_classes==real_vals)
    # perfect score every time
    assert roc_auc_mean==1


def test_safe_regression_score_filters_when_invalid_fraction_within_threshold():
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=0.50))

    y_real = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([0.0, 1.0, np.nan, 3.0], dtype=float)
    score = perf_data._safe_regression_score('r2', y_real, y_pred)

    assert np.isclose(score, 1.0)


def test_safe_regression_score_penalizes_when_invalid_fraction_exceeds_threshold():
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=0.10))

    y_real = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([0.0, 1.0, np.nan, 3.0], dtype=float)
    score = perf_data._safe_regression_score('r2', y_real, y_pred)

    assert score == -1.0e12


@pytest.mark.parametrize(
    'score_type,expected_penalty',
    [('mae', 1.0e12), ('rmse', 1.0e12)],
)
def test_safe_regression_score_penalty_direction_for_loss_metrics(score_type, expected_penalty):
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=0.10))

    y_real = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([0.0, np.nan, np.nan, 3.0], dtype=float)
    score = perf_data._safe_regression_score(score_type, y_real, y_pred)

    assert score == expected_penalty


def test_safe_classification_score_filters_when_invalid_fraction_within_threshold():
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=0.50))

    y_real = np.array([0, 1, 1, 0], dtype=int)
    y_pred = np.array([0, 1, np.nan, 0], dtype=float)
    score = perf_data._safe_classification_score('accuracy', y_real, y_pred)

    assert np.isclose(score, 1.0)


def test_safe_classification_score_penalizes_when_invalid_fraction_exceeds_threshold():
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=0.10))

    y_real = np.array([0, 1, 1, 0], dtype=int)
    y_pred = np.array([0, 1, np.nan, 0], dtype=float)
    score = perf_data._safe_classification_score('accuracy', y_real, y_pred)

    assert score == 0.0


@pytest.mark.parametrize(
    'score_type,y_real,y_pred,expected_value',
    [
        ('roc_auc', np.array([0, 1, 0, 1], dtype=int), np.array([0.1, 0.9, np.nan, 0.8], dtype=float), 1.0),
        ('avg_precision', np.array([0, 1, 0, 1], dtype=int), np.array([0.1, 0.9, np.nan, 0.8], dtype=float), 1.0),
    ],
)
def test_safe_classification_probability_metrics_filter_small_invalid_fraction(score_type, y_real, y_pred, expected_value):
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=0.50))

    score = perf_data._safe_classification_score(score_type, y_real, y_pred)
    assert np.isclose(score, expected_value)


def test_safe_classification_cross_entropy_filter_small_invalid_fraction():
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=0.50))

    y_real = np.array([0, 1, 0, 1], dtype=int)
    y_pred = np.array([0.2, 0.8, np.nan, 0.7], dtype=float)
    score = perf_data._safe_classification_score('cross_entropy', y_real, y_pred)
    expected = perf_data.log_loss(np.array([0, 1, 1]), np.array([0.2, 0.8, 0.7]))
    assert np.isclose(score, expected)


def test_safe_classification_cross_entropy_penalty_when_invalid_fraction_exceeds_threshold():
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=0.10))

    y_real = np.array([0, 1, 0, 1], dtype=int)
    y_pred = np.array([0.2, np.nan, np.nan, 0.7], dtype=float)
    score = perf_data._safe_classification_score('cross_entropy', y_real, y_pred)
    assert score == 1.0e12


def test_safe_regression_score_returns_penalty_for_all_invalid_rows():
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=0.99))

    y_real = np.array([np.nan, np.nan], dtype=float)
    y_pred = np.array([np.nan, np.nan], dtype=float)
    score = perf_data._safe_regression_score('r2', y_real, y_pred)
    assert score == -1.0e12


def test_safe_classification_score_returns_penalty_for_shape_mismatch():
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=0.50))

    y_real = np.array([0, 1, 0], dtype=int)
    y_pred = np.array([0.1, np.nan], dtype=float)
    score = perf_data._safe_classification_score('accuracy', y_real, y_pred)
    assert score == 0.0


def test_coerce_invalid_pred_classes_binary_sets_wrong_class_for_invalid_rows():
    y_real = np.array([0, 1, 1, 0], dtype=int)
    y_pred = np.array([0, 1, 0, 0], dtype=int)
    class_probs = np.array([0.1, np.inf, 0.8, np.nan], dtype=float)

    coerced = perf_data._coerce_invalid_pred_classes(y_real, y_pred, class_probs, 2)

    assert np.array_equal(coerced, np.array([0, 0, 0, 1]))


def test_safe_confusion_matrix_uses_coerced_classes_not_zeroed_when_real_is_finite():
    y_real = np.array([0, 1, 1, 0], dtype=int)
    y_pred = np.array([0, 0, 0, 1], dtype=int)

    cm = perf_data._safe_confusion_matrix(y_real, y_pred, 2)
    assert cm == [[1, 1], [2, 0]]


def test_max_invalid_pred_frac_clamps_out_of_range_values():
    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=2.5))
    assert perf_data._get_active_invalid_pred_frac_threshold() == 1.0

    perf_data._configure_invalid_pred_frac_threshold(SimpleNamespace(max_invalid_pred_frac=-0.2))
    assert perf_data._get_active_invalid_pred_frac_threshold() == 0.0


def test_simple_regression_prediction_results_include_invalid_prediction_fields():
    class _RegDataset:
        def __init__(self):
            self.ids = np.array([0, 1, 2, 3])
            self.y = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
            self.w = np.ones((4, 1), dtype=float)

    class _RegModelDataset:
        def __init__(self):
            dset = _RegDataset()
            self.train_valid_dsets = [(dset, dset)]
            self.dataset = dset
            self.test_dset = dset
            self._responses = dset.y

        def get_untransformed_responses(self, ids):
            return self._responses[np.array(ids, dtype=int)]

    model_dataset = _RegModelDataset()
    perf = perf_data.SimpleRegressionPerfData(model_dataset, subset='train')
    real_vals = perf.get_real_values()
    ids = model_dataset.train_valid_dsets[0][0].ids
    pred_vals = real_vals.copy()
    pred_vals[0, 0] = np.nan

    perf.accumulate_preds(pred_vals, ids)
    pred_results = perf.get_prediction_results()

    for key in [
        'invalid_prediction_count',
        'invalid_prediction_fraction',
        'invalid_prediction_threshold',
        'task_invalid_prediction_counts',
        'task_invalid_prediction_fractions',
    ]:
        assert key in pred_results

    assert pred_results['invalid_prediction_count'] == 1
    assert np.isclose(pred_results['invalid_prediction_fraction'], 1.0 / len(real_vals))
    assert len(pred_results['task_invalid_prediction_counts']) == perf.num_tasks


def test_simple_classification_prediction_results_include_invalid_prediction_fields():
    class _ClsDataset:
        def __init__(self):
            self.ids = np.array([0, 1, 2, 3])
            self.y = np.array([[0], [1], [0], [1]], dtype=int)
            self.w = np.ones((4, 1), dtype=float)

    class _ClsModelDataset:
        def __init__(self):
            dset = _ClsDataset()
            self.train_valid_dsets = [(dset, dset)]
            self.dataset = dset
            self.test_dset = dset
            self._responses = dset.y

        def get_untransformed_responses(self, ids):
            return self._responses[np.array(ids, dtype=int)]

    model_dataset = _ClsModelDataset()
    perf = perf_data.SimpleClassificationPerfData(model_dataset, subset='train')
    ids = model_dataset.train_valid_dsets[0][0].ids

    pred_vals = np.array([
        [[1.0, 0.0]],
        [[0.0, 1.0]],
        [[1.0, 0.0]],
        [[0.0, 1.0]],
    ], dtype=float)
    pred_vals[0, 0, 1] = np.nan

    perf.accumulate_preds(pred_vals, ids=ids)
    pred_results = perf.get_prediction_results()

    for key in [
        'invalid_prediction_count',
        'invalid_prediction_fraction',
        'invalid_prediction_threshold',
        'task_invalid_prediction_counts',
        'task_invalid_prediction_fractions',
    ]:
        assert key in pred_results

    assert pred_results['invalid_prediction_count'] == 1
    assert np.isclose(pred_results['invalid_prediction_fraction'], 0.25)


def test_simple_hybrid_prediction_results_include_invalid_prediction_fields():
    class _HybridDataset:
        def __init__(self):
            self.ids = np.array([0, 1, 2, 3])
            self.y = np.array([
                [7.0, np.nan],
                [6.5, np.nan],
                [0.8, 50.0],
                [0.2, 50.0],
            ], dtype=float)
            self.w = np.ones((4, 2), dtype=float)

    class _HybridModelDataset:
        def __init__(self):
            dset = _HybridDataset()
            self.train_valid_dsets = [(dset, dset)]
            self.dataset = dset
            self.test_dset = dset
            self._responses = dset.y

        def get_untransformed_responses(self, ids):
            return self._responses[np.array(ids, dtype=int)]

    model_dataset = _HybridModelDataset()
    perf = perf_data.SimpleHybridPerfData(model_dataset, subset='train', is_ki=False)
    pred_vals = model_dataset._responses.copy()
    pred_vals[0, 0] = np.nan

    perf.accumulate_preds(pred_vals, ids=model_dataset.train_valid_dsets[0][0].ids)
    pred_results = perf.get_prediction_results()

    for key in [
        'invalid_prediction_count',
        'invalid_prediction_fraction',
        'invalid_prediction_threshold',
        'task_invalid_prediction_counts',
        'task_invalid_prediction_fractions',
    ]:
        assert key in pred_results

    assert pred_results['invalid_prediction_count'] == 1
    assert np.isclose(pred_results['invalid_prediction_fraction'], 0.25)


if __name__ == "__main__":
    test_KFoldRegressionPerfDataMulti()
    test_KFoldRegressionPerfData()
    test_SimpleClassificationPerfData()
    test_KFoldClassificationPerfData()
    test_SimpleRegressionPerfData()

# ****************************************************************************************
# Regression guard: create_perf_data must accept split_strategy='indexed'.
#
# IndexedSplitting (splitting.IndexedSplitting, split_strategy='indexed') produces a
# single train/valid/test partition, so it needs the same Simple*PerfData as
# 'train_valid_test'. create_perf_data originally matched 'train_valid_test'
# literally and raised ValueError('Unknown split_strategy indexed') at EpochManager
# construction, i.e. before epoch 0 of every streaming run using a scaffold presplit.
def _indexed_model_dataset(split_strategy):
    class _Dataset:
        def __init__(self):
            self.ids = np.array([0, 1, 2, 3])
            self.y = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
            self.w = np.ones((4, 1), dtype=float)

    class _ModelDataset:
        def __init__(self):
            dset = _Dataset()
            self.train_valid_dsets = [(dset, dset)]
            self.dataset = dset
            self.test_dset = dset
            self._responses = dset.y
            self.params = SimpleNamespace(
                split_strategy=split_strategy,
                prediction_type='regression',
                max_invalid_pred_frac=0.01,
            )

        def get_untransformed_responses(self, ids):
            return self._responses[np.array(ids, dtype=int)]

    return _ModelDataset()


@pytest.mark.parametrize('split_strategy', ['train_valid_test', 'indexed'])
@pytest.mark.parametrize('subset', ['train', 'valid', 'test'])
def test_create_perf_data_single_split_strategies(split_strategy, subset):
    """Both single-partition strategies must yield SimpleRegressionPerfData."""
    perf = perf_data.create_perf_data(
        'regression', _indexed_model_dataset(split_strategy), subset)
    assert isinstance(perf, perf_data.SimpleRegressionPerfData)


def test_create_perf_data_rejects_unknown_split_strategy():
    """A genuinely unknown strategy must still raise, so the guard stays meaningful."""
    with pytest.raises(ValueError, match='Unknown split_strategy'):
        perf_data.create_perf_data(
            'regression', _indexed_model_dataset('not_a_real_strategy'), 'train')


# EpochManager used to build one PerfData per subset per epoch up front, which duplicates
# the untransformed responses max_epochs times and reads them out of the dataset just as
# often. Only the current and best epochs are ever read back, so EpochPerfData keeps those
# two and derives each epoch's object from the previous one.
def _kfold_model_dataset():
    class _Dataset:
        def __init__(self):
            self.ids = np.array([0, 1, 2, 3])
            self.y = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
            self.w = np.ones((4, 1), dtype=float)

    class _ModelDataset:
        def __init__(self):
            self._dset = _Dataset()
            self.test_dset = self._dset
            self.params = SimpleNamespace(
                split_strategy='k_fold_cv',
                prediction_type='regression',
                max_invalid_pred_frac=0.01,
            )

        def combined_training_data(self):
            return self._dset

        def get_subset_responses_and_weights(self, subset):
            real_vals = dict([(id, self._dset.y[id]) for id in self._dset.ids])
            weights = dict([(id, self._dset.w[id]) for id in self._dset.ids])
            return real_vals, weights

    return _ModelDataset()


def test_next_epoch_shares_dataset_arrays_and_clears_predictions():
    """next_epoch must reuse the real values and drop the previous epoch's predictions."""
    first = perf_data.create_perf_data(
        'regression', _indexed_model_dataset('train_valid_test'), 'train')
    first.accumulate_preds(np.array([[0.0], [1.0], [2.0], [3.0]]), first.ids)

    second = first.next_epoch()

    assert second is not first
    assert second.real_vals is first.real_vals
    assert second.weights is first.weights
    assert second.ids is first.ids
    assert second.pred_vals is None
    assert second.perf_metrics == []
    # the epoch we moved on from must be untouched, since it may be the best epoch
    assert len(first.perf_metrics) == 1


def test_next_epoch_resets_kfold_prediction_dict():
    """The k-fold classes accumulate into a dict of predictions, which must start empty."""
    first = perf_data.create_perf_data('regression', _kfold_model_dataset(), 'train')
    first.accumulate_preds(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 1, 2, 3]))
    assert first.folds == 1

    second = first.next_epoch()

    assert second.real_vals is first.real_vals
    assert second.folds == 0
    assert sorted(second.pred_vals.keys()) == sorted(first.pred_vals.keys())
    assert all(preds.shape == (0, 1) for preds in second.pred_vals.values())


def test_epoch_perf_data_retains_only_current_and_best():
    """Indexing forward advances the epoch; only the marked best epoch survives behind it."""
    first = perf_data.create_perf_data(
        'regression', _indexed_model_dataset('train_valid_test'), 'train')
    epoch_perf_data = perf_data.EpochPerfData(first)

    assert epoch_perf_data[0] is first
    epoch_perf_data.mark_best(0)

    second = epoch_perf_data[1]
    assert second is not first
    assert epoch_perf_data[1] is second
    # epoch 0 was the best epoch, so it is still reachable
    assert epoch_perf_data[0] is first

    # epoch 1 was never the best, so moving to epoch 2 discards it
    third = epoch_perf_data[2]
    assert third is not second
    with pytest.raises(IndexError):
        epoch_perf_data[1]
    assert epoch_perf_data[0] is first


def test_epoch_perf_data_best_marked_before_subset_advances():
    """A subset accumulated after _set_best_epoch must advance to the new-best epoch,
    not return the previous best's snapshot.

    EpochManager.update_epoch marks the best epoch on all subsets during the valid
    update (EpochManager._set_best_epoch), before the test subset has advanced to that
    epoch (test is accumulated last). __getitem__ must not hand back the stale _best for
    an epoch that has not been trained yet; it must advance _current and accumulate into
    the fresh PerfData. Otherwise the new-best epoch's test predictions land in the
    previous best's object, and compute_perf_metrics returns perf_metrics[0] (the
    previous best's score) instead of the new best's.
    """
    first = perf_data.create_perf_data(
        'regression', _indexed_model_dataset('train_valid_test'), 'test')
    epd = perf_data.EpochPerfData(first)

    # Epoch 0 is the first best. Accumulate, then mark best (best == current here, so
    # _best is still None and the advance below snapshots epoch 0 into _best).
    epd[0].accumulate_preds(np.array([[0.0], [1.0], [2.0], [3.0]]), first.ids)  # r2 = 1.0
    epd.mark_best(0)

    # Advance past epoch 0; the current==best branch snapshots epoch 0 into _best.
    epd[1].accumulate_preds(np.array([[1.0], [1.0], [1.0], [1.0]]), first.ids)

    # Epoch 2 becomes a new best. _set_best_epoch marks it on every subset during the
    # valid update, BEFORE the test subset advances to epoch 2.
    epd.mark_best(2)

    # The test subset now reaches epoch 2. This must advance to a fresh PerfData, not
    # return the stale epoch-0 snapshot held in _best.
    new_best = epd[2]
    new_best.accumulate_preds(np.array([[3.0], [3.0], [3.0], [3.0]]), first.ids)  # r2 < 0

    # Advance once more so the new best is snapshotted into _best.
    epd[3].accumulate_preds(np.array([[0.0], [1.0], [2.0], [3.0]]), first.ids)

    best = epd[2]
    assert best is new_best
    # The best snapshot holds only the epoch-2 score, not epoch 0's prepended before it.
    assert len(best.perf_metrics) == 1
    r2, _ = best.compute_perf_metrics()
    assert r2 < 0  # epoch 2's score, not epoch 0's 1.0


def test_epoch_manager_builds_one_perf_data_per_subset():
    """EpochManager must read the untransformed responses once per subset, not once per epoch."""
    model_dataset = _indexed_model_dataset('train_valid_test')
    calls = []
    responses = model_dataset.get_untransformed_responses
    model_dataset.get_untransformed_responses = lambda ids: (calls.append(ids), responses(ids))[1]

    wrapper = SimpleNamespace(
        params=SimpleNamespace(
            max_epochs=100,
            model_choice_score_type='r2',
            early_stopping_min_improvement=0.0,
            early_stopping_patience=10,
        ),
        log=logging.getLogger('test'),
    )
    perf_data.EpochManager(wrapper, prediction_type='regression', model_dataset=model_dataset)

    assert len(calls) == 3
    for subset in ['train', 'valid', 'test']:
        assert isinstance(getattr(wrapper, f'{subset}_perf_data'), perf_data.EpochPerfData)
