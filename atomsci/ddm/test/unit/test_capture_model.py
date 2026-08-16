"""Tests for the ``reuse_fit_train_preds`` capture factory
(``atomsci.ddm.pipeline._capture_model``) and its wiring through the train loops.

Three layers:

1. Factory dispatch (``make_capturing``): flag-off is a passthrough, flag-on
   returns a capturing subclass, unsupported classes raise.
2. Capture correctness at the model layer: with ``learning_rate=0`` and
   ``dropout=0`` (and ``batch_normalize=False`` for GraphConv, whose
   BatchNormalization differs train vs eval), the captured training-mode
   predictions equal ``model.predict`` exactly, padding rows are trimmed, and
   ids are aligned to the dataset. This isolates the plumbing (output
   selection, ``undo_transforms``, padding trim, id reorder) from weight
   evolution: at ``lr=0`` weights are frozen, so per-batch pre-step captured
   outputs equal the post-epoch predict outputs. A ``NormalizationTransformer``
   on y exercises ``undo_transforms``.
3. End-to-end through a streaming ``ModelPipeline``: at ``lr=0, dropout=0`` the
   flag is a no-op (identical train/valid/test epoch perf), and at ``lr>0,
   dropout>0`` the train column differs (train-mode vs eval-mode) while
   valid/test stay identical (the fit forward pass is unchanged by capture, so
   weights evolve identically and the inference predict pass agrees).

Run under ``.venv-cpu`` (deepchem is not in the base system Python).
"""
import os

import numpy as np
import pandas as pd
import pytest

import deepchem as dc
from deepchem.models import MultitaskRegressor

import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as model_pipeline
from atomsci.ddm.pipeline._capture_model import make_capturing


# --------------------------------------------------------------- tiny CSV helper

_SMILES_POOL = [
    'C', 'CC', 'CCO', 'CCN', 'CCCC', 'CCCCO', 'c1ccccc1', 'CCCl',
    'CCBr', 'CCC', 'CCCO', 'CCCN', 'CCCCN', 'CCCCCl', 'CCCCO', 'CC(O)C',
]


def _write_tiny_csv(path, n=40):
    rows = [
        {
            'compound_id': f'cmpd_{i}',
            'rdkit_smiles': _SMILES_POOL[i % len(_SMILES_POOL)],
            'y': float(i),
        }
        for i in range(n)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# --------------------------------------------------------------- factory dispatch


def test_make_capturing_flag_off_returns_class_unchanged():
    assert make_capturing(MultitaskRegressor, False) is MultitaskRegressor
    assert make_capturing(dc.models.GraphConvModel, False) is dc.models.GraphConvModel


def test_make_capturing_unsupported_class_raises():
    class NotADeepChemModel:
        pass

    with pytest.raises(TypeError, match='KerasModel or TorchModel'):
        make_capturing(NotADeepChemModel, True)


# The flag is argparse ``store_true``. ``dict_to_list`` must emit the bare flag
# (``--reuse_fit_train_preds``), not ``--reuse_fit_train_preds True`` (which
# argparse rejects with "unrecognized arguments: True"). This requires the flag
# to be listed in ``default_false``. Guards the config-file/dict path, the
# production way users set AMPL flags.
def test_reuse_fit_train_preds_parses_from_config_dict(tmp_path):
    csv = _write_tiny_csv(tmp_path / 'tiny.csv', n=10)

    on = parse.wrapper(_streaming_nn_params(
        csv, str(tmp_path), reuse_fit_train_preds='True'))
    assert on.reuse_fit_train_preds is True
    assert on.streaming is True

    off = parse.wrapper(_streaming_nn_params(
        csv, str(tmp_path), reuse_fit_train_preds='False'))
    assert off.reuse_fit_train_preds is False

    # Without streaming, postprocess resets the flag to False (the win only
    # exists under streaming) and warns.
    no_stream = parse.wrapper(_streaming_nn_params(
        csv, str(tmp_path), reuse_fit_train_preds='True', streaming='False'))
    assert no_stream.reuse_fit_train_preds is False


# ----------------------------------------------- capture correctness: torch backend


def test_torch_capture_matches_predict_lr0_dropout0(tmp_path):
    """Torch MultitaskRegressor: captured train preds == model.predict at lr=0,
    dropout=0, with a y NormalizationTransformer and a non-divisible batch size
    (exercises padding trim and undo_transforms)."""
    n_tasks, n_features, n, bs = 1, 16, 23, 8
    rng = np.random.RandomState(0)
    X = rng.randn(n, n_features).astype(np.float32)
    y = rng.randn(n, n_tasks).astype(np.float32)
    ids = np.array([f'mol{i}' for i in range(n)])
    dset = dc.data.NumpyDataset(X, y, ids=ids, w=np.ones((n, n_tasks)))

    transformer = dc.trans.NormalizationTransformer(
        transform_y=True, dataset=dset)
    dset_t = transformer.transform(dset)
    transformers = [transformer]

    CapturingCls = make_capturing(MultitaskRegressor, True)
    model = CapturingCls(
        n_tasks, n_features, layer_sizes=[32], dropouts=[0.0],
        learning_rate=0.0, batch_size=bs,
        model_dir=str(tmp_path / 'torch_cap'))
    model.fit(dset_t, nb_epoch=1)

    cap_preds, cap_ids = model.pop_captured_train_preds(transformers)
    pred = model.predict(dset_t, transformers)

    assert np.array_equal(cap_ids, dset.ids)
    assert cap_preds.shape == pred.shape
    assert np.allclose(cap_preds, pred, atol=1e-6)


# ----------------------------------------------- capture correctness: keras backend


def test_keras_capture_matches_predict_lr0_dropout0(tmp_path):
    """Keras GraphConvModel: captured train preds == model.predict at lr=0,
    dropout=0, batch_normalize=False (BatchNorm differs train vs eval, so it
    must be off to isolate the capture plumbing). ConvMol features and a
    non-divisible batch size exercise GraphConv's graph batching + padding."""
    n_tasks, n, bs = 1, 11, 4
    smiles = _SMILES_POOL[:n]
    feats = dc.feat.ConvMolFeaturizer().featurize(smiles)
    rng = np.random.RandomState(0)
    y = rng.randn(n, n_tasks).astype(np.float32)
    ids = np.array([f'mol{i}' for i in range(n)])
    dset = dc.data.NumpyDataset(feats, y, ids=ids, w=np.ones((n, n_tasks)))

    transformer = dc.trans.NormalizationTransformer(
        transform_y=True, dataset=dset)
    dset_t = transformer.transform(dset)
    transformers = [transformer]

    CapturingCls = make_capturing(dc.models.GraphConvModel, True)
    model = CapturingCls(
        n_tasks, batch_size=bs, learning_rate=0.0, dropout=0.0,
        batch_normalize=False, mode='regression',
        model_dir=str(tmp_path / 'keras_cap'))
    model.fit(dset_t, nb_epoch=1)

    cap_preds, cap_ids = model.pop_captured_train_preds(transformers)
    pred = model.predict(dset_t, transformers)

    assert np.array_equal(cap_ids, dset.ids)
    assert cap_preds.shape == pred.shape
    assert np.allclose(cap_preds, pred, atol=1e-5)


# ----------------------------------------------- end-to-end through ModelPipeline


def _streaming_nn_params(csv_path, result_dir, **overrides):
    params = dict(
        verbose='False',
        datastore='False',
        save_results='False',
        model_type='NN',
        featurizer='ecfp',
        ecfp_size=256,
        ecfp_radius=2,
        split_strategy='indexed',
        splitter='random',
        split_test_frac='0.2',
        split_valid_frac='0.2',
        transformers='True',
        id_col='compound_id',
        dataset_key=csv_path,
        response_cols='y',
        smiles_col='rdkit_smiles',
        max_epochs='2',
        early_stopping_patience='5',
        layer_sizes='32',
        batch_size='8',
        prediction_type='regression',
        result_dir=result_dir,
        streaming='True',
        uncertainty='False',
        seed='42',
    )
    params.update({k: str(v) for k, v in overrides.items()})
    return params


def _train_streaming_nn(csv_path, result_dir, reuse_flag, **overrides):
    params = _streaming_nn_params(
        csv_path, result_dir, reuse_fit_train_preds=reuse_flag, **overrides)
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()
    return mp


def test_reuse_fit_train_preds_flag_off_equals_on_dropout0(tmp_path):
    """At lr=0, dropout=0 the capture is a no-op: flag-on and flag-off produce
    identical train/valid/test epoch perf. Proves the full plumbing
    (config flag -> capturing recreate_model -> fit capture ->
    pop_captured_train_preds -> update_epoch) end to end."""
    csv = _write_tiny_csv(tmp_path / 'tiny.csv', n=40)
    common = dict(learning_rate='0.0', dropouts='0.0')
    off = _train_streaming_nn(csv, str(tmp_path / 'off'), 'False', **common)
    on = _train_streaming_nn(csv, str(tmp_path / 'on'), 'True', **common)

    w_off, w_on = off.model_wrapper, on.model_wrapper
    for subset in ['train', 'valid', 'test']:
        a = getattr(w_off, f'{subset}_epoch_perfs')
        b = getattr(w_on, f'{subset}_epoch_perfs')
        assert np.allclose(a, b, equal_nan=True), (
            f'{subset} epoch perf differs between flag-off and flag-on at '
            f'lr=0/dropout=0: {a} vs {b}')


def test_reuse_fit_train_preds_dropout_train_differs_valid_test_identical(tmp_path):
    """At lr>0, dropout>0 the documented semantics: the train column differs
    (flag-on uses training-mode fit outputs with dropout on; flag-off uses the
    inference predict pass with dropout off), while valid/test stay identical
    (both use the inference predict pass on weights that evolved identically,
    because capture does not alter the fit forward pass)."""
    csv = _write_tiny_csv(tmp_path / 'tiny.csv', n=40)
    common = dict(learning_rate='0.01', dropouts='0.4')
    off = _train_streaming_nn(csv, str(tmp_path / 'off'), 'False', **common)
    on = _train_streaming_nn(csv, str(tmp_path / 'on'), 'True', **common)

    w_off, w_on = off.model_wrapper, on.model_wrapper
    # valid/test: inference predict on identical weights -> identical.
    assert np.allclose(
        w_off.valid_epoch_perfs, w_on.valid_epoch_perfs, equal_nan=True), (
        f'valid perf differs: {w_off.valid_epoch_perfs} vs '
        f'{w_on.valid_epoch_perfs}')
    assert np.allclose(
        w_off.test_epoch_perfs, w_on.test_epoch_perfs, equal_nan=True), (
        f'test perf differs: {w_off.test_epoch_perfs} vs '
        f'{w_on.test_epoch_perfs}')
    # train: eval predict (dropout off) vs captured train-mode (dropout on).
    assert not np.allclose(
        w_off.train_epoch_perfs, w_on.train_epoch_perfs, equal_nan=True), (
        f'train perf unexpectedly identical at dropout=0.4: '
        f'{w_off.train_epoch_perfs} vs {w_on.train_epoch_perfs}')