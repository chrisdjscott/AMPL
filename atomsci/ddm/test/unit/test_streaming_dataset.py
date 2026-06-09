"""Unit tests for atomsci.ddm.pipeline.model_datasets.StreamingFileDataset and
_StreamingFeatureDataset.

The surface tests build tiny in-memory CSVs (so featurisation runs but stays
cheap) and exercise the DeepChem-Dataset-shaped methods directly. The
IndexedSplitting test composes the streaming dataset with the splitter.
Parser refusal tests drive parse_command_line with the same argument
combinations enumerated in the Phase 5 verification matrix.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pytest

from atomsci.ddm.pipeline import model_datasets as md
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.pipeline import splitting as split


# ----------------------------------------------------------------------- fixtures


_SMILES_POOL = ['C', 'CC', 'CCO', 'CCN', 'CCCC', 'CCCCO', 'c1ccccc1', 'CCCl']


def _write_tiny_csv(tmp_path, n=12):
    """Write a small CSV of `(compound_id, rdkit_smiles, y)` rows."""
    rows = [
        {
            'compound_id': f'cmpd_{i}',
            'rdkit_smiles': _SMILES_POOL[i % len(_SMILES_POOL)],
            'y': float(i),
        }
        for i in range(n)
    ]
    p = tmp_path / 'tiny.csv'
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def _streaming_params(csv_path, **overrides):
    """Minimal params Namespace for StreamingFileDataset.

    Carries the attributes touched by ModelDataset.__init__,
    FileDataset.__init__, get_featurized_data, and create_featurization.
    """
    defaults = dict(
        # ModelDataset / FileDataset init
        dataset_name=None,
        dataset_key=csv_path,
        output_dir=os.path.dirname(csv_path),
        previously_split=False,
        split_uuid=None,
        previously_featurized=True,
        # get_featurized_data
        max_dataset_rows=0,
        response_cols=['y'],
        prediction_type='regression',
        model_type='NN',
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        date_col=None,
        min_compound_number=1,
        # Featurization (ecfp)
        featurizer='ecfp',
        ecfp_size=256,
        ecfp_radius=2,
        # Streaming switch
        streaming=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_streaming_dataset(tmp_path, n=12, **overrides):
    csv = _write_tiny_csv(tmp_path, n=n)
    params = _streaming_params(csv, **overrides)
    dset = md.StreamingFileDataset(params, featurization=None)
    dset.get_featurized_data()
    return dset


# ---------------------------------------------------- _StreamingFeatureDataset


def test_len_matches_csv(tmp_path):
    dset = _make_streaming_dataset(tmp_path, n=12)
    assert len(dset.dataset) == 12


def test_y_w_ids_eager_shapes(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=10).dataset
    assert d.y.shape == (10, 1)
    assert d.w.shape == (10, 1)
    assert d.ids.shape == (10,)
    assert list(d.ids) == [f'cmpd_{i}' for i in range(10)]


def test_X_raises_not_implemented(tmp_path):
    d = _make_streaming_dataset(tmp_path).dataset
    with pytest.raises(NotImplementedError, match='does not materialise'):
        _ = d.X


def test_n_features_matches_ecfp_size(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=5, ecfp_size=512).dataset
    assert d.n_features == 512


def test_get_shape_uses_n_features(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=6).dataset
    x_shape, y_shape, w_shape, ids_shape = d.get_shape()
    assert x_shape == (6, d.n_features)
    assert y_shape == (6, 1)
    assert w_shape == (6, 1)
    assert ids_shape == (6,)


# ---------------------------------------------------- iterbatches


def test_iterbatches_deterministic_shapes_and_coverage(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=12).dataset
    batches = list(d.iterbatches(batch_size=4, epochs=1, deterministic=True))
    assert len(batches) == 3
    seen = []
    for X_b, y_b, w_b, ids_b in batches:
        assert X_b.shape == (4, d.n_features)
        assert y_b.shape == (4, 1)
        assert w_b.shape == (4, 1)
        assert ids_b.shape == (4,)
        seen.extend(list(ids_b))
    assert seen == [f'cmpd_{i}' for i in range(12)]


def test_iterbatches_pads_short_final_batch(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=12).dataset
    batches = list(d.iterbatches(
        batch_size=5, epochs=1, deterministic=True, pad_batches=True))
    assert len(batches) == 3
    last_X, _, last_w, _ = batches[-1]
    assert last_X.shape == (5, d.n_features)
    # The last two real rows (positions 10, 11) carry weight 1; padded rows are zero-weighted.
    assert np.allclose(last_w[2:], 0.0)


def test_iterbatches_nondeterministic_shuffles(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=12).dataset
    np.random.seed(0)
    ids_shuf_0 = list(next(d.iterbatches(batch_size=12, deterministic=False))[3])
    np.random.seed(1)
    ids_shuf_1 = list(next(d.iterbatches(batch_size=12, deterministic=False))[3])
    ids_det = list(next(d.iterbatches(batch_size=12, deterministic=True))[3])
    assert ids_shuf_0 != ids_det or ids_shuf_1 != ids_det
    assert ids_shuf_0 != ids_shuf_1


def test_iterbatches_epochs_runs_twice(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=4).dataset
    batches = list(d.iterbatches(batch_size=4, epochs=2, deterministic=True))
    assert len(batches) == 2


# ---------------------------------------------------- itersamples


def test_itersamples_yields_zero_X_and_real_y_w_id(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=4).dataset
    samples = list(d.itersamples())
    assert len(samples) == 4
    for i, (X_i, y_i, w_i, id_i) in enumerate(samples):
        assert X_i.shape == (d.n_features,)
        assert np.all(X_i == 0)
        assert np.array_equal(y_i, d.y[i])
        assert np.array_equal(w_i, d.w[i])
        assert id_i == d.ids[i]


# ---------------------------------------------------- select


def test_select_returns_sibling_with_reindexed_rows(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=12).dataset
    sub = d.select([1, 3, 5])
    assert isinstance(sub, md._StreamingFeatureDataset)
    assert list(sub.ids) == ['cmpd_1', 'cmpd_3', 'cmpd_5']
    assert np.array_equal(sub.y, d.y[[1, 3, 5]])
    assert np.array_equal(sub.w, d.w[[1, 3, 5]])
    assert sub.n_features == d.n_features
    # The original is unchanged.
    assert len(d) == 12


def test_select_iterbatches_runs(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=12).dataset
    sub = d.select([2, 4, 6, 8])
    X_b, _, _, ids_b = next(sub.iterbatches(batch_size=4, deterministic=True))
    assert X_b.shape == (4, d.n_features)
    assert list(ids_b) == ['cmpd_2', 'cmpd_4', 'cmpd_6', 'cmpd_8']


# ---------------------------------------------------- transform


class _AddConstantTransformer:
    """Adds 10.0 to y; leaves X, w, ids untouched (mirrors transform_array contract)."""

    def transform_array(self, X, y, w, ids):
        return X, y + 10.0, w, ids


def test_transform_returns_wrapper_with_eager_y_rewritten(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=6).dataset
    original_y = d.y.copy()
    wrapped = d.transform(_AddConstantTransformer())
    assert isinstance(wrapped, md._StreamingFeatureDataset)
    assert np.allclose(wrapped.y, original_y + 10.0)
    # Original dataset is untouched.
    assert np.allclose(d.y, original_y)


def test_transform_applies_per_batch_inside_iterbatches(tmp_path):
    d = _make_streaming_dataset(tmp_path, n=4).dataset
    wrapped = d.transform(_AddConstantTransformer())
    _, y_b, _, _ = next(wrapped.iterbatches(batch_size=4, deterministic=True))
    assert np.allclose(y_b, d.y + 10.0)


# ---------------------------------------------------- StreamingFileDataset wiring


def test_get_featurized_data_populates_model_dataset_state(tmp_path):
    dset = _make_streaming_dataset(tmp_path, n=10)
    assert isinstance(dset.dataset, md._StreamingFeatureDataset)
    assert dset.n_features == 256
    assert dset.vals.shape == (10, 1)
    assert set(dset.attr.index) == {f'cmpd_{i}' for i in range(10)}
    assert len(dset.untransformed_response_dict) == 10


def test_save_featurized_data_is_noop(tmp_path):
    dset = _make_streaming_dataset(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    dset.save_featurized_data(featurized_dset_df=None)
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after


def test_load_featurized_data_refuses(tmp_path):
    dset = _make_streaming_dataset(tmp_path)
    with pytest.raises(NotImplementedError, match='previously_featurized'):
        dset.load_featurized_data()


# ---------------------------------------------------- IndexedSplitting contract


def test_indexed_splitting_round_trip(tmp_path):
    dset = _make_streaming_dataset(tmp_path, n=20)
    sp_params = argparse.Namespace(
        production=False,
        split_strategy='indexed',
        splitter='scaffold',
        split_valid_frac=0.2,
        split_test_frac=0.2,
        split_train_num=None,
        split_valid_num=None,
        split_test_num=None,
    )
    splitter = split.IndexedSplitting(sp_params)
    (tv,), test, _, _ = splitter.split_dataset(
        dset.dataset, dset.attr, smiles_col='rdkit_smiles')
    train, valid = tv
    assert isinstance(train, md._StreamingFeatureDataset)
    assert isinstance(valid, md._StreamingFeatureDataset)
    assert isinstance(test, md._StreamingFeatureDataset)
    assert len(train) + len(valid) + len(test) == 20
    train_ids, valid_ids, test_ids = set(train.ids), set(valid.ids), set(test.ids)
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)


# ---------------------------------------------------- parse_command_line refusals


_BASE_ARGS = ['--dataset_key', '/tmp/streaming_test_dummy.csv', '--bucket', 'public']


def _parse(*extra):
    return parse.parse_command_line(_BASE_ARGS + list(extra))


def test_streaming_alone_parses():
    p = _parse('--streaming')
    assert p.streaming is True


def test_streaming_with_previously_split_raises():
    with pytest.raises(Exception, match='previously_split'):
        _parse('--streaming', '--previously_split')


def test_streaming_with_k_fold_cv_raises():
    with pytest.raises(Exception, match='k_fold_cv'):
        _parse('--streaming', '--split_strategy', 'k_fold_cv')


def test_streaming_with_datastore_raises():
    with pytest.raises(Exception, match='datastore'):
        _parse('--streaming', '--datastore')


def test_streaming_with_descriptor_featurizer_default_transformers_raises():
    with pytest.raises(Exception, match='transformers'):
        _parse('--streaming', '--featurizer', 'descriptors')


def test_streaming_with_descriptor_featurizer_transformers_off_parses():
    p = _parse('--streaming', '--featurizer', 'descriptors', '--transformers')
    assert p.streaming is True
    assert p.transformers is False


def test_non_streaming_with_each_forbidden_combination_unaffected():
    # Without --streaming none of the four refusals fire.
    assert _parse('--previously_split').previously_split is True
    assert _parse('--split_strategy', 'k_fold_cv').split_strategy == 'k_fold_cv'
    assert _parse('--datastore').datastore is True
    assert _parse('--featurizer', 'descriptors').featurizer == 'descriptors'


# ---------------------------------------------------- delaney end-to-end round-trip


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
_DELANEY_CSV = os.path.join(_REPO_ROOT, 'delaney-processed.csv')


@pytest.mark.skipif(
    not os.path.exists(_DELANEY_CSV),
    reason='delaney-processed.csv not present in repo root',
)
def test_create_and_load_model_dataset_round_trip_on_delaney():
    params = argparse.Namespace(
        dataset_name=None,
        dataset_key=_DELANEY_CSV,
        output_dir=os.path.dirname(_DELANEY_CSV),
        previously_split=False,
        split_uuid=None,
        previously_featurized=True,
        max_dataset_rows=0,
        response_cols=['measured log solubility in mols per litre'],
        prediction_type='regression',
        model_type='NN',
        id_col='Compound ID',
        smiles_col='smiles',
        date_col=None,
        min_compound_number=1,
        featurizer='ecfp',
        ecfp_size=256,
        ecfp_radius=2,
        streaming=True,
        datastore=False,
    )
    model_dataset = md.create_and_load_model_dataset(params)
    assert isinstance(model_dataset, md.StreamingFileDataset)
    assert isinstance(model_dataset.dataset, md._StreamingFeatureDataset)
    expected_rows = sum(1 for _ in open(_DELANEY_CSV)) - 1
    assert len(model_dataset.dataset) == expected_rows
    X_b, y_b, w_b, ids_b = next(
        model_dataset.dataset.iterbatches(batch_size=8, deterministic=True))
    assert X_b.shape == (8, 256)
    assert y_b.shape == (8, 1)
    assert w_b.shape == (8, 1)
    assert ids_b.shape == (8,)
