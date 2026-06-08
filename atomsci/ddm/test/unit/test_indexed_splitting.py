"""Unit tests for atomsci.ddm.pipeline.splitting.IndexedSplitting.

The tests build small in-memory NumpyDataset and attr_df fixtures and call
IndexedSplitting.split_dataset directly, so they do not depend on featurisation
or the wider ModelDataset pipeline. Factory-level wiring is exercised via
create_splitting.
"""
import argparse

import numpy as np
import pandas as pd
import pytest

from deepchem.data import NumpyDataset

from atomsci.ddm.pipeline import splitting as split


def _make_params(**overrides):
    """Build a minimal params Namespace for IndexedSplitting tests."""
    defaults = dict(
        production=False,
        split_strategy='indexed',
        splitter='scaffold',  # ignored under indexed, but base Splitting expects something
        split_valid_frac=0.1,
        split_test_frac=0.1,
        split_train_num=None,
        split_valid_num=None,
        split_test_num=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_dataset(n, n_tasks=1):
    """Build a tiny NumpyDataset and matching attr_df with unique ids."""
    X = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    y = np.arange(n * n_tasks, dtype=np.float32).reshape(n, n_tasks)
    w = np.ones((n, n_tasks), dtype=np.float32)
    ids = np.array([f"cmpd_{i}" for i in range(n)])
    dataset = NumpyDataset(X=X, y=y, w=w, ids=ids)
    attr_df = pd.DataFrame(
        {'rdkit_smiles': [f"C{'C' * (i % 5)}" for i in range(n)]},
        index=[f"cmpd_{i}" for i in range(n)],
    )
    return dataset, attr_df


# --------------------------------------------------------------------------- factory


def test_create_splitting_dispatches_to_indexed():
    params = _make_params()
    splitter = split.create_splitting(params)
    assert isinstance(splitter, split.IndexedSplitting)


def test_indexed_splitting_basic_attributes():
    params = _make_params()
    splitter = split.IndexedSplitting(params)
    assert splitter.split == 'indexed'
    assert splitter.num_folds == 1
    assert splitter.splitter is None
    assert splitter.needs_smiles() is False
    assert splitter.get_split_prefix() == 'indexed'
    assert splitter.get_split_prefix(parent='runs') == 'runs/indexed'


# --------------------------------------------------------------------------- fractions


def test_fraction_split_sizes():
    params = _make_params(split_valid_frac=0.1, split_test_frac=0.2)
    dataset, attr_df = _make_dataset(n=100)
    splitter = split.IndexedSplitting(params)
    (tv,), test, (tva,), test_attr = splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')
    train, valid = tv
    train_attr, valid_attr = tva
    assert len(train) == 70
    assert len(valid) == 10
    assert len(test) == 20
    assert len(train_attr) == 70
    assert len(valid_attr) == 10
    assert len(test_attr) == 20


def test_fraction_split_preserves_row_order():
    params = _make_params(split_valid_frac=0.1, split_test_frac=0.1)
    dataset, attr_df = _make_dataset(n=10)
    splitter = split.IndexedSplitting(params)
    (tv,), test, _, _ = splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')
    train, valid = tv
    assert list(train.ids) == [f"cmpd_{i}" for i in range(8)]
    assert list(valid.ids) == ['cmpd_8']
    assert list(test.ids) == ['cmpd_9']


def test_fraction_split_returns_matching_attr_rows():
    params = _make_params(split_valid_frac=0.2, split_test_frac=0.2)
    dataset, attr_df = _make_dataset(n=20)
    splitter = split.IndexedSplitting(params)
    (tv,), test, (tva,), test_attr = splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')
    train, valid = tv
    train_attr, valid_attr = tva
    assert list(train.ids) == list(train_attr.index)
    assert list(valid.ids) == list(valid_attr.index)
    assert list(test.ids) == list(test_attr.index)


def test_fraction_split_no_train_room_raises():
    params = _make_params(split_valid_frac=0.6, split_test_frac=0.5)
    dataset, attr_df = _make_dataset(n=10)
    splitter = split.IndexedSplitting(params)
    with pytest.raises(ValueError, match="no room for a training set"):
        splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')


# --------------------------------------------------------------------------- counts


def test_count_split_sizes():
    params = _make_params(split_train_num=70, split_valid_num=15, split_test_num=15)
    dataset, attr_df = _make_dataset(n=100)
    splitter = split.IndexedSplitting(params)
    (tv,), test, _, _ = splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')
    train, valid = tv
    assert len(train) == 70
    assert len(valid) == 15
    assert len(test) == 15


def test_count_split_takes_priority_over_fractions():
    """When all three counts are set, fractions are ignored."""
    params = _make_params(
        split_valid_frac=0.5, split_test_frac=0.4,  # would normally produce 10/50/40
        split_train_num=80, split_valid_num=15, split_test_num=5,
    )
    dataset, attr_df = _make_dataset(n=100)
    splitter = split.IndexedSplitting(params)
    (tv,), test, _, _ = splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')
    train, valid = tv
    assert len(train) == 80
    assert len(valid) == 15
    assert len(test) == 5


def test_count_split_zero_test_size():
    params = _make_params(split_train_num=80, split_valid_num=20, split_test_num=0)
    dataset, attr_df = _make_dataset(n=100)
    splitter = split.IndexedSplitting(params)
    (tv,), test, _, _ = splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')
    train, valid = tv
    assert len(train) == 80
    assert len(valid) == 20
    assert len(test) == 0


def test_count_sum_mismatch_raises():
    params = _make_params(split_train_num=50, split_valid_num=20, split_test_num=20)
    dataset, attr_df = _make_dataset(n=100)  # sums to 90, not 100
    splitter = split.IndexedSplitting(params)
    with pytest.raises(ValueError, match="sum to 90 but the dataset has 100 rows"):
        splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')


def test_negative_count_raises():
    params = _make_params(split_train_num=80, split_valid_num=-5, split_test_num=25)
    dataset, attr_df = _make_dataset(n=100)
    splitter = split.IndexedSplitting(params)
    with pytest.raises(ValueError, match="non-negative"):
        splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')


def test_partial_counts_raises():
    """At the splitter level, only counts set on 0 or 3 paths are accepted."""
    params = _make_params(split_train_num=70, split_valid_num=20, split_test_num=None)
    dataset, attr_df = _make_dataset(n=100)
    splitter = split.IndexedSplitting(params)
    with pytest.raises(ValueError, match="set together, or none of them"):
        splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')


# --------------------------------------------------------------------------- uniqueness


def test_duplicate_ids_raise():
    params = _make_params()
    dataset, attr_df = _make_dataset(n=10)
    # Stamp a duplicate id into the dataset
    ids = np.array(dataset.ids).copy()
    ids[5] = ids[0]
    dataset = NumpyDataset(X=dataset.X, y=dataset.y, w=dataset.w, ids=ids)
    splitter = split.IndexedSplitting(params)
    with pytest.raises(ValueError, match="unique compound ids"):
        splitter.split_dataset(dataset, attr_df, smiles_col='rdkit_smiles')
