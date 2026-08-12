"""Integration tests for the streaming feature cache (CachingStreamingFileDataset).

These build a real featurised dataset on a tiny in-memory CSV (so the cache is
exercised against actual featurizers, but cheaply) and assert the Phase 6
success criteria:

* Cold run produces features identical to the canonical featurizer output (the
  same ``featurize_smiles`` call the eager and non-cached streaming paths use),
  for BOTH a graph featurizer (graphconv) and a fixed-width one (ecfp).
* A warm cache serves epoch 2 with zero featurise calls (spy on
  ``featurize_smiles``), and the warm features match the cold ones.
* An invalid molecule flows through the cache: dropped from the dataset, and
  still zero featurise calls on the warm run.
* A featurizer-config change lands in a separate cache dir and re-featurises.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pytest

from atomsci.ddm.pipeline import featurization as feat
from atomsci.ddm.pipeline import model_datasets as md


# ----------------------------------------------------------------------- fixtures


_VALID_SMILES = ['CCO', 'CCN', 'CCCC', 'CCCCO', 'c1ccccc1', 'CCCl', 'CCCCN', 'OCCO']


def _write_csv(tmp_path, smiles):
    rows = [
        {'compound_id': f'cmpd_{i}', 'rdkit_smiles': s, 'y': float(i)}
        for i, s in enumerate(smiles)
    ]
    p = tmp_path / 'mols.csv'
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def _params(csv_path, cache_dir, featurizer='ecfp', **overrides):
    defaults = dict(
        dataset_name=None,
        dataset_key=csv_path,
        output_dir=os.path.dirname(csv_path),
        previously_split=False,
        split_uuid=None,
        previously_featurized=True,
        max_dataset_rows=0,
        response_cols=['y'],
        prediction_type='regression',
        model_type='NN',
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        date_col=None,
        min_compound_number=1,
        featurizer=featurizer,
        ecfp_size=256,
        ecfp_radius=2,
        streaming=True,
        datastore=False,
        feature_cache_dir=str(cache_dir),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _build(tmp_path, cache_dir, smiles, featurizer='ecfp', **overrides):
    csv = _write_csv(tmp_path, smiles)
    params = _params(csv, cache_dir, featurizer=featurizer, **overrides)
    dset = md.CachingStreamingFileDataset(params, featurization=None)
    dset.get_featurized_data()
    return dset


def _full_batch_X(dset):
    """Deterministic, full-length featurised batch (rows in source order)."""
    return next(dset.dataset.iterbatches(
        batch_size=len(dset.dataset), deterministic=True))[0]


def _reference_features(dset, smiles):
    """Canonical featurizer output over ``smiles`` (valid-only, in order).

    This is exactly what ``featurize_smiles`` returns for the eager FileDataset
    and the non-cached StreamingFileDataset, so equality here is parity with
    both.
    """
    df = pd.DataFrame({'rdkit_smiles': smiles})
    features, _ = feat.featurize_smiles(
        df, featurizer=dset.featurization.featurizer_obj, smiles_col='rdkit_smiles')
    return features


def _features_equal(a, b, featurizer):
    if featurizer == 'ecfp':
        return np.array_equal(a, b)
    # Graph featurizers yield a 1-D object array of graph objects; compare the
    # per-molecule numeric atom/node feature matrices.
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        ax = x.get_atom_features() if hasattr(x, 'get_atom_features') else x.node_features
        ay = y.get_atom_features() if hasattr(y, 'get_atom_features') else y.node_features
        if not np.array_equal(ax, ay):
            return False
    return True


# ------------------------------------------------------------ cold-vs-reference


@pytest.mark.parametrize('featurizer', ['ecfp', 'graphconv'])
def test_cold_cache_matches_reference_features(tmp_path, featurizer):
    cache_dir = tmp_path / 'cache'
    dset = _build(tmp_path, cache_dir, _VALID_SMILES, featurizer=featurizer)

    X_cold = _full_batch_X(dset)
    reference = _reference_features(dset, _VALID_SMILES)
    assert _features_equal(X_cold, reference, featurizer)


# ----------------------------------------------------------- warm zero-featurise


@pytest.mark.parametrize('featurizer', ['ecfp', 'graphconv'])
def test_warm_run_does_zero_featurise_calls(tmp_path, featurizer, monkeypatch):
    cache_dir = tmp_path / 'cache'
    # Cold build warms the cache.
    cold = _build(tmp_path, cache_dir, _VALID_SMILES, featurizer=featurizer)
    X_cold = _full_batch_X(cold)

    # Spy on the featurizer; a fully warm cache must not call it again.
    calls = {'n': 0}
    original = feat.featurize_smiles

    def spy(*args, **kwargs):
        calls['n'] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(feat, 'featurize_smiles', spy)

    warm = _build(tmp_path, cache_dir, _VALID_SMILES, featurizer=featurizer)
    assert calls['n'] == 0, "warm _scan_validity re-featurised at init"

    # A full epoch through the cached per-batch seam must also stay at zero.
    for _ in warm.dataset.iterbatches(batch_size=3, deterministic=False):
        pass
    assert calls['n'] == 0, "warm iterbatches re-featurised a batch"

    X_warm = _full_batch_X(warm)
    assert _features_equal(X_warm, X_cold, featurizer)


# ------------------------------------------------------- invalid row via cache


def test_invalid_row_flows_through_cache_and_warm_skips_it(tmp_path, monkeypatch):
    # MolGraphConvFeaturizer rejects the single-atom methane 'C'; it must be
    # dropped from the dataset and round-trip through the cache as invalid.
    smiles = ['C', 'CCO', 'c1ccccc1', 'CCN']
    n_valid = 3
    cache_dir = tmp_path / 'cache'

    cold = _build(tmp_path, cache_dir, smiles, featurizer='MolGraphConvFeaturizer')
    assert len(cold.dataset) == n_valid
    assert 'cmpd_0' not in set(cold.dataset.ids)  # the invalid methane row

    calls = {'n': 0}
    original = feat.featurize_smiles

    def spy(*args, **kwargs):
        calls['n'] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(feat, 'featurize_smiles', spy)

    warm = _build(tmp_path, cache_dir, smiles, featurizer='MolGraphConvFeaturizer')
    assert calls['n'] == 0, "cached invalid row triggered a re-featurise"
    assert len(warm.dataset) == n_valid
    assert _features_equal(_full_batch_X(warm), _full_batch_X(cold),
                           'MolGraphConvFeaturizer')


# --------------------------------------------------------- config invalidation


def test_config_change_uses_fresh_cache_and_refeaturises(tmp_path, monkeypatch):
    cache_dir = tmp_path / 'cache'
    first = _build(tmp_path, cache_dir, _VALID_SMILES, featurizer='ecfp', ecfp_size=256)
    assert first.n_features == 256

    calls = {'n': 0}
    original = feat.featurize_smiles

    def spy(*args, **kwargs):
        calls['n'] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(feat, 'featurize_smiles', spy)

    # Same cache root, different featurizer config -> different config-hash dir.
    second = _build(tmp_path, cache_dir, _VALID_SMILES, featurizer='ecfp', ecfp_size=512)
    assert second.n_features == 512
    assert second._feature_cache.cache_dir != first._feature_cache.cache_dir
    assert calls['n'] > 0, "config change did not re-featurise into a fresh dir"
