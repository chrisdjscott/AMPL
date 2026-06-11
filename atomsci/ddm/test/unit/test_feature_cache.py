"""Unit tests for atomsci.ddm.pipeline.feature_cache.FeatureCache.

Standalone: these use plain numpy arrays and small Python objects as feature
units, so the cache is exercised without invoking any real featurizer.
"""
import os

import numpy as np
import pytest

from atomsci.ddm.pipeline import feature_cache as fc


# ----------------------------------------------------------------------- helpers


_FEAT_TYPE = 'ecfp'
_CONFIG = {'ecfp_specific': {'ecfp_radius': 2, 'ecfp_size': 1024}}


def _open(tmp_path, **overrides):
    kw = dict(feat_type=_FEAT_TYPE, config_metadata=_CONFIG, n_features=1024)
    kw.update(overrides)
    return fc.FeatureCache(str(tmp_path), **kw)


def _vec(seed, width=4):
    return np.arange(seed, seed + width, dtype=np.float32)


# --------------------------------------------------------------------- config hash


def test_config_hash_stable_and_order_independent():
    a = fc.compute_config_hash('ecfp', {'x': 1, 'y': 2})
    b = fc.compute_config_hash('ecfp', {'y': 2, 'x': 1})
    assert a == b
    assert a != fc.compute_config_hash('ecfp', {'x': 1, 'y': 3})
    assert a != fc.compute_config_hash('graphconv', {'x': 1, 'y': 2})


# --------------------------------------------------------------------- get / put


def test_put_then_get_round_trip(tmp_path):
    cache = _open(tmp_path)
    smiles = ['CCO', 'CCN', 'c1ccccc1']
    units = [_vec(0), _vec(10), _vec(20)]
    cache.put(smiles, units, [True, True, True])

    features, is_valid, miss = cache.get(smiles)
    assert miss.tolist() == [False, False, False]
    assert is_valid.tolist() == [True, True, True]
    for got, want in zip(features, units):
        np.testing.assert_array_equal(got, want)


def test_get_miss_for_unknown_smiles(tmp_path):
    cache = _open(tmp_path)
    cache.put(['CCO'], [_vec(0)], [True])

    features, is_valid, miss = cache.get(['CCO', 'UNKNOWN'])
    assert miss.tolist() == [False, True]
    assert is_valid.tolist() == [True, False]
    assert features[1] is None
    np.testing.assert_array_equal(features[0], _vec(0))


def test_partial_hit_and_subsequent_put(tmp_path):
    cache = _open(tmp_path)
    cache.put(['CCO'], [_vec(0)], [True])

    _, _, miss = cache.get(['CCO', 'CCN'])
    assert miss.tolist() == [False, True]

    cache.put(['CCN'], [_vec(10)], [True])
    features, _, miss = cache.get(['CCO', 'CCN'])
    assert miss.tolist() == [False, False]
    np.testing.assert_array_equal(features[1], _vec(10))


def test_put_is_idempotent_for_existing_keys(tmp_path):
    cache = _open(tmp_path)
    cache.put(['CCO'], [_vec(0)], [True])
    size_after_first = os.path.getsize(cache.blob_path)

    # Re-putting the same SMILES with a different value must be a no-op.
    cache.put(['CCO'], [_vec(99)], [True])
    assert os.path.getsize(cache.blob_path) == size_after_first

    features, _, _ = cache.get(['CCO'])
    np.testing.assert_array_equal(features[0], _vec(0))


# ----------------------------------------------------------------- invalid rows


def test_invalid_molecule_round_trip(tmp_path):
    cache = _open(tmp_path)
    cache.put(['CCO', 'BADSMILES'], [_vec(0), None], [True, False])

    features, is_valid, miss = cache.get(['CCO', 'BADSMILES'])
    assert miss.tolist() == [False, False]
    assert is_valid.tolist() == [True, False]
    assert features[1] is None
    np.testing.assert_array_equal(features[0], _vec(0))


# -------------------------------------------------------------- index persistence


def test_index_persists_across_reopen(tmp_path):
    cache = _open(tmp_path)
    cache.put(['CCO', 'CCN'], [_vec(0), _vec(10)], [True, True])
    cache.close()

    reopened = _open(tmp_path)
    features, _, miss = reopened.get(['CCO', 'CCN'])
    assert miss.tolist() == [False, False]
    np.testing.assert_array_equal(features[0], _vec(0))
    np.testing.assert_array_equal(features[1], _vec(10))


def test_index_rebuilds_from_blob_when_index_missing(tmp_path):
    cache = _open(tmp_path)
    cache.put(['CCO', 'BAD', 'CCN'], [_vec(0), None, _vec(10)],
              [True, False, True])
    cache.close()

    os.remove(cache.index_path)
    assert not os.path.exists(cache.index_path)

    reopened = _open(tmp_path)
    features, is_valid, miss = reopened.get(['CCO', 'BAD', 'CCN'])
    assert miss.tolist() == [False, False, False]
    assert is_valid.tolist() == [True, False, True]
    np.testing.assert_array_equal(features[0], _vec(0))
    np.testing.assert_array_equal(features[2], _vec(10))


# ----------------------------------------------------------------- invalidation


def test_config_change_uses_separate_dir(tmp_path):
    cache = _open(tmp_path)
    cache.put(['CCO'], [_vec(0)], [True])

    other_config = {'ecfp_specific': {'ecfp_radius': 3, 'ecfp_size': 1024}}
    other = _open(tmp_path, config_metadata=other_config)
    assert other.cache_dir != cache.cache_dir

    _, _, miss = other.get(['CCO'])
    assert miss.tolist() == [True]


# ------------------------------------------------------------------- read-only


def test_open_readonly_reads_warm_cache(tmp_path):
    writer = _open(tmp_path)
    writer.put(['CCO', 'CCN'], [_vec(0), _vec(10)], [True, True])
    writer.close()

    reader = fc.FeatureCache.open_readonly(str(tmp_path), _FEAT_TYPE, _CONFIG)
    features, _, miss = reader.get(['CCO', 'CCN'])
    assert miss.tolist() == [False, False]
    np.testing.assert_array_equal(features[0], _vec(0))


def test_readonly_put_raises(tmp_path):
    writer = _open(tmp_path)
    writer.put(['CCO'], [_vec(0)], [True])
    writer.close()

    reader = fc.FeatureCache.open_readonly(str(tmp_path), _FEAT_TYPE, _CONFIG)
    with pytest.raises(RuntimeError):
        reader.put(['CCN'], [_vec(10)], [True])


def test_readonly_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fc.FeatureCache.open_readonly(str(tmp_path), _FEAT_TYPE, _CONFIG)


# --------------------------------------------------------------- meta validation


def test_meta_mismatch_raises(tmp_path):
    writer = _open(tmp_path)
    writer.put(['CCO'], [_vec(0)], [True])
    cache_dir = writer.cache_dir
    writer.close()

    # Corrupt meta.json's feat_type while leaving the hash-named dir intact.
    import json
    meta_path = os.path.join(cache_dir, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    meta['feat_type'] = 'graphconv'
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    with pytest.raises(ValueError):
        _open(tmp_path)


# --------------------------------------------------- shared blob, second handle


def test_second_handle_sees_writes_via_pread(tmp_path):
    writer = _open(tmp_path)
    writer.put(['CCO'], [_vec(0)], [True])

    reader = fc.FeatureCache.open_readonly(str(tmp_path), _FEAT_TYPE, _CONFIG)
    features, _, miss = reader.get(['CCO'])
    assert miss.tolist() == [False]
    np.testing.assert_array_equal(features[0], _vec(0))
