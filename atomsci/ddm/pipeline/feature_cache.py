"""Disk-backed feature cache for streaming datasets.

Stores one pickled feature object per molecule so that graph featurizers
(ConvMol / MolGraphConv) and fixed-width numeric featurizers (ECFP,
descriptors) share a single code path. Used by the streaming dataset to
avoid re-featurising every batch on epochs 2..N.

Backend is a custom append-only blob plus an in-RAM ``{smiles: offset}``
index. Reads are plain :func:`os.pread` on a stable file, the most robust
pattern when the cache lives on a shared cluster filesystem (NFS/Lustre)
where LMDB's mmap+byte-range locking and SQLite-over-NFS are unsafe.
Concurrent readers of a warm blob are safe; writes are serialised by an
:func:`fcntl.flock` advisory lock under a single-warming-writer model.

Layout::

    <cache_root>/<config_hash>/
      blob.dat   append-only records (see _RECORD format below)
      index.pkl  {smiles: (payload_offset, payload_len, is_valid)}
      meta.json  {format_version, feat_type, config, n_features}
      .lock      fcntl.flock target for append serialisation

Each blob record is::

    [u32 smiles_len][smiles utf8][u8 is_valid][u64 payload_len][pickle bytes]

Invalid molecules store ``is_valid=0`` and ``payload_len=0``. The embedded
SMILES header makes ``index.pkl`` a rebuildable accelerator: a stale or
missing index is reconstructed by scanning the blob.
"""

import hashlib
import json
import logging
import os
import pickle
import struct

import numpy as np

try:
    import fcntl
except ImportError:
    fcntl = None

log = logging.getLogger('ATOM')

FORMAT_VERSION = 1

# u32 smiles_len  ... then u8 is_valid + u64 payload_len after the smiles bytes.
_LEN_HEADER = struct.Struct('<I')
_VALID_HEADER = struct.Struct('<BQ')


# ****************************************************************************************
def compute_config_hash(feat_type, config_metadata):
    """Return a stable short hash identifying a featurizer configuration.

    The hash names the cache subdirectory, so any change to ``feat_type`` or
    the featurizer-specific metadata (ecfp_radius/size, whitelist featurizer
    params) maps to a fresh directory and thus automatic invalidation.

    Args:
        feat_type (str): the featurizer type (e.g. ``'ecfp'``, ``'graphconv'``).
        config_metadata (dict): featurizer-specific metadata, as returned by
            ``DynamicFeaturization.get_feature_specific_metadata``.

    Returns:
        str: a 16-character hex digest.
    """
    payload = json.dumps(
        {'feat_type': feat_type, 'config': config_metadata},
        sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


# ****************************************************************************************
class FeatureCache:
    """Append-only per-molecule feature store keyed by raw SMILES string.

    Open a writable cache with the constructor (creates the directory on
    first use); open an existing warm cache read-only with
    :meth:`open_readonly`. A read-only handle takes no lock, never writes,
    and serves :meth:`get` via :func:`os.pread`.
    """

    def __init__(self, cache_root, feat_type, config_metadata,
                 n_features=None, readonly=False):
        """Open (and create, if writable) the cache for one featurizer config.

        Args:
            cache_root (str): parent directory holding per-config subdirs.
            feat_type (str): featurizer type, stored and validated in meta.json.
            config_metadata (dict): featurizer-specific metadata; part of the
                config hash and validated against meta.json on open.
            n_features (int): feature width, recorded in meta.json for info.
            readonly (bool): if True, the cache dir must already exist and the
                handle refuses :meth:`put`.
        """
        self.feat_type = feat_type
        self.config_metadata = config_metadata
        self.n_features = n_features
        self.readonly = readonly
        self.config_hash = compute_config_hash(feat_type, config_metadata)
        self.cache_dir = os.path.join(cache_root, self.config_hash)
        self.blob_path = os.path.join(self.cache_dir, 'blob.dat')
        self.index_path = os.path.join(self.cache_dir, 'index.pkl')
        self.meta_path = os.path.join(self.cache_dir, 'meta.json')
        self.lock_path = os.path.join(self.cache_dir, '.lock')
        self._index = {}
        self._read_fd = None
        self._open()

    # ----------------------------------------------------------------- factory
    @classmethod
    def open_readonly(cls, cache_root, feat_type, config_metadata):
        """Open an existing warm cache read-only (no lock, pread only)."""
        return cls(cache_root, feat_type, config_metadata, readonly=True)

    # -------------------------------------------------------------------- open
    def _open(self):
        if self.readonly:
            if not os.path.isdir(self.cache_dir):
                raise FileNotFoundError(
                    "Read-only feature cache directory does not exist: %s"
                    % self.cache_dir)
            self._validate_meta()
            self._load_index()
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        if os.path.exists(self.meta_path):
            self._validate_meta()
        else:
            self._write_meta()
        self._load_index()

    def _open_read_fd(self):
        if self._read_fd is None and os.path.exists(self.blob_path):
            self._read_fd = os.open(self.blob_path, os.O_RDONLY)

    # -------------------------------------------------------------------- meta
    def _meta_dict(self):
        return {
            'format_version': FORMAT_VERSION,
            'feat_type': self.feat_type,
            'config': self.config_metadata,
            'n_features': self.n_features,
        }

    def _write_meta(self):
        with open(self.meta_path, 'w') as f:
            json.dump(self._meta_dict(), f, sort_keys=True, default=str)

    def _validate_meta(self):
        with open(self.meta_path) as f:
            meta = json.load(f)
        if meta.get('format_version') != FORMAT_VERSION:
            raise ValueError(
                "Feature cache format version mismatch in %s: found %r, "
                "expected %d." % (self.meta_path, meta.get('format_version'),
                                  FORMAT_VERSION))
        if meta.get('feat_type') != self.feat_type:
            raise ValueError(
                "Feature cache feat_type mismatch in %s: found %r, expected %r."
                % (self.meta_path, meta.get('feat_type'), self.feat_type))
        # Compare config through a JSON round-trip so types match the stored form.
        want = json.loads(json.dumps(self.config_metadata, default=str))
        if meta.get('config') != want:
            raise ValueError(
                "Feature cache config mismatch in %s (hash collision or "
                "corruption): found %r, expected %r."
                % (self.meta_path, meta.get('config'), want))
        if self.n_features is None:
            self.n_features = meta.get('n_features')

    # ------------------------------------------------------------------- index
    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                self._index = pickle.load(f)
        elif os.path.exists(self.blob_path):
            self._rebuild_index_from_blob()
        else:
            self._index = {}
        self._open_read_fd()

    def _rebuild_index_from_blob(self):
        """Reconstruct the in-RAM index by scanning every record in the blob."""
        index = {}
        with open(self.blob_path, 'rb') as f:
            while True:
                head = f.read(_LEN_HEADER.size)
                if len(head) < _LEN_HEADER.size:
                    break
                (smiles_len,) = _LEN_HEADER.unpack(head)
                smiles = f.read(smiles_len).decode('utf-8')
                is_valid, payload_len = _VALID_HEADER.unpack(
                    f.read(_VALID_HEADER.size))
                payload_offset = f.tell()
                f.seek(payload_len, os.SEEK_CUR)
                index[smiles] = (payload_offset, payload_len, bool(is_valid))
        self._index = index

    def _persist_index(self):
        tmp_path = '%s.tmp.%d' % (self.index_path, os.getpid())
        with open(tmp_path, 'wb') as f:
            pickle.dump(self._index, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.index_path)

    # --------------------------------------------------------------------- get
    def get(self, smiles_list):
        """Look up cached features for a list of SMILES.

        Args:
            smiles_list (sequence of str): per-molecule keys.

        Returns:
            tuple ``(features, is_valid, miss_mask)``:
                features (list): length ``len(smiles_list)``; the unpickled
                    feature unit for cached valid molecules, ``None`` for
                    misses and for cached-invalid molecules.
                is_valid (np.ndarray of bool): cached validity per molecule;
                    ``False`` where ``miss_mask`` is True (validity unknown).
                miss_mask (np.ndarray of bool): True where the SMILES is not
                    in the cache.
        """
        n = len(smiles_list)
        features = [None] * n
        is_valid = np.zeros(n, dtype=bool)
        miss_mask = np.zeros(n, dtype=bool)
        for i, smiles in enumerate(smiles_list):
            entry = self._index.get(smiles)
            if entry is None:
                miss_mask[i] = True
                continue
            offset, payload_len, valid = entry
            is_valid[i] = valid
            if valid:
                self._open_read_fd()
                raw = os.pread(self._read_fd, payload_len, offset)
                features[i] = pickle.loads(raw)
        return features, is_valid, miss_mask

    # --------------------------------------------------------------------- put
    def put(self, smiles_list, units, is_valid):
        """Append per-molecule features for cache misses.

        SMILES already present in the index are skipped (idempotent). Appends
        are serialised by an ``fcntl.flock`` advisory lock; the blob is
        fsynced and the index persisted atomically before the lock releases.

        Args:
            smiles_list (sequence of str): per-molecule keys.
            units (sequence): feature unit per molecule, or ``None`` for an
                invalid molecule (its ``is_valid`` entry must be False).
            is_valid (sequence of bool): validity per molecule.
        """
        if self.readonly:
            raise RuntimeError("put() called on a read-only feature cache: %s"
                               % self.cache_dir)

        lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            offset_cursor = (os.path.getsize(self.blob_path)
                             if os.path.exists(self.blob_path) else 0)
            with open(self.blob_path, 'ab') as bf:
                for smiles, unit, valid in zip(smiles_list, units, is_valid):
                    if smiles in self._index:
                        continue
                    smiles_bytes = smiles.encode('utf-8')
                    valid = bool(valid)
                    payload = (pickle.dumps(unit, protocol=pickle.HIGHEST_PROTOCOL)
                               if valid else b'')
                    bf.write(_LEN_HEADER.pack(len(smiles_bytes)))
                    bf.write(smiles_bytes)
                    bf.write(_VALID_HEADER.pack(1 if valid else 0, len(payload)))
                    bf.write(payload)
                    payload_offset = (offset_cursor + _LEN_HEADER.size
                                      + len(smiles_bytes) + _VALID_HEADER.size)
                    self._index[smiles] = (payload_offset, len(payload), valid)
                    offset_cursor = payload_offset + len(payload)
                bf.flush()
                os.fsync(bf.fileno())
            self._persist_index()
        finally:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        self._open_read_fd()

    # ------------------------------------------------------------------- close
    def close(self):
        """Close the blob read descriptor."""
        if self._read_fd is not None:
            os.close(self._read_fd)
            self._read_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ****************************************************************************************
def featurise_with_cache(cache, df, featurise_misses, smiles_col):
    """Featurise a DataFrame slice through a :class:`FeatureCache`.

    Reads cached per-molecule features for the rows already in the cache and
    computes only the misses via ``featurise_misses``, writing them back. The
    return value matches :func:`featurization.featurize_smiles`, so callers can
    drop this in wherever they would have called the featurizer directly.

    Args:
        cache (FeatureCache): the cache handle.
        df (pd.DataFrame): rows to featurise.
        featurise_misses (callable): ``featurise_misses(miss_df) ->
            (features, is_valid)`` with the same contract as
            ``featurize_smiles``: ``features`` holds the valid rows only,
            ``is_valid`` is a bool array over every row of ``miss_df``.
        smiles_col (str): the SMILES column name, used as the per-molecule key.

    Returns:
        tuple ``(features, is_valid)``:
            features (np.ndarray): valid-only array (2D float for fixed-width
                featurizers, 1D object for graph), reassembled from the stored
                per-molecule units.
            is_valid (np.ndarray of bool): length equal to ``len(df)``.
    """
    smiles_list = list(df[smiles_col].values)
    units, is_valid, miss_mask = cache.get(smiles_list)

    miss_positions = np.flatnonzero(miss_mask)
    if miss_positions.size:
        miss_df = df.iloc[miss_positions]
        miss_feats, miss_valid = featurise_misses(miss_df)
        miss_smiles = [smiles_list[p] for p in miss_positions]
        # miss_feats holds rows for valid misses only, in order; map them back
        # to a per-molecule unit (None for an invalid molecule).
        miss_units = [None] * miss_positions.size
        k = 0
        for m, valid in enumerate(miss_valid):
            if valid:
                miss_units[m] = miss_feats[k]
                k += 1
        cache.put(miss_smiles, miss_units, miss_valid)
        for m, pos in enumerate(miss_positions):
            units[pos] = miss_units[m]
            is_valid[pos] = miss_valid[m]

    valid_units = [u for u, valid in zip(units, is_valid) if valid]
    features = np.array(valid_units)
    return features, is_valid
