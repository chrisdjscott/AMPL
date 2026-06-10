# StreamingFileDataset implementation notes

Companion to `AMPL_STREAMING_DATASET_PLAN.md`. Records the trade-offs
taken inside the implementation (not in the public design doc) so
future readers do not have to reverse-engineer them.

## What streaming is and is not

`StreamingFileDataset` removes the **memory peak** of holding the full
featurised matrix in RAM during training. Across epochs, peak memory
stays bounded by `batch_size * n_features` instead of
`len(dataset) * n_features`.

It does **not** remove the cost of running the featurizer over the
dataset once. Init still does an `O(N)` featurise pass (`_scan_validity`,
see below). For expensive featurizers, the follow-up caching PR is the
right answer; PR #2 deliberately keeps a single overridable per-batch
hook (`_StreamingFeatureDataset._featurise_batch`) and a separate
init-time seam (`StreamingFileDataset._scan_validity`) so a caching
subclass can intercept both.

## `_scan_validity`: why it featurises everything at init

AMPL's downstream code (`perf_data`, `NormalizationTransformerMissingData`,
splitters) reads `.ids` / `.y` / `.w` as fixed-length eager arrays and
assumes they line up with every batch `iterbatches` will emit. The
eager `DynamicFeaturization.featurize_data` honours this contract by
filtering its `is_valid` mask **before** building the dataset surface.

Some featurizers reject inputs that RDKit happily parses:

| Featurizer | Rejects |
|---|---|
| `MolGraphConvFeaturizer` | single-atom SMILES (e.g. methane `C`), some disconnected fragments |
| `ConvMolFeaturizer` | invalid valence / SMILES that survive parsing but fail valence checks |
| `ECFP` / `RDKitDescriptors` | typically accepts anything `MolFromSmiles` returns |

A cheap RDKit-only pre-filter (`MolFromSmiles is not None`) is
insufficient because it misses the featurizer-specific rejections in
the first two rows. The Phase 7 AttentiveFP integration test surfaced
exactly this case: row 0 of curated Delaney is methane, RDKit parses
it cleanly, `MolGraphConvFeaturizer` refuses, the streaming dataset's
eager `.ids` advertised 1116 rows while the per-batch
`_materialise_positions` dropped that row, and `perf_data` crashed
when it indexed the missing row.

`_scan_validity` solves this by running the featurizer over every row
in chunks (default 512), keeping only the `is_valid` mask and (from
the first valid chunk) the feature width. Features themselves are
discarded after each chunk so memory peak stays at
`chunk_size * n_features`. The mask filters `dset_df` / `vals` / `w` /
`ids` / `attr` before the `_StreamingFeatureDataset` is built, putting
the streaming surface back on the same contract as the eager one.

A wall-time log line is emitted at INFO level after the pass so users
can see whether the init cost is acceptable for their featurizer and
dataset size. Grep for `_scan_validity featurised` in pipeline logs.

## Alternatives considered (and not taken)

These are the alternatives ruled out for PR #2. Each may be worth
revisiting later under its own PR if `_scan_validity` becomes a
measured bottleneck.

### 1. RDKit-only pre-filter

Run `MolFromSmiles` over every row at init, skip the featurise pass.
Cheap, but lets featurizer-specific rejections leak through to
per-batch. The downstream `IndexError` returns. Rejected.

### 2. Lazy validity discovery during the first epoch

Treat `.ids` / `.y` / `.w` as provisional until the first epoch has
seen every row, then prune. Breaks the dataset-is-immutable invariant
that splitters and the transformer-fit pass rely on
(`NormalizationTransformerMissingData` fits before training starts,
so its dataset has to be final). Substantially more state machine
than the bug warrants. Rejected.

### 3. Push the filter downstream

Teach `perf_data` and the transformer fit code to tolerate batches
whose length disagrees with `.ids`. Larger blast radius, touches code
that is orthogonal to streaming, and the eager path already provides
the cleaner invariant. Rejected.

### 4. Parallelise `_scan_validity`

The scan loop is embarrassingly parallel across chunks (each chunk's
featurise is independent). A `multiprocessing.Pool` or
`concurrent.futures.ProcessPoolExecutor` would reduce wall time on
expensive featurizers. Two reasons to defer:

* For the featurizers PR #2 targets (ECFP, GraphConv, MolGraphConv)
  the scan is cheap relative to a single training epoch (Phase 7
  Delaney run: ~15s for 1116 rows with `MolGraphConvFeaturizer`).
* Parallelisation introduces a serialisation requirement on the
  featurizer object and exposes us to the usual fork-vs-spawn issues
  with RDKit/DGL/TF. Wider testing surface than PR #2 should take on.

Reopen if profiling on a large CSV (>100k rows) with an expensive
featurizer shows scan dominating wall time.

### 5. Make `_scan_validity` opt-out

Add a flag (`--skip_validity_scan`, default off) so callers who know
their featurizer accepts every row in their dataset can skip the
pass. Saves init time at the cost of a sharper foot-gun: per-batch
size mismatches would resurface for callers who turn it on
incorrectly. Plausible but adds a knob. Worth doing alongside the
caching PR, where the cache already has to track validity per row
anyway and the scan can be replaced by a cache lookup.

## Caching follow-up: seams to override

The two extension points a `CachingStreamingFileDataset` (or
`--cache_dir` wrapper) needs to intercept:

1. **`StreamingFileDataset._scan_validity`** — at init. Replace with a
   cache lookup: if every row in the dataset already has a cached
   `(features, is_valid)`, skip the featurise pass entirely. Otherwise
   featurise the missing rows, write to cache, build the mask from
   the union.
2. **`_StreamingFeatureDataset._featurise_batch`** — at iteration
   time. Replace with a cache read (and write on miss for the
   first-epoch cold case).

Both seams already exist in the code (`_scan_validity` is the chunked
loop, `_featurise_batch` calls `feat.featurize_smiles`). Neither
requires a refactor before the cache PR can land.

## Other implementation notes

### `transform` keeps `_y_raw` / `_w_raw`

`_StreamingFeatureDataset.transform(transformer)` rewrites the eager
`.y` / `.w` at construction (so perf-metric and balancing-transformer
code paths read the post-transform values directly) AND applies the
transformer chain per batch inside `iterbatches`. To stop the chain
from compounding on itself, `_y_raw` / `_w_raw` keep the pre-chain
arrays. `_materialise_positions` slices the raw arrays; `iterbatches`
applies the chain on top. `select` / `transform` forward the raw
arrays. See Phase 6 Decision log in `.planning/...` for the original
bug.

### `.X` raises `NotImplementedError`

A hard refuse is preferred over "materialise on demand with a
warning". The latter would invite silent OOM regressions for callers
that read `.X` once and assume cheap (the `combined_training_data`
path, legacy code paths, RF `tree.predict(dataset.X)`). Parser-level
validation already refuses the combinations where `.X` access is
unavoidable (k-fold CV, feature transformers with descriptor
featurizers, `previously_split=True`, `datastore=True`).

### `itersamples` yields a zero placeholder for X

AMPL's only `itersamples` consumer is
`transformations.get_statistics_missing_ydata`, which reads `y` / `w`
only. Featurising every row inside `itersamples` would defeat
streaming. The placeholder is documented and a hard contract — any
new consumer that reads X from `itersamples` should either work with
zeros (rare) or use `iterbatches` instead.
