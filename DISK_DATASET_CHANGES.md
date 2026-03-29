# Branch Changes Summary: `disk-dataset-option-mlflow`

This document summarizes the changes in the `disk-dataset-option-mlflow` branch compared to the master branch.

## Overview

This branch contains **18 commits** that add two major features to the AMPL pipeline:

1. **DiskDataset Support** - Memory-efficient handling of large datasets using DeepChem's DiskDataset
2. **MLflow Integration** - Experiment tracking and logging via MLflow REST API

---

## 1. DiskDataset Support (Memory-Efficient Large Dataset Handling)

### New Command-Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use_disk_dataset` | flag | False | Enable DiskDataset instead of NumpyDataset for memory efficiency with large datasets |
| `--disk_dataset_root` | string | None | Root directory for storing disk datasets (defaults to temporary directories) |
| `--shard_size` | int | 10000 | Number of compounds per shard when using DiskDataset |

### New Split Strategy

| Parameter | Description |
|-----------|-------------|
| `simple_train_valid_test` | A simplified splitter that assumes input data is pre-shuffled and splits by row order (first N rows = train, next M = valid, etc.). Much faster for large datasets. |

### Split Number Parameters (for `simple_train_valid_test`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `--split_train_num` | int | Number of compounds in train split |
| `--split_valid_num` | int | Number of compounds in valid split |
| `--split_test_num` | int | Number of compounds in test split |

### Key Implementation Details

#### New File: `atomsci/ddm/pipeline/utils.py`
Utility functions for memory monitoring and debugging:
- `get_memory_usage()` - Returns current process memory usage in GiB
- `tracemalloc_snapshot()` - Takes memory allocation snapshots
- `sizeof_convmol()` - Estimates size of ConvMol objects

#### `model_datasets.py`
- Added `shard_generator()` method for chunked data loading and featurization
- DiskDataset creation with configurable shard size
- Incompatibility with `previously_featurized` option (raises RuntimeError)

#### `splitting.py`
- New `SimpleTrainValidTestSplitting` class for efficient splitting
- Supports both fraction-based and number-based splitting
- Handles DiskDataset-specific operations (moving datasets after splitting)
- Added extensive timing and memory logging

#### `perf_data.py`
- New `LazyArray` class for memory-efficient array storage
- Stores arrays on disk and loads them lazily when accessed
- Reduces memory footprint during training with large datasets
- Integrates with DiskDataset mode

#### `model_wrapper.py`
- DiskDataset handling in training loops
- Proper dataset path management after transformations
- Memory usage logging throughout training

---

## 2. MLflow Integration (Experiment Tracking)

### New Command-Line Parameter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use_mlflow` | flag | False | Enable MLflow tracking server logging |

### Environment Variables Required

| Variable | Description |
|----------|-------------|
| `MLFLOW_TRACKING_URI` | URL of the MLflow tracking server |
| `MLFLOW_EXPERIMENT_NAME` | Name of the experiment |
| `MLFLOW_RUN_NAME` | Name of the run |
| `MLFLOW_TRACKING_USERNAME` | Username for authentication |
| `MLFLOW_TRACKING_PASSWORD` | Password for authentication |

### New File: `atomsci/ddm/pipeline/mlflow_utils.py`

A complete MLflow REST API client with the following functions:

#### Experiment Management
- `get_or_create_experiment()` - Retrieves existing or creates new experiment
- `create_run(experiment_id)` - Creates a new run within an experiment
- `end_run(run_id)` - Marks run as finished

#### Logging Functions
- `log_metric(run_id, metric_name, metric_value, step=0)` - Logs metrics (e.g., training performance per epoch)
- `log_param(run_id, param_name, param_value)` - Logs parameters (e.g., dataset info, model config)
- `log_artifact(run_id, local_path, artifact_path)` - Uploads files (e.g., model tarballs)
- `set_tag(run_id, tag_name, tag_value)` - Sets tags (e.g., model_type, featurizer)

### MLflow Integration in `model_pipeline.py`

The training workflow now logs:

#### Dataset Loading & Featurization
- Dataset load/featurization time
- Memory usage after loading
- Dataset length
- Dataset key/path
- DiskDataset usage flag and shard size

#### Dataset Splitting
- Splitting time
- Memory usage after splitting

#### Training
- Training time
- Memory usage after training
- Per-epoch metrics (train_perf, valid_perf)

#### Final Metrics
- MAE, R², RMS scores for train/valid/test sets
- Number of compounds in each subset
- Dataset hash and result directory

---

## 3. Enhanced Debugging & Logging

Extensive debug logging has been added throughout the pipeline:

- Memory usage tracking at key stages (dataset loading, splitting, training)
- Timing measurements for dataset operations
- Dataset path logging for DiskDatasets
- Transformer creation and application logging
- Epoch manager performance tracking

---

## Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `mlflow_utils.py` | +198 (new) | MLflow REST API client |
| `utils.py` | +53 (new) | Memory monitoring utilities |
| `splitting.py` | +290/-6 | SimpleTrainValidTestSplitting, DiskDataset handling |
| `model_datasets.py` | +202/-13 | Shard generator, DiskDataset support |
| `model_pipeline.py` | +166/-38 | MLflow integration, timing/memory logging |
| `model_wrapper.py` | +100/-3 | DiskDataset training support |
| `perf_data.py` | +106/-3 | LazyArray class for memory efficiency |
| `parameter_parser.py` | +29/-2 | New CLI parameters |
| `transformations.py` | +14 | Debug logging |
| `featurization.py` | +2 | Debug logging |

**Total: 10 files, +1079/-81 lines**

---

## Usage Examples

### Using DiskDataset for Large Datasets

```bash
python -m atomsci.ddm.pipeline.model_pipeline \
  --use_disk_dataset \
  --disk_dataset_root /path/to/storage \
  --shard_size 10000 \
  --split_strategy simple_train_valid_test \
  --split_train_num 80000 \
  --split_valid_num 10000 \
  --split_test_num 10000 \
  --dataset_key large_dataset.csv \
  ...
```

### Using MLflow Tracking

```bash
# Set environment variables
export MLFLOW_TRACKING_URI=https://mlflow.example.com
export MLFLOW_EXPERIMENT_NAME=my_experiment
export MLFLOW_RUN_NAME=my_run
export MLFLOW_TRACKING_USERNAME=user
export MLFLOW_TRACKING_PASSWORD=pass

# Run pipeline with MLflow
python -m atomsci.ddm.pipeline.model_pipeline \
  --use_mlflow \
  --use_disk_dataset \
  ...
```

---

## Git Commit History

```
e58802f option to load presplit data
f3f0586 fixed an mlflow logging block that wasn't disabled when mlflow was
a55866e working on logging artifacts
9b35204 saving best metrics to mlflow
2ca32a2 fix leftover MLFLOW_LOADED
28386be converting mlflow to use rest api
79f7dd8 set experiment name and tracking uri by env vars
cc19db9 start testing mlflow
c431701 implement LazyArray in perf_data to avoid large memory allocations
0ba54a0 caching disk dataset
899dc92 adding simple dataset splitter for large disk datasets
c891455 add timings around splitting
692ea5c identifying shard caching issue
ab7016d debugging
cbba643 debugging
21808a4 debugging
c5a343b debugging sharded disk dataset
21013e0 fix running with numpy dataset
31b0664 working on option to use disk dataset
```

---

## Notes & Limitations

1. **DiskDataset incompatibility**: Cannot be used with `previously_featurized` option
2. **max_dataset_rows**: Setting this > 0 is unlikely to work properly with DiskDataset
3. **TODO items in code**:
   - Combined training data needs DiskDataset conversion
   - Temporary directory cleanup for DiskDataset root
   - Some commented-out artifact logging code
