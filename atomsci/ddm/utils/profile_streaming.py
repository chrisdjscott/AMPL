#!/usr/bin/env python
"""Break a streaming training run's per-epoch wall clock into its components.

Sizes the throughput follow-ups (STREAMING_PERFORMANCE_FOLLOWUPS.md items B, C
and D) by measuring where an epoch actually goes: the gradient pass, the three
metric passes, and inside each of those, per-batch featurisation, cache reads,
ConvMol agglomeration and model compute.

Usage:
    python -m atomsci.ddm.utils.profile_streaming --config <config.json> \
        --epochs 6 --out profile.json [--param key=value ...]

The config is an AMPL parameter JSON. Any --param overrides are applied on top,
so the same file that produced a benchmark run can be reused with a smaller
max_epochs.
"""

import argparse
import gc
import json
import os
import time
from collections import defaultdict

import deepchem as dc

import atomsci.ddm.pipeline.model_datasets as model_datasets
import atomsci.ddm.pipeline.model_pipeline as model_pipeline
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.perf_data as perf_data


class Timings:
    """Accumulates elapsed time and counts per (epoch, phase, component)."""

    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.epoch = 0
        self.phase = 'setup'

    def add(self, component, elapsed, count=1):
        self.totals[(self.epoch, self.phase, component)] += elapsed
        self.counts[(self.epoch, self.phase, component)] += count

    def records(self):
        keys = sorted(set(self.totals) | set(self.counts))
        return [{'epoch': epoch, 'phase': phase, 'component': component,
                 'seconds': round(self.totals[(epoch, phase, component)], 4),
                 'count': self.counts[(epoch, phase, component)]}
                for epoch, phase, component in keys]


TIMINGS = Timings()


def _timed(component, func):
    """Wrap func so its wall time lands in the current epoch and phase."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            TIMINGS.add(component, time.perf_counter() - start)
    return wrapper


def _instrument_iterbatches(cls):
    """Time the generator itself, i.e. featurisation plus per-batch transformers.

    Agglomeration and model compute happen in the consumer, so they are not
    counted here: this is the cost of producing (X, y, w, ids).
    """
    original = cls.iterbatches

    def iterbatches(self, *args, **kwargs):
        generator = original(self, *args, **kwargs)
        while True:
            start = time.perf_counter()
            try:
                batch = next(generator)
            except StopIteration:
                TIMINGS.add('iterbatches', time.perf_counter() - start, count=0)
                return
            TIMINGS.add('iterbatches', time.perf_counter() - start)
            TIMINGS.add('rows', 0.0, count=len(batch[0]))
            yield batch

    cls.iterbatches = iterbatches


def _instrument_featurise(cls, component):
    """Time a class's own _featurise_batch, not one it inherits."""
    if '_featurise_batch' not in vars(cls):
        return
    original = vars(cls)['_featurise_batch']

    def featurise_batch(self, dset_df_slice, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original(self, dset_df_slice, *args, **kwargs)
        finally:
            TIMINGS.add(component, time.perf_counter() - start, count=len(dset_df_slice))

    cls._featurise_batch = featurise_batch


def _instrument_epoch_phases():
    """Tag every timing with the phase it belongs to: the fit pass or a metric pass."""
    original_fit = dc.models.KerasModel.fit
    original_torch_fit = dc.models.TorchModel.fit
    original_accumulate = perf_data.EpochManager.accumulate

    def make_fit(original):
        def fit(self, dataset, *args, **kwargs):
            TIMINGS.phase = 'fit'
            start = time.perf_counter()
            try:
                return original(self, dataset, *args, **kwargs)
            finally:
                TIMINGS.add('phase_total', time.perf_counter() - start)
        return fit

    def accumulate(self, ei, subset, dset, train_pred=None):
        TIMINGS.epoch = ei
        TIMINGS.phase = f'predict_{subset}'
        start = time.perf_counter()
        try:
            return original_accumulate(self, ei, subset, dset, train_pred=train_pred)
        finally:
            TIMINGS.add('phase_total', time.perf_counter() - start)
            # the next fit call belongs to the following epoch
            TIMINGS.epoch = ei + 1

    dc.models.KerasModel.fit = make_fit(original_fit)
    dc.models.TorchModel.fit = make_fit(original_torch_fit)
    perf_data.EpochManager.accumulate = accumulate


def instrument():
    """Patch every seam that contributes to an epoch's wall clock."""
    _instrument_epoch_phases()
    _instrument_iterbatches(model_datasets._StreamingFeatureDataset)
    # the base class featurises; the caching subclass wraps it with cache reads,
    # so timing both gives the cache-read cost as the difference
    _instrument_featurise(model_datasets._StreamingFeatureDataset, 'featurise')
    _instrument_featurise(model_datasets.CachingStreamingFeatureDataset, 'featurise_or_cache_read')
    dc.feat.mol_graphs.ConvMol.agglomerate_mols = staticmethod(
        _timed('agglomerate_mols', dc.feat.mol_graphs.ConvMol.agglomerate_mols))


def summarise(records, epochs):
    """Mean per-epoch seconds by phase and component, skipping epoch 0.

    Epoch 0 pays the cache-warm and any lazy setup, so it is reported separately
    rather than averaged in.
    """
    steady = [r for r in records if r['epoch'] > 0 and r['epoch'] < epochs]
    n_epochs = len(set(r['epoch'] for r in steady)) or 1

    by_phase = defaultdict(lambda: defaultdict(float))
    for record in steady:
        by_phase[record['phase']][record['component']] += record['seconds'] / n_epochs

    summary = {'steady_state_epochs': n_epochs, 'phases': {}}
    for phase, components in sorted(by_phase.items()):
        total = components.get('phase_total', 0.0)
        data_path = components.get('iterbatches', 0.0) + components.get('agglomerate_mols', 0.0)
        summary['phases'][phase] = {
            'seconds': round(total, 3),
            'iterbatches': round(components.get('iterbatches', 0.0), 3),
            'featurise': round(components.get('featurise', 0.0), 3),
            'featurise_or_cache_read': round(components.get('featurise_or_cache_read', 0.0), 3),
            'agglomerate_mols': round(components.get('agglomerate_mols', 0.0), 3),
            'compute_and_other': round(total - data_path, 3),
        }

    epoch_total = sum(p['seconds'] for p in summary['phases'].values())
    data_path = sum(p['iterbatches'] + p['agglomerate_mols'] for p in summary['phases'].values())
    predict_train = summary['phases'].get('predict_train', {}).get('seconds', 0.0)
    featurise = sum(p['featurise_or_cache_read'] or p['featurise']
                    for p in summary['phases'].values())
    agglomerate = sum(p['agglomerate_mols'] for p in summary['phases'].values())

    summary['epoch_seconds'] = round(epoch_total, 3)
    summary['data_path_seconds'] = round(data_path, 3)
    summary['compute_and_other_seconds'] = round(epoch_total - data_path, 3)
    # what each follow-up would remove from an epoch, at face value
    summary['sizing'] = {
        'reuse_fit_pass_for_train_perf': round(predict_train, 3),
        'parallel_cpu_work_8_way': round((featurise + agglomerate) * 7 / 8, 3),
        'cache_agglomerated_batches': round(featurise + agglomerate, 3),
        'prefetch_upper_bound': round(min(data_path, epoch_total - data_path), 3),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='AMPL parameter JSON')
    parser.add_argument('--epochs', type=int, default=6, help='epochs to run')
    parser.add_argument('--out', required=True, help='where to write the profile JSON')
    parser.add_argument('--param', action='append', default=[],
                        help='key=value override applied on top of the config')
    parser.add_argument('--gc', choices=['default', 'frozen', 'disabled'], default='default',
                        help='garbage collector mode. Unpickling a batch of graph features '
                             'creates tens of thousands of tracked numpy arrays, so generational '
                             'collection can dominate the read path. "frozen" moves the objects '
                             'alive at startup out of the way; "disabled" measures the ceiling.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    config['max_epochs'] = args.epochs
    config['early_stopping_patience'] = args.epochs
    for override in args.param:
        key, _, value = override.partition('=')
        config[key] = value

    instrument()

    params = parse.wrapper(config)
    start = time.perf_counter()
    pipeline = model_pipeline.ModelPipeline(params)
    if args.gc == 'frozen':
        gc.collect()
        gc.freeze()
    elif args.gc == 'disabled':
        gc.disable()
    pipeline.train_model()
    wall = time.perf_counter() - start

    records = TIMINGS.records()
    profile = {
        'config': os.path.abspath(args.config),
        'epochs': args.epochs,
        'wall_seconds': round(wall, 2),
        'summary': summarise(records, args.epochs),
        'records': records,
    }
    with open(args.out, 'w') as f:
        json.dump(profile, f, indent=2)

    summary = profile['summary']
    print(f"\nwall {wall:.1f} s over {args.epochs} epochs, "
          f"steady-state epoch {summary['epoch_seconds']:.1f} s")
    for phase, values in summary['phases'].items():
        print(f"  {phase:16s} {values['seconds']:8.2f} s  "
              f"iterbatches {values['iterbatches']:7.2f}  "
              f"featurise {values['featurise']:7.2f}  "
              f"cache_read {values['featurise_or_cache_read']:7.2f}  "
              f"agglomerate {values['agglomerate_mols']:7.2f}  "
              f"compute/other {values['compute_and_other']:7.2f}")
    print('  sizing (seconds removed from an epoch):')
    for option, seconds in summary['sizing'].items():
        print(f"    {option:34s} {seconds:8.2f}")


if __name__ == '__main__':
    main()
