"""Backend-dispatched capture factory for the ``reuse_fit_train_preds`` flag.

When the flag is on, :func:`make_capturing` returns a dynamic subclass of the
model class whose fit pass also captures per-batch training-mode outputs,
aligned to dataset ids, so ``train_perf`` can be computed without a separate
inference-mode ``model.predict`` pass. valid and test are unchanged: they
always run the inference predict pass.

This is a DeepChem 2.8.0-coupled fork of ``KerasModel.fit_generator`` /
``TorchModel.fit_generator`` (and ``KerasModel._create_gradient_fn``). The fit
loops below are copied verbatim from deepchem 2.8.0 with a capture block added,
so **a deepchem upgrade is a re-verification event**: re-diff both
``fit_generator`` copies against the new deepchem source, and re-run
``atomsci/ddm/test/unit/test_capture_model.py``, whose lr=0 / dropout=0 tests
assert the captured predictions equal ``model.predict`` exactly.

Two further couplings to be aware of:

- The capture only overrides ``fit_generator``, which it requires to be driven
  by the id-carrying ``default_generator`` below. ``fit_on_batch`` passes raw
  3-tuples straight to ``fit_generator`` and would therefore fail on a
  capturing model. AMPL never calls it.
- :meth:`_CaptureMixin.pop_captured_train_preds` assumes only the last batch of
  an epoch is padded (see the note on its padding trim).
"""
import logging
import time
from collections.abc import Sequence as SequenceCollection

import numpy as np
import tensorflow as tf
import torch

from deepchem.models.keras_model import KerasModel
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.trans import undo_transforms

logger = logging.getLogger(__name__)


class _IdRecorder:
    """Dataset proxy that records the ids of each batch its ``iterbatches`` yields.

    The model backends' ``default_generator`` iterate ``dataset.iterbatches``
    and discard the per-batch ids. This proxy wraps the dataset so the ids
    travel alongside the batches the backend already produces, without
    replicating any model-specific batching (e.g. GraphConv ``ConvMol``
    agglomeration). Other attribute access delegates to the wrapped dataset.
    """

    def __init__(self, dataset, sink):
        # Bypass __getattr__ for our own attributes.
        self.__dict__['_ds'] = dataset
        self.__dict__['_sink'] = sink

    def iterbatches(self, **kwargs):
        for batch in self._ds.iterbatches(**kwargs):
            self._sink.append(batch[3])
            yield batch

    def __getattr__(self, name):
        return getattr(self._ds, name)


class _CaptureMixin:
    """Shared capture plumbing for the Keras and Torch backends.

    The backend-specific subclass supplies ``fit_generator`` (and Keras adds
    ``_create_gradient_fn``). This mixin supplies ``default_generator``
    (id-carrying for fit, unchanged for predict), ``fit`` (buffer reset and
    dataset reference), and ``pop_captured_train_preds`` (post-processing
    identical to ``model.predict``).
    """

    # Class-level defaults so pop_captured_train_preds before the first fit
    # fails with the message below rather than an opaque AttributeError.
    _capture_dataset = None
    _capture_train_size = 0
    _captured_outputs = None
    _captured_ids = None

    def fit(self, dataset, nb_epoch=10, **kwargs):
        self._capture_dataset = dataset
        self._capture_train_size = len(dataset)
        self._captured_outputs = []
        self._captured_ids = []
        return super().fit(dataset, nb_epoch=nb_epoch, **kwargs)

    def default_generator(self, dataset, epochs=1, mode='fit',
                          deterministic=True, pad_batches=True):
        # Predict path: delegate unchanged so valid/test stay on the inference
        # predict pass (3-tuple batches, no capture).
        if mode != 'fit':
            yield from super().default_generator(
                dataset, epochs=epochs, mode=mode, deterministic=deterministic,
                pad_batches=pad_batches)
            return
        # Fit path: record ids via a proxy and re-attach them to each batch the
        # backend's own default_generator already produces (4-tuple). This
        # works for any backend default_generator, including GraphConv's graph
        # batching, because the proxy only intercepts iterbatches.
        ids_for_batches = []
        proxy = _IdRecorder(dataset, ids_for_batches)
        for batch in super().default_generator(
                proxy, epochs=epochs, mode=mode, deterministic=deterministic,
                pad_batches=pad_batches):
            yield (*batch, ids_for_batches.pop(0))

    def pop_captured_train_preds(self, transformers):
        """Return captured train predictions in dataset-id order, transformer-undone.

        Mirrors ``model.predict`` post-processing per batch: select
        ``_prediction_outputs`` (None means use the full output list, as predict
        does), assert a single output, then ``undo_transforms``. Padded rows are
        trimmed by count from the tail of each epoch's last batch: ``pad_batch``
        re-tiles the first rows at the end, so padded ids are real duplicates and
        cannot be filtered by id value. Finally the predictions are reordered to
        dataset-id order so ``accumulate_preds(preds, dset.ids)`` aligns
        correctly (fit batches are shuffled, not in dataset order).

        Returns ``(preds, ids)``: ``preds`` aligned to
        ``self._capture_dataset.ids``; ``ids`` is that same id array (for
        alignment tests).

        Assumes the AMPL fit pattern of one epoch per ``fit`` call
        (``nb_epoch=1``); under multi-epoch ``fit`` the last epoch's prediction
        for a duplicated id wins the reorder.
        """
        if not self._captured_outputs:
            raise RuntimeError(
                "No captured train predictions to pop. pop_captured_train_preds "
                "must be called after a fit() on this model, once per fit, and "
                "only on a model built with make_capturing(cls, True).")
        outputs = self._captured_outputs
        ids = self._captured_ids
        n = self._capture_train_size
        bs = self.batch_size
        # Trim the padded tail of each epoch's last batch by count.
        #
        # This assumes the last batch of an epoch is the only padded one, which
        # holds because every batch before it is exactly batch_size rows. Note
        # StreamingFileDataset.iterbatches pads ANY short batch, and
        # _materialise_positions can shorten a batch by dropping rows whose
        # SMILES fail to featurise -- but StreamingFileDataset pre-scans
        # validity and drops those rows at construction, so no mid-epoch batch
        # is ever short. If that pre-scan ever goes away, this trim leaves
        # untrimmed duplicates and the reorder below raises KeyError on the
        # dropped ids.
        nbpe = (n + bs - 1) // bs  # batches per epoch (ceil)
        if nbpe > 0:
            real_last = n - (nbpe - 1) * bs
            nepochs = len(outputs) // nbpe
            if 0 < real_last < bs:
                for e in range(nepochs):
                    idx = (e + 1) * nbpe - 1
                    outputs[idx] = outputs[idx][:real_last]
                    ids[idx] = ids[idx][:real_last]
        # undo_transforms per batch (mirrors predict), then concatenate.
        undone = [undo_transforms(o, transformers) for o in outputs]
        preds = np.concatenate(undone, axis=0)
        all_ids = np.concatenate(ids, axis=0)
        # Reorder to dataset id order.
        target_ids = np.asarray(self._capture_dataset.ids)
        pred_by_id = dict(zip(all_ids, preds))
        ordered = np.stack([pred_by_id[i] for i in target_ids])
        self._captured_outputs = []
        self._captured_ids = []
        return ordered, target_ids


class KerasCaptureMixin(_CaptureMixin):
    """KerasModel capture: overrides ``_create_gradient_fn`` so the tf.function
    returns the raw training-mode outputs, and ``fit_generator`` to collect them."""

    def _create_gradient_fn(self, variables):
        @tf.function(experimental_relax_shapes=True)
        def apply_gradient_for_batch(inputs, labels, weights, loss):
            with tf.GradientTape() as tape:
                outputs = self.model(inputs, training=True)
                if tf.is_tensor(outputs):
                    outputs = [outputs]
                raw_outputs = tuple(outputs)  # before _loss_outputs rebinding
                if self._loss_outputs is not None:
                    outputs = [outputs[i] for i in self._loss_outputs]
                batch_loss = loss(outputs, labels, weights)
            if variables is None:
                vars = self.model.trainable_variables
            else:
                vars = variables
            grads = tape.gradient(batch_loss, vars)
            self._tf_optimizer.apply_gradients(zip(grads, vars))
            self._global_step.assign_add(1)
            return (batch_loss,) + raw_outputs

        return apply_gradient_for_batch

    def fit_generator(self, generator, max_checkpoints_to_keep=5,
                      checkpoint_interval=1000, restore=False, variables=None,
                      loss=None, callbacks=[], all_losses=None):
        # DeepChem 2.8.0 KerasModel.fit_generator with per-batch output capture.
        # The setup and loop are copied verbatim; the capture block is marked.
        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
        self._ensure_built()
        if checkpoint_interval > 0:
            manager = tf.train.CheckpointManager(self._checkpoint,
                                                 self.model_dir,
                                                 max_checkpoints_to_keep)
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        if loss is None:
            loss = self._loss_fn
        var_key = None
        if variables is not None:
            var_key = tuple(v.ref() for v in variables)
            zero_grads = [tf.zeros(v.shape) for v in variables]
            self._tf_optimizer.apply_gradients(zip(zero_grads, variables))
        if var_key not in self._gradient_fn_for_vars:
            self._gradient_fn_for_vars[var_key] = self._create_gradient_fn(
                variables)
        apply_gradient_for_batch = self._gradient_fn_for_vars[var_key]
        time1 = time.time()

        for batch in generator:
            inputs, labels, weights, ids_b = batch  # 4-tuple (capture: was 3-tuple)
            self._create_training_ops((inputs, labels, weights))
            if restore:
                self.restore()
                restore = False
            inputs, labels, weights = self._prepare_batch(
                (inputs, labels, weights))
            if len(inputs) == 1:
                inputs = inputs[0]
            result = apply_gradient_for_batch(inputs, labels, weights, loss)
            batch_loss = result[0]
            raw_outputs = list(result[1:])
            # --- capture training-mode outputs (mirror predict post-processing) ---
            out_np = [t.numpy() for t in raw_outputs]
            if self._prediction_outputs is not None:
                sel = [out_np[i] for i in self._prediction_outputs]
            else:
                sel = out_np
            if len(sel) != 1:
                raise ValueError(
                    "reuse_fit_train_preds capture supports single-output "
                    f"models only; got {len(sel)} prediction outputs.")
            self._captured_outputs.append(np.asarray(sel[0]))
            self._captured_ids.append(np.asarray(ids_b))
            # --- end capture ---
            current_step = self._global_step.numpy()
            avg_loss += batch_loss
            averaged_batches += 1
            should_log = (current_step % self.log_frequency == 0)
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                logger.info('Ending global_step %d: Average loss %g' %
                            (current_step, avg_loss))
                if all_losses is not None:
                    all_losses.append(avg_loss)
                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0
            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                manager.save()
            for c in callbacks:
                c(self, current_step)
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard('loss', batch_loss,
                                                current_step)
            if (self.wandb_logger is not None) and should_log:
                all_data = dict({'train/loss': batch_loss})
                self.wandb_logger.log_data(all_data, step=current_step)

        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            logger.info('Ending global_step %d: Average loss %g' %
                        (current_step, avg_loss))
            if all_losses is not None:
                all_losses.append(avg_loss)
            last_avg_loss = avg_loss
        if checkpoint_interval > 0:
            manager.save()
        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return last_avg_loss


class TorchCaptureMixin(_CaptureMixin):
    """TorchModel capture: ``fit_generator`` grabs ``self.model(inputs)`` before
    the fit's ``_loss_outputs`` rebinding."""

    def fit_generator(self, generator, max_checkpoints_to_keep=5,
                      checkpoint_interval=1000, restore=False, variables=None,
                      loss=None, callbacks=[], all_losses=None):
        # DeepChem 2.8.0 TorchModel.fit_generator with per-batch output capture.
        # The setup and loop are copied verbatim; the capture block is marked.
        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
        self._ensure_built()
        self.model.train()
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        if loss is None:
            loss = self._loss_fn
        if variables is None:
            optimizer = self._pytorch_optimizer
            lr_schedule = self._lr_schedule
        else:
            var_key = tuple(variables)
            if var_key in self._optimizer_for_vars:
                optimizer, lr_schedule = self._optimizer_for_vars[var_key]
            else:
                optimizer = self.optimizer._create_pytorch_optimizer(variables)
                if isinstance(self.optimizer.learning_rate,
                              LearningRateSchedule):
                    lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                        optimizer)
                else:
                    lr_schedule = None
                self._optimizer_for_vars[var_key] = (optimizer, lr_schedule)
        time1 = time.time()

        for batch in generator:
            inputs, labels, weights, ids_b = batch  # 4-tuple (capture: was 3-tuple)
            if restore:
                self.restore()
                restore = False
            inputs, labels, weights = self._prepare_batch(
                (inputs, labels, weights))
            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = inputs[0]
            optimizer.zero_grad()
            outputs = self.model(inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            raw_outputs = list(outputs)  # before _loss_outputs rebinding
            if self._loss_outputs is not None:
                outputs = [outputs[i] for i in self._loss_outputs]
            batch_loss = loss(outputs, labels, weights)
            batch_loss.backward()
            optimizer.step()
            if lr_schedule is not None:
                lr_schedule.step()
            self._global_step += 1
            current_step = self._global_step
            # --- capture training-mode outputs (mirror predict post-processing) ---
            out_np = [t.detach().cpu().numpy() for t in raw_outputs]
            if self._prediction_outputs is not None:
                sel = [out_np[i] for i in self._prediction_outputs]
            else:
                sel = out_np
            if len(sel) != 1:
                raise ValueError(
                    "reuse_fit_train_preds capture supports single-output "
                    f"models only; got {len(sel)} prediction outputs.")
            self._captured_outputs.append(np.asarray(sel[0]))
            self._captured_ids.append(np.asarray(ids_b))
            # --- end capture ---
            avg_loss += batch_loss
            averaged_batches += 1
            should_log = (current_step % self.log_frequency == 0)
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                logger.info('Ending global_step %d: Average loss %g' %
                            (current_step, avg_loss))
                if all_losses is not None:
                    all_losses.append(avg_loss)
                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0
            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                self.save_checkpoint(max_checkpoints_to_keep)
            for c in callbacks:
                c(self, current_step)
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard('loss', batch_loss,
                                                current_step)
            if (self.wandb_logger is not None) and should_log:
                all_data = dict({'train/loss': batch_loss})
                self.wandb_logger.log_data(all_data, step=current_step)

        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            logger.info('Ending global_step %d: Average loss %g' %
                        (current_step, avg_loss))
            if all_losses is not None:
                all_losses.append(avg_loss)
            last_avg_loss = avg_loss
        if checkpoint_interval > 0:
            self.save_checkpoint(max_checkpoints_to_keep)
        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return last_avg_loss


_capture_cache = {}


def make_capturing(model_cls, flag):
    """Return a subclass of ``model_cls`` that captures fit-pass train preds.

    When ``flag`` is false, returns ``model_cls`` unchanged (no capture,
    byte-identical behaviour). When true, returns a dynamic subclass whose fit
    pass captures per-batch training-mode outputs aligned to dataset ids,
    dispatched by backend (KerasModel or TorchModel). The subclass exposes
    ``pop_captured_train_preds(transformers)`` to read and clear the buffer.
    """
    if not flag:
        return model_cls
    if issubclass(model_cls, KerasModel):
        mixin = KerasCaptureMixin
    elif issubclass(model_cls, TorchModel):
        mixin = TorchCaptureMixin
    else:
        raise TypeError(
            "reuse_fit_train_preds capture supports KerasModel or TorchModel "
            f"subclasses, got {model_cls!r}")
    key = (mixin, model_cls)
    cls = _capture_cache.get(key)
    if cls is None:
        cls = type(f"Capturing{model_cls.__name__}", (mixin, model_cls), {})
        _capture_cache[key] = cls
    return cls