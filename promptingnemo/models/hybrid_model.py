"""Custom Hybrid RNN-T/CTC BPE model with tag classifier support.

Extends NeMo's EncDecHybridRNNTCTCBPEModel with:
- TrailingTagClassifier (AGE, GENDER, EMOTION, INTENT)
- Per-family WER tracking
- FlexibleSaveRestoreConnector for vocab-size mismatches
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist

try:
    from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
except ImportError:
    DALIOutputs = tuple()

from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.core.classes.mixins import AccessMixin
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector


class FlexibleSaveRestoreConnector(SaveRestoreConnector):
    def __init__(self):
        super().__init__()
        self._skipped_state = {}

    def load_instance_with_state_dict(self, instance, state_dict, strict):
        model_state = instance.state_dict()
        skipped = []
        self._skipped_state = {}
        for key in list(state_dict.keys()):
            if key in model_state and state_dict[key].shape != model_state[key].shape:
                skipped.append(
                    f"{key}: ckpt {list(state_dict[key].shape)} vs model {list(model_state[key].shape)}"
                )
                self._skipped_state[key] = state_dict.pop(key)
        if skipped:
            logging.warning("Skipped %d size-mismatched keys:\n  %s", len(skipped), "\n  ".join(skipped))
        instance.load_state_dict(state_dict, strict=False)
        instance._set_model_restore_state(is_being_restored=False)


class CustomEncDecHybridRNNTCTCBPEModel(EncDecHybridRNNTCTCBPEModel):
    _save_restore_connector = FlexibleSaveRestoreConnector()

    def setup_custom_loss(self):
        self._val_family_stats = defaultdict(lambda: {'errors': 0.0, 'words': 0})
        self._validation_dataset_ref = None

    def setup_tag_classifier(self, encoder_dim, category_sizes, weight=0.5,
                             hidden_dim=256, dropout=0.3):
        from scripts.asr.meta_asr.tag_classifier import TrailingTagClassifier
        self.use_tag_classifier = True
        self.tag_classifier_weight = weight
        self._tag_category_names = sorted(category_sizes.keys())

        self.tag_classifier = TrailingTagClassifier(
            encoder_dim, category_sizes, hidden_dim=hidden_dim, dropout=dropout,
        )
        self.tag_classifier.requires_grad_(True)

        def _capture_encoder_output(module, input, output):
            if isinstance(output, tuple):
                self._last_encoder_output = output[0]
            else:
                self._last_encoder_output = output

        self._encoder_hook = self.encoder.register_forward_hook(_capture_encoder_output)

        total_cls_params = sum(p.numel() for p in self.tag_classifier.parameters())
        logging.info(
            "Tag classifier enabled: %d categories %s, %d params, weight=%.2f",
            len(category_sizes), list(category_sizes.keys()), total_cls_params, weight,
        )

    def training_step(self, batch, batch_nb):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        sample_ids = None
        if len(batch) == 5:
            signal, signal_len, transcript, transcript_len, sample_ids = batch
        else:
            signal, signal_len, transcript, transcript_len = batch

        tag_labels = None
        if getattr(self, 'use_tag_classifier', False) and sample_ids is not None:
            tag_labels = self._tag_labels[sample_ids]

        # Encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # RNN-T loss
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        compute_wer = (sample_id + 1) % log_every_n_steps == 0

        if not self.joint.fuse_loss_wer:
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            rnnt_loss = self.loss(
                log_probs=joint, targets=transcript,
                input_lengths=encoded_len, target_lengths=target_length
            )
            rnnt_loss = self.add_auxiliary_losses(rnnt_loss)
        else:
            rnnt_loss, wer, _, _ = self.joint(
                encoder_outputs=encoded, decoder_outputs=decoder,
                encoder_lengths=encoded_len, transcripts=transcript,
                transcript_lengths=transcript_len, compute_wer=compute_wer,
            )
            rnnt_loss = self.add_auxiliary_losses(rnnt_loss)

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        # CTC loss
        loss_value = rnnt_loss
        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript,
                input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_rnnt_loss'] = rnnt_loss
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * rnnt_loss + self.ctc_loss_weight * ctc_loss

        # InterCTC
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )
        tensorboard_logs.update(additional_logs)

        # Tag classifier loss
        if getattr(self, 'use_tag_classifier', False) and tag_labels is not None:
            from scripts.asr.meta_asr.tag_classifier import compute_tag_classification_loss
            with torch.cuda.amp.autocast(enabled=False):
                enc = self._last_encoder_output.float().transpose(1, 2)
                tag_logits = self.tag_classifier(enc, encoded_len)
                tag_loss = compute_tag_classification_loss(
                    tag_logits, tag_labels,
                    class_weights=getattr(self, '_tag_class_weights', None),
                )
            if torch.isfinite(tag_loss):
                tag_loss = tag_loss.clamp(max=10.0)
                loss_value = loss_value + self.tag_classifier_weight * tag_loss
            tensorboard_logs['tag_cls_loss'] = tag_loss

        tensorboard_logs['train_loss'] = loss_value

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        self.log_dict(tensorboard_logs)

        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sample_ids = None
        if isinstance(batch, DALIOutputs):
            signal, signal_len, transcript, transcript_len = batch
        else:
            if len(batch) == 5:
                signal, signal_len, transcript, transcript_len, extra = batch
                if torch.is_tensor(extra) and not extra.dtype.is_floating_point:
                    sample_ids = extra
            else:
                signal, signal_len, transcript, transcript_len = batch

        # Call parent validation_pass for RNN-T + CTC metrics
        core_batch = (signal, signal_len, transcript, transcript_len)
        tensorboard_logs = self.validation_pass(core_batch, batch_idx, dataloader_idx)

        # Tag classifier validation
        if getattr(self, 'use_tag_classifier', False) and sample_ids is not None:
            val_tag_labels = getattr(self, '_val_tag_labels', None)
            if val_tag_labels is not None:
                batch_tag_labels = val_tag_labels[sample_ids]
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    enc = self._last_encoder_output.float().transpose(1, 2)
                    encoded_len = tensorboard_logs.get('encoded_len', None)
                    if encoded_len is None:
                        encoded, encoded_len = self.forward(
                            input_signal=signal, input_signal_length=signal_len
                        )
                    tag_logits = self.tag_classifier(enc, encoded_len)
                    from scripts.asr.meta_asr.tag_classifier import compute_tag_classification_loss
                    tag_loss = compute_tag_classification_loss(
                        tag_logits, batch_tag_labels,
                        class_weights=getattr(self, '_tag_class_weights', None),
                    )
                    tensorboard_logs['val_tag_cls_loss'] = tag_loss
                    self.log('val_tag_cls_loss', tag_loss, prog_bar=True)

                    for cat_idx, cat_name in enumerate(self._tag_category_names):
                        if cat_name in tag_logits:
                            preds = tag_logits[cat_name].argmax(dim=-1)
                            labels = batch_tag_labels[:, cat_idx]
                            valid_mask = labels >= 0
                            correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
                            total = valid_mask.sum().item()
                            self._val_tag_correct[cat_name] += correct
                            self._val_tag_total[cat_name] += total

        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(tensorboard_logs)
        else:
            self.validation_step_outputs.append(tensorboard_logs)

        return tensorboard_logs

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        if getattr(self, 'use_tag_classifier', False):
            self._val_tag_correct = defaultdict(int)
            self._val_tag_total = defaultdict(int)

    def on_validation_epoch_end(self):
        result = super().on_validation_epoch_end()

        if getattr(self, 'use_tag_classifier', False) and getattr(self, '_val_tag_total', None):
            for cat_name in self._tag_category_names:
                total = self._val_tag_total.get(cat_name, 0)
                if total > 0:
                    acc = self._val_tag_correct[cat_name] / total
                    self.log(f'val_tag_acc_{cat_name}', acc, prog_bar=False)

        return result
