"""Custom CTC BPE model with keyword loss and per-family WER tracking."""

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

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.core.classes.mixins import AccessMixin

from promptingnemo.tokenizer.aggregate import _family_name_for_lang


class CustomEncDecCTCModelBPE(EncDecCTCModelBPE):
    def setup_custom_loss(self):
        self.use_keyword_loss = self.cfg.get('use_keyword_loss', False)
        self.keyword_loss_weight = self.cfg.get('keyword_loss_weight', 0.3)
        self.keyword_loss_warmup_steps = self.cfg.get('keyword_loss_warmup_steps', 0)
        self.keyword_token_ids = set()
        self._val_family_stats = defaultdict(lambda: {'errors': 0.0, 'words': 0})
        self._validation_dataset_ref = None

        self.use_family_loss_weights = self.cfg.get('use_family_loss_weights', False)
        self.family_loss_weights = None

        if self.use_family_loss_weights:
            logging.info("Using language family weighted loss. Overriding CTC loss reduction to 'none'.")
            if not hasattr(self, 'loss') or not isinstance(self.loss, CTCLoss):
                self.loss = CTCLoss(
                    num_classes=self.decoder.num_classes, zero_infinity=True, reduction='mean_batch'
                )

            # Re-create loss with reduction='none' to get per-sample losses
            self.loss = CTCLoss(
                num_classes=self.decoder.num_classes,
                zero_infinity=self.loss.ctc_loss.zero_infinity,
                reduction='none'
            )

    def set_keyword_token_ids(self, keyword_ids):
        self.keyword_token_ids = set(keyword_ids)
        logging.info(f"Set {len(self.keyword_token_ids)} keyword token IDs for custom loss.")

    def set_family_loss_weights(self, weights: Dict[str, float]):
        """Stores the pre-computed language family weights."""
        if self.use_family_loss_weights:
            self.family_loss_weights = weights
            logging.info("Successfully set language family loss weights on the model.")

    def training_step(self, batch, batch_idx):
        sample_ids = None
        if len(batch) == 5:  # audio, audio_len, transcript, transcript_len, sample_id
            audio_signal, audio_signal_len, transcript, transcript_len, sample_ids = batch
        else:
            audio_signal, audio_signal_len, transcript, transcript_len = batch

        log_probs, encoded_len, greedy_predictions = self.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        # --- Language Family Weighted Loss ---
        if self.use_family_loss_weights:
            if sample_ids is not None and self.family_loss_weights:
                # `loss_value` is unreduced here (per-sample)
                dataset = self._train_dl.dataset
                batch_weights_list = [
                    self.family_loss_weights.get(_family_name_for_lang(dataset.language_ids[idx.item()]), 1.0)
                    for idx in sample_ids
                ]
                weights_tensor = torch.tensor(
                    batch_weights_list, device=loss_value.device, dtype=loss_value.dtype
                )

                # Apply weights and then compute the mean
                loss_value = (loss_value * weights_tensor).mean()
                self.log('family_weighted_loss', loss_value, on_step=True, on_epoch=False, prog_bar=True)
            else:
                # Fallback: if weights or sample_ids are missing, just take the mean
                loss_value = loss_value.mean()

        # --- Keyword Loss (applied after family weighting) ---
        if self.use_keyword_loss and self.training and len(self.keyword_token_ids) > 0:
            current_step = self.trainer.global_step
            if self.keyword_loss_warmup_steps > 0 and current_step < self.keyword_loss_warmup_steps:
                current_keyword_loss_weight = (
                    current_step / self.keyword_loss_warmup_steps
                ) * self.keyword_loss_weight
            else:
                current_keyword_loss_weight = self.keyword_loss_weight

            self.log('current_keyword_loss_weight', current_keyword_loss_weight, on_step=True, on_epoch=False, prog_bar=True)

            keyword_targets = []
            keyword_target_lengths = []
            for i in range(transcript.size(0)):
                target = transcript[i][: transcript_len[i]]
                keyword_target = [
                    token_id.item() for token_id in target if token_id.item() in self.keyword_token_ids
                ]

                if len(keyword_target) > 0:
                    keyword_targets.append(
                        torch.tensor(keyword_target, dtype=torch.long, device=transcript.device)
                    )
                    keyword_target_lengths.append(len(keyword_target))

            if keyword_targets:
                keyword_targets_padded = torch.nn.utils.rnn.pad_sequence(
                    keyword_targets, batch_first=True, padding_value=self.tokenizer.pad_id
                )
                keyword_target_lengths_tensor = torch.tensor(
                    keyword_target_lengths, dtype=torch.long, device=transcript.device
                )

                if keyword_targets_padded.numel() > 0:
                    keyword_loss = self.loss(
                        log_probs=log_probs,
                        targets=keyword_targets_padded,
                        input_lengths=encoded_len,
                        target_lengths=keyword_target_lengths_tensor,
                    )
                    loss_value = (
                        1 - current_keyword_loss_weight
                    ) * loss_value + current_keyword_loss_weight * keyword_loss

        self.log('train_loss', loss_value)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])

        return {'loss': loss_value}

    def _get_dataset_for_prefix(self, prefix: str):
        if prefix == 'val':
            dataset = getattr(self, '_validation_dataset_ref', None)
            if dataset is None:
                data_loader = getattr(self, '_validation_dl', None)
                dataset = getattr(data_loader, 'dataset', None) if data_loader is not None else None
                self._validation_dataset_ref = dataset
            return dataset
        return None

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_family_stats = defaultdict(lambda: {'errors': 0.0, 'words': 0})

    def _accumulate_family_wer(self, sample_ids, log_probs, encoded_len, transcript, transcript_len):
        dataset = self._get_dataset_for_prefix('val')
        if dataset is None or not hasattr(dataset, 'language_ids') or sample_ids is None:
            return

        if isinstance(sample_ids, torch.Tensor):
            sample_indices = sample_ids.detach().cpu().view(-1).tolist()
        else:
            sample_indices = [int(idx) for idx in sample_ids]

        if not sample_indices:
            return

        with torch.no_grad():
            try:
                hypotheses = self.decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=log_probs.detach(),
                    decoder_lengths=encoded_len.detach(),
                    return_hypotheses=False,
                )
            except RuntimeError:
                hypotheses = self.decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=log_probs.detach().cpu(),
                    decoder_lengths=encoded_len.detach().cpu(),
                    return_hypotheses=False,
                )

            targets_cpu = transcript.detach().cpu()
            target_lens_cpu = transcript_len.detach().cpu()

            limit = min(len(sample_indices), len(hypotheses), targets_cpu.size(0))

            for idx in range(limit):
                sample_index = sample_indices[idx]
                if sample_index >= len(dataset.language_ids):
                    continue
                lang_code = dataset.language_ids[sample_index]
                family = _family_name_for_lang(lang_code)
                hyp_obj = hypotheses[idx]
                hyp_text = hyp_obj.text if hasattr(hyp_obj, 'text') else str(hyp_obj)
                target_tokens = targets_cpu[idx][: target_lens_cpu[idx]].tolist()
                reference_text = self._decode_target_tokens(target_tokens)
                wer_value, words, *_ = word_error_rate_detail(
                    [hyp_text], [reference_text], use_cer=self.wer.use_cer
                )
                if not math.isfinite(wer_value) or words <= 0:
                    continue
                errors = wer_value * words
                stats = self._val_family_stats[family]
                stats['errors'] += float(errors)
                stats['words'] += int(round(words))

    def _log_family_metrics(self, prefix: str):
        if not getattr(self, '_val_family_stats', None):
            return

        aggregated = {}
        device = self.device if isinstance(self.device, torch.device) else torch.device('cpu')

        for family, stats in self._val_family_stats.items():
            tensor = torch.tensor([stats['errors'], stats['words']], dtype=torch.float32, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            aggregated[family] = tensor.cpu()

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        total_errors = 0.0
        total_words = 0.0
        for family, tensor in aggregated.items():
            errors = float(tensor[0].item())
            words = float(tensor[1].item())
            if words <= 0:
                continue
            total_errors += errors
            total_words += words
            self.log(f"{prefix}_wer_{family}", errors / words, prog_bar=False, sync_dist=False)

        if total_words > 0:
            self.log(f"{prefix}_wer_combined", total_errors / total_words, prog_bar=False, sync_dist=False)

    def on_validation_epoch_end(self):
        result = super().on_validation_epoch_end()
        self._log_family_metrics('val')
        self._val_family_stats = defaultdict(lambda: {'errors': 0.0, 'words': 0})
        return result

    def _decode_target_tokens(self, token_ids: List[int]) -> str:
        decoding = getattr(self, 'decoding', None)
        if decoding is not None:
            if hasattr(decoding, 'decode_ids_to_str'):
                return decoding.decode_ids_to_str(token_ids)
            tokens = None
            if hasattr(decoding, 'decode_ids_to_tokens'):
                tokens = decoding.decode_ids_to_tokens(token_ids)
            if tokens is not None:
                if tokens and isinstance(tokens[0], str):
                    return ''.join(tokens).replace('\u2581', ' ').strip()
                if hasattr(decoding, 'decode_tokens_to_str'):
                    return decoding.decode_tokens_to_str(tokens)
                return ''.join(tokens).replace('\u2581', ' ').strip()
        tokenizer = getattr(self, 'tokenizer', None)
        if tokenizer is not None:
            if hasattr(tokenizer, 'ids_to_text'):
                return tokenizer.ids_to_text(token_ids)
            if hasattr(tokenizer, 'ids_to_tokens'):
                tokens = tokenizer.ids_to_tokens(token_ids)
                return ''.join(tokens).replace('\u2581', ' ').strip()
        return ' '.join(str(t) for t in token_ids)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sample_ids = None

        if isinstance(batch, DALIOutputs):
            signal, signal_len, transcript, transcript_len = batch
            core_batch = batch
        else:
            if len(batch) == 6:
                signal, signal_len, transcript, transcript_len, _, sample_ids = batch
            elif len(batch) == 5:
                signal, signal_len, transcript, transcript_len, extra = batch
                if torch.is_tensor(extra):
                    if extra.dtype.is_floating_point:
                        sample_ids = None
                    else:
                        sample_ids = extra
                elif isinstance(extra, (float, np.floating)):
                    sample_ids = None
                else:
                    sample_ids = extra
            else:
                signal, signal_len, transcript, transcript_len = batch
            core_batch = (signal, signal_len, transcript, transcript_len)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        if isinstance(core_batch, DALIOutputs) and core_batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(
                input_signal=signal, input_signal_length=signal_len
            )

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        loss_value, metrics = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            log_wer_num_denom=True,
            log_prefix="val_",
        )

        self.wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        self.wer.reset()
        metrics.update({'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom, 'val_wer': wer})

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32, device=log_probs.device))

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if sample_ids is not None:
            self._accumulate_family_wer(sample_ids, log_probs, encoded_len, transcript, transcript_len)

        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)

        return metrics
