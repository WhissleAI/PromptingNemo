"""Knowledge-distillation CTC model: student trained against a frozen teacher.

The student sees noisy audio while the teacher processes clean audio.
Multi-objective loss combines CTC, KL-divergence (logit-level KD),
hidden-state MSE matching, and the existing tag classifier head.
Loss weights anneal over training to shift from teacher mimicry to
independent learning.
"""

import logging
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE
from promptingnemo.tokenizer.aggregate import _family_name_for_lang


class HiddenStateProjector(nn.Module):
    """Projects student hidden states to teacher dimensionality for MSE matching."""

    def __init__(self, student_dim: int, teacher_dim: int, num_layers: int):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(student_dim, teacher_dim) for _ in range(num_layers)
        ])

    def forward(self, student_hiddens: List[torch.Tensor], layer_indices: List[int]) -> List[torch.Tensor]:
        projected = []
        for i, idx in enumerate(layer_indices):
            projected.append(self.projectors[i](student_hiddens[idx]))
        return projected


def _add_white_noise(audio: torch.Tensor, snr_low: float = 5.0, snr_high: float = 30.0) -> torch.Tensor:
    """Add white noise to audio at a random SNR in [snr_low, snr_high] dB."""
    snr_db = random.uniform(snr_low, snr_high)
    signal_power = audio.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-10)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(audio) * noise_power.sqrt()
    return audio + noise


class DistillCTCModel(CustomEncDecCTCModelBPE):
    """Student CTC model with knowledge distillation from a frozen teacher.

    Configured via a `distillation` section in the OmegaConf config:
        distillation:
            temperature: 2.0
            alpha_ctc: 0.3
            beta_kd_logit: 0.4
            gamma_hidden: 0.2
            delta_tag: 0.1
            anneal_start_step: 100000
            anneal_end_step: 150000
            student_match_layers: [2, 5, 8, 11]
            teacher_match_layers: [4, 10, 16, 22]
            noise_snr_low: 5.0
            noise_snr_high: 30.0
    """

    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.teacher = None
        self.encoder_proj = None
        self._projector = None
        self._teacher_hooks = []
        self._student_hooks = []
        self._teacher_hiddens = {}
        self._student_hiddens = {}

    def setup_tag_classifier(self, encoder_dim, vocabulary, categories=None, weight=0.5,
                             special_token_prefixes=None):
        """Set up the trailing tag classification head."""
        from scripts.asr.meta_asr.tag_classifier import (
            TrailingTagClassifier, build_trailing_tag_maps, build_all_special_token_ids,
            build_sp_boundary_ids,
        )
        self.use_tag_classifier = True
        self.tag_classifier_weight = weight

        trailing_tag_ids, category_to_id, category_sizes = build_trailing_tag_maps(
            vocabulary, categories=categories
        )
        self._trailing_tag_ids = trailing_tag_ids
        self._category_to_id = category_to_id
        self._category_names = sorted(category_sizes.keys())
        self._category_sizes = category_sizes
        self._sp_boundary_ids = build_sp_boundary_ids(vocabulary)

        if special_token_prefixes:
            self._all_special_token_ids = build_all_special_token_ids(
                vocabulary, special_token_prefixes
            )
        else:
            self._all_special_token_ids = trailing_tag_ids

        self.tag_classifier = TrailingTagClassifier(encoder_dim, category_sizes)
        self.tag_classifier.requires_grad_(True)

        def _capture_encoder_output(module, input, output):
            if isinstance(output, tuple):
                self._last_encoder_output = output[0]
            else:
                self._last_encoder_output = output

        self._encoder_hook = self.encoder.register_forward_hook(_capture_encoder_output)
        self._val_tag_preds = {cat: [] for cat in self._category_names}
        self._val_tag_labels = {cat: [] for cat in self._category_names}

        from nemo.utils import logging as nemo_logging
        total_cls_params = sum(p.numel() for p in self.tag_classifier.parameters())
        nemo_logging.info(
            "Tag classifier enabled: %d categories %s, %d total params, weight=%.2f",
            len(category_sizes), list(category_sizes.keys()), total_cls_params, weight,
        )

    def setup_distillation(self, teacher: CustomEncDecCTCModelBPE, distill_cfg):
        """Attach frozen teacher and configure distillation losses."""
        # Store teacher outside nn.Module system so its weights are excluded
        # from state_dict() — avoids checkpoint save/load mismatches.
        object.__setattr__(self, 'teacher', teacher)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.temperature = distill_cfg.get('temperature', 2.0)
        self.alpha_ctc = distill_cfg.get('alpha_ctc', 0.3)
        self.beta_kd = distill_cfg.get('beta_kd_logit', 0.4)
        self.gamma_hidden = distill_cfg.get('gamma_hidden', 0.2)
        self.delta_tag = distill_cfg.get('delta_tag', 0.1)
        self.anneal_start = distill_cfg.get('anneal_start_step', 100000)
        self.anneal_end = distill_cfg.get('anneal_end_step', 150000)
        self.noise_snr_low = distill_cfg.get('noise_snr_low', 5.0)
        self.noise_snr_high = distill_cfg.get('noise_snr_high', 30.0)
        self.tag_grad_through_encoder = distill_cfg.get('tag_grad_through_encoder', False)

        self.student_match_layers = list(distill_cfg.get('student_match_layers', [2, 5, 8, 11]))
        self.teacher_match_layers = list(distill_cfg.get('teacher_match_layers', [4, 10, 16, 22]))

        student_dim = self.encoder._feat_out
        teacher_dim = self.teacher.encoder._feat_out

        if self.gamma_hidden > 0 and self.student_match_layers:
            self._projector = HiddenStateProjector(
                student_dim, teacher_dim, len(self.student_match_layers)
            )
            self._register_hidden_hooks()

        logging.info(
            "Distillation configured: T=%.1f, alpha=%.2f, beta=%.2f, gamma=%.2f, delta=%.2f, "
            "noise SNR=[%.0f, %.0f]dB, anneal steps=[%d, %d]",
            self.temperature, self.alpha_ctc, self.beta_kd, self.gamma_hidden, self.delta_tag,
            self.noise_snr_low, self.noise_snr_high, self.anneal_start, self.anneal_end,
        )

    def forward(self, input_signal=None, input_signal_length=None,
                processed_signal=None, processed_signal_length=None):
        has_input_signal = input_signal is not None
        if has_input_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )
        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )
        if self.encoder_proj is not None:
            encoded = self.encoder_proj(encoded)
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, encoded_len, greedy_predictions

    def _register_hidden_hooks(self):
        """Register forward hooks to capture intermediate hidden states."""
        student_layers = self._get_layers(self.encoder)
        teacher_layers = self._get_layers(self.teacher.encoder)

        for idx in self.student_match_layers:
            if idx < len(student_layers):
                hook = student_layers[idx].register_forward_hook(
                    self._make_capture_hook(self._student_hiddens, idx)
                )
                self._student_hooks.append(hook)

        for idx in self.teacher_match_layers:
            if idx < len(teacher_layers):
                hook = teacher_layers[idx].register_forward_hook(
                    self._make_capture_hook(self._teacher_hiddens, idx)
                )
                self._teacher_hooks.append(hook)

    @staticmethod
    def _get_layers(encoder) -> list:
        for attr in ['layers', 'conformer_layers']:
            candidate = getattr(encoder, attr, None)
            if candidate is not None and hasattr(candidate, '__len__'):
                return list(candidate)
        if hasattr(encoder, '_modules'):
            for key, module in encoder._modules.items():
                if hasattr(module, '__len__') and len(module) > 4:
                    return list(module)
        return []

    @staticmethod
    def _make_capture_hook(storage: dict, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[layer_idx] = output[0].detach() if not storage.get('_keep_grad') else output[0]
            else:
                storage[layer_idx] = output.detach() if not storage.get('_keep_grad') else output
        return hook

    def _get_annealed_weights(self, step: int) -> Tuple[float, float]:
        """Linearly anneal beta_kd down and alpha_ctc up."""
        if step < self.anneal_start:
            return self.alpha_ctc, self.beta_kd

        if step >= self.anneal_end:
            return self.alpha_ctc + self.beta_kd - 0.1, 0.1

        progress = (step - self.anneal_start) / max(self.anneal_end - self.anneal_start, 1)
        target_beta = 0.1
        target_alpha = self.alpha_ctc + (self.beta_kd - target_beta)
        alpha = self.alpha_ctc + progress * (target_alpha - self.alpha_ctc)
        beta = self.beta_kd + progress * (target_beta - self.beta_kd)
        return alpha, beta

    def _compute_kd_loss(self, student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor,
                          student_lengths: torch.Tensor, teacher_lengths: torch.Tensor) -> torch.Tensor:
        """Frame-level KL divergence at temperature T.

        Inputs are log-probabilities (decoder already applies log_softmax).
        We exponentiate to recover logits (shift-invariant), then apply temperature.
        """
        T = self.temperature
        min_len = min(student_log_probs.size(1), teacher_log_probs.size(1))
        s_lp = student_log_probs[:, :min_len, :]
        t_lp = teacher_log_probs[:, :min_len, :]
        # Align vocab dimension (student may have extra tokens from extension)
        min_vocab = min(s_lp.size(-1), t_lp.size(-1))
        s_lp = s_lp[:, :, :min_vocab]
        t_lp = t_lp[:, :, :min_vocab]
        # log_probs -> probs, then apply temperature
        s_probs = s_lp.exp()
        t_probs = t_lp.exp()
        # Apply temperature: raise probs to power 1/T and renormalize
        s_tempered = (s_probs ** (1.0 / T))
        s_tempered = s_tempered / s_tempered.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        t_tempered = (t_probs ** (1.0 / T))
        t_tempered = t_tempered / t_tempered.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        s_log = torch.log(s_tempered.clamp(min=1e-10))
        t_soft = t_tempered

        # Mask padded frames
        B = student_log_probs.size(0)
        max_len = min_len
        mask = torch.arange(max_len, device=student_log_probs.device).unsqueeze(0) < torch.minimum(
            student_lengths, teacher_lengths
        ).unsqueeze(1)
        mask = mask.unsqueeze(-1)

        kl = F.kl_div(s_log, t_soft, reduction='none') * mask
        return kl.sum() / mask.sum().clamp(min=1) * (T ** 2)

    def _compute_hidden_loss(self) -> torch.Tensor:
        """MSE between projected student and teacher hidden states."""
        if not self._projector or not self._student_hiddens or not self._teacher_hiddens:
            return torch.tensor(0.0, device=self.device)

        student_states = []
        for idx in self.student_match_layers:
            h = self._student_hiddens.get(idx)
            if h is not None:
                if h.dim() == 3:
                    student_states.append(h)
                else:
                    student_states.append(h.transpose(1, 2))

        teacher_states = []
        for idx in self.teacher_match_layers:
            h = self._teacher_hiddens.get(idx)
            if h is not None:
                if h.dim() == 3:
                    teacher_states.append(h)
                else:
                    teacher_states.append(h.transpose(1, 2))

        if not student_states or not teacher_states:
            return torch.tensor(0.0, device=self.device)

        n_pairs = min(len(student_states), len(teacher_states), len(self._projector.projectors))
        projected = self._projector(student_states[:n_pairs], list(range(n_pairs)))

        loss = torch.tensor(0.0, device=self.device)
        for proj, teacher_h in zip(projected, teacher_states[:n_pairs]):
            min_t = min(proj.size(1), teacher_h.size(1))
            loss = loss + F.mse_loss(proj[:, :min_t, :], teacher_h[:, :min_t, :])

        return loss / max(n_pairs, 1)

    def training_step(self, batch, batch_idx):
        sample_ids = None
        if len(batch) == 5:
            audio_signal, audio_signal_len, transcript, transcript_len, sample_ids = batch
        else:
            audio_signal, audio_signal_len, transcript, transcript_len = batch

        # --- Dual-head: strip trailing tags ---
        if getattr(self, 'use_tag_classifier', False):
            from scripts.asr.meta_asr.tag_classifier import strip_trailing_tags_and_get_labels, \
                masked_mean_pool, compute_tag_classification_loss
            ctc_transcript, ctc_transcript_len, tag_labels = strip_trailing_tags_and_get_labels(
                transcript, transcript_len,
                self._trailing_tag_ids, self._category_to_id, self._category_names,
                all_special_ids=self._all_special_token_ids,
                sp_boundary_ids=getattr(self, '_sp_boundary_ids', None),
            )
        else:
            ctc_transcript = transcript
            ctc_transcript_len = transcript_len
            tag_labels = None

        # --- Teacher forward (clean audio, no grad) ---
        self._teacher_hiddens.clear()
        with torch.no_grad():
            teacher_log_probs, teacher_encoded_len, _ = self.teacher.forward(
                input_signal=audio_signal, input_signal_length=audio_signal_len
            )

        # --- Student forward (noisy audio) ---
        noisy_audio = _add_white_noise(audio_signal, self.noise_snr_low, self.noise_snr_high)
        self._student_hiddens['_keep_grad'] = True
        self._student_hiddens.clear()
        self._student_hiddens['_keep_grad'] = True

        student_log_probs, student_encoded_len, greedy_predictions = self.forward(
            input_signal=noisy_audio, input_signal_length=audio_signal_len
        )

        # --- Loss 1: CTC on hard labels ---
        ctc_loss = self.loss(
            log_probs=student_log_probs, targets=ctc_transcript,
            input_lengths=student_encoded_len, target_lengths=ctc_transcript_len,
        )
        if ctc_loss.dim() > 0:
            ctc_loss = ctc_loss.mean()

        # --- Loss 2: KL divergence (logit-level KD) ---
        kd_loss = self._compute_kd_loss(
            student_log_probs, teacher_log_probs,
            student_encoded_len, teacher_encoded_len,
        )

        # --- Loss 3: Hidden state matching ---
        hidden_loss = self._compute_hidden_loss()

        # --- Loss 4: Tag classifier ---
        tag_loss = torch.tensor(0.0, device=self.device)
        if getattr(self, 'use_tag_classifier', False) and tag_labels is not None:
            enc = self._last_encoder_output  # [B, D, T]
            B, D, T = enc.shape
            time_mask = torch.arange(T, device=enc.device).unsqueeze(0) < student_encoded_len.unsqueeze(1)
            time_mask = time_mask.unsqueeze(1).float()  # [B, 1, T]
            pooled = (enc * time_mask).sum(dim=2) / time_mask.sum(dim=2).clamp(min=1)  # [B, D]
            if not getattr(self, 'tag_grad_through_encoder', False):
                pooled = pooled.detach()
            tag_logits = self.tag_classifier(pooled)
            tag_loss = compute_tag_classification_loss(tag_logits, tag_labels)

        # --- Combine with progressive annealing ---
        step = self.trainer.global_step
        alpha, beta = self._get_annealed_weights(step)

        total_loss = (
            alpha * ctc_loss
            + beta * kd_loss
            + self.gamma_hidden * hidden_loss
            + self.delta_tag * tag_loss
        )

        # --- Logging ---
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('ctc_loss', ctc_loss, on_step=True)
        self.log('kd_loss', kd_loss, on_step=True)
        self.log('hidden_loss', hidden_loss, on_step=True)
        self.log('alpha_ctc', alpha, on_step=True)
        self.log('beta_kd', beta, on_step=True)
        if tag_loss.item() > 0:
            self.log('tag_cls_loss', tag_loss, on_step=True, prog_bar=True)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])

        self._student_hiddens.clear()
        self._teacher_hiddens.clear()

        return {'loss': total_loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._student_hiddens.clear()
        self._teacher_hiddens.clear()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_family_cer_stats = defaultdict(lambda: {'errors': 0, 'chars': 0})

    def on_validation_epoch_end(self):
        result = super().on_validation_epoch_end()
        self._log_family_cer()
        self._val_family_cer_stats = defaultdict(lambda: {'errors': 0, 'chars': 0})
        return result

    def _log_family_cer(self):
        stats = getattr(self, '_val_family_cer_stats', None)
        if not stats:
            return
        total_errors = 0
        total_chars = 0
        for family, s in stats.items():
            if s['chars'] > 0:
                cer = s['errors'] / s['chars']
                self.log(f'val_cer_{family}', cer, prog_bar=False, sync_dist=False)
                total_errors += s['errors']
                total_chars += s['chars']
        if total_chars > 0:
            self.log('val_cer_combined', total_errors / total_chars, prog_bar=False, sync_dist=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Override to strip trailing tags before WER and compute extra metrics."""
        sample_ids = None
        if len(batch) == 5:
            signal, signal_len, transcript, transcript_len, sample_ids = batch
        elif len(batch) == 6:
            signal, signal_len, transcript, transcript_len, _, sample_ids = batch
        else:
            signal, signal_len, transcript, transcript_len = batch

        # Strip trailing tags for CTC loss, strip ALL specials for WER
        if getattr(self, 'use_tag_classifier', False) and hasattr(self, '_trailing_tag_ids'):
            from scripts.asr.meta_asr.tag_classifier import (
                strip_trailing_tags_and_get_labels, strip_all_special_from_targets,
            )
            ctc_transcript, ctc_transcript_len, _ = strip_trailing_tags_and_get_labels(
                transcript, transcript_len,
                self._trailing_tag_ids, self._category_to_id, self._category_names,
                all_special_ids=self._all_special_token_ids,
                sp_boundary_ids=getattr(self, '_sp_boundary_ids', None),
            )
            wer_transcript, wer_transcript_len = strip_all_special_from_targets(
                ctc_transcript, ctc_transcript_len,
                self._all_special_token_ids,
                sp_boundary_ids=getattr(self, '_sp_boundary_ids', None),
            )
            if batch_idx == 0:
                logging.info(
                    "val strip: orig=%s trailing_stripped=%s wer_stripped=%s",
                    transcript_len[:2].tolist(), ctc_transcript_len[:2].tolist(),
                    wer_transcript_len[:2].tolist(),
                )
        else:
            ctc_transcript, ctc_transcript_len = transcript, transcript_len
            wer_transcript, wer_transcript_len = transcript, transcript_len

        # Pass fully-stripped targets for WER computation in super
        stripped_batch = (signal, signal_len, wer_transcript, wer_transcript_len)
        if sample_ids is not None:
            stripped_batch = (signal, signal_len, wer_transcript, wer_transcript_len, sample_ids)

        metrics = super().validation_step(stripped_batch, batch_idx, dataloader_idx)

        # --- Extra metrics: per-sample CER with per-family breakdown ---
        with torch.no_grad():
            log_probs, encoded_len, greedy_preds = self.forward(
                input_signal=signal, input_signal_length=signal_len
            )

        vocab = list(self.decoder.vocabulary)
        blank_id = len(vocab)
        batch_pred_lens = []
        batch_ref_lens = []
        batch_cer_errors = 0
        batch_cer_chars = 0

        dataset = self._get_dataset_for_prefix('val')
        has_lang = (dataset is not None and hasattr(dataset, 'language_ids')
                    and sample_ids is not None)
        if has_lang:
            if isinstance(sample_ids, torch.Tensor):
                sample_indices = sample_ids.detach().cpu().view(-1).tolist()
            else:
                sample_indices = [int(idx) for idx in sample_ids]
        else:
            sample_indices = None

        for b in range(signal.size(0)):
            pred_ids = greedy_preds[b, :encoded_len[b]].cpu().tolist()
            decoded = []
            prev = None
            for tok_id in pred_ids:
                if tok_id != blank_id and tok_id != prev and tok_id < len(vocab):
                    decoded.append(vocab[tok_id])
                prev = tok_id
            pred_text = ''.join(decoded).replace('▁', ' ').strip()

            ref_ids = wer_transcript[b, :wer_transcript_len[b]].cpu().tolist()
            ref_tokens = [vocab[t] for t in ref_ids if t < len(vocab)]
            ref_text = ''.join(ref_tokens).replace('▁', ' ').strip()

            batch_pred_lens.append(len(pred_text.split()))
            batch_ref_lens.append(len(ref_text.split()))

            pred_chars = list(pred_text.replace(' ', ''))
            ref_chars = list(ref_text.replace(' ', ''))
            n_ref = len(ref_chars)
            r, h = n_ref, len(pred_chars)
            if r == 0:
                edit_dist = h
            else:
                d = list(range(h + 1))
                for i in range(1, r + 1):
                    nd = [i] + [0] * h
                    for j in range(1, h + 1):
                        cost = 0 if ref_chars[i-1] == pred_chars[j-1] else 1
                        nd[j] = min(nd[j-1] + 1, d[j] + 1, d[j-1] + cost)
                    d = nd
                edit_dist = d[h]

            batch_cer_errors += edit_dist
            batch_cer_chars += n_ref

            if sample_indices is not None and b < len(sample_indices):
                si = sample_indices[b]
                if si < len(dataset.language_ids):
                    family = _family_name_for_lang(dataset.language_ids[si])
                    self._val_family_cer_stats[family]['errors'] += edit_dist
                    self._val_family_cer_stats[family]['chars'] += n_ref

        avg_pred_len = sum(batch_pred_lens) / max(len(batch_pred_lens), 1)
        avg_ref_len = sum(batch_ref_lens) / max(len(batch_ref_lens), 1)
        len_ratio = avg_pred_len / max(avg_ref_len, 1)
        cer = batch_cer_errors / max(batch_cer_chars, 1)

        self.log('val_cer', cer, prog_bar=True, sync_dist=False)
        self.log('val_pred_len', avg_pred_len, sync_dist=False)
        self.log('val_ref_len', avg_ref_len, sync_dist=False)
        self.log('val_len_ratio', len_ratio, prog_bar=True, sync_dist=False)

        return metrics

