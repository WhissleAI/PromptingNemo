"""Text CTC Tagger v3: XLM-RoBERTa encoder (frozen) → upsample → CTC.

Uses a pretrained multilingual encoder for rich text representations,
with CTC output over the STT-meta-1B aggregate vocabulary.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import AutoModel

log = logging.getLogger(__name__)

TAG_PREFIXES = (
    'ENTITY_', 'INTENT_', 'EMOTION_', 'GENDER_', 'AGE_',
    'DIALECT_', 'KEYWORD_', 'LANG_', 'OTHER_', 'ROLE_',
    'SPEAKER_', 'TURN_', 'FAMILY_',
)
EXACT_TAG_TOKENS = {'END', 'TURN_CHANGE'}


def _is_tag(word: str) -> bool:
    if word in EXACT_TAG_TOKENS:
        return True
    return any(word.startswith(p) for p in TAG_PREFIXES)


class LearnedUpsampler(nn.Module):
    def __init__(self, hidden_dim: int, factor: int = 3):
        super().__init__()
        self.factor = factor
        self.upsample = nn.ConvTranspose1d(
            hidden_dim, hidden_dim,
            kernel_size=factor, stride=factor,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        return x, lengths * self.factor


class TextCTCTaggerV3(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        encoder_name = cfg.get('encoder_name', 'xlm-roberta-base')
        output_vocab_size = cfg['output_vocab_size']
        upsample_factor = cfg.get('upsample_factor', 3)
        freeze_encoder = cfg.get('freeze_encoder', True)
        unfreeze_top_n = cfg.get('unfreeze_top_n', 0)

        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if unfreeze_top_n > 0:
                layers = self.encoder.encoder.layer[-unfreeze_top_n:]
                for layer in layers:
                    for param in layer.parameters():
                        param.requires_grad = True
                log.info("Unfroze top %d encoder layers", unfreeze_top_n)

        self.proj = nn.Linear(encoder_dim, cfg.get('hidden_dim', 512))
        self.proj_norm = nn.LayerNorm(cfg.get('hidden_dim', 512))
        self.upsampler = LearnedUpsampler(cfg.get('hidden_dim', 512), upsample_factor)
        self.decoder_proj = nn.Linear(cfg.get('hidden_dim', 512), output_vocab_size + 1)
        self.blank_id = output_vocab_size
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, reduction='mean', zero_infinity=True)

        self.output_tokenizer = None
        self._val_wer_errors = 0
        self._val_wer_words = 0
        self._val_tag_tp = 0
        self._val_tag_fp = 0
        self._val_tag_fn = 0

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info("TextCTCTaggerV3: %d total params, %d trainable", total, trainable)

    def set_output_tokenizer(self, tokenizer):
        self.output_tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        with torch.set_grad_enabled(any(p.requires_grad for p in self.encoder.parameters())):
            enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state
        hidden = self.proj_norm(F.gelu(self.proj(hidden)))
        lengths = attention_mask.sum(dim=1)
        upsampled, up_lengths = self.upsampler(hidden, lengths)
        logits = self.decoder_proj(upsampled)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, up_lengths

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, target_ids, target_lengths = batch
        log_probs, enc_lengths = self.forward(input_ids, attention_mask)
        log_probs_t = log_probs.permute(1, 0, 2)
        loss = self.ctc_loss(log_probs_t, target_ids, enc_lengths, target_lengths)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, target_ids, target_lengths = batch
        log_probs, enc_lengths = self.forward(input_ids, attention_mask)
        log_probs_t = log_probs.permute(1, 0, 2)
        loss = self.ctc_loss(log_probs_t, target_ids, enc_lengths, target_lengths)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        if self.output_tokenizer is not None:
            greedy_preds = self._greedy_decode(log_probs, enc_lengths)
            for i in range(len(greedy_preds)):
                tgt_ids = target_ids[i, :target_lengths[i]].tolist()
                pred_text = self.output_tokenizer.decode(greedy_preds[i])
                ref_text = self.output_tokenizer.decode(tgt_ids)
                pred_words = [w for w in pred_text.split() if not _is_tag(w)]
                ref_words = [w for w in ref_text.split() if not _is_tag(w)]
                self._val_wer_errors += self._edit_distance(pred_words, ref_words)
                self._val_wer_words += max(len(ref_words), 1)
                pred_tags = set(w for w in pred_text.split() if _is_tag(w))
                ref_tags = set(w for w in ref_text.split() if _is_tag(w))
                self._val_tag_tp += len(pred_tags & ref_tags)
                self._val_tag_fp += len(pred_tags - ref_tags)
                self._val_tag_fn += len(ref_tags - pred_tags)
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        if self._val_wer_words > 0:
            wer = self._val_wer_errors / self._val_wer_words
            self.log('val_wer', wer, prog_bar=True)
            tp = self._val_tag_tp
            prec = tp / max(tp + self._val_tag_fp, 1)
            rec = tp / max(tp + self._val_tag_fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            self.log('val_tag_f1', f1, prog_bar=True)
            log.info(
                "Validation @ step %d: val_wer=%.4f, tag_f1=%.4f (P=%.4f R=%.4f)",
                self.trainer.global_step, wer, f1, prec, rec,
            )
        self._val_wer_errors = 0
        self._val_wer_words = 0
        self._val_tag_tp = 0
        self._val_tag_fp = 0
        self._val_tag_fn = 0

    def _greedy_decode(self, log_probs, lengths):
        predictions = log_probs.argmax(dim=-1)
        results = []
        for i in range(predictions.shape[0]):
            seq = predictions[i, :lengths[i]].tolist()
            decoded = []
            prev = self.blank_id
            for tok in seq:
                if tok != self.blank_id and tok != prev:
                    decoded.append(tok)
                prev = tok
            results.append(decoded)
        return results

    @staticmethod
    def _edit_distance(hyp, ref):
        n, m = len(hyp), len(ref)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, m + 1):
                cost = 0 if hyp[i - 1] == ref[j - 1] else 1
                prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
        return dp[m]

    def configure_optimizers(self):
        lr = self.cfg.get('lr', 5e-4)
        weight_decay = self.cfg.get('weight_decay', 0.01)
        warmup_steps = self.cfg.get('warmup_steps', 2000)
        max_steps = self.cfg.get('max_steps', 200000)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1},
        }
