"""Text CTC Tagger: character input -> tagged text output via CTC loss.

Architecture:
  CharacterEmbedding -> LearnedUpsampler -> TransformerEncoder -> Linear -> CTC

This mirrors the meta-ASR streaming architecture but replaces audio frames
with character embeddings. Because characters outnumber output subword+tag
tokens, the CTC alignment constraint (input_len >= output_len) is satisfied.
Streaming inference uses causal attention masks, same as streaming meta-ASR.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

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


class CharacterEmbedding(nn.Module):
    """Character-level embedding with learned positional encoding."""

    def __init__(self, char_vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.projection = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: (B, T_chars) character indices
        Returns:
            (B, T_chars, hidden_dim)
        """
        B, T = char_ids.shape
        positions = torch.arange(T, device=char_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(char_ids) + self.pos_encoding(positions)
        x = self.projection(x)
        x = self.layer_norm(x)
        return self.dropout(x)


class LearnedUpsampler(nn.Module):
    """Learnable upsampling via transposed 1D convolution.

    Ensures CTC constraint: upsampled_len = char_len * factor >= target_len.
    """

    def __init__(self, hidden_dim: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.upsample = nn.ConvTranspose1d(
            hidden_dim, hidden_dim,
            kernel_size=factor, stride=factor,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C)
            lengths: (B,) original lengths
        Returns:
            (B, T*factor, C), new_lengths
        """
        # (B, T, C) -> (B, C, T) for conv
        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        # (B, C, T*factor) -> (B, T*factor, C)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        new_lengths = lengths * self.factor
        return x, new_lengths


class TextCTCTagger(pl.LightningModule):
    """Text CTC Tagger model.

    Takes character sequences as input and produces tagged text via CTC decoding.
    Supports compositional tag vocabulary for zero-shot tag generalization.

    Config keys:
        char_vocab_size: int - number of unique characters + special tokens
        embed_dim: int - character embedding dimension (default 128)
        hidden_dim: int - encoder hidden dimension (default 256)
        num_heads: int - attention heads (default 4)
        num_layers: int - transformer layers (default 4)
        ffn_dim: int - feedforward dimension (default 1024)
        dropout: float - dropout rate (default 0.1)
        upsample_factor: int - character upsampling factor (default 2)
        vocab_size: int - output vocabulary size (SP + tag pieces, without blank)
        max_text_length: int - maximum input character length (default 1024)
        causal: bool - use causal attention for streaming (default False)
        lr: float - learning rate (default 1e-3)
        weight_decay: float - AdamW weight decay (default 0.01)
        warmup_steps: int - linear warmup steps (default 5000)
        max_steps: int - total training steps for cosine schedule (default 500000)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        char_vocab_size = cfg['char_vocab_size']
        embed_dim = cfg.get('embed_dim', 128)
        hidden_dim = cfg.get('hidden_dim', 256)
        num_heads = cfg.get('num_heads', 4)
        num_layers = cfg.get('num_layers', 4)
        ffn_dim = cfg.get('ffn_dim', 1024)
        dropout = cfg.get('dropout', 0.1)
        upsample_factor = cfg.get('upsample_factor', 2)
        vocab_size = cfg['vocab_size']
        max_len = cfg.get('max_text_length', 1024) * upsample_factor
        self.causal = cfg.get('causal', False)

        self.char_embedding = CharacterEmbedding(
            char_vocab_size, embed_dim, hidden_dim, max_len, dropout
        )
        self.upsampler = LearnedUpsampler(hidden_dim, upsample_factor)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # +1 for CTC blank
        self.decoder_proj = nn.Linear(hidden_dim, vocab_size + 1)
        self.blank_id = vocab_size

        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, reduction='mean', zero_infinity=True)

        self._init_weights()

        self.tokenizer = None
        self._val_wer_errors = 0
        self._val_wer_words = 0
        self._val_tag_tp = 0
        self._val_tag_fp = 0
        self._val_tag_fn = 0

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info("TextCTCTagger: %d params (%d trainable)", total_params, trainable)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _make_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """True where padded."""
        arange = torch.arange(max_len, device=lengths.device)
        return arange.unsqueeze(0) >= lengths.unsqueeze(1)

    def forward(
        self,
        char_ids: torch.Tensor,
        char_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            char_ids: (B, T_chars) input character indices
            char_lengths: (B,) actual lengths

        Returns:
            log_probs: (B, T_up, vocab_size+1)
            encoded_lengths: (B,) upsampled lengths
        """
        embedded = self.char_embedding(char_ids)
        upsampled, up_lengths = self.upsampler(embedded, char_lengths)

        B, T_up, C = upsampled.shape
        pad_mask = self._make_padding_mask(up_lengths, T_up)
        attn_mask = self._make_causal_mask(T_up, upsampled.device) if self.causal else None

        encoded = self.encoder(upsampled, mask=attn_mask, src_key_padding_mask=pad_mask)
        logits = self.decoder_proj(encoded)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, up_lengths

    def training_step(self, batch, batch_idx):
        char_ids, char_lengths, token_ids, token_lengths = batch
        log_probs, encoded_lengths = self.forward(char_ids, char_lengths)

        # CTC loss expects (T, B, C)
        log_probs_t = log_probs.permute(1, 0, 2)

        loss = self.ctc_loss(log_probs_t, token_ids, encoded_lengths, token_lengths)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        char_ids, char_lengths, token_ids, token_lengths = batch
        log_probs, encoded_lengths = self.forward(char_ids, char_lengths)

        log_probs_t = log_probs.permute(1, 0, 2)
        loss = self.ctc_loss(log_probs_t, token_ids, encoded_lengths, token_lengths)

        greedy_preds = self._greedy_decode(log_probs, encoded_lengths)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        if self.tokenizer is not None:
            for i in range(len(greedy_preds)):
                target_ids = token_ids[i, :token_lengths[i]].tolist()
                pred_text = self.tokenizer.decode(greedy_preds[i])
                ref_text = self.tokenizer.decode(target_ids)

                pred_words = [w for w in pred_text.split() if not _is_tag(w)]
                ref_words = [w for w in ref_text.split() if not _is_tag(w)]
                errors = self._edit_distance(pred_words, ref_words)
                self._val_wer_errors += errors
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
            tag_precision = tp / max(tp + self._val_tag_fp, 1)
            tag_recall = tp / max(tp + self._val_tag_fn, 1)
            tag_f1 = 2 * tag_precision * tag_recall / max(tag_precision + tag_recall, 1e-8)
            self.log('val_tag_f1', tag_f1, prog_bar=True)
            self.log('val_tag_precision', tag_precision)
            self.log('val_tag_recall', tag_recall)

            log.info(
                "Validation @ step %d: val_wer=%.4f, tag_f1=%.4f (P=%.4f R=%.4f)",
                self.trainer.global_step, wer, tag_f1, tag_precision, tag_recall,
            )

        self._val_wer_errors = 0
        self._val_wer_words = 0
        self._val_tag_tp = 0
        self._val_tag_fp = 0
        self._val_tag_fn = 0

    @staticmethod
    def _edit_distance(hyp: List[str], ref: List[str]) -> int:
        n, m = len(hyp), len(ref)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, m + 1):
                cost = 0 if hyp[i - 1] == ref[j - 1] else 1
                prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
        return dp[m]

    def _greedy_decode(
        self, log_probs: torch.Tensor, lengths: torch.Tensor
    ) -> List[List[int]]:
        """CTC greedy decode: collapse repeated tokens and remove blanks."""
        predictions = log_probs.argmax(dim=-1)  # (B, T)
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

    def configure_optimizers(self):
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 0.01)
        warmup_steps = self.cfg.get('warmup_steps', 5000)
        max_steps = self.cfg.get('max_steps', 500000)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }
