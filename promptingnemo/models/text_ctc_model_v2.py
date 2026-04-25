"""Text CTC Tagger v2: subword input → Conformer encoder (causal) → CTC.

Improvements over v1:
  - Subword input (SentencePiece) instead of characters — no need to learn tokenization
  - Conformer encoder with causal convolutions for streaming compatibility
  - Higher upsampling factor to compensate for fewer input tokens
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


# ---------------------------------------------------------------------------
# Conformer building blocks
# ---------------------------------------------------------------------------

class FeedForwardModule(nn.Module):
    """Conformer feed-forward module with expansion factor."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class CausalConvModule(nn.Module):
    """Conformer convolution module with causal (left-only) padding for streaming."""

    def __init__(self, hidden_dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.pointwise1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            groups=hidden_dim,
            padding=0,
        )
        self.causal_padding = kernel_size - 1
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.SiLU()
        self.pointwise2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.pointwise1(x)
        x = F.glu(x, dim=-1)
        x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        x = F.pad(x, (self.causal_padding, 0))
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return residual + x


class CausalMultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal mask for streaming."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                causal: bool = True) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        if causal:
            T = x.shape[1]
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
        else:
            attn_mask = None
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=pad_mask)
        x = self.dropout(x)
        return residual + x


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN(½) → MHSA → Conv → FFN(½) → LayerNorm."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 conv_kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(hidden_dim, ffn_dim, dropout)
        self.attn = CausalMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.conv = CausalConvModule(hidden_dim, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(hidden_dim, ffn_dim, dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                causal: bool = True) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attn(x, pad_mask=pad_mask, causal=causal)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.layer_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 num_layers: int, conv_kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, ffn_dim, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                causal: bool = True) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask, causal=causal)
        return x


# ---------------------------------------------------------------------------
# Input embedding and upsampler
# ---------------------------------------------------------------------------

class SubwordEmbedding(nn.Module):
    """Subword embedding with learned positional encoding."""

    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(token_ids) + self.pos_encoding(positions)
        x = self.layer_norm(x)
        return self.dropout(x)


class LearnedUpsampler(nn.Module):
    """Learnable upsampling via transposed 1D convolution."""

    def __init__(self, hidden_dim: int, factor: int = 3):
        super().__init__()
        self.factor = factor
        self.upsample = nn.ConvTranspose1d(
            hidden_dim, hidden_dim,
            kernel_size=factor, stride=factor,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        new_lengths = lengths * self.factor
        return x, new_lengths


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TextCTCTaggerV2(pl.LightningModule):
    """Text CTC Tagger v2 with Conformer encoder and subword input.

    Config keys:
        vocab_size: int - aggregate vocabulary size (shared input+output, without blank)
        hidden_dim: int - encoder hidden dimension (default 256)
        num_heads: int - attention heads (default 4)
        num_layers: int - conformer layers (default 6)
        ffn_dim: int - feedforward dimension (default 1024)
        conv_kernel_size: int - conformer conv kernel size (default 31)
        dropout: float - dropout rate (default 0.1)
        upsample_factor: int - subword upsampling factor (default 3)
        max_input_length: int - maximum input subword length (default 512)
        causal: bool - use causal attention/conv for streaming (default True)
        lr: float - learning rate (default 1e-3)
        weight_decay: float - AdamW weight decay (default 0.01)
        warmup_steps: int - linear warmup steps (default 5000)
        max_steps: int - total training steps for cosine schedule (default 500000)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        vocab_size = cfg['vocab_size']
        hidden_dim = cfg.get('hidden_dim', 256)
        num_heads = cfg.get('num_heads', 4)
        num_layers = cfg.get('num_layers', 6)
        ffn_dim = cfg.get('ffn_dim', 1024)
        conv_kernel_size = cfg.get('conv_kernel_size', 31)
        dropout = cfg.get('dropout', 0.1)
        upsample_factor = cfg.get('upsample_factor', 3)
        max_len = cfg.get('max_input_length', 512) * upsample_factor
        self.causal = cfg.get('causal', True)

        self.subword_embedding = SubwordEmbedding(
            vocab_size, hidden_dim, max_len, dropout
        )
        self.upsampler = LearnedUpsampler(hidden_dim, upsample_factor)
        self.encoder = ConformerEncoder(
            hidden_dim, num_heads, ffn_dim, num_layers, conv_kernel_size, dropout
        )

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
        log.info("TextCTCTaggerV2: %d params (%d trainable)", total_params, trainable)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        arange = torch.arange(max_len, device=lengths.device)
        return arange.unsqueeze(0) >= lengths.unsqueeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.subword_embedding(input_ids)
        upsampled, up_lengths = self.upsampler(embedded, input_lengths)

        B, T_up, C = upsampled.shape
        pad_mask = self._make_padding_mask(up_lengths, T_up)

        encoded = self.encoder(upsampled, pad_mask=pad_mask, causal=self.causal)
        logits = self.decoder_proj(encoded)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, up_lengths

    def training_step(self, batch, batch_idx):
        input_ids, input_lengths, target_ids, target_lengths = batch
        log_probs, encoded_lengths = self.forward(input_ids, input_lengths)

        log_probs_t = log_probs.permute(1, 0, 2)
        loss = self.ctc_loss(log_probs_t, target_ids, encoded_lengths, target_lengths)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, input_lengths, target_ids, target_lengths = batch
        log_probs, encoded_lengths = self.forward(input_ids, input_lengths)

        log_probs_t = log_probs.permute(1, 0, 2)
        loss = self.ctc_loss(log_probs_t, target_ids, encoded_lengths, target_lengths)

        greedy_preds = self._greedy_decode(log_probs, encoded_lengths)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        if self.tokenizer is not None:
            for i in range(len(greedy_preds)):
                tgt_ids = target_ids[i, :target_lengths[i]].tolist()
                pred_text = self.tokenizer.decode(greedy_preds[i])
                ref_text = self.tokenizer.decode(tgt_ids)

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
