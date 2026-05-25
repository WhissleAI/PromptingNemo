"""
Next-gen Audio-Visual Meta-ASR CTC model with multi-head output.

Extends AVEncDecCTCModelBPE with:
  - Cross-attention fusion (audio queries, video keys/values) instead of
    concatenated self-attention — enables attention heat map extraction
  - Head 1: CTC decoder with rich inline + trailing tags
  - Head 2: Scene classifier (scene_type, noise_level, num_speakers)
  - Head 3: Speaker attribute classifier via TrailingTagClassifier
  - Head 4: Attention map export for video heat map visualization

Visual encoder upgraded from CLIP ViT-L/14 (768d) to SigLIP 2 So400m (1152d)
with patch-level tokens for spatial attention maps.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict

from promptingnemo.models.av_ctc_model import AVEncDecCTCModelBPE

log = logging.getLogger(__name__)


class CrossModalFusionLayer(nn.Module):
    """Single layer of cross-modal fusion: self-attn on audio, cross-attn audio→video, FFN."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_key_padding_mask: Optional[torch.Tensor] = None,
        video_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio: (B, T, D) audio token representations
            video: (B, F, D) video token representations
            audio_key_padding_mask: (B, T) True for padded positions
            video_key_padding_mask: (B, F) True for padded positions

        Returns:
            audio: (B, T, D) updated audio representations
            cross_attn_weights: (B, T, F) attention weights from audio→video
        """
        # Self-attention on audio
        normed = self.norm1(audio)
        sa_out, _ = self.self_attn(
            normed, normed, normed,
            key_padding_mask=audio_key_padding_mask,
        )
        audio = audio + self.dropout(sa_out)

        # Cross-attention: audio queries attend to video keys/values
        normed = self.norm2(audio)
        ca_out, cross_attn_weights = self.cross_attn(
            normed, video, video,
            key_padding_mask=video_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights: (B, H, T, F)
        )
        audio = audio + self.dropout(ca_out)

        # Feed-forward network
        audio = audio + self.ffn(self.norm3(audio))

        return audio, cross_attn_weights


class SceneClassifier(nn.Module):
    """Multi-category scene classifier on pooled fused representation."""

    def __init__(self, input_dim: int, category_sizes: Dict[str, int]):
        super().__init__()
        self.category_names = sorted(category_sizes.keys())
        self.heads = nn.ModuleDict({
            cat: nn.Linear(input_dim, n_classes)
            for cat, n_classes in sorted(category_sizes.items())
        })

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {cat: head(pooled) for cat, head in self.heads.items()}


class AVMetaCTCModelBPE(AVEncDecCTCModelBPE):
    """Next-gen AV model with cross-attention fusion, multi-head output, attention maps.

    Config keys (in addition to parent AVEncDecCTCModelBPE):
        fusion_type: str - "cross_attention" (new) or "self_attention" (parent default)
        fusion_num_layers: int - number of fusion layers (default 6)
        video_feat_dim: int - SigLIP feature dim (default 1152)
        scene_classifier: dict - {category: num_classes} for Head 2
        scene_classifier_weight: float - loss weight for scene classification (default 0.3)
        attention_entropy_weight: float - attention sparsity regularization (default 0.01)
    """

    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg, trainer)

        self.fusion_type = cfg.get("fusion_type", "cross_attention")
        self._attn_maps = None
        self._last_fused_pooled = None

        if self.use_video_modality and self.fusion_type == "cross_attention":
            self._build_cross_attn_fusion(cfg)

        # Head 2: Scene classifier
        scene_cfg = cfg.get("scene_classifier", None)
        if scene_cfg:
            scene_categories = dict(scene_cfg) if not isinstance(scene_cfg, dict) else scene_cfg
            d_model = cfg.get("fusion_d_model", 512)
            self.scene_cls = SceneClassifier(d_model, scene_categories)
            self.scene_cls_weight = cfg.get("scene_classifier_weight", 0.3)
            log.info("Built scene classifier: %s (weight=%.2f)",
                     scene_categories, self.scene_cls_weight)
        else:
            self.scene_cls = None
            self.scene_cls_weight = 0.0

        self.attn_entropy_weight = cfg.get("attention_entropy_weight", 0.01)

    def _build_cross_attn_fusion(self, cfg: DictConfig):
        """Build cross-attention fusion replacing parent's self-attention fusion."""
        d_model = cfg.get("fusion_d_model", 512)
        nhead = cfg.get("fusion_nhead", 8)
        num_layers = cfg.get("fusion_num_layers", 6)
        dropout = cfg.get("fusion_dropout", 0.1)
        video_feat_dim = cfg.get("video_feat_dim", 1152)
        encoder_dim = self.encoder._feat_out

        # Remove parent's self-attention fusion components
        if hasattr(self, "av_encoder"):
            del self.av_encoder

        # Override video linear projection for new feature dim
        self.v_linear = nn.Linear(video_feat_dim, d_model)

        # Build cross-attention fusion layers
        self.cross_attn_layers = nn.ModuleList([
            CrossModalFusionLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self._fusion_d_model = d_model

        log.info(
            "Built cross-attention fusion: d_model=%d, nhead=%d, layers=%d, "
            "dropout=%.2f, video_dim=%d",
            d_model, nhead, num_layers, dropout, video_feat_dim,
        )

    def _fuse_modalities(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        video_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse audio and video with cross-attention, storing attention maps.

        If fusion_type is "self_attention", delegates to parent's implementation.
        """
        if self.fusion_type != "cross_attention":
            return super()._fuse_modalities(encoded, encoded_len, video_feats)

        B, C, T = encoded.shape

        # (B, C, T) -> (B, T, C)
        audio_tokens = encoded.permute(0, 2, 1)

        # Project to fusion dimension
        audio_proj = self.a_linear(audio_tokens)  # (B, T, d_model)
        video_proj = self.v_linear(video_feats)    # (B, F, d_model)

        F_vid = video_proj.shape[1]

        # Add modality embeddings
        audio_modal = self.a_modal_embs(torch.zeros(1, dtype=torch.long, device=encoded.device))
        video_modal = self.v_modal_embs(torch.zeros(1, dtype=torch.long, device=encoded.device))
        audio_proj = audio_proj + audio_modal.unsqueeze(0)
        video_proj = video_proj + video_modal.unsqueeze(0)

        # Add positional encodings
        a_positions = torch.arange(T, device=encoded.device)
        v_positions = torch.arange(F_vid, device=encoded.device)
        audio_proj = audio_proj + self.a_pos_enc(a_positions).unsqueeze(0)
        video_proj = video_proj + self.v_pos_enc(v_positions).unsqueeze(0)

        # Build padding masks
        audio_padding_mask = torch.arange(T, device=encoded.device).unsqueeze(0) >= encoded_len.unsqueeze(1)

        # Run cross-attention fusion layers
        all_attn_weights = []
        audio_out = audio_proj
        for layer in self.cross_attn_layers:
            audio_out, attn_weights = layer(
                audio_out, video_proj,
                audio_key_padding_mask=audio_padding_mask,
            )
            all_attn_weights.append(attn_weights)

        # Store attention maps for heat map export: (num_layers, B, H, T, F)
        self._attn_maps = torch.stack(all_attn_weights, dim=0)

        # Store pooled fused representation for classifier heads
        # Mean pool over valid audio positions in fusion space
        time_mask = (~audio_padding_mask).unsqueeze(-1).float()  # (B, T, 1)
        self._last_fused_pooled = (audio_out * time_mask).sum(dim=1) / time_mask.sum(dim=1).clamp(min=1)

        # Project back to encoder dimension
        fused_audio = self.fusion_out_proj(audio_out)  # (B, T, C)

        # (B, T, C) -> (B, C, T)
        return fused_audio.permute(0, 2, 1)

    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """Return cross-attention maps from the last forward pass.

        Returns:
            Tensor of shape (num_layers, B, H, T_audio, F_video) or None
        """
        return self._attn_maps

    def get_attention_heatmaps(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """Get averaged attention heat maps for visualization.

        Args:
            layer_idx: Which fusion layer's attention to use (-1 = last)

        Returns:
            Tensor of shape (B, T_audio, F_video) — attention weights
            averaged across heads
        """
        if self._attn_maps is None:
            return None
        attn = self._attn_maps[layer_idx]  # (B, H, T, F)
        return attn.mean(dim=1)  # (B, T, F) — average across heads

    def _compute_scene_loss(self, scene_labels: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute scene classification loss (Head 2)."""
        if self.scene_cls is None or self._last_fused_pooled is None or scene_labels is None:
            return torch.tensor(0.0, device=self._last_fused_pooled.device if self._last_fused_pooled is not None else "cpu")

        scene_logits = self.scene_cls(self._last_fused_pooled)
        total_loss = torch.tensor(0.0, device=self._last_fused_pooled.device)
        n_cats = 0

        for cat_name, logits in scene_logits.items():
            if cat_name in scene_labels:
                labels = scene_labels[cat_name]
                total_loss = total_loss + F.cross_entropy(logits, labels)
                n_cats += 1

        return total_loss / max(n_cats, 1)

    def _compute_attention_entropy_loss(self) -> torch.Tensor:
        """Compute attention entropy regularization to encourage sparsity.

        Low entropy = model focuses on specific video regions (good for heat maps).
        """
        if self._attn_maps is None:
            return torch.tensor(0.0)

        # Use last layer's attention, averaged across heads
        attn = self._attn_maps[-1].mean(dim=1)  # (B, T, F)
        # Clamp to avoid log(0)
        attn = attn.clamp(min=1e-8)
        entropy = -(attn * attn.log()).sum(dim=-1).mean()
        return entropy

    def training_step(self, batch, batch_idx):
        """Multi-head training step.

        Batch format: (signal, signal_len, video_feats, transcript, transcript_len)
        Optionally with scene labels appended.
        """
        if len(batch) == 5:
            signal, signal_len, video_input, transcript, transcript_len = batch
            scene_labels = None
        elif len(batch) == 6:
            signal, signal_len, video_input, transcript, transcript_len, scene_labels = batch
        elif len(batch) == 4:
            signal, signal_len, transcript, transcript_len = batch
            video_input = None
            scene_labels = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        # Forward pass (computes fusion, stores attention maps and pooled repr)
        log_probs, encoded_len, _ = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            video_input_signal=video_input if self.use_video_modality else None,
        )

        # Head 1: CTC loss
        ctc_loss = self.loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )

        # Head 2: Scene classification loss
        scene_loss = self._compute_scene_loss(scene_labels)

        # Head 3: Tag classification loss (computed by trainer.py via setup_tag_classifier)
        # The trainer hooks into the model's tag_classifier attribute
        tag_loss = torch.tensor(0.0, device=ctc_loss.device)
        if hasattr(self, "tag_classifier") and self.tag_classifier is not None:
            from scripts.asr.meta_asr.tag_classifier import (
                masked_mean_pool,
                compute_tag_classification_loss,
            )
            # Get the encoder output for tag classification
            # Use the fused pooled representation if available, else pool from encoded
            if self._last_fused_pooled is not None:
                pooled = self._last_fused_pooled
            else:
                # Fallback: pool from the log_probs encoder input
                encoded_for_pool = self.encoder(
                    audio_signal=self.preprocessor(input_signal=signal, length=signal_len)[0],
                    length=signal_len,
                )[0]
                pooled = masked_mean_pool(encoded_for_pool, encoded_len, input_format='BDT')

            tag_logits = self.tag_classifier(pooled)
            if hasattr(self, "_tag_labels") and self._tag_labels is not None:
                tag_loss = compute_tag_classification_loss(tag_logits, self._tag_labels)

        # Head 4: Attention entropy regularization
        attn_entropy = self._compute_attention_entropy_loss()

        # Combined loss
        total_loss = (
            ctc_loss
            + self.scene_cls_weight * scene_loss
            + self.attn_entropy_weight * attn_entropy
        )

        # Tag loss is added by the trainer if tag_classifier is set up
        if tag_loss.item() > 0:
            tag_weight = getattr(self, "_tag_classifier_weight", 0.5)
            total_loss = total_loss + tag_weight * tag_loss

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_ctc_loss", ctc_loss, on_step=True, on_epoch=True)
        self.log("train_scene_loss", scene_loss, on_step=True, on_epoch=True)
        self.log("train_attn_entropy", attn_entropy, on_step=True, on_epoch=True)
        self.log("learning_rate", self._optimizer.param_groups[0]["lr"])

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 5:
            signal, signal_len, video_input, transcript, transcript_len = batch
        elif len(batch) == 4:
            signal, signal_len, transcript, transcript_len = batch
            video_input = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        log_probs, encoded_len, predictions = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            video_input_signal=video_input if self.use_video_modality else None,
        )

        loss_value = self.loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )

        # Standard NeMo WER
        self.wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        self.wer.reset()

        # AV-aware WER
        hypotheses, references = self._decode_for_av_wer(
            log_probs, encoded_len, transcript, transcript_len
        )
        if hypotheses and references:
            self.av_wer.update(hypotheses, references)

        # Scene classification accuracy
        scene_acc = {}
        if self.scene_cls is not None and self._last_fused_pooled is not None:
            scene_logits = self.scene_cls(self._last_fused_pooled)
            for cat_name, logits in scene_logits.items():
                preds = logits.argmax(dim=-1)
                scene_acc[f"val_scene_{cat_name}_acc"] = preds.float().mean()

        metrics = {
            "val_loss": loss_value,
            "val_wer": wer,
            "val_wer_num": wer_num,
            "val_wer_denom": wer_denom,
            **scene_acc,
        }

        self.log("global_step", torch.tensor(self.trainer.global_step, dtype=torch.float32, device=log_probs.device))

        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)

        return metrics

    def on_validation_epoch_end(self):
        result = super().on_validation_epoch_end()

        # Scene accuracy is logged per-step; aggregate if needed
        return result

    def setup_tag_classifier(self, encoder_dim, category_sizes, weight=0.5):
        """Set up the TrailingTagClassifier (Head 3).

        Called by trainer.py when use_tag_classifier is enabled.
        Uses the fused pooled representation instead of raw encoder output.
        """
        from scripts.asr.meta_asr.tag_classifier import TrailingTagClassifier

        # Use fusion dim for input since we pool from fused representation
        input_dim = self._fusion_d_model if hasattr(self, "_fusion_d_model") else encoder_dim
        self.tag_classifier = TrailingTagClassifier(input_dim, category_sizes)
        self._tag_classifier_weight = weight
        log.info("Set up tag classifier on fused output (dim=%d, weight=%.2f): %s",
                 input_dim, weight, category_sizes)
