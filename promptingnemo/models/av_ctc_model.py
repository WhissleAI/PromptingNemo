"""
Audio-Visual CTC model for noisy speech recognition.

Implements the AV-UNI-SNR architecture from:
  "Visual-Aware Speech Recognition for Noisy Scenarios"
  Darur & Singla, EMNLP 2025
  https://aclanthology.org/2025.emnlp-main.845/

Architecture:
  - Pretrained Conformer CTC encoder (frozen, with optional linear adapters)
  - CLIP ViT-L/14 visual encoder (pre-extracted features, frozen)
  - Linear projections + modality embeddings + positional encodings
  - Transformer encoder for cross-modal self-attention (configurable layers/heads)
  - After fusion, only audio-aligned positions are retained -> CTC decoder
  - Noise label (<N\\d+>) appended as final token in transcript
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, open_dict

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.core.classes.mixins import AccessMixin

from promptingnemo.eval.av_wer import AVWordErrorRate, separate_labels_from_text

log = logging.getLogger(__name__)


class AVEncDecCTCModelBPE(EncDecCTCModelBPE):
    """Audio-Visual CTC model with cross-modal fusion.

    Wraps a pretrained EncDecCTCModelBPE as the audio backbone and adds
    a transformer-based fusion module that attends over both audio encoder
    outputs and pre-extracted CLIP visual features.

    Config keys (under model.*):
        use_video_modality: bool - enable AV fusion (False = audio-only baseline)
        pretrained_model_path: str - path to .nemo checkpoint for audio backbone
        freeze_encoder: bool - freeze conformer encoder weights (default True)
        use_adapters: bool - insert linear adapters into conformer (default False)
        adapter_dim: int - adapter bottleneck dimension (default 64)
        fusion_d_model: int - fusion transformer hidden dim (default 512)
        fusion_nhead: int - number of attention heads (default 8)
        fusion_num_layers: int - number of transformer encoder layers (default 4)
        fusion_dropout: float - dropout in fusion transformer (default 0.1)
        video_feat_dim: int - CLIP feature dimension (default 768)
    """

    def __init__(self, cfg: DictConfig, trainer=None):
        # Load pretrained audio model first to inherit its config
        pretrained_path = cfg.get("pretrained_model_path", None)
        if pretrained_path:
            log.info("Loading pretrained audio model from %s", pretrained_path)
            a_model = EncDecCTCModelBPE.restore_from(pretrained_path, map_location="cpu")
            # Merge pretrained config into our cfg, preserving our overrides
            with open_dict(cfg):
                for key in ["preprocessor", "encoder", "decoder", "tokenizer"]:
                    if key not in cfg or cfg[key] is None:
                        cfg[key] = a_model.cfg[key]
        else:
            a_model = None

        super().__init__(cfg=cfg, trainer=trainer)

        # Copy pretrained weights if available
        if a_model is not None:
            self.load_state_dict(a_model.state_dict(), strict=False)
            del a_model

        self.use_video_modality = cfg.get("use_video_modality", False)
        self.freeze_encoder_weights = cfg.get("freeze_encoder", True)

        # Freeze conformer encoder
        if self.freeze_encoder_weights:
            for param in self.encoder.parameters():
                param.requires_grad = False
            log.info("Conformer encoder weights frozen.")

        # Optional linear adapters in conformer layers
        self.use_adapters = cfg.get("use_adapters", False)
        if self.use_adapters:
            self._insert_adapters(cfg.get("adapter_dim", 64))

        # Build fusion components
        if self.use_video_modality:
            self._build_fusion(cfg)

        # AV-aware WER metric
        self.av_wer = AVWordErrorRate()

    def _insert_adapters(self, adapter_dim: int):
        """Insert lightweight linear adapters after each conformer layer.

        Each adapter is: Linear(hidden, bottleneck) -> ReLU -> Linear(bottleneck, hidden)
        with a residual connection. Only adapter parameters are trainable.
        """
        encoder_dim = self.encoder._feat_out
        adapters = nn.ModuleList()

        # Find conformer layers — NeMo stores them in different attributes
        # depending on version; try common locations
        conformer_layers = None
        for attr in ["layers", "conformer_layers", "_modules"]:
            candidate = getattr(self.encoder, attr, None)
            if candidate is not None and hasattr(candidate, "__len__"):
                conformer_layers = candidate
                break

        if conformer_layers is None:
            log.warning("Could not locate conformer layers for adapter insertion.")
            return

        n_layers = len(conformer_layers)
        for _ in range(n_layers):
            adapter = nn.Sequential(
                nn.Linear(encoder_dim, adapter_dim),
                nn.ReLU(),
                nn.Linear(adapter_dim, encoder_dim),
            )
            # Small init so adapters start near identity
            nn.init.zeros_(adapter[2].weight)
            nn.init.zeros_(adapter[2].bias)
            adapters.append(adapter)

        self.adapters = adapters
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = True

        log.info(
            "Inserted %d linear adapters (dim=%d) into conformer encoder.",
            n_layers,
            adapter_dim,
        )

    def _build_fusion(self, cfg: DictConfig):
        """Build cross-modal fusion transformer components."""
        d_model = cfg.get("fusion_d_model", 512)
        nhead = cfg.get("fusion_nhead", 8)
        num_layers = cfg.get("fusion_num_layers", 4)
        dropout = cfg.get("fusion_dropout", 0.1)
        video_feat_dim = cfg.get("video_feat_dim", 768)
        encoder_dim = self.encoder._feat_out

        # Linear projections into shared fusion space
        self.a_linear = nn.Linear(encoder_dim, d_model)
        self.v_linear = nn.Linear(video_feat_dim, d_model)

        # Modality embeddings (learned, one per modality)
        self.a_modal_embs = nn.Embedding(1, d_model)
        self.v_modal_embs = nn.Embedding(1, d_model)

        # Positional encodings (learned)
        self.a_pos_enc = nn.Embedding(10000, d_model)
        self.v_pos_enc = nn.Embedding(10000, d_model)

        # Transformer encoder for cross-modal self-attention
        av_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.av_encoder = nn.TransformerEncoder(av_encoder_layer, num_layers=num_layers)

        # Project back from fusion dim to encoder dim for the CTC decoder
        self.fusion_out_proj = nn.Linear(d_model, encoder_dim)

        self._fusion_d_model = d_model

        log.info(
            "Built AV fusion transformer: d_model=%d, nhead=%d, layers=%d, dropout=%.2f, video_dim=%d",
            d_model,
            nhead,
            num_layers,
            dropout,
            video_feat_dim,
        )

    def _apply_adapters(self, encoded: torch.Tensor) -> torch.Tensor:
        """Apply adapters as a post-hoc residual on the full encoder output.

        This is a simplified approach: since NeMo's conformer doesn't expose
        per-layer hooks easily, we apply all adapters sequentially on the
        final encoder output. Each adapter adds a residual.
        """
        if not hasattr(self, "adapters"):
            return encoded
        # encoded: (B, C, T) — adapters work on (B, T, C)
        x = encoded.permute(0, 2, 1)
        for adapter in self.adapters:
            x = x + adapter(x)
        return x.permute(0, 2, 1)

    def _fuse_modalities(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        video_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse audio encoder outputs with video features.

        Args:
            encoded: Audio encoder output (B, C, T)
            encoded_len: Lengths of audio sequences (B,)
            video_feats: Pre-extracted CLIP features (B, F, D) where F = num frames

        Returns:
            Fused representation (B, C, T) — only audio-aligned positions retained
        """
        B, C, T = encoded.shape

        # (B, C, T) -> (B, T, C)
        audio_tokens = encoded.permute(0, 2, 1)

        # Project to fusion dimension
        audio_proj = self.a_linear(audio_tokens)  # (B, T, d_model)
        video_proj = self.v_linear(video_feats)    # (B, F, d_model)

        F_vid = video_proj.shape[1]

        # Add modality embeddings (broadcast across time/frame dimension)
        audio_modal = self.a_modal_embs(torch.zeros(1, dtype=torch.long, device=encoded.device))  # (1, d_model)
        video_modal = self.v_modal_embs(torch.zeros(1, dtype=torch.long, device=encoded.device))  # (1, d_model)
        audio_proj = audio_proj + audio_modal.unsqueeze(0)  # broadcast to (B, T, d_model)
        video_proj = video_proj + video_modal.unsqueeze(0)  # broadcast to (B, F, d_model)

        # Add positional encodings
        a_positions = torch.arange(T, device=encoded.device)
        v_positions = torch.arange(F_vid, device=encoded.device)
        audio_proj = audio_proj + self.a_pos_enc(a_positions).unsqueeze(0)  # (1, T, d_model) broadcast
        video_proj = video_proj + self.v_pos_enc(v_positions).unsqueeze(0)  # (1, F, d_model) broadcast

        # Concatenate [audio; video] along sequence dimension
        combined = torch.cat([audio_proj, video_proj], dim=1)  # (B, T+F, d_model)

        # Run through fusion transformer
        fused = self.av_encoder(combined)  # (B, T+F, d_model)

        # Keep only audio-aligned positions
        fused_audio = fused[:, :T, :]  # (B, T, d_model)

        # Project back to encoder dimension
        fused_audio = self.fusion_out_proj(fused_audio)  # (B, T, C)

        # (B, T, C) -> (B, C, T)
        return fused_audio.permute(0, 2, 1)

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        video_input_signal=None,
    ):
        """Forward pass with optional video modality fusion.

        Args:
            input_signal: Raw audio waveform (B, S)
            input_signal_length: Audio lengths (B,)
            processed_signal: Pre-processed audio features (B, C, T) — alternative to input_signal
            processed_signal_length: Processed signal lengths (B,)
            video_input_signal: Pre-extracted CLIP visual features (B, F, D)

        Returns:
            (log_probs, encoded_len, greedy_predictions)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None

        if not has_input_signal and not has_processed_signal:
            raise ValueError("Either input_signal or processed_signal must be provided.")

        # Preprocess audio to mel spectrogram
        if has_input_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )

        # Spec augmentation during training
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )

        # Run through conformer encoder
        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )

        # Apply adapters if present
        if self.use_adapters:
            encoded = self._apply_adapters(encoded)

        # Cross-modal fusion with video features
        if self.use_video_modality and video_input_signal is not None:
            encoded = self._fuse_modalities(encoded, encoded_len, video_input_signal)

        # CTC decoder
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions

    def training_step(self, batch, batch_idx):
        # AV batch: (signal, signal_len, video_feats, transcript, transcript_len)
        if len(batch) == 5:
            signal, signal_len, video_input, transcript, transcript_len = batch
        elif len(batch) == 4:
            signal, signal_len, transcript, transcript_len = batch
            video_input = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        log_probs, encoded_len, _ = self.forward(
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

        self.log("train_loss", loss_value, on_step=True, on_epoch=True, prog_bar=True)
        self.log("learning_rate", self._optimizer.param_groups[0]["lr"])

        return {"loss": loss_value}

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

        # AV-aware WER (labelled, unlabelled, noise accuracy)
        hypotheses, references = self._decode_for_av_wer(
            log_probs, encoded_len, transcript, transcript_len
        )
        if hypotheses and references:
            self.av_wer.update(hypotheses, references)

        metrics = {
            "val_loss": loss_value,
            "val_wer": wer,
            "val_wer_num": wer_num,
            "val_wer_denom": wer_denom,
        }

        self.log("global_step", torch.tensor(self.trainer.global_step, dtype=torch.float32, device=log_probs.device))

        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)

        return metrics

    def on_validation_epoch_end(self):
        result = super().on_validation_epoch_end()

        # Log AV-specific metrics
        av_metrics = self.av_wer.compute()
        self.log("val_labelled_wer", av_metrics["labelled_wer"], prog_bar=True)
        self.log("val_unlabelled_wer", av_metrics["unlabelled_wer"], prog_bar=True)
        self.log("val_noise_label_acc", av_metrics["noise_label_accuracy"], prog_bar=True)
        self.av_wer.reset()

        return result

    def _decode_for_av_wer(
        self,
        log_probs: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
    ) -> Tuple[List[str], List[str]]:
        """Decode log probs and target tokens to text for AV WER computation."""
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

        hyp_texts = []
        for h in hypotheses:
            hyp_texts.append(h.text if hasattr(h, "text") else str(h))

        ref_texts = []
        targets_cpu = transcript.detach().cpu()
        target_lens_cpu = transcript_len.detach().cpu()
        for i in range(targets_cpu.shape[0]):
            token_ids = targets_cpu[i][: target_lens_cpu[i]].tolist()
            ref_texts.append(self._decode_target_tokens(token_ids))

        return hyp_texts, ref_texts

    def _decode_target_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text string."""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            if hasattr(tokenizer, "ids_to_text"):
                return tokenizer.ids_to_text(token_ids)
            if hasattr(tokenizer, "ids_to_tokens"):
                tokens = tokenizer.ids_to_tokens(token_ids)
                return "".join(tokens).replace("\u2581", " ").strip()
        return " ".join(str(t) for t in token_ids)
