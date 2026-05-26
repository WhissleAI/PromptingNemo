#!/usr/bin/env python3
"""
META-TTS Training Script.

Fine-tunes F5-TTS on metadata-tagged speech data for controllable TTS.
Extends the pretrained F5TTS_v1_Base model with:
  - Custom vocabulary including metadata tokens (AGE_, GER_, EMOTION_, INTENT_)
  - Speaker ID tokens (SPK_xxx) for consistent voice generation
  - All European language characters

Based on F5-TTS training pipeline (Hydra + HuggingFace Accelerate).

Usage:
    # Single GPU
    python train.py --config-name config

    # Multi-GPU with Accelerate
    accelerate launch train.py --config-name config

    # Override config values
    accelerate launch train.py --config-name config \
        ++datasets.batch_size_per_gpu=9600 \
        ++optim.learning_rate=5e-6
"""

import json
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import CustomDataset
from f5_tts.model.utils import get_tokenizer

try:
    from datasets import load_from_disk
except ImportError:
    from datasets import Dataset as _HFDataset
    load_from_disk = _HFDataset.load_from_disk


def extend_text_embedding(model, old_vocab_size, new_vocab_size):
    """Extend text embedding layer to accommodate new vocabulary tokens.

    New token embeddings are initialized as the mean of existing embeddings
    to provide a reasonable starting point for fine-tuning.
    """
    if new_vocab_size <= old_vocab_size:
        return

    old_embed = model.transformer.text_embed.text_embed
    old_weight = old_embed.weight.data

    mean_embed = old_weight[:old_vocab_size].mean(dim=0)

    new_embed = torch.nn.Embedding(new_vocab_size + 1, old_embed.embedding_dim)
    new_embed.weight.data[:old_vocab_size + 1] = old_weight[:old_vocab_size + 1]

    for i in range(old_vocab_size + 1, new_vocab_size + 1):
        noise = torch.randn_like(mean_embed) * 0.01
        new_embed.weight.data[i] = mean_embed + noise

    new_embed = new_embed.to(old_embed.weight.device, dtype=old_embed.weight.dtype)
    model.transformer.text_embed.text_embed = new_embed

    print(f"  Extended text embedding: {old_vocab_size} → {new_vocab_size} tokens")


def load_pretrained_checkpoint(model, ckpt_path=None):
    """Load pretrained F5-TTS checkpoint for fine-tuning."""
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading pretrained checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "ema_model_state_dict" in ckpt:
            state_dict = ckpt["ema_model_state_dict"]
        else:
            state_dict = ckpt

        text_embed_key = "transformer.text_embed.text_embed.weight"
        if text_embed_key in state_dict:
            pretrained_vocab_size = state_dict[text_embed_key].shape[0]
            current_vocab_size = model.transformer.text_embed.text_embed.weight.shape[0]

            if pretrained_vocab_size != current_vocab_size:
                print(f"  Vocab size mismatch: pretrained={pretrained_vocab_size}, "
                      f"current={current_vocab_size}")
                min_size = min(pretrained_vocab_size, current_vocab_size)
                model.transformer.text_embed.text_embed.weight.data[:min_size] = \
                    state_dict[text_embed_key][:min_size]
                del state_dict[text_embed_key]
                print(f"  Copied {min_size} embedding rows, rest initialized from mean")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"    {k}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"    {k}")

        return True

    try:
        from huggingface_hub import hf_hub_download
        print("Downloading F5TTS_v1_Base pretrained checkpoint from HuggingFace...")
        ckpt_path = hf_hub_download(
            repo_id="SWivid/F5-TTS",
            filename="F5TTS_v1_Base/model_1250000.safetensors",
        )
        print(f"  Downloaded to: {ckpt_path}")

        from safetensors.torch import load_file
        raw_state_dict = load_file(ckpt_path)

        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith("ema_model."):
                state_dict[k[len("ema_model."):]] = v
            else:
                state_dict[k] = v

        text_embed_key = "transformer.text_embed.text_embed.weight"
        pretrained_vocab_size = state_dict[text_embed_key].shape[0]
        current_vocab_size = model.transformer.text_embed.text_embed.weight.shape[0]

        if pretrained_vocab_size != current_vocab_size:
            min_size = min(pretrained_vocab_size, current_vocab_size)
            model.transformer.text_embed.text_embed.weight.data[:min_size] = \
                state_dict[text_embed_key][:min_size]
            del state_dict[text_embed_key]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded pretrained weights (missing={len(missing)}, unexpected={len(unexpected)})")
        return True

    except Exception as e:
        print(f"  Could not load pretrained checkpoint: {e}")
        print("  Training from scratch.")
        return False


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):
    model_cfg = cfg
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    wandb_project = model_cfg.ckpts.get("wandb_project", "Meta-TTS")
    wandb_run_name = model_cfg.ckpts.get("wandb_run_name", "MetaTTS_v1")
    wandb_resume_id = model_cfg.ckpts.get("wandb_resume_id", None)

    tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, "custom")
    print(f"Vocabulary size: {vocab_size}")

    n_meta_tokens = sum(1 for k in vocab_char_map if "_" in k and len(k) > 3)
    n_spk_tokens = sum(1 for k in vocab_char_map if k.startswith("SPK_"))
    n_char_tokens = vocab_size - n_meta_tokens - n_spk_tokens
    print(f"  Character tokens: {n_char_tokens}")
    print(f"  Metadata tokens: {n_meta_tokens - n_spk_tokens}")
    print(f"  Speaker tokens: {n_spk_tokens}")

    model = CFM(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            mel_dim=model_cfg.model.mel_spec.n_mel_channels,
        ),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    meta_tts_cfg = model_cfg.get("meta_tts", {})
    pretrained_ckpt = meta_tts_cfg.get("pretrained_ckpt", None)
    load_pretrained_checkpoint(model, pretrained_ckpt)

    use_ema = meta_tts_cfg.get("use_ema", False)

    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=model_cfg.ckpts.save_dir,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    dataset_path = model_cfg.datasets.name
    raw_dataset = load_from_disk(f"{dataset_path}/raw")
    with open(f"{dataset_path}/duration.json", "r") as f:
        durations = json.load(f)["duration"]
    train_dataset = CustomDataset(
        raw_dataset,
        durations=durations,
        preprocessed_mel=False,
        **OmegaConf.to_container(model_cfg.model.mel_spec, resolve=True),
    )

    print(f"\nDataset: {len(train_dataset)} samples")
    print(f"Starting training...")

    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()
