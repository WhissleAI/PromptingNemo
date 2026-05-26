#!/usr/bin/env python3
"""
Fine-tune whisper-large-v3-turbo with hybrid CTC + seq2seq loss.

Architecture:
  Audio → Whisper Encoder (32 layers, 1280d)
              ↓
     ┌────────┴─────────┐
     ↓                  ↓
  CTC Head          Whisper Decoder (4 layers)
  (Linear)             ↓
     ↓            Full annotated text
  Fast CTC        (entities + meta tags)
  (plain text)

Usage:
  # Single node:
  python finetune_whisper.py --config /mnt/nfs/experiments/finetune_whisper_multilingual.yaml

  # Multi-node (2 nodes):
  # Node 0 (master):
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=MASTER_IP --master_port=29500 finetune_whisper.py --config config.yaml
  # Node 1:
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=MASTER_IP --master_port=29500 finetune_whisper.py --config config.yaml
"""
import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def is_dist():
    return dist.is_initialized()

def is_main():
    return not is_dist() or dist.get_rank() == 0

def world_size():
    return dist.get_world_size() if is_dist() else 1


# ---------------------------------------------------------------------------
# Model: Whisper + CTC head
# ---------------------------------------------------------------------------

class WhisperWithCTC(nn.Module):
    """Whisper encoder-decoder with an additional CTC head on the encoder."""

    def __init__(self, whisper_model, ctc_vocab_size: int):
        super().__init__()
        self.whisper = whisper_model
        encoder_dim = whisper_model.config.d_model  # 1280 for large-v3-turbo
        self.ctc_proj = nn.Linear(encoder_dim, ctc_vocab_size)
        self.ctc_blank_id = ctc_vocab_size - 1

    def forward(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
    ):
        # Encoder forward
        encoder_outputs = self.whisper.model.encoder(
            input_features,
            attention_mask=attention_mask,
        )
        encoder_hidden = encoder_outputs.last_hidden_state  # (B, T, 1280)

        # CTC logits from encoder
        ctc_logits = self.ctc_proj(encoder_hidden)  # (B, T, ctc_vocab)

        # Decoder forward (seq2seq)
        seq2seq_output = self.whisper(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        return {
            'ctc_logits': ctc_logits,
            'seq2seq_loss': seq2seq_output.loss,
            'seq2seq_logits': seq2seq_output.logits,
            'encoder_hidden': encoder_hidden,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultilingualASRDataset(torch.utils.data.Dataset):
    """Load from JSONL manifest, compute mel features, tokenize text."""

    def __init__(
        self,
        manifest_path: str,
        processor,
        tokenizer,
        max_duration: float = 30.0,
        sample_rate: int = 16000,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.samples = []

        with open(manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                dur = sample.get('duration', 0)
                if 0.5 <= dur <= max_duration:
                    self.samples.append(sample)

        log.info("Loaded %d samples from %s", len(self.samples), manifest_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample['audio_filepath']
        text = sample['text']
        lang_family = sample.get('lang_family', 'ENGLISH')

        # Load audio via soundfile (avoids torchcodec dependency)
        import soundfile as sf
        import numpy as np
        waveform_np, sr = sf.read(audio_path, dtype='float32')
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=1)
        if sr != self.sample_rate:
            import torchaudio
            waveform = torch.from_numpy(waveform_np)
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        else:
            waveform = torch.from_numpy(waveform_np)

        # Compute mel features using Whisper's processor
        input_features = self.processor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features.squeeze(0)  # (80, 3000) for 30s

        # Tokenize full annotated text for seq2seq target
        labels = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=448,
        ).input_ids.squeeze(0)

        return {
            'input_features': input_features,
            'labels': labels,
            'lang_family': lang_family,
            'text': text,
        }


def collate_fn(batch, pad_token_id=-100):
    """Collate batch with padding."""
    input_features = torch.stack([b['input_features'] for b in batch])

    # Pad labels
    max_label_len = max(b['labels'].size(0) for b in batch)
    labels = torch.full((len(batch), max_label_len), pad_token_id, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :b['labels'].size(0)] = b['labels']

    return {
        'input_features': input_features,
        'labels': labels,
        'lang_families': [b['lang_family'] for b in batch],
        'ref_texts': [b['text'] for b in batch],
    }


# ---------------------------------------------------------------------------
# Family-balanced sampler
# ---------------------------------------------------------------------------

class FamilyBalancedSampler(torch.utils.data.Sampler):
    """Temperature-based sampling that upweights low-resource families.

    In distributed mode, each rank gets a disjoint shard of samples per
    family so no duplicates are seen across GPUs in the same step.
    """

    def __init__(self, dataset, temperature: float = 2.0, seed: int = 42,
                 rank: int = 0, num_replicas: int = 1, epoch: int = 0):
        self.dataset = dataset
        self.temperature = temperature
        self.rank = rank
        self.num_replicas = num_replicas
        self.epoch = epoch
        self.rng = torch.Generator()

        # Group indices by family
        self.family_indices = {}
        for i, sample in enumerate(dataset.samples):
            fam = sample.get('lang_family', 'ENGLISH')
            if fam not in self.family_indices:
                self.family_indices[fam] = []
            self.family_indices[fam].append(i)

        # Compute sampling weights using temperature
        family_sizes = {f: len(ids) for f, ids in self.family_indices.items()}
        total = sum(family_sizes.values())
        self.family_probs = {}
        for fam, size in family_sizes.items():
            raw_prob = (size / total) ** (1.0 / temperature)
            self.family_probs[fam] = raw_prob
        prob_sum = sum(self.family_probs.values())
        for fam in self.family_probs:
            self.family_probs[fam] /= prob_sum

        self._samples_per_rank = len(self.dataset) // self.num_replicas

        if rank == 0:
            log.info("Family sampling probabilities (T=%.1f):", temperature)
            for fam in sorted(self.family_probs):
                log.info("  %-25s: %.3f (n=%d)", fam, self.family_probs[fam], family_sizes[fam])

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        self.rng.manual_seed(self.epoch * 1000 + self.rank + 42)
        families = list(self.family_probs.keys())
        probs = [self.family_probs[f] for f in families]

        for _ in range(self._samples_per_rank):
            fam_idx = torch.multinomial(
                torch.tensor(probs), 1, generator=self.rng
            ).item()
            fam = families[fam_idx]
            indices = self.family_indices[fam]
            idx = indices[torch.randint(len(indices), (1,), generator=self.rng).item()]
            yield idx

    def __len__(self):
        return self._samples_per_rank


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_ctc_loss(ctc_logits, labels, tokenizer, blank_id):
    """Compute CTC loss from encoder logits and seq2seq labels."""
    B, T, V = ctc_logits.shape

    # CTC needs (T, B, V) input
    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)

    # Input lengths — all frames available from encoder
    input_lengths = torch.full((B,), T, dtype=torch.long, device=ctc_logits.device)

    # Target: use the seq2seq labels but filter out special decoder tokens
    # (Whisper's <|startoftranscript|>, <|endoftext|>, language tokens, etc.)
    ctc_targets = []
    ctc_target_lengths = []
    for i in range(B):
        # Filter: keep only actual text tokens (not padding, not Whisper control tokens)
        label_seq = labels[i]
        valid_tokens = label_seq[(label_seq >= 0) & (label_seq != tokenizer.pad_token_id)]
        # Remove Whisper special tokens (IDs < 50257 for standard tokens)
        # Keep our added special tokens (ENTITY_*, AGE_*, etc.)
        text_tokens = valid_tokens[valid_tokens < blank_id]
        ctc_targets.append(text_tokens)
        ctc_target_lengths.append(len(text_tokens))

    if not ctc_targets or max(ctc_target_lengths) == 0:
        return torch.tensor(0.0, device=ctc_logits.device)

    max_target_len = max(ctc_target_lengths)
    padded_targets = torch.zeros(B, max_target_len, dtype=torch.long, device=ctc_logits.device)
    for i, t in enumerate(ctc_targets):
        if len(t) > 0:
            padded_targets[i, :len(t)] = t

    target_lengths = torch.tensor(ctc_target_lengths, dtype=torch.long, device=ctc_logits.device)

    loss = F.ctc_loss(
        log_probs, padded_targets, input_lengths, target_lengths,
        blank=blank_id, reduction='mean', zero_infinity=True,
    )
    return loss


FAMILY_LOSS_WEIGHTS = {
    'ENGLISH': 1.0,
    'EUROPEAN': 1.0,
    'SLAVIC': 1.2,
    'INDO_ARYAN': 1.0,
    'MANDARIN': 1.5,
    'DRAVIDIAN': 2.0,
    'INDIAN_LOW_RESOURCE': 2.0,
}


def train_step(
    model, batch, tokenizer, ctc_weight, device,
    family_loss_weights=None,
):
    """Single training step with hybrid CTC + seq2seq loss."""
    input_features = batch['input_features'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(
        input_features=input_features,
        labels=labels,
    )

    # Access ctc_blank_id from unwrapped model when using DDP
    base = model.module if hasattr(model, 'module') else model
    seq2seq_loss = outputs['seq2seq_loss']
    ctc_loss = compute_ctc_loss(
        outputs['ctc_logits'], labels, tokenizer, base.ctc_blank_id,
    )

    total_loss = ctc_weight * ctc_loss + (1 - ctc_weight) * seq2seq_loss

    # Per-family loss weighting: upweight low-resource families
    if family_loss_weights and 'lang_families' in batch:
        families = batch['lang_families']
        weights = [family_loss_weights.get(f, 1.0) for f in families]
        batch_weight = sum(weights) / len(weights)
        total_loss = total_loss * batch_weight

    return {
        'loss': total_loss,
        'ctc_loss': ctc_loss.item(),
        'seq2seq_loss': seq2seq_loss.item(),
        'lang_families': batch.get('lang_families', []),
    }


def setup_model_and_tokenizer(config):
    """Load Whisper, extend tokenizer, add CTC head."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

    pretrained = config['model']['pretrained']
    log.info("Loading %s ...", pretrained)

    processor = WhisperProcessor.from_pretrained(pretrained)
    tokenizer = WhisperTokenizer.from_pretrained(pretrained)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        pretrained,
        torch_dtype=torch.bfloat16,
    )

    # Load special tokens
    special_tokens_path = config['model'].get('special_tokens_path', '')
    special_tokens = []
    if special_tokens_path and os.path.exists(special_tokens_path):
        with open(special_tokens_path) as f:
            special_tokens = yaml.safe_load(f)
        log.info("Loaded %d special tokens from %s", len(special_tokens), special_tokens_path)

    # Add LANG_ tokens for each family
    lang_tokens = [f'LANG_{f}' for f in [
        'ENGLISH', 'EUROPEAN', 'SLAVIC', 'INDO_ARYAN',
        'MANDARIN', 'DRAVIDIAN', 'INDIAN_LOW_RESOURCE',
    ]]
    special_tokens.extend(lang_tokens)

    # Extend tokenizer
    num_added = tokenizer.add_tokens(special_tokens)
    log.info("Added %d special tokens to tokenizer (vocab: %d → %d)",
             num_added, len(tokenizer) - num_added, len(tokenizer))

    # Resize model embeddings
    whisper_model.resize_token_embeddings(len(tokenizer))
    log.info("Resized model embeddings to %d", len(tokenizer))

    # Initialize new token embeddings with small random values
    with torch.no_grad():
        embed = whisper_model.model.decoder.embed_tokens
        # New tokens are at the end
        if num_added > 0:
            nn.init.normal_(embed.weight[-num_added:], mean=0.0, std=0.02)

    # Update processor's tokenizer
    processor.tokenizer = tokenizer

    # Enable gradient checkpointing to fit encoder gradients in 40GB
    whisper_model.gradient_checkpointing_enable()
    log.info("Gradient checkpointing enabled")

    # Create hybrid model with CTC head
    ctc_vocab_size = len(tokenizer) + 1  # +1 for blank
    model = WhisperWithCTC(whisper_model, ctc_vocab_size)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ctc_params = sum(p.numel() for p in model.ctc_proj.parameters())
    log.info("Model: %.1fM total params, %.1fM trainable, CTC head: %.1fM",
             total_params / 1e6, trainable_params / 1e6, ctc_params / 1e6)

    return model, processor, tokenizer


def finetune(config_path: str):
    """Main fine-tuning entry point. Supports single-GPU and multi-node DDP."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ---- DDP init (torchrun sets LOCAL_RANK etc.) ----
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    ws = int(os.environ.get('WORLD_SIZE', 1))

    if ws > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        log.info("DDP initialised: rank=%d/%d, local_rank=%d", rank, ws, local_rank)
    else:
        log.info("Single-GPU mode")

    device = torch.device('cuda', local_rank)

    model, processor, tokenizer = setup_model_and_tokenizer(config)
    model = model.to(device)

    # Data
    train_cfg = config['training']
    data_dir = train_cfg['data_dir']
    train_manifest = os.path.join(data_dir, train_cfg['train_manifest'])
    valid_manifest = os.path.join(data_dir, train_cfg['valid_manifest'])

    train_dataset = MultilingualASRDataset(
        train_manifest, processor, tokenizer,
        max_duration=train_cfg.get('max_duration', 30.0),
    )
    valid_dataset = MultilingualASRDataset(
        valid_manifest, processor, tokenizer,
        max_duration=train_cfg.get('max_duration', 30.0),
    )

    # Sampler — distributed-aware
    temperature = train_cfg.get('family_sampling_temperature', 2.0)
    train_sampler = FamilyBalancedSampler(
        train_dataset, temperature=temperature,
        rank=rank, num_replicas=ws,
    )

    batch_size = train_cfg.get('batch_size', 16)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=train_cfg.get('num_workers', 8),
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id or -100),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 8),
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id or -100),
    )

    # Freeze encoder initially (before DDP wrapping)
    model_cfg = config['model']
    freeze_encoder_steps = model_cfg.get('freeze_encoder_steps', 1000)
    if freeze_encoder_steps > 0:
        for param in model.whisper.model.encoder.parameters():
            param.requires_grad = False
        if is_main():
            log.info("Encoder frozen for first %d steps", freeze_encoder_steps)

    # ---- Wrap in DDP ----
    raw_model = model  # keep reference to unwrapped model
    if ws > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        if is_main():
            log.info("Model wrapped in DDP (world_size=%d)", ws)

    # Optimizer
    optim_cfg = train_cfg.get('optim', {})
    lr = optim_cfg.get('lr', 1e-5)
    weight_decay = optim_cfg.get('weight_decay', 0.01)
    warmup_steps = optim_cfg.get('warmup_steps', 2000)
    max_steps = train_cfg.get('max_steps', 200000)
    accumulate = train_cfg.get('accumulate_grad_batches', 1)

    # Scale accumulation: with 2 GPUs each doing batch=16, effective = 16*2*accumulate
    # To keep effective batch 256 with 2 GPUs: accumulate = 256 / (16*2) = 8
    effective_accumulate = max(1, accumulate // ws)
    if is_main() and effective_accumulate != accumulate:
        log.info("Adjusted accumulate_grad_batches: %d → %d (world_size=%d, effective_batch=%d)",
                 accumulate, effective_accumulate, ws, batch_size * ws * effective_accumulate)
    accumulate = effective_accumulate

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR scheduler: linear warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # CTC weight annealing
    ctc_weight_init = model_cfg.get('ctc_weight', 0.3)
    ctc_weight_final = model_cfg.get('ctc_weight_final', 0.1)
    ctc_anneal_steps = model_cfg.get('ctc_anneal_steps', 50000)

    # Experiment setup
    exp_cfg = config.get('experiment', {})
    exp_dir = os.path.join(
        exp_cfg.get('exp_dir', '/mnt/nfs/experiments'),
        exp_cfg.get('exp_name', 'whisper-turbo-multilingual'),
    )
    if is_main():
        os.makedirs(exp_dir, exist_ok=True)
        log.info("Experiment dir: %s", exp_dir)

    # Mixed precision — bf16 on A100 doesn't need GradScaler
    autocast_dtype = torch.bfloat16

    # Resume from checkpoint if available (all ranks load the same checkpoint)
    global_step = 0
    best_val_loss = float('inf')

    resume_ckpt = config.get('resume_from_checkpoint', '')
    if not resume_ckpt:
        import glob
        ckpts = sorted(glob.glob(os.path.join(exp_dir, 'checkpoint_step*.pt')))
        if ckpts:
            resume_ckpt = ckpts[-1]

    if resume_ckpt and os.path.exists(resume_ckpt):
        if is_main():
            log.info("Resuming from checkpoint: %s", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        global_step = ckpt.get('step', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        for _ in range(global_step):
            scheduler.step()
        if global_step >= freeze_encoder_steps:
            for param in raw_model.whisper.model.encoder.parameters():
                param.requires_grad = True
            if is_main():
                log.info("Encoder already unfrozen (resumed past step %d)", freeze_encoder_steps)
        if is_main():
            log.info("Resumed at step %d, best_val_loss=%.3f", global_step, best_val_loss)

    if ws > 1:
        dist.barrier()

    optimizer.zero_grad()

    if is_main():
        log.info("Starting training: max_steps=%d, batch=%d×%d GPUs, accumulate=%d, effective_batch=%d, lr=%g",
                 max_steps, batch_size, ws, accumulate, batch_size * ws * accumulate, lr)

    epoch = 0
    while global_step < max_steps:
        model.train()
        train_sampler.set_epoch(epoch)
        epoch += 1
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            # Unfreeze encoder after warmup (on raw model, DDP syncs automatically)
            if global_step == freeze_encoder_steps and freeze_encoder_steps > 0:
                for param in raw_model.whisper.model.encoder.parameters():
                    param.requires_grad = True
                if is_main():
                    log.info("Step %d: Encoder unfrozen", global_step)

            # Anneal CTC weight
            if global_step < ctc_anneal_steps:
                ctc_weight = ctc_weight_init - (ctc_weight_init - ctc_weight_final) * (global_step / ctc_anneal_steps)
            else:
                ctc_weight = ctc_weight_final

            # Forward — train_step uses raw model for attribute access,
            # but we call model (DDP-wrapped) so gradients sync
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                result = train_step(
                    model, batch, tokenizer, ctc_weight, device,
                    family_loss_weights=FAMILY_LOSS_WEIGHTS,
                )
                loss = result['loss'] / accumulate

            # Backward
            loss.backward()

            if (batch_idx + 1) % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                epoch_losses.append(result['loss'].item() if isinstance(result['loss'], torch.Tensor) else result['loss'] * accumulate)

                # Logging (rank 0 only)
                if is_main() and global_step % 10 == 0:
                    avg_loss = sum(epoch_losses[-10:]) / len(epoch_losses[-10:])
                    log.info(
                        "step=%d loss=%.3f ctc=%.3f seq2seq=%.3f ctc_w=%.3f lr=%.2e",
                        global_step, avg_loss,
                        result['ctc_loss'], result['seq2seq_loss'],
                        ctc_weight, scheduler.get_last_lr()[0],
                    )

                # Validation (all ranks run forward pass, rank 0 logs)
                eval_every = exp_cfg.get('eval_every_n_steps', 5000)
                if global_step % eval_every == 0:
                    val_model = raw_model  # validate on unwrapped model
                    val_loss = validate(val_model, valid_loader, tokenizer, ctc_weight, device, autocast_dtype)

                    if is_main():
                        log.info("step=%d val_loss=%.3f", global_step, val_loss)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_path = os.path.join(exp_dir, f'best_step{global_step}.pt')
                            torch.save({
                                'step': global_step,
                                'model_state_dict': raw_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss': val_loss,
                                'config': config,
                            }, save_path)
                            log.info("Saved best model to %s (val_loss=%.3f)", save_path, val_loss)

                    if ws > 1:
                        dist.barrier()

                # Periodic checkpoint (rank 0 only)
                ckpt_interval = 1000 if global_step < 10000 else 10000
                if is_main() and global_step % ckpt_interval == 0:
                    save_path = os.path.join(exp_dir, f'checkpoint_step{global_step}.pt')
                    torch.save({
                        'step': global_step,
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                    }, save_path)
                    log.info("Checkpoint saved: %s", save_path)

    if is_main():
        log.info("Training complete. Best val_loss=%.3f", best_val_loss)

    if ws > 1:
        dist.destroy_process_group()


TAG_CATEGORIES = ['AGE', 'GENDER', 'EMOTION', 'INTENT']
RE_TAG = re.compile(r'\b(AGE_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+)\b')
RE_ENTITY_SPAN = re.compile(r'ENTITY_(\w+)\s+.*?\s+END')
RE_LANG_PREFIX = re.compile(r'^LANG_\S+\s*')
RE_ALL_TAGS = re.compile(r'\b(LANG_\S+|AGE_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+|ENTITY_\S+|END)\b')


def _compute_wer(ref: str, hyp: str) -> tuple[int, int]:
    """Word-level edit distance. Returns (errors, ref_len)."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    r_len = len(ref_words)
    if r_len == 0:
        return len(hyp_words), 1

    d = list(range(len(hyp_words) + 1))
    for i in range(1, r_len + 1):
        prev = d[:]
        d[0] = i
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[j] = prev[j - 1]
            else:
                d[j] = 1 + min(prev[j], d[j - 1], prev[j - 1])
    return d[-1], r_len


def _extract_tags(text: str) -> dict[str, str]:
    """Extract tag values by category from annotated text."""
    tags = {}
    for match in RE_TAG.finditer(text):
        tag = match.group(1)
        cat = tag.split('_')[0]
        tags[cat] = tag
    return tags


def _strip_tags(text: str) -> str:
    """Strip all tags to get plain transcript text."""
    text = RE_LANG_PREFIX.sub('', text)
    text = RE_ALL_TAGS.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


def validate(model, valid_loader, tokenizer, ctc_weight, device, autocast_dtype):
    """Run validation with per-family WER and tag accuracy."""
    model.eval()
    total_loss = 0
    count = 0

    family_metrics = {}

    with torch.no_grad():
        for batch in valid_loader:
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                result = train_step(model, batch, tokenizer, ctc_weight, device)

            loss_val = result['loss'].item() if isinstance(result['loss'], torch.Tensor) else result['loss']
            total_loss += loss_val
            count += 1

            # Decode predictions via seq2seq greedy
            input_features = batch['input_features'].to(device)
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                generated = model.whisper.generate(
                    input_features,
                    max_new_tokens=440,
                    return_timestamps=False,
                )
            hyp_texts = tokenizer.batch_decode(generated, skip_special_tokens=False)

            ref_texts = batch.get('ref_texts', [])
            families = batch.get('lang_families', [])

            for ref, hyp, fam in zip(ref_texts, hyp_texts, families):
                if fam not in family_metrics:
                    family_metrics[fam] = {
                        'wer_errors': 0, 'wer_words': 0, 'loss_sum': 0.0, 'n': 0,
                        'tag_correct': {c: 0 for c in TAG_CATEGORIES},
                        'tag_total': {c: 0 for c in TAG_CATEGORIES},
                        'entity_tp': 0, 'entity_fp': 0, 'entity_fn': 0,
                    }
                fm = family_metrics[fam]
                fm['loss_sum'] += loss_val
                fm['n'] += 1

                # WER on plain text (tags stripped)
                ref_plain = _strip_tags(ref)
                hyp_plain = _strip_tags(hyp)
                errors, ref_len = _compute_wer(ref_plain, hyp_plain)
                fm['wer_errors'] += errors
                fm['wer_words'] += ref_len

                # Tag accuracy per category
                ref_tags = _extract_tags(ref)
                hyp_tags = _extract_tags(hyp)
                for cat in TAG_CATEGORIES:
                    if cat in ref_tags:
                        fm['tag_total'][cat] += 1
                        if hyp_tags.get(cat) == ref_tags[cat]:
                            fm['tag_correct'][cat] += 1

                # Entity detection (type-level F1)
                ref_entities = set(RE_ENTITY_SPAN.findall(ref))
                hyp_entities = set(RE_ENTITY_SPAN.findall(hyp))
                fm['entity_tp'] += len(ref_entities & hyp_entities)
                fm['entity_fp'] += len(hyp_entities - ref_entities)
                fm['entity_fn'] += len(ref_entities - hyp_entities)

            if count >= 100:
                break

    # Log per-family metrics
    total_wer_err, total_wer_words = 0, 0
    log.info("  %-22s %7s %7s %7s %7s %7s %7s", "family", "WER%", "AGE%", "GEN%", "EMO%", "INT%", "Ent-F1%")
    for fam in sorted(family_metrics.keys()):
        fm = family_metrics[fam]
        wer = 100 * fm['wer_errors'] / max(fm['wer_words'], 1)
        total_wer_err += fm['wer_errors']
        total_wer_words += fm['wer_words']

        tag_accs = []
        for cat in TAG_CATEGORIES:
            acc = 100 * fm['tag_correct'][cat] / max(fm['tag_total'][cat], 1)
            tag_accs.append(acc)

        tp, fp, fn = fm['entity_tp'], fm['entity_fp'], fm['entity_fn']
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 200 * prec * rec / max(prec + rec, 1e-9)

        log.info("  %-22s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%%",
                 fam, wer, *tag_accs, f1)

    overall_wer = 100 * total_wer_err / max(total_wer_words, 1)
    log.info("  %-22s %6.1f%%", "OVERALL_WER", overall_wer)

    model.train()
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def transcribe(model, processor, tokenizer, audio_path: str, mode: str = 'rich', device='cuda'):
    """Transcribe a single audio file.

    Args:
        mode: 'fast' for CTC-only (plain text), 'rich' for seq2seq (full annotations)
    """
    import soundfile as sf
    import numpy as np
    waveform_np, sr = sf.read(audio_path, dtype='float32')
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=1)
    if sr != 16000:
        import torchaudio
        waveform = torch.from_numpy(waveform_np)
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    else:
        waveform = torch.from_numpy(waveform_np)

    input_features = processor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt",
    ).input_features.to(device)

    model.eval()
    with torch.no_grad():
        if mode == 'fast':
            # CTC greedy decoding on encoder
            encoder_out = model.whisper.model.encoder(input_features)
            ctc_logits = model.ctc_proj(encoder_out.last_hidden_state)
            pred_ids = ctc_logits.argmax(dim=-1).squeeze(0)
            # Remove blanks and consecutive duplicates
            tokens = []
            prev = model.ctc_blank_id
            for t in pred_ids.tolist():
                if t != model.ctc_blank_id and t != prev:
                    tokens.append(t)
                prev = t
            return tokenizer.decode(tokens, skip_special_tokens=False)

        else:  # 'rich' mode
            generated = model.whisper.generate(
                input_features,
                max_new_tokens=440,
                language=None,  # auto-detect
                return_timestamps=False,
            )
            return tokenizer.batch_decode(generated, skip_special_tokens=False)[0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with hybrid CTC+seq2seq")
    parser.add_argument('--config', required=True, help='YAML config file')
    parser.add_argument('--mode', default='train', choices=['train', 'test'],
                        help='Train or test mode')
    parser.add_argument('--audio', help='Audio file for test mode')
    args = parser.parse_args()

    if args.mode == 'train':
        finetune(args.config)
    elif args.mode == 'test':
        if not args.audio:
            print("--audio required for test mode")
            sys.exit(1)
        with open(args.config) as f:
            config = yaml.safe_load(f)
        model, processor, tokenizer = setup_model_and_tokenizer(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        print("Fast (CTC):", transcribe(model, processor, tokenizer, args.audio, 'fast', device))
        print("Rich (seq2seq):", transcribe(model, processor, tokenizer, args.audio, 'rich', device))
