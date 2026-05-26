#!/usr/bin/env python3
"""
META-TTS Inference Script.

Generate speech with metadata tag conditioning and optional voice cloning.

Two modes:
  1. Reference mode: provide ref audio → voice identity from reference
  2. Speaker-ID mode: provide SPK_xxx token → deterministic voice, no ref needed

Usage:
    # Reference audio mode (voice cloning + tag control)
    python infer.py \
        --checkpoint /mnt/nfs/experiments/meta-tts-v1/model_last.pt \
        --vocab /mnt/nfs/data/meta_tts_euro/vocab.txt \
        --ref-audio reference.wav \
        --ref-text "This is a reference sentence." \
        --text "I understand your frustration." \
        --tags "AGE_30_45 GER_FEMALE EMOTION_HAP INTENT_INFORM" \
        --output output.wav

    # Speaker ID mode (no reference audio needed)
    python infer.py \
        --checkpoint /mnt/nfs/experiments/meta-tts-v1/model_last.pt \
        --vocab /mnt/nfs/data/meta_tts_euro/vocab.txt \
        --speaker-id SPK_cv_en_12345 \
        --text "I understand your frustration." \
        --tags "AGE_30_45 GER_FEMALE EMOTION_HAP INTENT_INFORM" \
        --output output.wav

    # Batch mode from manifest
    python infer.py \
        --checkpoint /mnt/nfs/experiments/meta-tts-v1/model_last.pt \
        --vocab /mnt/nfs/data/meta_tts_euro/vocab.txt \
        --manifest batch.jsonl \
        --output-dir outputs/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from f5_tts.model import CFM
from f5_tts.model.backbones.dit import DiT
from f5_tts.model.utils import get_tokenizer


def load_model(checkpoint_path, vocab_path, device="cuda"):
    """Load trained META-TTS model."""
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")

    model = CFM(
        transformer=DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_num_embeds=vocab_size,
            mel_dim=100,
            text_mask_padding=True,
            conv_layers=4,
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=24000,
            n_mel_channels=100,
            hop_length=256,
            win_length=1024,
            n_fft=1024,
            mel_spec_type="vocos",
        ),
        vocab_char_map=vocab_char_map,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "ema_model_state_dict" in ckpt:
        state_dict = ckpt["ema_model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    return model, vocab_char_map


def load_vocoder(device="cuda"):
    """Load Vocos vocoder for mel → waveform conversion."""
    try:
        from vocos import Vocos
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        vocoder = vocoder.to(device).eval()
        print("  Vocos vocoder loaded")
        return vocoder
    except ImportError:
        print("  Warning: vocos not installed. Install with: pip install vocos")
        print("  Output will be mel spectrogram only (no audio).")
        return None


def load_reference_audio(ref_audio_path, target_sr=24000):
    """Load and resample reference audio."""
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    return audio.squeeze(0)


def build_tagged_text(text, tags=None, speaker_id=None):
    """Build tagged text in META-TTS format: SPK_xxx TAGS clean_text."""
    parts = []
    if speaker_id:
        parts.append(speaker_id)
    if tags:
        if isinstance(tags, str):
            parts.extend(tags.split())
        elif isinstance(tags, list):
            parts.extend(tags)
    parts.append(text)
    return " ".join(parts)


@torch.no_grad()
def generate(
    model,
    vocoder,
    text,
    ref_audio=None,
    tags=None,
    speaker_id=None,
    steps=32,
    cfg_strength=2.0,
    sway_coef=-1.0,
    speed=1.0,
    device="cuda",
):
    """Generate speech from tagged text."""
    tagged_text = build_tagged_text(text, tags, speaker_id)
    print(f"  Input: {tagged_text[:100]}...")

    if ref_audio is not None:
        ref_audio = ref_audio.to(device)
        cond = model.mel_spec(ref_audio.unsqueeze(0))
        cond = cond.permute(0, 2, 1)
        ref_mel_len = cond.shape[1]
    else:
        ref_mel_len = 0
        cond = torch.zeros(1, 0, 100, device=device)

    target_duration = int(ref_mel_len + len(tagged_text) * 12 / speed)

    generated, _ = model.sample(
        cond=cond,
        text=[tagged_text],
        duration=target_duration,
        steps=steps,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_coef,
    )

    if ref_mel_len > 0:
        generated = generated[:, ref_mel_len:]

    if vocoder is not None:
        generated_mel = generated.permute(0, 2, 1)
        audio = vocoder.decode(generated_mel)
        return audio.squeeze().cpu()

    return generated.cpu()


def main():
    parser = argparse.ArgumentParser(description="META-TTS Inference")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--vocab", required=True, help="Path to vocab.txt")
    parser.add_argument("--ref-audio", default=None, help="Reference audio for voice cloning")
    parser.add_argument("--ref-text", default="", help="Transcript of reference audio")
    parser.add_argument("--text", default=None, help="Text to synthesize")
    parser.add_argument("--tags", default=None, help="Metadata tags (space-separated)")
    parser.add_argument("--speaker-id", default=None, help="Speaker ID token (e.g., SPK_cv_en_12345)")
    parser.add_argument("--output", default="output.wav", help="Output audio path")
    parser.add_argument("--manifest", default=None, help="JSONL manifest for batch inference")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for batch mode")
    parser.add_argument("--steps", type=int, default=32, help="ODE solver steps")
    parser.add_argument("--cfg-strength", type=float, default=2.0, help="Classifier-free guidance strength")
    parser.add_argument("--sway-coef", type=float, default=-1.0, help="Sway sampling coefficient")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (1.0 = normal)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    if not args.manifest and not args.text:
        parser.error("Either --text or --manifest is required")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Device: {device}")

    model, vocab_map = load_model(args.checkpoint, args.vocab, device)
    vocoder = load_vocoder(device)

    if args.manifest:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        with open(args.manifest, "r") as f:
            items = [json.loads(line) for line in f if line.strip()]

        print(f"\nBatch inference: {len(items)} items")

        for i, item in enumerate(items):
            text = item["text"]
            tags = item.get("tags", args.tags)
            speaker_id = item.get("speaker_id", args.speaker_id)
            ref_audio_path = item.get("ref_audio", args.ref_audio)
            output_name = item.get("output", f"output_{i:04d}.wav")

            ref_audio = None
            if ref_audio_path:
                ref_audio = load_reference_audio(ref_audio_path)

            print(f"\n[{i + 1}/{len(items)}] Generating: {text[:60]}...")
            t0 = time.time()

            audio = generate(
                model, vocoder, text,
                ref_audio=ref_audio,
                tags=tags,
                speaker_id=speaker_id,
                steps=args.steps,
                cfg_strength=args.cfg_strength,
                sway_coef=args.sway_coef,
                speed=args.speed,
                device=device,
            )

            if isinstance(audio, torch.Tensor) and audio.ndim == 1:
                out_path = Path(args.output_dir) / output_name
                sf.write(str(out_path), audio.numpy(), 24000)
                print(f"  Saved: {out_path} ({time.time() - t0:.2f}s)")

    else:
        ref_audio = None
        if args.ref_audio:
            ref_audio = load_reference_audio(args.ref_audio)
            print(f"Reference audio: {args.ref_audio} ({ref_audio.shape[-1] / 24000:.2f}s)")

        print(f"\nGenerating...")
        t0 = time.time()

        audio = generate(
            model, vocoder, args.text,
            ref_audio=ref_audio,
            tags=args.tags,
            speaker_id=args.speaker_id,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            sway_coef=args.sway_coef,
            speed=args.speed,
            device=device,
        )

        if isinstance(audio, torch.Tensor) and audio.ndim == 1:
            sf.write(args.output, audio.numpy(), 24000)
            duration = audio.shape[-1] / 24000
            elapsed = time.time() - t0
            print(f"\nSaved: {args.output}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Generation time: {elapsed:.2f}s")
            print(f"  RTF: {elapsed / duration:.3f}x")
        else:
            print("  Warning: No vocoder available, mel spectrogram saved as .pt")
            torch.save(audio, args.output.replace(".wav", ".pt"))


if __name__ == "__main__":
    main()
