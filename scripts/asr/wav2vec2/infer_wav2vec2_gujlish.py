#!/usr/bin/env python3
"""
Inference script for WhissleAI Gujlish wav2vec2 Meta-ASR model.

Transcribes audio files and extracts metadata tags (AGE, GENDER, EMOTION, INTENT, ENTITY).
Supports both HuggingFace (.safetensors) and PyTorch Lightning (.ckpt) checkpoints.

Usage:
    # From HuggingFace hub
    python infer_wav2vec2_gujlish.py --model WhissleAI/speech-tagger_gujlish_wav2vec2_meta --audio test.wav

    # From local safetensors
    python infer_wav2vec2_gujlish.py --model ./final_model --audio test.wav

    # From .ckpt checkpoint
    python infer_wav2vec2_gujlish.py --model ./checkpoints/best.ckpt \
        --pretrained CLSRIL-23.pt --vocab ./vocab.json --audio test.wav

    # Batch from JSONL manifest
    python infer_wav2vec2_gujlish.py --model WhissleAI/speech-tagger_gujlish_wav2vec2_meta \
        --manifest test.json --output results.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
TAG_PREFIXES = ("ENTITY_", "INTENT_", "EMOTION_", "GENDER_", "AGE_")


class MetaASRTokenizer:
    """Character-level tokenizer with atomic meta-tag tokens."""

    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.pad_id = vocab.get("<pad>", 0)
        self.unk_id = vocab.get("<unk>", 1)
        self.space_id = vocab.get("|", 2)
        self.tag_tokens = sorted(
            [t for t in vocab if any(t.startswith(p) for p in TAG_PREFIXES) or t == "END"],
            key=len, reverse=True,
        )

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            t = self.id_to_token.get(i, "")
            if t in ("<pad>", "<unk>", "<ctc>"):
                continue
            if t == "|":
                tokens.append(" ")
            else:
                tokens.append(t)
        return "".join(tokens).strip()


def ctc_greedy_decode(logits: torch.Tensor, pad_id: int = 0) -> List[List[int]]:
    """CTC greedy decoding: argmax + collapse repeats + remove blanks."""
    pred_ids = torch.argmax(logits, dim=-1)
    results = []
    for seq in pred_ids:
        collapsed = []
        prev = -1
        for p in seq.tolist():
            if p != prev and p != pad_id:
                collapsed.append(p)
            prev = p
        results.append(collapsed)
    return results


def parse_meta_tags(text: str) -> Dict:
    """Separate transcript text from meta tags."""
    words = text.split()
    transcript_words = []
    tags = {"age": None, "gender": None, "emotion": None, "intent": None, "entities": []}

    entity_buffer = []
    in_entity = False

    for word in words:
        if word.startswith("AGE_"):
            tags["age"] = word
        elif word.startswith("GENDER_"):
            tags["gender"] = word
        elif word.startswith("EMOTION_"):
            tags["emotion"] = word
        elif word.startswith("INTENT_"):
            tags["intent"] = word
        elif word.startswith("ENTITY_"):
            in_entity = True
            entity_buffer = [word]
        elif word == "END" and in_entity:
            entity_buffer.append(word)
            tags["entities"].append(" ".join(entity_buffer))
            entity_buffer = []
            in_entity = False
        elif in_entity:
            entity_buffer.append(word)
        else:
            transcript_words.append(word)

    return {
        "transcript": " ".join(transcript_words),
        "tags": tags,
    }


def detect_architecture(state_dict: dict) -> dict:
    encoder_layers = set()
    for key in state_dict:
        m = re.match(r"encoder\.layers\.(\d+)\.", key)
        if m:
            encoder_layers.add(int(m.group(1)))
    num_layers = max(encoder_layers) + 1 if encoder_layers else 12

    hidden_size = 768
    for key in state_dict:
        if "encoder.layers.0.self_attn.k_proj.weight" in key:
            hidden_size = state_dict[key].shape[0]
            break

    num_heads = hidden_size // 64
    intermediate_size = 3072
    for key in state_dict:
        if "encoder.layers.0.fc1.weight" in key:
            intermediate_size = state_dict[key].shape[0]
            break

    conv_layers = set()
    for key in state_dict:
        m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.", key)
        if m:
            conv_layers.add(int(m.group(1)))
    num_conv = max(conv_layers) + 1 if conv_layers else 7

    conv_dim, conv_kernel, conv_stride = [], [], []
    for i in range(num_conv):
        w_key = f"feature_extractor.conv_layers.{i}.0.weight"
        if w_key in state_dict:
            w = state_dict[w_key]
            conv_dim.append(w.shape[0])
            conv_kernel.append(w.shape[2])
            conv_stride.append(1)
    if num_conv == 7:
        conv_stride = [5, 2, 2, 2, 2, 2, 2]
    if not conv_dim:
        conv_dim = [512] * 7
        conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        conv_stride = [5, 2, 2, 2, 2, 2, 2]

    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "intermediate_size": intermediate_size,
        "conv_dim": conv_dim,
        "conv_kernel": conv_kernel,
        "conv_stride": conv_stride,
    }


def map_fairseq_key(key: str) -> Optional[str]:
    m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", key)
    if m:
        return f"wav2vec2.feature_extractor.conv_layers.{m.group(1)}.conv.{m.group(2)}"
    m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.2\.(\d+)\.(weight|bias)", key)
    if m:
        return f"wav2vec2.feature_extractor.conv_layers.{m.group(1)}.layer_norm.{m.group(3)}"
    m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.2\.(weight|bias)", key)
    if m:
        return f"wav2vec2.feature_extractor.conv_layers.{m.group(1)}.layer_norm.{m.group(2)}"
    if key.startswith("post_extract_proj."):
        return key.replace("post_extract_proj.", "wav2vec2.feature_projection.projection.")
    if key.startswith("layer_norm."):
        return key.replace("layer_norm.", "wav2vec2.feature_projection.layer_norm.")
    if key == "mask_emb":
        return "wav2vec2.masked_spec_embed"
    m = re.match(r"encoder\.pos_conv\.0\.(weight|bias)", key)
    if m:
        return f"wav2vec2.encoder.pos_conv_embed.conv.{m.group(1)}"
    if key.startswith("encoder.layer_norm."):
        return key.replace("encoder.layer_norm.", "wav2vec2.encoder.layer_norm.")
    m = re.match(r"encoder\.layers\.(\d+)\.(.*)", key)
    if m:
        layer_idx = m.group(1)
        rest = m.group(2)
        mappings = {
            "self_attn.k_proj.": "attention.k_proj.",
            "self_attn.v_proj.": "attention.v_proj.",
            "self_attn.q_proj.": "attention.q_proj.",
            "self_attn.out_proj.": "attention.out_proj.",
            "self_attn_layer_norm.": "layer_norm.",
            "fc1.": "feed_forward.intermediate_dense.",
            "fc2.": "feed_forward.output_dense.",
            "final_layer_norm.": "final_layer_norm.",
        }
        for src, dst in mappings.items():
            if rest.startswith(src):
                return f"wav2vec2.encoder.layers.{layer_idx}.{rest.replace(src, dst, 1)}"
    return None


def load_model_from_ckpt(ckpt_path: str, pretrained_path: str, vocab: dict):
    """Load from PyTorch Lightning .ckpt checkpoint."""
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    log.info("Loading pretrained fairseq model for architecture: %s", pretrained_path)
    pt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    fairseq_state = pt.get("model", pt)
    arch = detect_architecture(fairseq_state)

    config = Wav2Vec2Config(
        hidden_size=arch["hidden_size"],
        num_hidden_layers=arch["num_hidden_layers"],
        num_attention_heads=arch["num_attention_heads"],
        intermediate_size=arch["intermediate_size"],
        conv_dim=arch["conv_dim"],
        conv_kernel=arch["conv_kernel"],
        conv_stride=arch["conv_stride"],
        vocab_size=len(vocab),
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = Wav2Vec2ForCTC(config)

    log.info("Loading fine-tuned checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    cleaned = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    return model


def load_model_hf(model_path: str):
    """Load from HuggingFace format (local dir or hub ID)."""
    from transformers import Wav2Vec2ForCTC
    log.info("Loading HuggingFace model: %s", model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    return model


def load_vocab(model_path: str) -> dict:
    """Load vocab.json from a model directory or HF hub."""
    if os.path.isdir(model_path):
        vocab_path = os.path.join(model_path, "vocab.json")
    else:
        from huggingface_hub import hf_hub_download
        vocab_path = hf_hub_download(model_path, "vocab.json")
    with open(vocab_path, encoding="utf-8") as f:
        return json.load(f)


def load_audio(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform


def transcribe(
    model,
    tokenizer: MetaASRTokenizer,
    audio_paths: List[str],
    device: str = "cpu",
    batch_size: int = 1,
) -> List[Dict]:
    model.eval()
    model.to(device)
    results = []

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        waveforms = [load_audio(p) for p in batch_paths]

        max_len = max(w.shape[0] for w in waveforms)
        padded = torch.zeros(len(waveforms), max_len)
        attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.long)
        for j, w in enumerate(waveforms):
            padded[j, : w.shape[0]] = w
            attention_mask[j, : w.shape[0]] = 1

        padded = padded.to(device)
        attention_mask = attention_mask.to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_values=padded, attention_mask=attention_mask)
        latency = (time.time() - t0) / len(batch_paths)

        decoded = ctc_greedy_decode(outputs.logits, tokenizer.pad_id)

        for j, (path, ids) in enumerate(zip(batch_paths, decoded)):
            raw_text = tokenizer.decode(ids)
            parsed = parse_meta_tags(raw_text)
            duration = waveforms[j].shape[0] / SAMPLE_RATE
            results.append({
                "audio_filepath": path,
                "raw_output": raw_text,
                "transcript": parsed["transcript"],
                "tags": parsed["tags"],
                "duration_sec": round(duration, 2),
                "latency_sec": round(latency, 4),
                "rtf": round(latency / max(duration, 0.01), 4),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="WhissleAI Gujlish wav2vec2 Meta-ASR Inference")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID, local dir with safetensors, or .ckpt path")
    parser.add_argument("--pretrained", default=None,
                        help="Path to CLSRIL-23.pt (required only for .ckpt loading)")
    parser.add_argument("--vocab", default=None,
                        help="Path to vocab.json (auto-detected for HF models)")
    parser.add_argument("--audio", nargs="+", default=None,
                        help="Audio file(s) to transcribe")
    parser.add_argument("--manifest", default=None,
                        help="JSONL manifest with audio_filepath entries")
    parser.add_argument("--output", default=None,
                        help="Output JSONL file for results")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None,
                        help="Device (cpu/cuda, auto-detected if omitted)")
    args = parser.parse_args()

    if not args.audio and not args.manifest:
        parser.error("Provide --audio or --manifest")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    if args.model.endswith(".ckpt"):
        if not args.pretrained:
            parser.error("--pretrained required for .ckpt loading")
        vocab_path = args.vocab
        if not vocab_path:
            parser.error("--vocab required for .ckpt loading")
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        model = load_model_from_ckpt(args.model, args.pretrained, vocab)
    else:
        vocab = load_vocab(args.model)
        model = load_model_hf(args.model)

    tokenizer = MetaASRTokenizer(vocab)
    log.info("Vocab: %d tokens (%d tags)", len(vocab),
             len([t for t in vocab if any(t.startswith(p) for p in TAG_PREFIXES)]))

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %.1fM parameters", total_params / 1e6)

    audio_paths = []
    if args.audio:
        audio_paths = args.audio
    elif args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                audio_paths.append(entry["audio_filepath"])

    log.info("Transcribing %d files...", len(audio_paths))
    results = transcribe(model, tokenizer, audio_paths, device=device, batch_size=args.batch_size)

    for r in results:
        print(f"\n{'='*60}")
        print(f"File: {r['audio_filepath']}")
        print(f"Duration: {r['duration_sec']}s | Latency: {r['latency_sec']}s | RTF: {r['rtf']}")
        print(f"Transcript: {r['transcript']}")
        if r['tags']['age']:
            print(f"  Age: {r['tags']['age']}")
        if r['tags']['gender']:
            print(f"  Gender: {r['tags']['gender']}")
        if r['tags']['emotion']:
            print(f"  Emotion: {r['tags']['emotion']}")
        if r['tags']['intent']:
            print(f"  Intent: {r['tags']['intent']}")
        if r['tags']['entities']:
            print(f"  Entities: {r['tags']['entities']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("Results written to %s", args.output)

    log.info("Done. Avg RTF: %.4f", np.mean([r["rtf"] for r in results]))


if __name__ == "__main__":
    main()
