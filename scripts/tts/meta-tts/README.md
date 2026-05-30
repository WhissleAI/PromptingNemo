# META-TTS Pipeline

Train controllable text-to-speech models conditioned on metadata tags (age,
gender, emotion, intent) and speaker identity. Built on top of
[F5-TTS](https://github.com/SWivid/F5-TTS), META-TTS extends the pretrained
F5TTS_v1_Base model with a custom vocabulary that includes metadata tokens
and per-speaker ID tokens for consistent voice generation across inferences.

## Overview

The META-TTS pipeline has four stages:

```
download_audio.py  -->  prepare_data.py  -->  train.py  -->  infer.py
   (raw audio)        (Arrow dataset +     (fine-tune     (generate
                       vocab + durations)   F5-TTS)        speech)
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 16 GB+ VRAM (A100 recommended for training)
- F5-TTS installed (`pip install f5-tts`)
- HuggingFace Accelerate for multi-GPU training
- Dependencies: `pip install -r requirements.txt`

## Stage 1: Download Audio

Download public audio datasets required for META-TTS training. Sources include
CommonVoice 15.0, Multilingual LibriSpeech (MLS), LibriSpeech, and
People's Speech, covering six European languages.

```bash
python download_audio.py \
    --audio-root /mnt/nfs/data/tts_audio \
    --languages en,de,es,fr,it,pt
```

**What it does:**
- Downloads CommonVoice, MLS, LibriSpeech, and People's Speech archives
- Extracts audio files to a structured directory layout
- Supports resume on interrupted downloads (via wget)
- Matches the `audio_filepath` references in `WhissleAI/Meta_STT_EURO_Set1`

## Stage 2: Prepare Data

Convert the WhissleAI/Meta_STT_EURO_Set1 HuggingFace dataset into an
F5-TTS-compatible Arrow dataset with rearranged metadata tags and speaker IDs.

```bash
python prepare_data.py \
    --audio-root /mnt/nfs/data/tts_audio \
    --output-dir /mnt/nfs/data/meta_tts_euro \
    --split train \
    --workers 16
```

**What it does:**
- Reads the HuggingFace parquet dataset
- Resolves audio file paths to local NFS locations
- Extracts speaker IDs from audio paths (e.g., `SPK_cv_en_12345`)
- Rearranges metadata tags from trailing position to the front of the text
  (required for F5-TTS conditioning)
- Creates a custom vocabulary file (`vocab.txt`) with metadata tokens, speaker
  tokens, and all European language characters
- Computes per-utterance durations and writes `duration.json`
- Outputs an F5-TTS-compatible HuggingFace Arrow dataset

**Output structure:**

```
<output_dir>/
├── raw/              # HF Arrow dataset with (audio_path, text) columns
├── duration.json     # {"duration": [float, ...]}
└── vocab.txt         # Custom vocabulary: metadata + speaker tokens + characters
```

## Stage 3: Train

Fine-tune F5-TTS on the prepared metadata-tagged dataset. The training script
extends the pretrained model's text embedding layer to accommodate the custom
vocabulary (new token embeddings are initialized as the mean of existing
embeddings).

```bash
# Single GPU
python train.py --config-name config

# Multi-GPU with Accelerate
accelerate launch train.py --config-name config

# Override config values
accelerate launch train.py --config-name config \
    ++datasets.batch_size_per_gpu=9600 \
    ++optim.learning_rate=5e-6
```

**Key configuration** (`config.yaml`):

| Section | Key Settings |
|---------|-------------|
| `datasets` | `batch_size_per_gpu`, `batch_size_type` (frame), `num_workers` |
| `optim` | `learning_rate`, `num_warmup_updates`, `grad_accumulation_steps` |
| `model` | `tokenizer: custom`, `tokenizer_path` (points to `vocab.txt`), `backbone: DiT` |
| `ckpts` | `wandb_project`, `save_per_updates`, `save_dir` |
| `meta_tts` | `pretrained_ckpt`, `tag_tokens_in_text`, `freeze_vocoder` |

## Stage 4: Inference

Generate speech with metadata tag conditioning and optional voice cloning.
Two modes are available:

**Reference audio mode** -- provide a reference WAV for voice cloning:

```bash
python infer.py \
    --checkpoint /path/to/model_last.pt \
    --vocab /path/to/vocab.txt \
    --ref-audio reference.wav \
    --ref-text "This is a reference sentence." \
    --text "I understand your frustration." \
    --tags "AGE_30_45 GER_FEMALE EMOTION_HAP INTENT_INFORM" \
    --output output.wav
```

**Speaker ID mode** -- deterministic voice from a speaker token, no reference needed:

```bash
python infer.py \
    --checkpoint /path/to/model_last.pt \
    --vocab /path/to/vocab.txt \
    --speaker-id SPK_cv_en_12345 \
    --text "I understand your frustration." \
    --tags "AGE_30_45 GER_FEMALE EMOTION_HAP INTENT_INFORM" \
    --output output.wav
```

**Batch mode** -- process a JSONL manifest:

```bash
python infer.py \
    --checkpoint /path/to/model_last.pt \
    --vocab /path/to/vocab.txt \
    --manifest batch.jsonl \
    --output-dir outputs/
```

## Files

| File | Description |
|------|-------------|
| `download_audio.py` | Download CommonVoice, MLS, LibriSpeech, and People's Speech audio |
| `prepare_data.py` | Build F5-TTS Arrow dataset with metadata tags and speaker tokens |
| `train.py` | Fine-tune F5-TTS with custom vocabulary and metadata conditioning |
| `infer.py` | Generate speech with tag control and voice cloning |
| `config.yaml` | Hydra config for model, optimizer, dataset, and checkpoint settings |
| `requirements.txt` | Python dependencies |
