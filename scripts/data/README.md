# Data Processing Scripts

Scripts for acquiring, processing, and preparing speech and video data for
Meta-ASR and related model training. Organized by data modality and processing
stage.

## Directory Overview

```
scripts/data/
├── audio/              # Audio data processing
│   ├── 1person/        # Single-speaker datasets
│   │   ├── real/       # Real speech (CommonVoice, LibriSpeech, SLURP, AI4Bharat, etc.)
│   │   └── synthetic/  # Synthetic tagged speech via TTS
│   ├── 2person/        # Two-speaker conversation processing
│   ├── emotion/        # Emotion classification datasets (IEMOCAP)
│   ├── tts-finetunning/# TTS fine-tuning data preparation
│   └── utils/          # Audio format conversion helpers
├── video/              # Video data processing
│   ├── avspeech/       # AVSpeech dataset annotation
│   ├── ted/            # TED Talk video download and transcription
│   ├── youtube/        # YouTube download + Gemini transcription pipeline
│   └── utils/          # Video processing utilities
├── meta-asr/           # Meta-ASR dataset preparation
│   ├── annote-entent/  # Per-language entity/intent annotation (European, Slavic)
│   ├── shrutilipi-process/ # Indic ShrutiLipi parquet processing (18 languages)
│   ├── madASR/         # MadASR 2.0 Indic dataset processing
│   ├── process-data-en/    # English dataset processors (NPTEL, OpenSR, YouTube)
│   ├── process-data-mls/   # MLS/LibriSpeech/CommonVoice European language processors
│   ├── models/         # Utility models (age/gender classification, NER tagging)
│   ├── fast_api/       # Annotation review web interface
│   ├── docker-process/ # Docker-based text normalization
│   └── utils/          # Entity classification, Gemini transcription, keyword extraction
├── hf/                 # HuggingFace format converters
├── meta-mt/            # Meta machine translation (Gemini-based)
├── noise/              # Noise data download and preparation
└── utils/              # General-purpose data utilities
```

## audio/ -- Audio Data Processing

### audio/1person/real/

Process single-speaker real speech datasets into NeMo-format manifests.

| Subdirectory / File | Description |
|---------------------|-------------|
| `AI4Bharat/` | Process AI4Bharat Hindi/Indic datasets; includes WER evaluation and NER scoring |
| `annotate_sentence/` | LLM-based sentence annotation using GCP Gemini, OpenAI ChatGPT, or custom GCP models |
| `AudioEmotionClassification/` | Audio emotion classifier training (collator, models, trainer) |
| `data_download/` | CommonVoice downloader, audio format conversion scripts |
| `gyaanvaami/` | Process Gyaanvaami Indic speech data from CommonVoice November 2024 |
| `commonvoice_downloader.py` | Download CommonVoice datasets by language |
| `process_cv.py` | Process CommonVoice data into NeMo manifests |
| `process_cv_eu.py` | Process European CommonVoice languages |
| `process_cv_nov24.py` | Process CommonVoice November 2024 release |
| `process_libre.py` | Process LibriSpeech into NeMo manifests |
| `process_slurp.py` | Process SLURP (IoT/smart-home) dataset into tagged NeMo manifests |
| `process_INSuperb.py` | Process IN-Superb Indic speech benchmark |
| `process_indic-voices.py` | Process IndicVoices dataset |

### audio/1person/synthetic/

Generate synthetic tagged speech data using TTS systems.

| File | Description |
|------|-------------|
| `create_synthetic_tagged_text.py` | Generate synthetic sentences with metadata tags for TTS input |
| `create_tts_manifest.py` | Create NeMo manifests from TTS-generated audio |
| `create_tts_manifest_xtts.py` | Create manifests using XTTS (Coqui) for multi-speaker synthesis |
| `clean_synthetic_text.py` | Clean and filter synthetic tagged text |
| `clean_synthetic_text_slurp_extended.py` | Clean synthetic text for extended SLURP domain |
| `clean_keep_frequent_tags.py` | Filter synthetic data to keep only frequently occurring tags |
| `add_noise.py` | Add ambient noise to clean audio for data augmentation |
| `real_data_tag.py` | Apply tags to real speech data based on synthetic tag distributions |

### audio/2person/

Pipeline for processing two-speaker conversations:

1. `01_create_manifest_raw.py` -- Create initial manifest from raw audio
2. `02_ctm2segments.py` -- Convert CTM (time-marked) format to segments
3. `03_annotate_turns.py` -- Annotate speaker turns
4. `04_split_and_emotion.py` -- Split by speaker and add emotion labels

### audio/emotion/

| File | Description |
|------|-------------|
| `iemocap_convert_to_wav.py` | Convert IEMOCAP audio files to WAV format |
| `iemocap_manifest.py` | Create NeMo manifest from IEMOCAP with emotion labels |

### audio/tts-finetunning/

| File | Description |
|------|-------------|
| `finetune_tts.py` | Fine-tune TTS models on custom speaker data |
| `create_data_hub.py` | Create HuggingFace dataset from TTS training data |
| `push_to_hub.py` | Upload processed TTS datasets to HuggingFace Hub |
| `webm_wav.py` | Convert WebM audio files to WAV format |

### audio/utils/

| File | Description |
|------|-------------|
| `csv2manifest.py` | Convert CSV-format transcription data to NeMo JSONL manifests |

## video/ -- Video Data Processing

### video/youtube/

Three-stage pipeline for creating speech datasets from YouTube:

1. `01_youtube_downloader.py` -- Download videos by search query
2. `02_bucket_gemini_transcribe.py` -- Upload to GCS and transcribe with Gemini
3. `03_segment_and_upload.py` -- Segment long audio into short utterances

### video/ted/

| File | Description |
|------|-------------|
| `get_mp4.py` | Download TED Talk videos by ID |
| `get_segments_and_transcription.py` | Extract segments with aligned transcriptions |

### video/avspeech/

| File | Description |
|------|-------------|
| `video_annotation.py` | Annotate AVSpeech video segments |

### video/utils/

| File | Description |
|------|-------------|
| `gcp_annotate.py` | Annotate video data using GCP AI services |
| `process_via_audio.py` | Extract and process audio tracks from video files |

## meta-asr/ -- Meta-ASR Dataset Preparation

Language-specific data processing pipelines for building Meta-ASR training sets.

### annote-entent/

LLM-based entity and intent annotation scripts, organized by language family:

- `european/` -- French, German, Italian, Portuguese, Spanish
- `salvic/` -- Bulgarian, Belarusian, Czech, Georgian, Macedonian, Polish, Russian, Serbian, Slovak, Slovenian, Ukrainian

### shrutilipi-process/

Process ShrutiLipi parquet datasets for 18 Indic languages: Assamese, Bengali,
Dogri, Gujarati, Hindi, Kannada, Konkani, Maithili, Malayalam, Marathi, Nepali,
Odia, Punjabi, Sanskrit, Tamil, Telugu, and more.

### madASR/

Process MadASR 2.0 datasets with HuggingFace integration (`hf-process/`
subdirectory) for Bengali, Bhojpuri, Chhattisgarhi, Kannada, Magahi, Maithili,
Marathi, and Telugu.

### process-data-en/

English dataset processors for NPTEL lectures, AV Speech, CommonVoice,
OpenSR, YouTube data, and VILS.

### process-data-mls/

European language processors for MLS (Multilingual LibriSpeech) and CommonVoice
datasets: French, German, Portuguese, Spanish, Italian, Russian, Bulgarian.

## hf/ -- HuggingFace Format Converters

| File | Description |
|------|-------------|
| `paraquet2manifest.py` | Convert HuggingFace Parquet datasets to NeMo JSONL manifests |

## meta-mt/ -- Meta Machine Translation

| File | Description |
|------|-------------|
| `translate_gemini.py` | Translate tagged text using Gemini while preserving metadata tag structure |

## utils/ -- General Utilities

| File | Description |
|------|-------------|
| `verify_audio_manifest.py` | Verify all audio files in a manifest exist and are readable |
| `fix_encoding_eu.py` | Fix character encoding issues in European language manifests |
| `fixed_librespeech_paths.py` | Fix LibriSpeech audio file paths in manifests |
| `gcp/gemini.py` | GCP Gemini API client for transcription and annotation |
| `gcp/transcribe.py` | Transcribe audio using GCP Speech-to-Text API |

## Getting Started

Most scripts follow a common pattern: read a NeMo JSONL manifest, process
entries (download audio, annotate, normalize), and write an output manifest.

```bash
# Example: process CommonVoice data for a new language
cd scripts/data/audio/1person/real
python process_cv.py \
    --language hi \
    --output-dir /data/hindi \
    --split train

# Example: create synthetic tagged data
cd scripts/data/audio/1person/synthetic
python create_synthetic_tagged_text.py \
    --domain iot \
    --output synthetic_iot.jsonl

# Example: convert HuggingFace parquet to NeMo manifest
cd scripts/data/hf
python paraquet2manifest.py \
    --dataset WhissleAI/Meta_STT_EN_Set1 \
    --output /data/english/train.jsonl
```

For the recommended data preparation workflow using the higher-level utility
scripts, see [`scripts/asr/meta-asr/utils/README.md`](../asr/meta-asr/utils/README.md).
