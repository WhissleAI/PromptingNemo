# WhissleAI Ecosystem

PromptingNemo is part of a broader set of tools for building Speech-2-Action systems.

## Core Repositories

### [PromptingNemo](https://github.com/WhissleAI/PromptingNemo) (this repo)
Training toolkit for meta-ASR models. Handles tokenizer training, model fine-tuning, evaluation, and export.

### [NeMo-W](https://github.com/WhissleAI/NeMo-W)
WhissleAI's fork of NVIDIA NeMo. Contains the base model architectures (FastConformer, Parakeet) and CTC training infrastructure that PromptingNemo builds on top of. The `CustomEncDecCTCModelBPE` in PromptingNemo extends NeMo's `EncDecCTCModelBPE`.

The [`audio-visual-balu`](https://github.com/WhissleAI/NeMo-W/tree/audio-visual-balu/balu_codes) branch contains the original research prototype for Audio-Visual ASR (EMNLP 2025). This prototype demonstrated CLIP-based visual feature fusion for noisy speech recognition and was subsequently productionized into PromptingNemo's `promptingnemo/models/av_ctc_model.py` and `recipes/av_asr/` modules.

### [espnet-w](https://github.com/WhissleAI/espnet-w)
WhissleAI's fork of ESPnet. Provides an alternative training backend with the `egs2/` recipe system. PromptingNemo's `recipes/espnet_asr/` provides a bridge recipe for training meta-ASR models using ESPnet's infrastructure.

### [whissle-annotator](https://github.com/WhissleAI/whissle-annotator)
YAML-driven data annotation pipeline with 10 stages:

```
s01_ingest → s02_diarize → s03_transcribe → s04_audio_classify
→ s05_entity_intent → s06_visual_extract → s07_visual_classify
→ s08_crossmodal → s09_quality_check → s10_merge_finalize
```

Supports multiple data connectors (LibriSpeech, YouTube, CommonVoice) and annotation backends (Whisper, Gemini, custom models). Output manifests feed directly into PromptingNemo training.

## SDKs & APIs

### [whissle_python_api](https://github.com/WhissleAI/whissle_python_api)
Python SDK for the Whissle API. Multi-modal speech processing client.

### [live_assist_js_sdk](https://github.com/WhissleAI/live_assist_js_sdk)
JavaScript SDK with React components for real-time speech processing in web applications.

## Data & Models on HuggingFace

### Pretrained Models
| Model | Languages | Parameters | Link |
|-------|-----------|------------|------|
| STT-meta-1B | Multilingual (20+ langs) | 600M | [WhissleAI/STT-meta-1B](https://huggingface.co/WhissleAI/STT-meta-1B) |
| stt_en_conformer_ctc_large_slurp | English (IoT) | 115M | [WhissleAI/stt_en_conformer_ctc_large_slurp](https://huggingface.co/WhissleAI/stt_en_conformer_ctc_large_slurp) |
| stt_hi_conformer_ctc_large_with_meta | Hindi | 110M | [WhissleAI/stt_hi_conformer_ctc_large_with_meta](https://huggingface.co/WhissleAI/stt_hi_conformer_ctc_large_with_meta) |

### Key Datasets
| Dataset | Language | Description |
|---------|----------|-------------|
| [Meta_STT_HI_Set1](https://huggingface.co/datasets/WhissleAI/Meta_STT_HI_Set1) | Hindi | Tagged Hindi speech with entities, emotion, age, gender |
| [Meta_STT_ZH_AIShell3](https://huggingface.co/datasets/WhissleAI/Meta_STT_ZH_AIShell3) | Mandarin | Tagged Chinese speech |
| [Meta_STT_EURO_Set1](https://huggingface.co/datasets/WhissleAI/Meta_STT_EURO_Set1) | European (5 langs) | Multi-language European speech |
| [Meta_STT_SLAVIC_CommonVoice](https://huggingface.co/datasets/WhissleAI/Meta_STT_SLAVIC_CommonVoice) | Slavic (10 langs) | CommonVoice with meta-tags |

| VANS (Visual-Aware Noisy Speech) | English | AudioSet noise videos mixed with People's Speech clean audio at variable SNRs. Used for Audio-Visual ASR training. |

See [all datasets](https://huggingface.co/WhissleAI) for the complete list (34 datasets).

## Other Tools

| Repository | Purpose |
|-----------|---------|
| [agentic_youtube_scrapper](https://github.com/WhissleAI/agentic_youtube_scrapper) | YouTube data collection with Google ADK |
| [Synthetic_ASR-NL_data](https://github.com/WhissleAI/Synthetic_ASR-NL_data) | Synthetic speech data generation |
| [kenlm-w](https://github.com/WhissleAI/kenlm-w) | Language model for ASR decoding |
| [piper](https://github.com/WhissleAI/piper) | Fast local TTS |
