# PromptingNemo

**Speech-2-Action: Train ASR models that transcribe and tag in a single pass**

## What is PromptingNemo?

PromptingNemo is a training toolkit for building Speech-2-Action models -- automatic speech recognition systems that output both transcription **and** structured meta-tags in a single CTC decoding pass. Instead of piping ASR output through separate NER, emotion classification, and speaker attribute models, PromptingNemo trains a single end-to-end model that emits inline entity tags (`ENTITY_PERSON_NAME John Smith END`), sentence-level attribute tags (`AGE_30_45 GENDER_FEMALE EMOTION_HAPPY`), intent labels, and dialect markers alongside the transcription itself.

The key insight is that CTC models can learn to produce structured tokens interleaved with speech tokens when the training data embeds those tags directly in the transcript text. PromptingNemo provides the full pipeline to make this work at scale: data annotation and normalization, aggregate multi-language tokenizer training, slim decoder pruning for efficient fine-tuning, tag weight initialization to prevent mid-utterance tag spam, and balanced multi-language batch sampling.

PromptingNemo builds on top of [NeMo-W](https://github.com/WhissleAI/NeMo-W) (WhissleAI's fork of NVIDIA NeMo) and supports models from the FastConformer / Parakeet family. It also provides a bridge recipe for [ESPnet-W](https://github.com/WhissleAI/espnet-w) as an alternative training backend. Pretrained models covering 20+ languages are available on [HuggingFace](https://huggingface.co/WhissleAI).

## Install

```bash
pip install -e .                    # Core (data prep, normalization, evaluation)
pip install -e ".[train]"           # + NeMo for GPU training
pip install -e ".[all]"             # Everything (train + export + dev tools)
```

**Using Docker** (recommended for training):

```bash
# Build the training image (from repo root)
docker build -t promptingnemo:training -f docker/Dockerfile.training .

# Interactive training shell with GPU
docker run --gpus all -it --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/experiments:/experiments \
  promptingnemo:training bash

# Or use docker compose for training + tensorboard
DATA_DIR=/path/to/data EXPERIMENTS_DIR=/path/to/experiments \
  docker compose run --rm training bash
```

## Quick Start

```bash
# 1. Prepare a tagged manifest (NeMo JSONL format)
python -c "
from promptingnemo.data.normalize import normalize_text
print(normalize_text('hello GER_FEMALE EMOTION_HAP AGE_60PLUS'))
# -> hello GENDER_FEMALE EMOTION_HAPPY AGE_60+
"

# 2. Train a model
cd recipes/meta_asr
python train.py --config conf/hindi.yaml --mode both
```

The `--mode` flag controls which stages to run:
- `both` -- train tokenizer, then train model (default)
- `tokenizer` -- train the aggregate tokenizer only
- `train` -- train the model only (requires tokenizer to exist)
- `validate_data` -- validate manifests without training

See the [Training Guide](docs/training_guide.md) for multi-GPU and cloud training tips.

## Audio-Visual ASR (NEW)

PromptingNemo now includes an Audio-Visual ASR extension for visual-aware noisy speech recognition. Based on the EMNLP 2025 paper by Darur & Singla, this extension uses CLIP visual features from video frames to contextualize noise sources, significantly improving ASR accuracy in noisy conditions. The best model (AV-UNI-SNR, 453M params) achieves 20.71 WER at 10dB SNR versus 26.99 for audio-only Conformer-CTC, and is competitive with Whisper Large V3 (1.55B params).

**Quick start:**

```bash
cd recipes/av_asr
python train.py --config conf/av_conformer_ctc.yaml
```

See [`recipes/av_asr/README.md`](recipes/av_asr/README.md) for the full training guide, and [`docs/audio_visual.md`](docs/audio_visual.md) for the architecture deep dive.

## Pretrained Models

| Model | Languages | Parameters | Link |
|-------|-----------|------------|------|
| STT-meta-1B | Multilingual (20+ langs) | 600M | [WhissleAI/STT-meta-1B](https://huggingface.co/WhissleAI/STT-meta-1B) |
| stt_en_conformer_ctc_large_slurp | English (IoT) | 115M | [WhissleAI/stt_en_conformer_ctc_large_slurp](https://huggingface.co/WhissleAI/stt_en_conformer_ctc_large_slurp) |
| stt_hi_conformer_ctc_large_with_meta | Hindi | 110M | [WhissleAI/stt_hi_conformer_ctc_large_with_meta](https://huggingface.co/WhissleAI/stt_hi_conformer_ctc_large_with_meta) |

Single-language models are also available for Bengali, Marathi, Punjabi, and others. See the full list on [HuggingFace](https://huggingface.co/WhissleAI).

**Inference:**

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("WhissleAI/stt_en_conformer_ctc_large_slurp")
transcriptions = model.transcribe(["file.wav"])
# -> "turn on the lights ENTITY_DEVICE lights END INTENT_COMMAND AGE_30_45 GENDER_MALE EMOTION_NEUTRAL"
```

## Recipes

| Recipe | Description |
|--------|-------------|
| [`recipes/meta_asr/`](recipes/meta_asr/) | Full meta-ASR training pipeline (tokenizer + model). Language configs in `conf/`. |
| [`recipes/data_prep/`](recipes/data_prep/) | Data preparation and manifest normalization utilities. |
| [`recipes/espnet_asr/`](recipes/espnet_asr/) | ESPnet-W integration for training meta-ASR with ESPnet's `egs2/` system. |
| [`recipes/av_asr/`](recipes/av_asr/) | Audio-Visual ASR training with CLIP visual features for noisy speech recognition. |

Additional training scripts and data tools are in [`scripts/asr/meta-asr/`](scripts/asr/meta-asr/), including evaluation scripts, data download utilities, and per-language YAML configs.

## Tag Schema

PromptingNemo models emit uppercase tag tokens in the transcription stream. There are two categories:

| Category | Format | Example Tags |
|----------|--------|-------------|
| **Inline** (entity spans) | `ENTITY_<TYPE> ... END` | `ENTITY_PERSON_NAME`, `ENTITY_ORGANIZATION`, `ENTITY_DATE` |
| **Sentence-level** (appended) | `<transcript> TAG_VALUE` | `AGE_30_45`, `GENDER_FEMALE`, `EMOTION_HAPPY`, `INTENT_INFORM`, `DIALECT_BIHAR` |

Use `promptingnemo.data.normalize.normalize_text()` to canonicalize legacy tag formats (e.g., `GER_MALE` to `GENDER_MALE`, `EMOTION_HAP` to `EMOTION_HAPPY`).

See [docs/tag_schema.md](docs/tag_schema.md) for the complete tag reference.

## WhissleAI Ecosystem

PromptingNemo is part of a broader set of tools for building Speech-2-Action systems:

| Repository | Description |
|-----------|-------------|
| [whissle-annotator](https://github.com/WhissleAI/whissle-annotator) | YAML-driven 10-stage annotation pipeline (ingest, diarize, transcribe, classify, NER, merge) |
| [NeMo-W](https://github.com/WhissleAI/NeMo-W) | Training backend -- WhissleAI's fork of NVIDIA NeMo with CTC meta-tag support |
| [espnet-w](https://github.com/WhissleAI/espnet-w) | Alternative training backend -- WhissleAI's fork of ESPnet |
| [HuggingFace](https://huggingface.co/WhissleAI) | Pretrained models and 34+ tagged speech datasets |
| [live_assist_js_sdk](https://github.com/WhissleAI/live_assist_js_sdk) | JavaScript SDK with React components for real-time speech processing |
| [whissle_python_api](https://github.com/WhissleAI/whissle_python_api) | Python SDK for the Whissle speech API |

See [docs/ecosystem.md](docs/ecosystem.md) for details on how these components connect.

## Architecture

```
promptingnemo/
  tokenizer/    Aggregate multi-language tokenizer training
  models/       Custom CTC model with slim decoder and tag weight init
  data/         Robust dataset loading, balanced sampling, tag normalization
  training/     Training orchestration and CLI
  eval/         WER + NER F1 evaluation
  export/       ONNX and HuggingFace export
```

The training pipeline flows through five stages: data preparation, tokenizer training, model training (with slim decoder and adapter support), evaluation, and export. See [docs/architecture.md](docs/architecture.md) for the full design and key decisions. For a complete walkthrough, see the [Training Guide](docs/training_guide.md).

## Publications

Karan, S., Shahab, J., Yeon-Jun, K., Andrej, L., Moreno, D. A., Srinivas, B., & Benjamin, S. (2023, December). *1-step Speech Understanding and Transcription Using CTC Loss.* In Proceedings of the 20th International Conference on Natural Language Processing (ICON) (pp. 370-377).

Karan, S., Mahnoosh, M., Daniel, P., Ryan, P., Srinivas, C. B., Yeon-Jun, K., & Srinivas, B. (2023, December). *Combining Pre trained Speech and Text Encoders for Continuous Spoken Language Processing.* In Proceedings of the 20th International Conference on Natural Language Processing (ICON) (pp. 832-842).

Darur, B. & Singla, K. (2025). *Visual-Aware Speech Recognition for Noisy Scenarios.* In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP). [https://aclanthology.org/2025.emnlp-main.845/](https://aclanthology.org/2025.emnlp-main.845/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

## License

This project is licensed under the Apache License 2.0. See [LICENCE](./LICENCE) for details.
