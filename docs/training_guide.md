# Training Guide

End-to-end guide for training a Speech-2-Action model with PromptingNemo -- from raw data to a deployed model that transcribes and tags in a single pass.

## Overview

The training pipeline has six stages:

```
1. Data Preparation     Download, annotate, normalize manifests
2. Tokenizer Training   Build per-language SentencePiece models + aggregate vocabulary
3. Model Training       Fine-tune with slim decoder, tag weight init, adapters
4. Evaluation           WER (transcription) + NER F1 (tagging accuracy)
5. Export               NeMo -> ONNX / HuggingFace
6. Deployment           Serve via ONNX runtime or NeMo inference
```

## Prerequisites

- Python 3.10+
- GPU with 16+ GB VRAM (A100, V100, T4, or similar)
- `pip install -e ".[all]"` or the Docker training image

```bash
# Docker (recommended — run from repo root)
docker build -t promptingnemo:training -f docker/Dockerfile.training .
docker run --gpus all -it --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/experiments:/experiments \
  promptingnemo:training bash
```

---

## 1. Data Preparation

### 1.1 Download from HuggingFace

WhissleAI provides pre-tagged datasets on HuggingFace. You can download them directly:

```python
from datasets import load_dataset

ds = load_dataset("WhissleAI/Meta_STT_HI_Set1")
```

Available datasets include:
- `WhissleAI/Meta_STT_HI_Set1` -- Hindi with entities, emotion, age, gender
- `WhissleAI/Meta_STT_ZH_AIShell3` -- Mandarin Chinese
- `WhissleAI/Meta_STT_EURO_Set1` -- 5 European languages
- `WhissleAI/Meta_STT_SLAVIC_CommonVoice` -- 10 Slavic languages

See the [full list on HuggingFace](https://huggingface.co/WhissleAI) (34+ datasets).

### 1.2 Annotate with whissle-annotator

For raw (un-tagged) audio data, use the [whissle-annotator](https://github.com/WhissleAI/whissle-annotator) pipeline to add NER, emotion, age, gender, intent, and dialect tags:

```bash
# Configure your annotation pipeline in a YAML file, then run:
# s01_ingest -> s02_diarize -> s03_transcribe -> s04_audio_classify
# -> s05_entity_intent -> s06_visual_extract -> s07_visual_classify
# -> s08_crossmodal -> s09_quality_check -> s10_merge_finalize
```

The annotator outputs NeMo JSONL manifests that feed directly into PromptingNemo training.

### 1.3 Normalize Tags

Raw annotated data often contains inconsistent tag formats. Normalize them to the canonical schema before training:

```python
from promptingnemo.data.normalize import normalize_text

# Fixes common issues:
# GER_MALE -> GENDER_MALE
# EMOTION_HAP -> EMOTION_HAPPY
# AGE_60PLUS -> AGE_60+
text = normalize_text("hello GER_FEMALE EMOTION_HAP AGE_60PLUS")
# -> "hello GENDER_FEMALE EMOTION_HAPPY AGE_60+"
```

For batch normalization of manifests, process each line:

```python
import json

with open("raw_manifest.json") as f_in, open("clean_manifest.json", "w") as f_out:
    for line in f_in:
        entry = json.loads(line)
        entry["text"] = normalize_text(entry["text"])
        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

### 1.4 Manifest Format

The training manifest is a JSONL file where each line is a JSON object:

```json
{"audio_filepath": "/data/audio/001.wav", "duration": 4.1, "text": "ENTITY_PERSON_NAME Rahul END called about the ENTITY_DATE meeting END EMOTION_HAPPY AGE_18_30 GENDER_MALE", "lang": "HINDI", "lang_family": "INDO_ARYAN"}
```

Required fields:
- `audio_filepath` -- absolute or relative path to the audio file (16kHz WAV recommended)
- `duration` -- audio duration in seconds
- `text` -- tagged transcription

Optional fields:
- `lang` -- language name (used for tokenizer training)
- `lang_family` -- language family grouping (used for balanced sampling)

### 1.5 Validate Manifests

Before training, validate that audio files exist and tags are well-formed:

```bash
cd recipes/meta_asr
python train.py --config conf/hindi.yaml --mode validate_data
```

This scans all manifests, drops entries with missing audio or malformed tags, and reports statistics.

---

## 2. Tokenizer Training

### 2.1 How the Aggregate Tokenizer Works

PromptingNemo uses an aggregate tokenizer that combines per-language SentencePiece models into a unified vocabulary. Each language family gets its own subword segmentation, but they all share common special tokens (tag tokens like `ENTITY_PERSON_NAME`, `EMOTION_HAPPY`, `AGE_30_45`, etc.).

This design allows a single model to handle multiple languages while maintaining language-specific subword tokenization quality.

### 2.2 Configuration

The tokenizer configuration lives in the training YAML file under `model.dynamic_tokenizer_params`:

```yaml
model:
  use_aggregate_tokenizer: true
  dynamic_tokenizer_params:
    type: bpe
    non_special_tokens_per_lang: 1024         # Subword vocab per language
    non_special_tokens_per_lang_overrides:
      ENGLISH: 1024
      MANDARIN: 3000                          # Larger for logographic scripts
    character_coverage: null
    character_coverage_overrides:
      MANDARIN: 0.991
  special_token_prefixes:
    - ENTITY_
    - INTENT_
    - EMOTION_
    - GENDER_
    - AGE_
    - KEYWORD_
    - LANG_
```

### 2.3 Train the Tokenizer

Run tokenizer-only mode:

```bash
cd recipes/meta_asr
python train.py --config conf/hindi.yaml --mode tokenizer
```

This will:
1. Scan the training manifest for all languages and special tokens
2. Train a SentencePiece model for each language, injecting shared special tokens
3. Build the aggregate vocabulary (union of all language vocabularies)
4. Save tokenizer metadata back to the config file

The resulting tokenizer files are stored in the directory specified by `model.new_tokenizer_folder`.

---

## 3. Model Training

### 3.1 Base Model

Training starts from a pretrained checkpoint. The recommended base model is [STT-meta-1B](https://huggingface.co/WhissleAI/STT-meta-1B) (600M parameters, FastConformer/Parakeet architecture, 20+ languages).

Download it:

```bash
# Using the GCP tooling:
cd recipes/meta_asr/gcp
./download-model.sh --model WhissleAI/STT-meta-1B

# Or manually:
huggingface-cli download WhissleAI/STT-meta-1B --local-dir /workspace/pretrained/
```

### 3.2 Slim Decoder

The pretrained model has an ~18K-token vocabulary covering all language families. When fine-tuning for a specific language family (e.g., `INDO_ARYAN`), the slim decoder removes tokens from non-target families, reducing the output space to ~10K tokens. This improves training efficiency and reduces spurious predictions.

Configure target families in the YAML:

```yaml
model:
  language_families:
    - INDO_ARYAN
```

The slim decoder:
- Removes transcription tokens unique to non-target language families
- Keeps all shared special tokens (entity, emotion, age, gender, intent tags)
- Preserves pretrained decoder weights for all kept tokens
- Rebuilds the CTC loss layer for the new vocabulary size

### 3.3 Tag Weight Initialization (`scale_down_tag_decoder_weights`)

After slim decoder pruning, sentence-level tags (`AGE_*`, `GENDER_*`, `EMOTION_*`, `INTENT_*`, `DIALECT_*`) can inherit disproportionately strong decoder weights from the pretrained model. This causes them to fire at every frame instead of only at end-of-utterance positions.

`scale_down_tag_decoder_weights()` re-initializes these tag token weights to small random values (default scale: 0.01), forcing the model to learn proper positional firing from the CTC loss signal during fine-tuning.

This step runs automatically during training -- no manual configuration needed.

### 3.4 Adapter Fine-Tuning

For lightweight fine-tuning without modifying the encoder weights, enable adapter layers:

```yaml
adapter:
  enabled: true
  name: lang_adapter
  dim: 128
  activation: swish
  norm_position: pre
  unfreeze_decoder: false    # true = freeze encoder, unfreeze decoder
```

When adapters are enabled:
- Linear adapter modules are inserted into each Conformer encoder layer
- The base encoder weights are frozen
- Only the adapter parameters and (optionally) the decoder are trained
- This is much faster and uses less memory than full fine-tuning

### 3.5 Training Configuration

Key training hyperparameters:

```yaml
training:
  batch_size: 76
  accumulate_grad_batches: 8    # Effective batch = 76 * 8 = 608
  max_duration: 22.0            # Max audio duration in seconds
  max_steps: 10000000
  num_workers: 12
  pin_memory: true

  # Keyword loss (optional, helps with entity recognition)
  use_keyword_loss: true
  keyword_loss_weight: 0.6
  keyword_loss_warmup_steps: 1000

  # Family-balanced loss weighting
  use_family_loss_weights: true

  # SpecAugment
  spec_augment:
    time_masks: 4
    time_width: 80

  # Optimizer
  optim:
    lr: 0.0001
    weight_decay: 0.0
    sched:
      warmup_steps: 5000
```

### 3.6 Run Training

Full pipeline (tokenizer + training):

```bash
cd recipes/meta_asr
python train.py --config conf/hindi.yaml --mode both
```

Training only (tokenizer already built):

```bash
python train.py --config conf/hindi.yaml --mode train
```

Resume from a checkpoint:

```bash
python train.py --config conf/hindi.yaml --mode train --resume_from /path/to/checkpoint.ckpt
```

Or use the CLI entry point directly:

```bash
promptingnemo --config conf/hindi.yaml --mode both
```

### 3.7 GCP Spot Instance Training

For cost-effective training on GCP, use the spot instance tooling:

```bash
cd recipes/meta_asr/gcp

# One-time setup
./create-training-disk.sh
./launch-experiment.sh --name my-exp --gpu t4
./setup-instance.sh

# Download model and data
./download-model.sh --model WhissleAI/STT-meta-1B
./download-data.sh --dataset WhissleAI/Meta_STT_HI_Set1 --lang HINDI

# Train
./run-finetune.sh --model WhissleAI/STT-meta-1B \
  --dataset WhissleAI/Meta_STT_HI_Set1 --lang HINDI \
  --mode adapter --name hindi-adapter-v1
```

Persistent disks preserve data across spot preemptions. Use `--resume_from` to continue training after interruptions.

### 3.8 Balanced Language Sampling

For multi-language training, `BalancedLanguageBatchSampler` ensures each batch contains proportional representation from all language families, preventing dominant languages from monopolizing training. It uses the `lang_family` field in the manifest and the `language_family_map` in the config.

### 3.9 Audio Augmentation

Training automatically applies:
- **White noise perturbation** (SNR range -90 to -46 dB)
- **Time shift perturbation** (100-500 ms)
- **SpecAugment** (configurable time masks)

---

## 4. Evaluation

### 4.1 Transcription Accuracy (WER)

Word Error Rate is computed on the transcription-only portion of the output (all uppercase tag tokens are stripped before scoring):

```python
from promptingnemo.eval.wer import multi_word_error_rate

tags = {
    "NER": ["ENTITY_PERSON_NAME", "ENTITY_ORGANIZATION", "ENTITY_DATE"],
    "END": ["END"],
}

wer, ner_metrics = multi_word_error_rate(
    hypotheses=predictions,
    references=ground_truth,
    tags=tags,
    score_type="label",
)
print(f"WER: {wer:.2%}")
```

### 4.2 Entity Tagging Accuracy (NER F1)

NER scoring extracts `ENTITY_<TYPE> ... END` spans from both predictions and ground truth, then computes precision, recall, and F1:

```python
print(f"Micro F1: {ner_metrics['overall_micro']['fscore']:.3f}")
print(f"Macro F1: {ner_metrics['overall_macro']['fscore']:.3f}")

# Per-entity-type breakdown
for tag, scores in ner_metrics.items():
    if tag.startswith("ENTITY_"):
        print(f"  {tag}: P={scores['precision']:.3f} R={scores['recall']:.3f} F1={scores['fscore']:.3f}")
```

The `score_type` parameter controls matching:
- `"label"` -- match entity type only (ignores the span text)
- `"exact"` -- match both entity type and span text

### 4.3 Running Inference on a Test Set

Use the inference utilities to transcribe a manifest:

```python
from promptingnemo.eval.inference import transcribe_manifest

transcribe_manifest(
    checkpoint_path="/workspace/experiments/best_model.nemo",
    input_jsonl="/data/test_manifest.json",
    output_jsonl="/data/predictions.json",
    batch_size=8,
    use_gpu=True,
)
```

### 4.4 Benchmarking (GCP)

After training on GCP, run the benchmark script:

```bash
cd recipes/meta_asr/gcp
./benchmark.sh --name hindi-adapter-v1
```

---

## 5. Export

### 5.1 ONNX Export

Convert a trained NeMo checkpoint to ONNX for deployment:

```python
from promptingnemo.export.to_onnx import export_nemo_to_onnx

export_nemo_to_onnx(
    nemo_model_path="/workspace/experiments/best_model.nemo",
    save_directory="/workspace/exported/onnx_model",
)
```

This produces:
- `model.onnx` -- the ONNX model
- `tokenizer/tokenizer.model` -- the SentencePiece tokenizer
- `magic.yaml` / `magic.txt` -- preprocessor configuration (sample rate, feature extraction params)

### 5.2 HuggingFace Export

Upload a trained model to HuggingFace Hub:

```python
from promptingnemo.export.to_hf import upload_nemo_to_hf

upload_nemo_to_hf(
    nemo_model_path="/workspace/experiments/best_model.nemo",
    repo_id="WhissleAI/my-new-model",
    hf_token="hf_xxx",
)
```

Or from the GCP tooling:

```bash
cd recipes/meta_asr/gcp
./upload-model.sh --name hindi-adapter-v1 \
  --hf-repo WhissleAI/stt_hi_conformer_ctc_large_with_meta_v2 \
  --hf-token hf_xxx
```

---

## 6. Deployment

### 6.1 ONNX Runtime Inference

After exporting to ONNX, serve the model with ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
# Preprocess audio to log-mel features (see magic.yaml for params)
# Run inference
outputs = session.run(None, {"audio_signal": features, "length": lengths})
```

The ONNX model is also compatible with the [Whissle unified Docker image](https://github.com/WhissleAI/PromptingNemo/tree/main/docker) which bundles the ASR engine, Python agent server, and nginx gateway.

### 6.2 NeMo Inference

For direct NeMo inference (useful for prototyping):

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.restore_from("best_model.nemo")
transcriptions = model.transcribe(["audio.wav"])
```

### 6.3 Production SDKs

For production applications, use the Whissle SDKs:

- **JavaScript/React:** [live_assist_js_sdk](https://github.com/WhissleAI/live_assist_js_sdk) -- real-time streaming ASR with entity/emotion extraction
- **Python:** [whissle_python_api](https://github.com/WhissleAI/whissle_python_api) -- REST/WebSocket client for the Whissle API

---

## Appendix: Example Configs

### Single-Language Fine-Tuning (Hindi)

```yaml
model:
  model_root: /workspace/pretrained/
  model_name: STT-meta-1B.nemo
  use_aggregate_tokenizer: true
  language_families:
    - INDO_ARYAN
  special_token_prefixes:
    - ENTITY_
    - EMOTION_
    - GENDER_
    - AGE_
    - INTENT_

training:
  data_dir: /workspace/data/hindi/
  train_manifest: train.json
  test_manifest: test.json
  batch_size: 32
  max_steps: 50000
  max_duration: 20.0

experiment:
  exp_dir: /workspace/experiments/
  exp_name: hindi-finetune-v1
  monitor: val_wer
  mode: min
  always_save_nemo: true
  save_top_k: 3
  every_n_train_steps: 5000
```

### Adapter Fine-Tuning (Mandarin)

```yaml
model:
  model_root: /workspace/pretrained/
  model_name: STT-meta-1B.nemo
  use_aggregate_tokenizer: true
  language_families:
    - MANDARIN
  dynamic_tokenizer_params:
    non_special_tokens_per_lang_overrides:
      MANDARIN: 3000
    character_coverage_overrides:
      MANDARIN: 0.991

adapter:
  enabled: true
  name: mandarin_adapter
  dim: 128
  activation: swish
  unfreeze_decoder: true

training:
  data_dir: /workspace/data/mandarin/
  train_manifest: train.json
  test_manifest: test.json
  batch_size: 48
  max_steps: 100000
  max_duration: 22.0

experiment:
  exp_dir: /workspace/experiments/
  exp_name: mandarin-adapter-v1
  monitor: val_wer
  mode: min
  always_save_nemo: true
```
