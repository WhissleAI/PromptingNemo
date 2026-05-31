# promptingnemo

Core Python package for the PromptingNemo Speech-2-Action training toolkit. Provides dataset loading, model definitions, tokenizer construction, training orchestration, evaluation metrics, and model export utilities for multilingual metadata-enriched ASR.

## Installation

```bash
# Core (data prep, normalization, tokenizer utilities, evaluation)
pip install -e .

# With NeMo for GPU training
pip install -e ".[train]"

# Everything (train + ONNX export + dev tools)
pip install -e ".[all]"
```

## Package Structure

```
promptingnemo/
├── data/           Dataset loading, normalization, sampling, tag parsing
├── models/         CTC models with keyword loss, AV fusion, decoder utilities
├── tokenizer/      Aggregate multi-language tokenizer, SentencePiece, dedup merge
├── training/       Training orchestration, CLI entry points
├── eval/           WER, NER F1, AV-WER, inference utilities
└── export/         ONNX and HuggingFace Hub export
```

---

## data -- Dataset Loading and Normalization

Tools for loading, filtering, and normalizing NeMo-format JSONL manifests. Supports audio, audio-visual, and text-only training pipelines.

| Module | Description |
|--------|-------------|
| `dataset.py` | `RobustAudioToBPEDataset` -- extends NeMo's `AudioToBPEDataset` with language-ID normalization, manifest validation, audio readability checks, and keyphrase oversampling. |
| `av_dataset.py` | `AVToBPEDataset` -- Audio-Visual dataset for CTC training. Loads clean audio, mixes with noise extracted from video files at configurable SNR ratios, and provides pre-extracted CLIP ViT-L/14 visual features. |
| `normalize.py` | Text normalization for meta-ASR tags. Fixes common annotation typos (`EMOTION_HAP` -> `EMOTION_HAPPY`, `GER_` -> `GENDER_`), normalizes age/gender/emotion tags to canonical forms. |
| `tag_parser.py` | Parses tagged text into (clean_text, tagged_text) pairs. Supports trailing tags and inline entities (`ENTITY_PERSON John END`). Includes compositional tag decomposition for zero-shot tag generalization. |
| `sampler.py` | `BalancedLanguageBatchSampler` -- temperature-scaled language-family-aware batch sampling for distributed training. |
| `manifest.py` | Manifest validation utilities. Multi-threaded audio readability checking. |
| `text_tagger_dataset.py` | `TextTaggerDataset` -- character-input dataset for text CTC tagger training. |
| `text_tagger_dataset_v2.py` | `TextTaggerDatasetV2` -- subword-input variant using the aggregate tokenizer. |
| `text_tagger_dataset_v3.py` | `TextTaggerDatasetV3` -- XLM-RoBERTa input with chunked trailing tags. |

**Quick example:**

```python
from promptingnemo.data.normalize import normalize_text, extract_tags

raw = "hello GER_FEMALE EMOTION_HAP AGE_60+"
print(normalize_text(raw))
# -> "hello GENDER_FEMALE EMOTION_HAPPY AGE_60PLUS"

emotions, ages = extract_tags(raw)
# emotions: ['EMOTION_HAPPY'], ages: ['AGE_60PLUS']
```

```python
from promptingnemo.data.tag_parser import parse_tagged_text, decompose_tag

clean, tagged = parse_tagged_text("ENTITY_PERSON John END said hello EMOTION_HAPPY")
# clean: "John said hello"
# tagged: "ENTITY_PERSON John END said hello EMOTION_HAPPY"

pieces = decompose_tag("INTENT_REPORT_SYMPTOM")
# -> ['INTENT_', 'REPORT', '_SYMPTOM']
```

---

## models -- CTC Models and Decoder Utilities

Model definitions for CTC-based ASR with metadata tagging and audio-visual fusion.

| Module | Description |
|--------|-------------|
| `ctc_model.py` | `CustomEncDecCTCModelBPE` -- extends NeMo's CTC BPE model with keyword loss (CTC on entity tokens only, with warmup), per-language-family WER/loss tracking during validation, and language-family-weighted CTC loss. Uses `FlexibleSaveRestoreConnector` for vocab-size-mismatched checkpoint loading. |
| `av_ctc_model.py` | `AVEncDecCTCModelBPE` -- Audio-Visual CTC model implementing the AV-UNI-SNR architecture (EMNLP 2025). Wraps a pretrained Conformer encoder with a Transformer fusion module over concatenated audio + CLIP ViT-L/14 visual features. |
| `decoder.py` | Decoder manipulation utilities: `scan_manifest_for_new_tokens` (find tags missing from vocabulary), `extend_decoder_for_new_tokens` (add new output tokens), `slim_decoder_for_training` (remove non-target language tokens), `scale_down_tag_decoder_weights` (re-initialize tag logits). |
| `text_ctc_model.py` | `TextCTCTagger` -- text-only CTC tagger. Architecture: CharacterEmbedding -> LearnedUpsampler -> TransformerEncoder -> Linear -> CTC. |
| `text_ctc_model_v2.py` | `TextCTCTaggerV2` -- subword-input variant with a Conformer encoder. |
| `text_ctc_model_v3.py` | `TextCTCTaggerV3` -- XLM-RoBERTa encoder as input, projecting to aggregate vocabulary. |

---

## tokenizer -- Aggregate Multi-Language Tokenizer

Trains per-language-family SentencePiece models and merges them into a single deduplicated vocabulary with shared special tokens.

| Module | Description |
|--------|-------------|
| `config.py` | Language family configuration and tokenizer metadata persistence. |
| `sentencepiece.py` | Trains a SentencePiece model from a manifest file with user-defined special symbols. |
| `aggregate.py` | Aggregate tokenizer training pipeline. Creates one SentencePiece model per language family and deduplicates into a global vocabulary. |
| `dedup_aggregate.py` | `DedupAggregateTokenizer` -- patches NeMo's `AggregateTokenizer` with deduplicated global vocabulary. |
| `meta_tokenizer.py` | `MetaTokenizer` -- lightweight wrapper using STT-meta-1B aggregate vocabulary without NeMo dependency. |
| `text_tagger_tokenizer.py` | `TextTaggerTokenizer` -- hybrid tokenizer for text CTC tagger with compositional tag decomposition. |

**Training a tokenizer from a config:**

```bash
promptingnemo --config recipes/meta_asr/conf/hindi.yaml --mode tokenizer
```

---

## training -- Trainer and CLI

Training orchestration, checkpoint loading, data setup, and CLI entry points.

| Module | Description |
|--------|-------------|
| `trainer.py` | `train_model` -- end-to-end training function. Handles checkpoint loading, tokenizer setup, decoder manipulation (slim/extend/rescale), balanced batch sampling, audio/spec augmentation, optimizer/scheduler configuration, and NeMo experiment manager integration. |
| `cli.py` | `main` -- command-line entry point. Modes: `both` (tokenizer + training), `tokenizer`, `train`, `validate_data`. |

**CLI usage:**

```bash
# Full pipeline: tokenizer + training
promptingnemo --config recipes/meta_asr/conf/hindi.yaml --mode both

# Tokenizer only
promptingnemo --config recipes/meta_asr/conf/hindi.yaml --mode tokenizer

# Training only (tokenizer already built)
promptingnemo --config recipes/meta_asr/conf/hindi.yaml --mode train

# Resume from checkpoint
promptingnemo --config recipes/meta_asr/conf/hindi.yaml --mode train \
    --resume_from /path/to/checkpoint.ckpt

# Validate manifests without training
promptingnemo --config recipes/meta_asr/conf/hindi.yaml --mode validate_data
```

---

## eval -- Evaluation Metrics

WER, NER F1, Audio-Visual WER metrics, and batch inference.

| Module | Description |
|--------|-------------|
| `wer.py` | Multi-metric WER evaluation with entity (NER) scoring. |
| `ner.py` | Named Entity Recognition scoring with per-label and overall precision/recall/F1. |
| `av_wer.py` | `AVWordErrorRate` -- torchmetrics-based metric for audio-visual ASR. |
| `inference.py` | `transcribe_manifest` -- batch inference using a NeMo CTC checkpoint. |

---

## export -- ONNX and HuggingFace Export

Export trained models for production inference or sharing.

| Module | Description |
|--------|-------------|
| `to_onnx.py` | Exports a `.nemo` checkpoint to ONNX format with config, vocabulary, and SentencePiece model files. |
| `to_hf.py` | Uploads a NeMo checkpoint to HuggingFace Hub with retry logic. |

**Export to ONNX:**

```bash
promptingnemo-export \
    --nemo-model /path/to/model.nemo \
    --output-dir /path/to/onnx_export \
    --opset 17
```

**Upload to HuggingFace:**

```python
from promptingnemo.export.to_hf import upload_nemo_to_hf

upload_nemo_to_hf(
    nemo_model_path="model.nemo",
    repo_id="WhissleAI/my-model",
    hf_token="hf_...",
)
```
