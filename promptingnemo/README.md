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
├── models/         CTC models with keyword loss, tag classifiers, AV fusion, distillation
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
| `dataset.py` | `RobustAudioToBPEDataset` -- extends NeMo's `AudioToBPEDataset` with language-ID normalization, manifest validation, audio readability checks, and keyphrase oversampling. Monkey-patches NeMo's base class. Also provides `patched_speech_collate_fn` that handles mixed mono/stereo audio and filters `None` samples. |
| `av_dataset.py` | `AVToBPEDataset` -- Audio-Visual dataset for CTC training. Loads clean audio, mixes with noise extracted from video files at configurable SNR ratios, and provides pre-extracted CLIP ViT-L/14 visual features. Implements the VANS pipeline from the EMNLP 2025 paper. |
| `normalize.py` | Text normalization for meta-ASR tags. Fixes common annotation typos (`EMOTION_HAP` -> `EMOTION_HAPPY`, `GER_` -> `GENDER_`), normalizes age/gender/emotion tags to canonical forms, filters forbidden tags, and detects concatenated tag errors. |
| `tag_parser.py` | Parses tagged text into (clean_text, tagged_text) pairs. Supports trailing tags (`AGE_30_45 GENDER_MALE`) and inline entities (`ENTITY_PERSON John END`). Includes compositional tag decomposition (`decompose_tag`) for zero-shot tag generalization and vocabulary construction utilities. |
| `sampler.py` | `BalancedLanguageBatchSampler` -- temperature-scaled language-family-aware batch sampling for distributed training. Uses inverse-frequency weighting with configurable temperature to balance representation across language families. Supports keyphrase oversampling weights. |
| `manifest.py` | Manifest validation utilities. Multi-threaded audio readability checking with `validate_manifest_file`, bulk validation via `validate_manifests` that updates config paths to point at cleaned copies. |
| `chunked_tag_utils.py` | Utilities for chunked/streaming-aware CTC targets. Splits tagged text into inline entity tags and trailing sentence tags, then repeats trailing tags at regular chunk intervals to teach causal models to emit them at streaming buffer boundaries. |
| `text_tagger_dataset.py` | `TextTaggerDataset` -- character-input dataset for text CTC tagger training. Converts clean text to character IDs (encoder input) and tagged text to subword+tag token IDs (CTC target). Enforces CTC alignment constraint via upsampling factor. |
| `text_tagger_dataset_v2.py` | `TextTaggerDatasetV2` -- subword-input variant using the STT-meta-1B aggregate tokenizer for both input and output, so output token IDs match the ASR model exactly. |
| `text_tagger_dataset_v3.py` | `TextTaggerDatasetV3` -- XLM-RoBERTa input with chunked trailing tags in the output. Uses jittered chunk sizes during training for robustness. |

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

Model definitions for CTC-based ASR with metadata tagging, audio-visual fusion, knowledge distillation, and text-only tagging.

| Module | Description |
|--------|-------------|
| `ctc_model.py` | `CustomEncDecCTCModelBPE` -- extends NeMo's CTC BPE model with: keyword loss (CTC on entity tokens only, with warmup), per-language-family WER/loss tracking during validation, language-family-weighted CTC loss (inverse frequency), and an optional `TrailingTagClassifier` dual-head for AGE/GENDER/EMOTION/INTENT. Uses `FlexibleSaveRestoreConnector` for vocab-size-mismatched checkpoint loading. |
| `av_ctc_model.py` | `AVEncDecCTCModelBPE` -- Audio-Visual CTC model implementing the AV-UNI-SNR architecture (EMNLP 2025). Wraps a pretrained Conformer encoder (frozen, with optional linear adapters) and adds a Transformer fusion module over concatenated audio + CLIP ViT-L/14 visual features. Audio-aligned positions are retained for CTC decoding. Reports labelled WER, unlabelled WER, and noise label accuracy. |
| `av_meta_ctc_model.py` | `AVMetaCTCModelBPE` -- next-gen AV model with cross-attention fusion (audio queries attend to video keys/values), four output heads (CTC decoder, scene classifier, speaker attribute classifier, attention map export), and SigLIP 2 So400m (1152d) visual features. Includes attention entropy regularization for sparse, interpretable heat maps. |
| `decoder.py` | Decoder manipulation utilities: `scan_manifest_for_new_tokens` (find tags missing from vocabulary), `extend_decoder_for_new_tokens` (add new output tokens while preserving CTC blank position), `slim_decoder_for_training` (remove non-target language tokens, reuse pretrained weights), `scale_down_tag_decoder_weights` (re-initialize tag logits to prevent domination). |
| `distill_ctc_model.py` | `DistillCTCModel` -- knowledge distillation student model. Teacher processes clean audio (frozen); student processes noise-augmented audio. Multi-objective loss: CTC on hard labels, KL-divergence at temperature T (logit-level KD), hidden-state MSE matching via learned projectors, and tag classifier. Loss weights anneal over training to shift from teacher mimicry to independent learning. |
| `hybrid_model.py` | `CustomEncDecHybridRNNTCTCBPEModel` -- extends NeMo's Hybrid RNN-T/CTC model with TrailingTagClassifier and per-family WER tracking. Combined loss: `(1 - ctc_weight) * rnnt_loss + ctc_weight * ctc_loss + tag_weight * tag_cls_loss`. |
| `text_ctc_model.py` | `TextCTCTagger` -- text-only CTC tagger (v1). Architecture: CharacterEmbedding -> LearnedUpsampler (transposed conv) -> TransformerEncoder -> Linear -> CTC. Supports causal attention masks for streaming inference. Reports WER on transcription words and tag F1 (precision/recall on tag tokens). |
| `text_ctc_model_v2.py` | `TextCTCTaggerV2` -- subword-input variant with a full Conformer encoder (causal convolutions, causal MHSA) instead of vanilla Transformer. Higher upsampling factor compensates for fewer input tokens. |
| `text_ctc_model_v3.py` | `TextCTCTaggerV3` -- XLM-RoBERTa encoder (frozen, with optional top-N layer unfreezing) as input encoder, projecting to the aggregate vocabulary output space via learned upsampler + linear CTC decoder. |
| `weight_init.py` | SVD-based weight transfer for model distillation. `init_student_from_teacher` compresses teacher conformer layers to student dimensions via truncated SVD. Supports evenly-spaced and first-N layer mapping strategies. Also provides `copy_decoder_weights` for same-vocabulary decoder transfer. |

**Key concept -- keyword loss:**

The keyword loss applies extra CTC weight to tag tokens during training,
ensuring the model learns to emit structured tags alongside transcription
text. A warmup schedule prevents instability during early training steps.

---

## tokenizer -- Aggregate Multi-Language Tokenizer

Trains per-language-family SentencePiece models and merges them into a single deduplicated vocabulary with shared special tokens.

| Module | Description |
|--------|-------------|
| `config.py` | Language family configuration and tokenizer metadata persistence. Maintains module-level `LANG_FAMILIES` and `LANG_TO_FAMILY` mappings. Provides YAML-based storage and loading for tokenizer configs, shared special tokens, and aggregate vocabularies. |
| `sentencepiece.py` | `train_sentencepiece_tokenizer` -- trains a SentencePiece model from a manifest file with user-defined special symbols (entity/intent/emotion tags). Includes retry logic for vocabulary size mismatches (too small or too large). |
| `aggregate.py` | Aggregate tokenizer training pipeline. `extract_langs_and_special_tokens` scans manifests for language IDs and special tokens. `train_aggregate_tokenizer` creates one SentencePiece model per language family, deduplicates into a global vocabulary, and stores the mapping. `setup_tokenizer` returns the appropriate tokenizer config. |
| `dedup_aggregate.py` | `DedupAggregateTokenizer` -- patches NeMo's `AggregateTokenizer` with deduplicated global vocabulary. Maps local per-family token IDs to global IDs, supports vocabulary extension for new special tokens, and provides `ids_to_words_and_langs` for per-word language attribution. |
| `meta_tokenizer.py` | `MetaTokenizer` -- lightweight wrapper using STT-meta-1B aggregate vocabulary without NeMo dependency. Loads per-family SentencePiece models, maps local IDs to global aggregate IDs, and handles tag tokens as atomic units during encoding. Used by the text tagger pipeline. |
| `text_tagger_tokenizer.py` | `TextTaggerTokenizer` -- hybrid tokenizer for text CTC tagger. Output vocabulary is `[SP subwords] + [compositional tag pieces] + [CTC blank]`. Tags are decomposed into compositional pieces (`INTENT_REPORT_SYMPTOM` -> `['INTENT_', 'REPORT', '_SYMPTOM']`) enabling zero-shot generalization to unseen tag combinations. |

**Training a tokenizer from a config:**

```bash
promptingnemo --config recipes/meta_asr/conf/hindi.yaml --mode tokenizer
```

---

## training -- Trainer and CLI

Training orchestration, checkpoint loading, data setup, and CLI entry points.

| Module | Description |
|--------|-------------|
| `trainer.py` | `train_model` -- end-to-end training function. Handles checkpoint loading with `FlexibleSaveRestoreConnector`, tokenizer setup (aggregate or single), decoder manipulation (slim/extend/rescale), `BalancedLanguageBatchSampler` injection, audio augmentation (white noise + shift or configurable pipeline), spec augmentation overrides, optimizer/scheduler configuration, tag classifier setup with class-weighted loss and minority-class oversampling, per-family loss weighting, and NeMo experiment manager integration. Also includes `ValidationMetricsPrinter` callback for structured validation reporting. |
| `cli.py` | `main` -- command-line entry point for Meta-ASR training. Modes: `both` (tokenizer + training), `tokenizer` (build/rebuild tokenizer only), `train` (training only), `validate_data` (manifest validation only). Parses YAML config, sets language families, configures dataset options, and dispatches to the appropriate pipeline stage. |
| `hybrid_trainer.py` | `train_hybrid_model` -- training function for Hybrid RNN-T/CTC models. Preserves both RNN-T and CTC decoders (unlike `trainer.py` which converts to pure CTC). Otherwise follows the same pipeline: checkpoint loading, tokenizer setup, tag classifier, balanced sampling, augmentation. |
| `hybrid_cli.py` | CLI for Hybrid RNN-T/CTC training. Supports `tokenizer` and `train` modes. Delegates tokenizer training to the standard CLI; training dispatches to `hybrid_trainer.py`. |

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

# Hybrid RNN-T/CTC training
python -m promptingnemo.training.hybrid_cli \
    --config recipes/meta_asr/conf/mega_zh_v1_hybrid.yaml --mode train
```

---

## eval -- Evaluation Metrics

WER, NER F1, Audio-Visual WER metrics, and batch inference.

| Module | Description |
|--------|-------------|
| `wer.py` | Multi-metric WER evaluation with entity (NER) scoring. Separates clean transcript from meta-tags, computes standard WER via NeMo, and entity-level precision/recall/F1 via `ner.py`. Supports both label-only and label+phrase scoring modes. |
| `ner.py` | Named Entity Recognition scoring. Computes per-label and overall (micro/macro) precision, recall, and F-score. Handles duplicate entity disambiguation and edit-distance-based fuzzy matching for entity phrases. |
| `av_wer.py` | `AVWordErrorRate` -- torchmetrics-based metric for audio-visual ASR. Computes labelled WER (including noise tags), unlabelled WER (tags stripped), and noise label classification accuracy. Noise tags follow the `<N\d+>` pattern from the VANS dataset. |
| `inference.py` | `transcribe_manifest` -- batch inference using a NeMo CTC checkpoint. Loads a JSONL manifest, transcribes audio files in batches, and writes predictions to an output JSONL. Handles NeMo Hypothesis objects, dicts, and nested predictions. |

---

## export -- ONNX and HuggingFace Export

Export trained models for production inference or sharing.

| Module | Description |
|--------|-------------|
| `to_onnx.py` | `export_nemo_to_onnx` -- exports a `.nemo` checkpoint to ONNX format compatible with the Whissle API (`decoder_onnx`). Produces `model.onnx`, `config.json` (preprocessor config), `vocabulary.json` (vocabulary + blank_id + tokenizer metadata), and per-language SentencePiece model files. Supports FP16 export and metadata-only extraction. |
| `to_hf.py` | `upload_nemo_to_hf` -- uploads a NeMo checkpoint to HuggingFace Hub with exponential-backoff retry logic. Creates the repository if needed and optionally uploads a README alongside the model. |

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
