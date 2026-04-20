# Architecture

## Library Structure

```
promptingnemo/
├── tokenizer/          Aggregate tokenizer training & management
│   ├── config.py       Load/store tokenizer YAML configs
│   ├── sentencepiece.py  Train per-language SentencePiece models
│   ├── aggregate.py    Build multi-language aggregate tokenizers
│   └── dedup_aggregate.py  Deduplicated aggregate tokenizer (NeMo monkey-patch)
├── models/             Custom NeMo model classes
│   ├── ctc_model.py    CustomEncDecCTCModelBPE (CTC with meta-tags)
│   └── decoder.py      Slim decoder, vocabulary extension, tag weight init
├── data/               Data loading & processing
│   ├── dataset.py      RobustAudioToBPEDataset (handles bad audio gracefully)
│   ├── sampler.py      BalancedLanguageBatchSampler (family-balanced batches)
│   ├── manifest.py     Manifest validation & scanning
│   └── normalize.py    Tag normalization (GER_→GENDER_, HAP→HAPPY, etc.)
├── training/           Training orchestration
│   ├── trainer.py      Model setup, data config, trainer.fit()
│   └── cli.py          Command-line interface & argument parsing
├── eval/               Evaluation
│   ├── inference.py    Model loading & transcription
│   ├── wer.py          WER computation with tag-aware scoring
│   └── ner.py          NER precision/recall/F1 scoring
└── export/             Model export (Whissle API compatible)
    ├── to_onnx.py      NeMo → ONNX + config.json + vocabulary.json
    └── to_hf.py        NeMo → HuggingFace format
```

## Training Pipeline

```
1. Data Preparation (recipes/data_prep/)
   ├── Download dataset from HuggingFace
   ├── Annotate with NER, emotion, age, gender, intent
   ├── Normalize tags to canonical format
   └── Output: NeMo JSONL manifest

2. Tokenizer Training (promptingnemo/tokenizer/)
   ├── Train per-language SentencePiece models
   ├── Inject shared special tokens (ENTITY_*, AGE_*, etc.)
   └── Build aggregate vocabulary across all languages

3. Model Training (promptingnemo/training/)
   ├── Load pretrained model (e.g., STT-meta-1B)
   ├── Slim decoder to target language families
   ├── Scale down tag decoder weights (prevent mid-utterance spam)
   ├── Setup adapter layers (optional)
   └── Train with CTC loss + optional keyword loss

4. Evaluation (promptingnemo/eval/)
   ├── Transcribe test set
   ├── Compute WER (transcription accuracy)
   └── Compute NER F1 (entity tagging accuracy)

5. Export (promptingnemo/export/)
   ├── NeMo → ONNX (config.json + vocabulary.json + model.onnx for Whissle API)
   └── NeMo → HuggingFace (for sharing)
```

## Key Design Decisions

### Aggregate Tokenizer
Each language family gets its own SentencePiece model, but they share a unified vocabulary with common special tokens. This allows a single model to handle multiple languages while maintaining language-specific subword segmentation.

### Slim Decoder
The pretrained model has an 18K-token vocabulary covering all language families. When fine-tuning for a specific family (e.g., INDO_ARYAN), the slim decoder removes tokens from non-target families, reducing the output space from ~18K to ~10K tokens. Pretrained weights for kept tokens are preserved.

### Tag Weight Initialization
After slim decoder pruning, sentence-level tags (AGE_*, GENDER_*, EMOTION_*, etc.) inherit pretrained decoder weights that can be disproportionately strong. `scale_down_tag_decoder_weights()` re-initializes these to small random values so they learn proper end-of-utterance positioning from the CTC loss signal.

### Balanced Language Sampling
For multi-language training, `BalancedLanguageBatchSampler` ensures each batch contains proportional representation from all language families, preventing dominant languages from monopolizing training.

---

## Audio-Visual Extension

PromptingNemo includes an Audio-Visual ASR extension that fuses visual context from video frames with audio features to improve speech recognition in noisy environments. This is based on the EMNLP 2025 paper "Visual-Aware Speech Recognition for Noisy Scenarios" by Darur & Singla ([paper](https://aclanthology.org/2025.emnlp-main.845/)).

### AV Model Architecture

The `AVEncDecCTCModelBPE` (in `promptingnemo/models/av_ctc_model.py`) extends the existing `CustomEncDecCTCModelBPE` with a cross-modal fusion pathway. The architecture consists of:

1. **Audio encoder** -- A pretrained Conformer CTC encoder (with optional adapters), producing per-frame audio embeddings.
2. **Visual encoder** -- A frozen CLIP ViT-L/14 model that extracts visual features from video frames. These features capture the visual scene context (e.g., noise source identification).
3. **Projection layers** -- Linear projections that map both audio features (`feat_out` -> 512) and CLIP features (768 -> 512) into a shared embedding space, with learned modality embeddings and positional encodings.
4. **Fusion encoder** -- A 4-layer Transformer encoder that performs cross-modal fusion via multi-head self-attention over concatenated audio and visual tokens.
5. **CTC decoder** -- After fusion, only the audio-aligned output tokens are extracted and passed to the CTC decoder for transcription. Noise labels are appended as the final token in the transcript (e.g., `<N12>`).

```
Audio --> Conformer Encoder --> Linear(feat_out, 512)  --+
                                                         +--> Concat --> Transformer Encoder --> CTC Decoder
Video --> CLIP ViT-L/14 --> Linear(768, 512)           --+
                             + Modality Embeddings
                             + Positional Encodings
```

The key insight is that visual features provide noise-source context (e.g., a barking dog, traffic, music) that helps the model disambiguate speech from noise. Importantly, models trained with visual awareness also improve audio-only inference -- the visual pathway teaches the encoder better noise-invariant representations even when visual input is absent at test time.

### New Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `AVEncDecCTCModelBPE` | `promptingnemo/models/av_ctc_model.py` | Audio-visual CTC model with CLIP fusion |
| `AVToBPEDataset` | `promptingnemo/data/av_dataset.py` | Dataset class for paired audio-video data with CLIP feature extraction |
| `AVWordErrorRate` | `promptingnemo/eval/av_wer.py` | WER metric with noise label accuracy scoring |

### How It Extends the Audio-Only Pipeline

The AV extension integrates with the existing PromptingNemo pipeline at three points:

- **Data loading**: `AVToBPEDataset` extends the base dataset to load video frames alongside audio, extracting CLIP features during data loading or from pre-computed feature files.
- **Model**: `AVEncDecCTCModelBPE` wraps the standard Conformer encoder and adds the visual branch and fusion Transformer. The pretrained audio encoder weights are fully preserved.
- **Evaluation**: `AVWordErrorRate` extends standard WER computation to additionally score noise label accuracy (the `<N##>` tokens appended to transcripts).

The training recipe lives in `recipes/av_asr/` with its own configs, data pipeline scripts, and training entry point. See [docs/audio_visual.md](audio_visual.md) for the full guide.
