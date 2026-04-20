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
└── export/             Model export
    ├── to_onnx.py      NeMo → ONNX conversion
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
   ├── NeMo → ONNX (for deployment)
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
