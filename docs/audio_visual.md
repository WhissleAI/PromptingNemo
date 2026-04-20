# Audio-Visual ASR: Visual-Aware Speech Recognition for Noisy Scenarios

## Paper Reference

Darur, B. & Singla, K. (2025). *Visual-Aware Speech Recognition for Noisy Scenarios.* In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP).

[https://aclanthology.org/2025.emnlp-main.845/](https://aclanthology.org/2025.emnlp-main.845/)

Research prototype: [NeMo-W `audio-visual-balu` branch](https://github.com/WhissleAI/NeMo-W/tree/audio-visual-balu/balu_codes)

---

## Motivation

Automatic speech recognition degrades significantly in noisy environments. Traditional approaches treat noise as a nuisance to be filtered out, but noise sources are often visually identifiable -- a barking dog, traffic, a crowd, music from a speaker. If the model can see the noise source, it can better separate speech from noise.

The key insight of this work is that CLIP visual features extracted from video frames of the noise source provide strong contextual cues for noise-aware ASR. Rather than building a separate noise classifier or audio enhancement front-end, the visual features are fused directly with the audio encoder's representations, allowing end-to-end learning of noise-robust speech recognition.

An additional finding is that models trained with visual awareness also improve in audio-only inference. The visual training signal teaches the encoder better noise-invariant representations that persist even when visual input is absent at test time.

---

## Architecture

### Overview

The Audio-Visual model (`AVEncDecCTCModelBPE`) extends PromptingNemo's `CustomEncDecCTCModelBPE` by adding a visual feature pathway and a cross-modal fusion Transformer.

```
Audio --> Conformer Encoder --> Linear(feat_out, 512)  --+
                                                         +--> Concat --> Transformer Encoder --> CTC Decoder
Video --> CLIP ViT-L/14 --> Linear(768, 512)           --+
                             + Modality Embeddings
                             + Positional Encodings
```

### Components

**1. Audio Encoder (Conformer CTC, pretrained)**

The audio pathway uses a standard Conformer CTC encoder, initialized from a pretrained checkpoint (e.g., STT-meta-1B or a Conformer-CTC-Large model). Optional adapter layers can be inserted for parameter-efficient fine-tuning. The encoder output is projected from `feat_out` dimensions to 512 via a linear layer.

**2. Visual Encoder (CLIP ViT-L/14, frozen)**

Video frames are passed through a frozen CLIP ViT-L/14 model to extract visual features. CLIP provides semantically rich representations that capture the visual scene, including noise source objects. The CLIP output (768-dim) is projected to 512 dimensions via a linear layer. The CLIP weights are kept frozen throughout training -- only the projection layer is trained.

**3. Cross-Modal Fusion (4-layer Transformer Encoder)**

Audio and visual tokens are concatenated along the sequence dimension. Learned modality embeddings (audio vs. visual) and positional encodings are added to distinguish token origin and position. A 4-layer Transformer encoder performs multi-head self-attention over the concatenated sequence, allowing audio and visual tokens to attend to each other.

After fusion, only the audio-aligned output positions are extracted and passed to the CTC decoder. This ensures the output sequence length matches the audio frames, preserving CTC alignment.

**4. CTC Decoder with Noise Labels**

The CTC decoder operates identically to the standard PromptingNemo decoder, with one addition: noise labels are appended as the final token in the transcript. Noise labels follow the format `<N##>` where `##` is the noise class index (e.g., `<N12>` for a specific AudioSet noise category). This allows the model to both transcribe speech and identify the noise source in a single pass.

Example output: `the cat sat on the mat <N12>`

---

## VANS Dataset Pipeline

The Visual-Aware Noisy Speech (VANS) dataset is constructed by mixing clean speech with noisy video audio at controlled signal-to-noise ratios (SNRs).

### Data Sources

- **Clean speech**: [People's Speech](https://mlcommons.org/peoples-speech/) corpus -- large-scale English speech dataset with transcriptions.
- **Noise videos**: [AudioSet](https://research.google.com/audioset/) -- a large-scale dataset of 10-second YouTube video clips with audio event labels. The video frames provide the visual features, and the audio provides the noise signal.

### Construction Process

1. **Select noise videos** from AudioSet, filtering for categories that represent identifiable noise sources (e.g., dog barking, traffic, music, machinery).
2. **Extract audio and video** from each noise clip. Video frames are sampled for CLIP feature extraction.
3. **Mix clean speech with noise audio** at variable SNRs (e.g., -5dB, 0dB, 5dB, 10dB, 15dB, 20dB). Each clean utterance can be mixed with multiple noise sources at different SNRs.
4. **Assign noise labels** (`<N##>`) to each mixed sample based on the AudioSet category of the noise source.
5. **Extract CLIP features** from the noise video frames using the frozen CLIP ViT-L/14 model. These are stored alongside the mixed audio.
6. **Generate manifests** in NeMo JSONL format, with additional fields for CLIP feature paths and noise labels.

---

## Training Workflow

Training follows three stages:

### Stage 1: Data Preparation

Prepare the VANS dataset or bring your own audio-visual noisy speech data. The `recipes/av_asr/` directory includes data pipeline scripts for constructing the dataset from AudioSet and People's Speech.

```bash
cd recipes/av_asr
# Follow the data preparation scripts in the recipe
```

### Stage 2: CLIP Feature Extraction

Pre-extract CLIP features from video frames to avoid redundant computation during training. The extracted features are stored as numpy arrays or tensor files.

```bash
# Extract CLIP features for the training set
python extract_clip_features.py --video_dir /data/noise_videos/ --output_dir /data/clip_features/
```

### Stage 3: Training

Train the AV-ASR model using the pre-extracted CLIP features and mixed audio:

```bash
cd recipes/av_asr
python train.py --config conf/av_conformer_ctc.yaml
```

The training config specifies:
- Pretrained audio encoder checkpoint
- CLIP feature directory
- SNR mixing parameters
- Fusion Transformer hyperparameters (layers, heads, hidden dim)
- CTC loss configuration with noise label tokens

---

## Evaluation

### WER (Word Error Rate)

WER is computed on the transcription portion only, stripping noise label tokens (`<N##>`) before scoring. The `AVWordErrorRate` metric in `promptingnemo/eval/av_wer.py` handles this automatically.

### Noise Label Accuracy

Noise label accuracy measures whether the model correctly identifies the noise source. The predicted `<N##>` token is compared against the ground truth noise label.

### Running Evaluation

```bash
cd recipes/av_asr
python eval.py --config conf/av_conformer_ctc.yaml --checkpoint /path/to/best_model.nemo
```

---

## Key Results

Results on the VANS test set at 10dB SNR:

| Model | Params | WER (10dB) | Notes |
|-------|--------|------------|-------|
| Conformer-CTC (audio-only) | ~120M | 26.99 | Baseline without visual features |
| AV-Conformer (audio-only trained) | ~450M | 23.11 | Larger model, still audio-only |
| AV-Conformer (AV-trained, audio-only inference) | ~450M | 22.29 | AV training improves audio-only inference |
| **AV-UNI-SNR** | **453M** | **20.71** | **Best model, full AV inference** |
| Whisper Large V3 | 1.55B | Competitive | 3.4x more parameters |

Key takeaways:
- The AV-UNI-SNR model achieves 20.71 WER at 10dB, a 23% relative improvement over the audio-only Conformer-CTC baseline (26.99).
- The AV model (453M params) is competitive with Whisper Large V3 (1.55B params), achieving comparable accuracy with roughly 3.4x fewer parameters.
- Visual-aware training improves audio-only inference: the AV-trained model achieves 22.29 WER in audio-only mode versus 23.11 for a model trained without visual features.

---

## Ablation Findings

The paper reports several ablation studies:

- **Fusion depth**: 4 Transformer layers provides the best trade-off between accuracy and computational cost. Fewer layers underperform; more layers show diminishing returns.
- **Visual encoder**: CLIP ViT-L/14 outperforms smaller CLIP variants and non-CLIP visual encoders, due to its strong semantic understanding of visual scenes.
- **SNR-aware training**: Training with variable SNR mixing (the "UNI-SNR" variant) outperforms fixed-SNR training, producing a model robust across a range of noise levels.
- **Noise labels**: Appending noise labels as CTC tokens provides a useful auxiliary signal that improves both transcription accuracy and noise classification.
- **Frozen vs. fine-tuned CLIP**: Keeping CLIP frozen performs comparably to fine-tuning it, while being significantly more memory-efficient.

---

## Connection to Speech-2-Action

The Audio-Visual extension fits into PromptingNemo's broader Speech-2-Action vision. Just as the base meta-ASR models append entity, emotion, and intent tags to transcriptions in a single CTC pass, the AV model appends noise source labels. This follows the same design principle: a single end-to-end model that produces transcription plus structured metadata, eliminating the need for separate classification pipelines.

In production scenarios, noise labels can be used for:
- Adaptive audio processing (adjusting noise suppression based on identified noise type)
- Context-aware dialogue systems (e.g., "I notice you're in a noisy environment with traffic")
- Quality monitoring and analytics (tracking noise conditions across calls or sessions)

---

## Future Work

Directions identified in the paper:

- **Multi-label noise classification**: Current models predict a single noise label per utterance. Real-world scenarios often involve multiple simultaneous noise sources.
- **Other visual cues**: Beyond noise source identification, visual features could encode speaker lip movements, gestures, and environmental context for further ASR improvements.
- **Low-resource and multilingual AV-ASR**: Extending the approach to languages beyond English, leveraging CLIP's multilingual visual understanding.
- **Streaming AV-ASR**: Adapting the architecture for real-time streaming inference with incremental visual feature processing.
- **Integration with meta-tags**: Combining noise labels with the existing entity, emotion, and intent tags for a unified multi-task output.

---

## Code Reference

| File | Description |
|------|-------------|
| `promptingnemo/models/av_ctc_model.py` | `AVEncDecCTCModelBPE` model class |
| `promptingnemo/data/av_dataset.py` | `AVToBPEDataset` for audio-visual data loading |
| `promptingnemo/eval/av_wer.py` | `AVWordErrorRate` metric with noise label scoring |
| `recipes/av_asr/` | Training recipe with configs, scripts, and data pipeline |
| `recipes/av_asr/conf/av_conformer_ctc.yaml` | Default training configuration |
