# Audio-Visual ASR Training Recipe

Train audio-visual ASR models that leverage CLIP visual features from video frames
to improve speech recognition in noisy conditions. This recipe implements the VANS
(Visual-Aware Noisy Speech) pipeline from the EMNLP 2025 paper, which mixes clean
speech with environmental noise at variable SNRs and appends a noise class label as
the final token in the transcript. A Transformer fusion module combines Conformer
audio encodings with CLIP ViT-L/14 visual features, enabling the model to use visual
context to disambiguate speech corrupted by noise.

## Reference

> **Visual-Aware Speech Recognition for Noisy Scenarios**
> Bala Durga Sai Darur, Karan Singla
> EMNLP 2025
>
> ```bibtex
> @inproceedings{darur2025visual,
>   title     = {Visual-Aware Speech Recognition for Noisy Scenarios},
>   author    = {Darur, Bala Durga Sai and Singla, Karan},
>   booktitle = {Proceedings of the 2025 Conference on Empirical Methods
>                in Natural Language Processing (EMNLP)},
>   year      = {2025}
> }
> ```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 24 GB+ VRAM (A100 recommended)
- PromptingNemo installed (`pip install -e ".[train]"` from the repo root)
- Additional Python packages:
  ```bash
  pip install open_clip_torch av yt-dlp
  ```
- NeMo pretrained model: `stt_en_conformer_ctc_large` (downloaded automatically by NeMo
  or manually placed under a model directory)
- Datasets:
  - **People's Speech** -- clean English speech corpus ([MLCommons](https://mlcommons.org/datasets/peoples-speech/))
  - **AudioSet** -- environmental audio with class labels ([Google Research](https://research.google.com/audioset/))

## Architecture

```
                        Audio Input
                            |
                    +-----------------+
                    | Mel Spectrogram |
                    +-----------------+
                            |
                   +-------------------+
                   | Conformer Encoder |  (18 layers, 512 dim, subsampling x4)
                   |   + Adapters      |  (linear, dim=64, swish)
                   +-------------------+
                            |
                      Linear (512)
                            |
        +-------------------+-------------------+
        |                                       |
        |                                Video Frames (5 fps)
        |                                       |
        |                              +------------------+
        |                              | CLIP ViT-L/14    |  (frozen, 768-dim features)
        |                              +------------------+
        |                                       |
        |                                Linear (768 -> 512)
        |                                       |
        +------- concat along time axis --------+
                            |
                 +------------------------+
                 | Transformer Fusion     |  (4 layers, 512 dim, 8 heads)
                 +------------------------+
                            |
                 +------------------------+
                 | CTC Decoder            |  (Linear -> vocab)
                 +------------------------+
                            |
                    Transcript + NOISE_LABEL
```

The model appends the noise class label (e.g., `NOISE_dog_bark`) as the final token
in the CTC output, enabling joint transcription and noise classification.

## Data Preparation

### Step 1: Create the VANS Dataset

The VANS dataset is constructed by mixing clean speech from People's Speech with
single-class noise segments from AudioSet:

```bash
# Full pipeline: filter AudioSet, download, mix with clean speech, align, split
./run.sh --stage 1 --stop-stage 1 \
  --peoples-speech-dir /data/peoples_speech \
  --audioset-dir /data/audioset \
  --output-dir /data/vans
```

This runs `local/prepare_vans.py` which:
1. Filters AudioSet for single-noise-label videos (excludes speech/voice classes,
   requires 750+ samples per class)
2. Downloads the filtered AudioSet segments via `yt-dlp`
3. Mixes clean speech with noise at the configured SNR (or uniform random SNR)
4. Uses NeMo forced alignment to trim audio/transcript pairs to 10 seconds
5. Appends the noise label as the final token (e.g., `hello world NOISE_dog_bark`)
6. Splits into train/val/test with balanced noise class distribution
7. Outputs NeMo JSONL manifests with fields: `audio_filepath`, `video_filepath`,
   `feature_file`, `text`, `duration`

### Step 2: Extract CLIP Features

Pre-extract CLIP ViT-L/14 features from video frames at 5 fps:

```bash
./run.sh --stage 2 --stop-stage 2 \
  --output-dir /data/vans
```

This runs `local/extract_features.py` which saves per-video `.npy` feature files
and updates the manifest with `feature_file` paths.

## Training

### AV-UNI-SNR (Best Model)

Trains with video features and uniform SNR sampling across [-5dB, +5dB]:

```bash
# Single GPU
./run.sh --stage 3 --stop-stage 3 \
  --config av_conformer_ctc \
  --snr rand \
  --gpus 1

# Multi-GPU (4x A100)
./run.sh --stage 3 --stop-stage 3 \
  --config av_conformer_ctc \
  --snr rand \
  --gpus 4
```

Or directly with the Python entry point:

```bash
python train.py \
  --config conf/av_conformer_ctc.yaml \
  --snr rand \
  --gpus 1
```

### AV-SNR (Fixed 10dB)

Trains with video features at a fixed 10dB SNR:

```bash
python train.py \
  --config conf/av_conformer_ctc.yaml \
  --snr 10.0 \
  --gpus 1
```

### A-UNI-SNR (Audio-Only Baseline)

Trains without video features for comparison:

```bash
python train.py \
  --config conf/audio_only_baseline.yaml \
  --snr rand \
  --gpus 1
```

### Resume from Checkpoint

```bash
python train.py \
  --config conf/av_conformer_ctc.yaml \
  --snr rand \
  --gpus 1 \
  --resume /path/to/checkpoint.ckpt
```

## Evaluation

Evaluate WER and noise label accuracy on the test set:

```bash
python -m promptingnemo.eval.av_asr \
  --config conf/av_conformer_ctc.yaml \
  --checkpoint /path/to/best_model.ckpt \
  --test-manifest /data/vans/test_manifest.json
```

Metrics reported:
- **WER**: Word Error Rate on the transcript (noise label token excluded)
- **Noise Label Accuracy**: Classification accuracy of the appended noise token

## Results

Results from the paper at 10dB SNR test condition (People's Speech + AudioSet):

| Model         | Video | SNR Training | WER (10dB) | Params |
|---------------|-------|--------------|------------|--------|
| A-SNR         | No    | Fixed 10dB   | 26.99      | 121M   |
| AV-SNR        | Yes   | Fixed 10dB   | 22.28      | 453M   |
| A-UNI-SNR     | No    | Uniform      | 24.54      | 121M   |
| **AV-UNI-SNR**| Yes   | Uniform      | **20.71**  | 453M   |
| Whisper Large V3 | No | --           | 19.81      | 1,550M |

Key findings:
- Visual features reduce WER by ~4-6 points absolute in noisy conditions
- Uniform SNR training improves generalization over fixed-SNR training
- AV-UNI-SNR (453M params) achieves competitive WER with Whisper Large V3 (1,550M params)

## Directory Structure

```
recipes/av_asr/
├── run.sh                      # Shell entry point with stage control
├── train.py                    # Python training entry point
├── conf/
│   ├── av_conformer_ctc.yaml   # AV-UNI-SNR config (best model)
│   └── audio_only_baseline.yaml # Audio-only baseline config
├── local/
│   ├── prepare_vans.py         # VANS dataset creation pipeline
│   └── extract_features.py     # CLIP feature extraction
└── README.md                   # This file
```

## Further Reading

- [PromptingNemo main README](../../README.md) -- installation, architecture overview
- [Meta-ASR recipe](../meta_asr/README.md) -- text-only ASR with metadata tagging
- [Data preparation recipe](../data_prep/README.md) -- manifest normalization
