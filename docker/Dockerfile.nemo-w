FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg sox espeak-ng curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    'nemo_toolkit[asr]==2.7.2' \
    torchaudio==2.6.0 \
    sentencepiece \
    omegaconf \
    pyyaml \
    tensorboard \
    'datasets>=3.0.0,<4.0.0' \
    soundfile \
    librosa \
    jiwer \
    rich \
    huggingface_hub

WORKDIR /workspace
