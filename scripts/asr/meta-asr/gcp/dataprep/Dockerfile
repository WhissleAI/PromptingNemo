FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg sox libsndfile1 git wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    'datasets==2.21.0' \
    'fsspec<2024.12' \
    soundfile \
    numpy \
    scipy \
    librosa \
    tqdm \
    huggingface_hub \
    pyarrow \
    transformers \
    google-generativeai \
    num2words && \
    pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

ENV HF_DATASETS_TRUST_REMOTE_CODE=1

WORKDIR /app
COPY scripts/ /app/scripts/

ENTRYPOINT ["python"]
