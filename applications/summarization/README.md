# Video Summarization (FastAPI)

This service exposes a FastAPI endpoint that:

- extracts audio from an uploaded video (via **ffmpeg**)
- runs **local speaker diarization** using **pyannote**
- slices audio per diarized segment and transcribes each segment using **Whissle STT**
- extracts a keyframe per segment and captions it using **Gemini**
- uses **Gemini** to generate a final video summary using segment transcripts + visual captions

## Requirements

- **ffmpeg** installed and available on `PATH`
- Python deps:

```bash
pip install -r applications/summarization/requirements.txt
```

## Environment variables

- **WHISSLE_AUTH_TOKEN**: your Whissle API token from `whissle.ai`
- **GOOGLE_API_KEY**: Gemini API key
- **HF_TOKEN**: HuggingFace token (needed if pyannote needs to fetch model weights; cached weights can also work)
- **PYANNOTE_MODEL**: optional, default `pyannote/speaker-diarization-3.1`
- **PYANNOTE_DEVICE**: optional, e.g. `cpu` or `cuda`
- **GEMINI_MODEL**: optional, default `gemini-1.5-flash`

## Run

From repo root:

```bash
python -m applications.summarization.app
```

Then open `http://127.0.0.1:8010/docs`.

## API

### `POST /summarize`

Multipart form-data:

- `video` (file): video file (mp4/mkv/mov/etc)
- optional form fields to override env vars:
  - `whissle_auth_token`
  - `hf_token`
  - `pyannote_model`
  - `pyannote_device`
  - `gemini_model`
  - `keep_artifacts` (0/1)

Example:

```bash
curl -X POST "http://127.0.0.1:8010/summarize" \
  -H "Accept: application/json" \
  -F "video=@/path/to/video.mp4"
```

### `POST /summarize_stream` (live progress)This streams **Server-Sent Events (SSE)** so you can see diarization progress, per-segment STT output, etc.

```bash
curl -N -X POST "http://127.0.0.1:8010/summarize_stream" \
  -H "Accept: text/event-stream" \
  -F "video=@/path/to/video.mp4"
```
