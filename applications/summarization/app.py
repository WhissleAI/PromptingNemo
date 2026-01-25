from __future__ import annotations

import os
import tempfile
import json
import threading
import queue
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import logging

try:
    # When running from repo root (recommended)
    from applications.summarization.video_summarization import (
        VideoSummarizationError,
        summarize_video_file,
    )
except Exception:  # pragma: no cover
    # When running from applications/summarization/ directly
    from video_summarization import (  # type: ignore
        VideoSummarizationError,
        summarize_video_file,
    )


# Load nearest .env if present (works whether launched from repo root or subdir).
load_dotenv(find_dotenv(usecwd=True))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

app = FastAPI(title="Video Summarization API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


@app.get("/")
def root():
    return {
        "message": "Video Summarization API is running",
        "docs": "/docs",
        "endpoints": {"status": "/status", "summarize": "/summarize", "summarize_stream": "/summarize_stream"},
    }


@app.get("/status")
def status():
    return {
        "whissle_configured": bool(_env("WHISSLE_AUTH_TOKEN")),
        "google_configured": bool(_env("GOOGLE_API_KEY")),
        "pyannote_model": _env("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"),
        "pyannote_device": _env("PYANNOTE_DEVICE"),
        "gemini_model": _env("GEMINI_MODEL", "gemini-1.5-flash"),
    }


@app.post("/summarize")
async def summarize(
    video: UploadFile = File(...),
    whissle_auth_token: Optional[str] = Form(None),
    pyannote_model: Optional[str] = Form(None),
    hf_token: Optional[str] = Form(None),
    pyannote_device: Optional[str] = Form(None),
    gemini_model: Optional[str] = Form(None),
    keep_artifacts: int = Form(0),
):
    """
    Upload a video file. The server extracts audio, diarizes via pyannote, runs Whissle STT per segment,
    captions keyframes via Gemini, and returns a final summary.

    Tokens can be passed via form fields (for one-off calls) or via env vars:
    - WHISSLE_AUTH_TOKEN
    - HF_TOKEN (for pyannote model download/cached access)
    - GOOGLE_API_KEY (Gemini)
    """
    # Resolve config: form override -> env fallback
    whissle_auth_token = whissle_auth_token or _env("WHISSLE_AUTH_TOKEN")
    pyannote_model = pyannote_model or _env("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
    hf_token = hf_token or _env("HF_TOKEN")
    pyannote_device = pyannote_device or _env("PYANNOTE_DEVICE")
    gemini_model = gemini_model or _env("GEMINI_MODEL", "gemini-1.5-flash")
    keep_artifacts_bool = bool(int(keep_artifacts))

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    try:
        with tempfile.TemporaryDirectory(prefix="video_api_") as td:
            video_path = Path(td) / f"input{suffix}"
            content = await video.read()
            video_path.write_bytes(content)

            result = summarize_video_file(
                video_path,
                whissle_auth_token=whissle_auth_token or "",
                pyannote_model=pyannote_model or "pyannote/speaker-diarization-3.1",
                hf_token=hf_token,
                pyannote_device=pyannote_device,
                keep_artifacts=keep_artifacts_bool,
                gemini_model=gemini_model or "gemini-1.5-flash",
            )
            return result
    except VideoSummarizationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {type(e).__name__}: {e}") from e


@app.post("/summarize_stream")
async def summarize_stream(
    video: UploadFile = File(...),
    whissle_auth_token: Optional[str] = Form(None),
    pyannote_model: Optional[str] = Form(None),
    hf_token: Optional[str] = Form(None),
    pyannote_device: Optional[str] = Form(None),
    gemini_model: Optional[str] = Form(None),
    keep_artifacts: int = Form(0),
):
    """
    Server-Sent Events (SSE) endpoint that streams progress updates.
    Use curl with `-N` to see events as they happen.
    """
    whissle_auth_token = whissle_auth_token or _env("WHISSLE_AUTH_TOKEN")
    pyannote_model = pyannote_model or _env("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
    hf_token = hf_token or _env("HF_TOKEN")
    pyannote_device = pyannote_device or _env("PYANNOTE_DEVICE")
    gemini_model = gemini_model or _env("GEMINI_MODEL", "gemini-1.5-flash")
    keep_artifacts_bool = bool(int(keep_artifacts))

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    content = await video.read()

    q: "queue.Queue[dict]" = queue.Queue()
    done_sentinel = {"event": "__DONE__"}

    def push(evt: dict) -> None:
        q.put(evt)

    def worker(video_bytes: bytes) -> None:
        try:
            with tempfile.TemporaryDirectory(prefix="video_api_") as td:
                video_path = Path(td) / f"input{suffix}"
                video_path.write_bytes(video_bytes)

                result = summarize_video_file(
                    video_path,
                    whissle_auth_token=whissle_auth_token or "",
                    pyannote_model=pyannote_model or "pyannote/speaker-diarization-3.1",
                    hf_token=hf_token,
                    pyannote_device=pyannote_device,
                    keep_artifacts=keep_artifacts_bool,
                    gemini_model=gemini_model or "gemini-1.5-flash",
                    progress_cb=push,
                )
                q.put({"event": "final_result", "payload": result})
        except VideoSummarizationError as e:
            q.put({"event": "error", "payload": {"message": str(e)}})
        except Exception as e:
            q.put({"event": "error", "payload": {"message": f"{type(e).__name__}: {e}"}})
        finally:
            q.put(done_sentinel)

    t = threading.Thread(target=worker, args=(content,), daemon=True)
    t.start()

    def sse_gen():
        # SSE format:
        # event: <name>\n
        # data: <json>\n\n
        while True:
            evt = q.get()
            if evt.get("event") == "__DONE__":
                break
            name = evt.get("event", "message")
            data = json.dumps(evt.get("payload", {}), ensure_ascii=False)
            yield f"event: {name}\n".encode("utf-8")
            yield f"data: {data}\n\n".encode("utf-8")

    return StreamingResponse(sse_gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8010"))
    # Prefer running from repo root with:
    #   python -m applications.summarization.app
    # For local dev, this also works from applications/summarization:
    #   python -m app
    uvicorn.run(app, host=host, port=port, reload=False)
