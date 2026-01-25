"""
Video summarization pipeline:

1) Extract mono 16kHz WAV from input video (ffmpeg).
2) Diarize speakers locally using pyannote.
3) For each diarized segment:
   - Cut segment audio.
   - Call Whissle STT on the segment audio.
   - Extract a representative keyframe from the video.
   - Use Gemini to caption the keyframe (visual context).
4) Use Gemini again to produce a final video summary from segment data.

This module is designed to be used by `applications/summarization/app.py`.
"""

from __future__ import annotations

import json
import os
import inspect
import logging
import base64
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from pydub import AudioSegment
import torch
import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None  # type: ignore

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None  # type: ignore

try:
    from pyannote.audio import Pipeline
except Exception:  # pragma: no cover
    Pipeline = None  # type: ignore

class VideoSummarizationError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentResult:
    idx: int
    speaker: str
    start_s: float
    end_s: float
    duration_s: float
    transcript: str
    keyframe_path: Optional[str]
    keyframe_caption: Optional[str]


def _run(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise VideoSummarizationError(
            f"Missing executable: {cmd[0]}. Please install it (e.g. ffmpeg)."
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="ignore")
        raise VideoSummarizationError(f"Command failed: {' '.join(cmd)}\n{stderr}") from e


def extract_audio_wav(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> None:
    """Extract mono WAV audio from a video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(wav_path),
    ]
    _run(cmd)


def diarize_audio(
    wav_path: Path,
    *,
    pyannote_model: str,
    hf_token: Optional[str],
    device: Optional[str] = None,
) -> List[Tuple[float, float, str]]:
    """
    Returns list of (start_s, end_s, speaker_label).
    Requires `pyannote.audio` installed and a diarization pipeline available.
    """
    if Pipeline is None:
        raise VideoSummarizationError(
            "pyannote.audio is not available. Install it and its dependencies."
        )

    # Load pipeline with a version-compatible token kwarg.
    try:
        sig = inspect.signature(Pipeline.from_pretrained)
        kwargs: Dict[str, Any] = {}
        if hf_token:
            if "use_auth_token" in sig.parameters:
                kwargs["use_auth_token"] = hf_token
            elif "auth_token" in sig.parameters:
                kwargs["auth_token"] = hf_token
            elif "token" in sig.parameters:
                kwargs["token"] = hf_token
        pipeline = Pipeline.from_pretrained(pyannote_model, **kwargs)
    except Exception as e:
        raise VideoSummarizationError(
            f"Failed to load pyannote pipeline '{pyannote_model}'. "
            f"Ensure model weights are accessible (HF cache) and HF_TOKEN is set if needed. Error: {e}"
        ) from e

    if device:
        try:
            pipeline.to(device)
        except Exception:
            # Not fatal (some versions/pipelines may not support .to())
            pass

    # Avoid pyannote's internal decoding (torchcodec warning on some macOS envs):
    # feed preloaded waveform instead.
    waveform: Optional[torch.Tensor] = None
    sample_rate: Optional[int] = None

    try:
        if sf is not None:
            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
            # soundfile returns shape (time, channels) when always_2d=True
            audio = np.transpose(audio)  # (channels, time)
            waveform = torch.from_numpy(audio)
            sample_rate = int(sr)
        elif torchaudio is not None:
            waveform, sr = torchaudio.load(str(wav_path))
            sample_rate = int(sr)
        else:
            raise VideoSummarizationError(
                "Cannot preload audio for pyannote (missing soundfile and torchaudio)."
            )
    except Exception as e:
        raise VideoSummarizationError(f"Failed to load WAV for diarization: {e}") from e

    diarization_out = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    def _safe_repr(obj: Any, limit: int = 500) -> str:
        try:
            s = repr(obj)
        except Exception:
            s = f"<unreprable {type(obj).__name__}>"
        return s if len(s) <= limit else s[:limit] + "…"

    # pyannote versions differ:
    # - some return `pyannote.core.Annotation` directly (has .itertracks)
    # - others return a wrapper (often called DiarizeOutput) holding the Annotation somewhere.
    annotation = None

    # Fast path: already iterable
    if hasattr(diarization_out, "itertracks"):
        annotation = diarization_out
    # Common dict-like shape
    elif isinstance(diarization_out, dict):
        for key in ("diarization", "annotation"):
            if key in diarization_out and hasattr(diarization_out[key], "itertracks"):
                annotation = diarization_out[key]
                break
    else:
        # Common attribute names across pyannote pipeline wrappers
        for attr in (
            "diarization",
            "annotation",
            "speaker_diarization",
            "exclusive_speaker_diarization",
            "predictions",
            "prediction",
            "result",
            "output",
        ):
            if hasattr(diarization_out, attr):
                candidate = getattr(diarization_out, attr)
                if hasattr(candidate, "itertracks"):
                    annotation = candidate
                    break

        # Some wrappers expose dict-style access
        if annotation is None and hasattr(diarization_out, "__getitem__"):
            for key in ("diarization", "annotation"):
                try:
                    candidate = diarization_out[key]  # type: ignore[index]
                except Exception:
                    continue
                if hasattr(candidate, "itertracks"):
                    annotation = candidate
                    break

    if annotation is None:
        # Print more context to help diagnose the exact object shape in your environment.
        attrs = []
        try:
            attrs = [a for a in dir(diarization_out) if not a.startswith("_")]
        except Exception:
            pass
        dkeys = []
        if hasattr(diarization_out, "__dict__"):
            try:
                dkeys = list(getattr(diarization_out, "__dict__", {}).keys())
            except Exception:
                dkeys = []

        logger.error(
            "pyannote diarization output unwrap failed. type=%s repr=%s public_attrs(sample)=%s __dict__(keys)=%s",
            type(diarization_out).__name__,
            _safe_repr(diarization_out),
            attrs[:40],
            dkeys,
        )
        raise VideoSummarizationError(
            "Unexpected diarization output type from pyannote pipeline "
            f"({type(diarization_out).__name__}); cannot iterate tracks. "
            "Check server logs for the object's available fields."
        )

    segments: List[Tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((float(turn.start), float(turn.end), str(speaker)))

    segments.sort(key=lambda x: (x[0], x[1], x[2]))
    return segments


def merge_adjacent_segments(
    segments: List[Tuple[float, float, str]],
    *,
    max_gap_s: float = 0.25,
    min_duration_s: float = 0.3,
) -> List[Tuple[float, float, str]]:
    if not segments:
        return []

    merged: List[Tuple[float, float, str]] = []
    cur_s, cur_e, cur_spk = segments[0]
    for s, e, spk in segments[1:]:
        gap = s - cur_e
        if spk == cur_spk and gap <= max_gap_s:
            cur_e = max(cur_e, e)
        else:
            if (cur_e - cur_s) >= min_duration_s:
                merged.append((cur_s, cur_e, cur_spk))
            cur_s, cur_e, cur_spk = s, e, spk
    if (cur_e - cur_s) >= min_duration_s:
        merged.append((cur_s, cur_e, cur_spk))
    return merged


def cut_wav_segment(
    wav_path: Path,
    out_wav_path: Path,
    start_s: float,
    end_s: float,
) -> None:
    audio = AudioSegment.from_wav(str(wav_path))
    start_ms = max(0, int(start_s * 1000))
    end_ms = max(start_ms + 1, int(end_s * 1000))
    seg = audio[start_ms:end_ms]
    seg.export(str(out_wav_path), format="wav")


def extract_keyframe(
    video_path: Path,
    out_image_path: Path,
    timestamp_s: float,
) -> None:
    """Extract a single frame using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, timestamp_s):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_image_path),
    ]
    _run(cmd)


def whissle_stt_segment(
    audio_path: Path,
    *,
    whissle_auth_token: str,
    url: str = "https://api.whissle.ai/v1/conversation/STT",
    timeout_s: int = 120,
) -> str:
    """
    Calls Whissle STT via the REST endpoint shown by the user.
    Expects multipart form-data with field name 'audio'.
    """
    params = {"auth_token": whissle_auth_token}
    headers = {"Accept": "*/*"}

    with open(audio_path, "rb") as f:
        files = {"audio": (audio_path.name, f, "audio/wav")}
        resp = requests.post(url, params=params, headers=headers, files=files, timeout=timeout_s)
    if resp.status_code >= 400:
        raise VideoSummarizationError(f"Whissle STT failed ({resp.status_code}): {resp.text[:500]}")

    try:
        data = resp.json()
        if isinstance(data, dict):
            for k in ("text", "transcript", "transcription"):
                v = data.get(k)
                if isinstance(v, str):
                    return v.strip()
            if isinstance(data.get("data"), dict):
                v = data["data"].get("text") or data["data"].get("transcript")
                if isinstance(v, str):
                    return v.strip()
        return json.dumps(data)[:2000]
    except Exception:
        return resp.text.strip()


def _require_gemini() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise VideoSummarizationError("GOOGLE_API_KEY is not set (required for Gemini).")


def _normalize_gemini_model(model_name: str) -> str:
    # Accept "gemini-1.5-flash" or "models/gemini-1.5-flash"
    model_name = (model_name or "").strip()
    if model_name.startswith("models/"):
        return model_name[len("models/") :]
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def _gemini_generate_text(
    *,
    model_name: str,
    prompt: str,
    image_bytes: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
    timeout_s: int = 180,
) -> str:
    """
    Minimal Gemini call over HTTP (no google SDK dependency).
    Uses Google AI Studio Generative Language API.
    """
    _require_gemini()
    api_key = os.getenv("GOOGLE_API_KEY") or ""
    model = _normalize_gemini_model(model_name) or "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    parts: List[Dict[str, Any]] = [{"text": prompt}]
    if image_bytes is not None:
        if not image_mime_type:
            image_mime_type = "image/jpeg"
        parts.append(
            {
                "inline_data": {
                    "mime_type": image_mime_type,
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                }
            }
        )

    body = {"contents": [{"parts": parts}]}
    resp = requests.post(url, params={"key": api_key}, json=body, timeout=timeout_s)
    if resp.status_code >= 400:
        raise VideoSummarizationError(f"Gemini call failed ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        out_parts = content.get("parts") or []
        texts = []
        for p in out_parts:
            t = p.get("text")
            if isinstance(t, str):
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return json.dumps(data)[:2000]


def gemini_caption_image(
    image_path: Path,
    *,
    model_name: str = "gemini-1.5-flash",
    timeout_s: int = 120,
) -> str:
    prompt = (
        "Describe this video frame in 1-2 sentences. "
        "Focus on key visual facts (people, actions, scene, text on screen if readable)."
    )
    image_bytes = image_path.read_bytes()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    return _gemini_generate_text(
        model_name=model_name,
        prompt=prompt,
        image_bytes=image_bytes,
        image_mime_type=mime,
        timeout_s=timeout_s,
    )


def gemini_summarize_video(
    segments: List[SegmentResult],
    *,
    model_name: str = "gemini-1.5-flash",
    timeout_s: int = 180,
) -> str:
    payload = [
        {
            "idx": s.idx,
            "speaker": s.speaker,
            "start_s": round(s.start_s, 3),
            "end_s": round(s.end_s, 3),
            "duration_s": round(s.duration_s, 3),
            "transcript": s.transcript,
            "keyframe_caption": s.keyframe_caption,
        }
        for s in segments
    ]

    prompt = (
        "You are given diarized video segments with per-segment speech transcripts and a keyframe caption.\n"
        "Write a concise video summary (5-10 bullet points) and then a 2-3 sentence overall abstract.\n"
        "Be faithful to the transcript; if something is unclear, say it's unclear.\n\n"
        f"SEGMENTS_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    return _gemini_generate_text(model_name=model_name, prompt=prompt, timeout_s=timeout_s)


def summarize_video_file(
    video_path: Path,
    *,
    whissle_auth_token: str,
    pyannote_model: str,
    hf_token: Optional[str],
    pyannote_device: Optional[str] = None,
    keep_artifacts: bool = False,
    gemini_model: str = "gemini-1.5-flash",
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint used by the API.
    Returns a dict containing `segments` and `summary`.
    """
    if not whissle_auth_token:
        raise VideoSummarizationError("Whissle auth token is missing. Set WHISSLE_AUTH_TOKEN.")

    def emit(event: str, **payload: Any) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb({"event": event, "payload": payload})
        except Exception:
            # Never break the pipeline due to progress reporting.
            pass

    work_dir_obj: Optional[tempfile.TemporaryDirectory[str]] = None
    if keep_artifacts:
        work_dir = Path(tempfile.mkdtemp(prefix="video_summarize_"))
    else:
        work_dir_obj = tempfile.TemporaryDirectory(prefix="video_summarize_")
        work_dir = Path(work_dir_obj.name)

    try:
        emit("start", video=str(video_path))
        wav_path = work_dir / "audio.wav"
        emit("extract_audio_started")
        extract_audio_wav(video_path, wav_path)
        emit("extract_audio_done", wav=str(wav_path))

        emit("diarization_started", model=pyannote_model)
        raw_segments = diarize_audio(
            wav_path,
            pyannote_model=pyannote_model,
            hf_token=hf_token,
            device=pyannote_device,
        )
        diarized = merge_adjacent_segments(raw_segments)
        emit("diarization_done", raw_segments=len(raw_segments), segments=len(diarized))

        seg_results: List[SegmentResult] = []
        for idx, (start_s, end_s, speaker) in enumerate(diarized):
            emit("segment_started", idx=idx, speaker=speaker, start_s=start_s, end_s=end_s)
            seg_wav = work_dir / f"seg_{idx:04d}.wav"
            cut_wav_segment(wav_path, seg_wav, start_s, end_s)

            emit("stt_started", idx=idx, speaker=speaker)
            transcript = whissle_stt_segment(seg_wav, whissle_auth_token=whissle_auth_token)
            emit("stt_done", idx=idx, transcript_preview=transcript[:160])

            mid_s = start_s + (end_s - start_s) / 2.0
            frame_path = work_dir / f"frame_{idx:04d}.jpg"
            keyframe_caption: Optional[str] = None
            keyframe_path: Optional[str] = None
            try:
                emit("keyframe_started", idx=idx, timestamp_s=mid_s)
                extract_keyframe(video_path, frame_path, mid_s)
                keyframe_path = str(frame_path)
                emit("keyframe_done", idx=idx, keyframe_path=keyframe_path)

                emit("caption_started", idx=idx)
                keyframe_caption = gemini_caption_image(frame_path, model_name=gemini_model)
                emit("caption_done", idx=idx, caption_preview=(keyframe_caption or "")[:160])
            except Exception:
                keyframe_path = None
                keyframe_caption = None
                emit("caption_skipped_or_failed", idx=idx)

            seg_results.append(
                SegmentResult(
                    idx=idx,
                    speaker=speaker,
                    start_s=start_s,
                    end_s=end_s,
                    duration_s=end_s - start_s,
                    transcript=transcript,
                    keyframe_path=keyframe_path,
                    keyframe_caption=keyframe_caption,
                )
            )
            emit("segment_done", idx=idx)

        emit("summary_started")
        summary = gemini_summarize_video(seg_results, model_name=gemini_model)
        emit("summary_done", summary_preview=(summary or "")[:200])

        result = {
            "artifacts_dir": str(work_dir) if keep_artifacts else None,
            "num_segments": len(seg_results),
            "segments": [
                {
                    "idx": s.idx,
                    "speaker": s.speaker,
                    "start_s": s.start_s,
                    "end_s": s.end_s,
                    "duration_s": s.duration_s,
                    "transcript": s.transcript,
                    "keyframe_path": s.keyframe_path,
                    "keyframe_caption": s.keyframe_caption,
                }
                for s in seg_results
            ],
            "summary": summary,
        }
        emit("done", num_segments=len(seg_results))
        return result
    finally:
        if work_dir_obj is not None:
            work_dir_obj.cleanup()
