#!/usr/bin/env python3
"""
competitive_analysis.py
=======================
Compare emotion labels between:
  1) A deployed HTTP model endpoint (multipart upload; returns JSON with label in `class`, e.g. {"class": "Happy"})
  2) A JSONL manifest with ground-truth emotion_label (one JSON object per line)
  3) (Optional) Whissle STT API (returns transcript with inline tags like EMOTION_*, INTENT_*, etc.)

Example (deployment):
  POST {deployment_url}
    headers: Authorization: Bearer <API_KEY>
    files:   content=@audio.wav
  response JSON: {"class": "..."}

Example (Whissle):
  POST https://api.whissle.ai/v1/conversation/STT?auth_token=<TOKEN>
    files: audio=@audio.wav
  response JSON:
    {"transcript": "... AGE_30_45 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM", ...}

This script normalizes emotion labels to a canonical form and reports match rates:
  - deployment emotion vs manifest GT emotion
  - (optional) deployment emotion vs Whissle EMOTION_* tag

Usage (manifest):
  python competitive_analysis.py \
    --manifest-jsonl /path/to/manifest.jsonl \
    --deployment-url "https://.../predict" \
    --deployment-api-key "$DEPLOYMENT_API_KEY" \
    --whissle-auth-token "$WHISSLE_AUTH_TOKEN" \
    --out-csv /tmp/emotion_comp.csv \
    --out-jsonl /tmp/emotion_comp.jsonl

Usage (skip Whissle):
  python competitive_analysis.py \
    --audio-dir /path/to/wavs \
    --deployment-url "https://.../predict" \
    --deployment-api-key "$DEPLOYMENT_API_KEY" \
    --skip-whissle \
    --out-csv /tmp/intent_comp.csv \
    --out-jsonl /tmp/intent_comp.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests  # type: ignore[import-not-found]
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'requests'. Install it with: pip install requests"
    ) from e


WHISSLE_STT_URL = "https://api.whissle.ai/v1/conversation/STT"

TAG_PATTERNS = {
    "AGE": re.compile(r"\b(AGE_\d+_\d+)\b"),
    # Whissle example uses GENDER_*, older eval code uses GER_*
    "GENDER": re.compile(r"\b((?:GENDER|GER)_[A-Z]+)\b"),
    "EMOTION": re.compile(r"\b(EMOTION_[A-Z]+)\b"),
    "INTENT": re.compile(r"\b(INTENT_[A-Z]+)\b"),
}


def _iter_audio_files(audio_dir: Path) -> Iterable[Path]:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".webm"}
    for p in sorted(audio_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _normalize_intent(label: Optional[str], intent_prefix: str = "INTENT_") -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    s = s.upper()
    if s.startswith(intent_prefix):
        return s
    # Common cases: "INFORM", "Intent_inform", etc.
    if re.fullmatch(r"[A-Z0-9_]+", s):
        return f"{intent_prefix}{s}"
    return s  # fallback: keep as-is if it contains spaces/symbols


def _canon_emotion(label: Optional[str]) -> Optional[str]:
    """
    Canonical emotion string for comparisons.
    Examples:
      "Happy" -> "HAPPY"
      "EMOTION_NEUTRAL" -> "NEUTRAL"
    """
    if label is None:
        return None
    s = str(label).strip().upper()
    if not s:
        return None
    if s.startswith("EMOTION_"):
        s = s[len("EMOTION_") :]
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    return s or None


def _extract_tags_from_transcript(transcript: str) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {k: None for k in TAG_PATTERNS.keys()}
    for k, pat in TAG_PATTERNS.items():
        m = pat.search(transcript or "")
        out[k] = m.group(1) if m else None
    return out


def call_deployment(
    *,
    url: str,
    api_key: str,
    audio_path: Path,
    timeout_s: float,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with audio_path.open("rb") as f:
        resp = requests.post(url, files={"content": f}, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def call_whissle_stt(
    *,
    auth_token: str,
    audio_path: Path,
    timeout_s: float,
) -> Dict[str, Any]:
    params = {"auth_token": auth_token}
    headers = {"Accept": "*/*"}
    with audio_path.open("rb") as f:
        files = {"audio": (audio_path.name, f)}
        resp = requests.post(WHISSLE_STT_URL, params=params, headers=headers, files=files, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


@dataclass
class Row:
    audio_filepath: str
    text: Optional[str]
    gt_emotion_label_raw: Optional[str]
    gt_emotion: Optional[str]
    deployment_label_raw: Optional[str]
    deployment_emotion: Optional[str]
    whissle_emotion: Optional[str]
    match_deployment_vs_gt: Optional[bool]
    match_deployment_vs_whissle: Optional[bool]
    whissle_transcript: Optional[str]
    whissle_duration_seconds: Optional[float]
    whissle_language: Optional[str]
    error: Optional[str]


@dataclass
class Sample:
    audio_path: Path
    text: Optional[str] = None
    emotion_label: Optional[str] = None


def _load_manifest_jsonl(path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "audio_filepath" not in obj:
                raise ValueError(f"Line {i}: missing key 'audio_filepath'")
            samples.append(
                Sample(
                    audio_path=Path(obj["audio_filepath"]),
                    text=obj.get("text"),
                    emotion_label=obj.get("emotion_label"),
                )
            )
    return samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare emotion labels between deployment, manifest GT, and optional Whissle STT tags.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest-jsonl", type=Path, help="JSONL manifest with audio_filepath,text,emotion_label")
    src.add_argument("--audio-file", type=Path, help="Single audio file to evaluate")
    src.add_argument("--audio-dir", type=Path, help="Directory of audio files (recursive)")

    p.add_argument("--deployment-url", required=True, help="Deployment URL (e.g. https://.../predict)")
    p.add_argument(
        "--deployment-api-key",
        default=os.getenv("DEPLOYMENT_API_KEY", ""),
        help="Deployment API key (or set env DEPLOYMENT_API_KEY)",
    )

    p.add_argument(
        "--whissle-auth-token",
        default=os.getenv("WHISSLE_AUTH_TOKEN", ""),
        help="Whissle auth token (or set env WHISSLE_AUTH_TOKEN)",
    )
    p.add_argument("--skip-whissle", action="store_true", help="Skip Whissle STT call entirely")

    p.add_argument(
        "--deployment-label-field",
        default="class",
        help="JSON field in deployment response that contains the label (default: class)",
    )
    p.add_argument(
        "--intent-prefix",
        default="INTENT_",
        help="Prefix used for normalized intents (default: INTENT_)",
    )
    p.add_argument("--timeout-s", type=float, default=60.0, help="HTTP timeout per request")

    p.add_argument("--out-csv", type=Path, required=True, help="Write per-file results to CSV")
    p.add_argument("--out-jsonl", type=Path, required=True, help="Write per-file results to JSONL (full payloads)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.deployment_api_key:
        raise SystemExit("Missing --deployment-api-key (or env DEPLOYMENT_API_KEY)")
    if (not args.skip_whissle) and (not args.whissle_auth_token):
        raise SystemExit("Missing --whissle-auth-token (or env WHISSLE_AUTH_TOKEN), or pass --skip-whissle")

    samples: List[Sample]
    if args.manifest_jsonl:
        samples = _load_manifest_jsonl(args.manifest_jsonl)
        if not samples:
            raise SystemExit(f"No samples found in manifest: {args.manifest_jsonl}")
    elif args.audio_file:
        samples = [Sample(audio_path=args.audio_file)]
    else:
        audio_files = list(_iter_audio_files(args.audio_dir))
        if not audio_files:
            raise SystemExit(f"No audio files found under: {args.audio_dir}")
        samples = [Sample(audio_path=p) for p in audio_files]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Row] = []
    n_ok_gt, n_match_gt = 0, 0
    n_ok_wh, n_match_wh = 0, 0
    conf_gt: Dict[str, Dict[str, int]] = {}

    with args.out_jsonl.open("w", encoding="utf-8") as jsonl_f:
        for sample in samples:
            audio_path = sample.audio_path
            deployment_resp: Optional[Dict[str, Any]] = None
            whissle_resp: Optional[Dict[str, Any]] = None
            err: Optional[str] = None

            try:
                deployment_resp = call_deployment(
                    url=args.deployment_url,
                    api_key=args.deployment_api_key,
                    audio_path=audio_path,
                    timeout_s=args.timeout_s,
                )
                if not args.skip_whissle:
                    whissle_resp = call_whissle_stt(
                        auth_token=args.whissle_auth_token,
                        audio_path=audio_path,
                        timeout_s=args.timeout_s,
                    )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

            gt_emotion_raw = sample.emotion_label
            gt_emotion = _canon_emotion(gt_emotion_raw)

            deployment_label_raw = None
            deployment_emotion = None
            whissle_emotion = None
            whissle_transcript = None
            whissle_duration = None
            whissle_lang = None

            if deployment_resp is not None:
                deployment_label_raw = deployment_resp.get(args.deployment_label_field)
                deployment_emotion = _canon_emotion(deployment_label_raw)

            if whissle_resp is not None:
                whissle_transcript = whissle_resp.get("transcript")
                whissle_duration = whissle_resp.get("duration_seconds")
                whissle_lang = whissle_resp.get("language")
                tags = _extract_tags_from_transcript(whissle_transcript or "")
                whissle_emotion = _canon_emotion(tags.get("EMOTION"))

            match_deployment_vs_gt: Optional[bool] = None
            match_deployment_vs_whissle: Optional[bool] = None

            if err is None:
                if gt_emotion and deployment_emotion:
                    match_deployment_vs_gt = (gt_emotion == deployment_emotion)
                    n_ok_gt += 1
                    if match_deployment_vs_gt:
                        n_match_gt += 1
                    conf_gt.setdefault(gt_emotion, {})
                    conf_gt[gt_emotion][deployment_emotion] = conf_gt[gt_emotion].get(deployment_emotion, 0) + 1

                if whissle_emotion and deployment_emotion:
                    match_deployment_vs_whissle = (whissle_emotion == deployment_emotion)
                    n_ok_wh += 1
                    if match_deployment_vs_whissle:
                        n_match_wh += 1

            row = Row(
                audio_filepath=str(audio_path),
                text=sample.text,
                gt_emotion_label_raw=str(gt_emotion_raw) if gt_emotion_raw is not None else None,
                gt_emotion=gt_emotion,
                deployment_label_raw=str(deployment_label_raw) if deployment_label_raw is not None else None,
                deployment_emotion=deployment_emotion,
                whissle_emotion=whissle_emotion,
                match_deployment_vs_gt=match_deployment_vs_gt,
                match_deployment_vs_whissle=match_deployment_vs_whissle,
                whissle_transcript=whissle_transcript,
                whissle_duration_seconds=whissle_duration,
                whissle_language=whissle_lang,
                error=err,
            )
            rows.append(row)

            # full JSONL record (includes raw payloads)
            jsonl_f.write(
                json.dumps(
                    {
                        "audio_filepath": row.audio_filepath,
                        "text": row.text,
                        "gt_emotion_label_raw": row.gt_emotion_label_raw,
                        "gt_emotion": row.gt_emotion,
                        "deployment": deployment_resp,
                        "whissle": whissle_resp,
                        "deployment_label_raw": row.deployment_label_raw,
                        "deployment_emotion": row.deployment_emotion,
                        "whissle_emotion": row.whissle_emotion,
                        "match_deployment_vs_gt": row.match_deployment_vs_gt,
                        "match_deployment_vs_whissle": row.match_deployment_vs_whissle,
                        "error": row.error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # CSV (human-friendly)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "audio_filepath",
                "text",
                "gt_emotion_label_raw",
                "gt_emotion",
                "deployment_label_raw",
                "deployment_emotion",
                "whissle_emotion",
                "match_deployment_vs_gt",
                "match_deployment_vs_whissle",
                "whissle_language",
                "whissle_duration_seconds",
                "whissle_transcript",
                "error",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "audio_filepath": r.audio_filepath,
                    "text": r.text,
                    "gt_emotion_label_raw": r.gt_emotion_label_raw,
                    "gt_emotion": r.gt_emotion,
                    "deployment_label_raw": r.deployment_label_raw,
                    "deployment_emotion": r.deployment_emotion,
                    "whissle_emotion": r.whissle_emotion,
                    "match_deployment_vs_gt": r.match_deployment_vs_gt,
                    "match_deployment_vs_whissle": r.match_deployment_vs_whissle,
                    "whissle_language": r.whissle_language,
                    "whissle_duration_seconds": r.whissle_duration_seconds,
                    "whissle_transcript": r.whissle_transcript,
                    "error": r.error,
                }
            )

    total = len(rows)
    print(f"Processed: {total}")
    if n_ok_gt:
        print(f"Deployment vs GT emotion: {n_match_gt}/{n_ok_gt} = {n_match_gt / n_ok_gt:.2%}")
    else:
        print("Deployment vs GT emotion: n/a (manifest missing emotion_label or requests failed)")

    if n_ok_wh:
        print(f"Deployment vs Whissle emotion: {n_match_wh}/{n_ok_wh} = {n_match_wh / n_ok_wh:.2%}")
    elif args.skip_whissle:
        print("Deployment vs Whissle emotion: skipped (--skip-whissle)")
    else:
        print("Deployment vs Whissle emotion: n/a (missing Whissle emotion tag or requests failed)")

    if conf_gt:
        print("\nConfusion (GT -> predicted) [top]:")
        for gt in sorted(conf_gt.keys()):
            preds = conf_gt[gt]
            top = sorted(preds.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            top_str = ", ".join([f"{p}:{c}" for p, c in top])
            print(f"  {gt}: {top_str}")
    print(f"Wrote CSV : {args.out_csv.resolve()}")
    print(f"Wrote JSONL: {args.out_jsonl.resolve()}")


if __name__ == "__main__":
    main()


