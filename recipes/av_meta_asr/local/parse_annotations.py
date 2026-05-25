#!/usr/bin/env python3
"""Parse SpeakerVid-5M annotations into structured metadata with rich tags.

Combines merge_anno (bbox, is_talking, speaker count), asr (transcripts),
anno (MLLM captions), and l_score (quality) into a single structured JSONL
output with:
  - Visual metadata tags (talking, facing, expression, body, activity, movement)
  - Scene classification (indoor/outdoor/studio/meeting/presentation)
  - Noise level estimation (clean/mild/heavy)
  - Inline event tokens extracted from MLLM captions (NOD, GESTURE, SMILE, etc.)

Usage:
    python parse_annotations.py \
        --annotations-dir /mnt/nfs/data/speakervid_5m/annotations \
        --output /mnt/nfs/data/speakervid_5m/metadata/parsed_annotations.jsonl \
        --workers 8
"""
import argparse
import json
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Visual metadata patterns ──

FACING_PATTERNS = {
    "VISUAL_FACING_FRONT": re.compile(
        r"facing\s+(the\s+)?camera|looking\s+(at|into)\s+(the\s+)?camera|frontal|front[- ]?facing", re.I),
    "VISUAL_FACING_SIDE": re.compile(
        r"side\s+(view|profile)|facing\s+(left|right|sideways)|profile\s+view|turned\s+(to\s+the\s+)?(left|right)", re.I),
    "VISUAL_FACING_AWAY": re.compile(
        r"facing\s+away|back\s+(to|of)\s+(the\s+)?camera|turned\s+away|looking\s+away", re.I),
}

BODY_PATTERNS = {
    "VISUAL_FULLBODY": re.compile(
        r"full[- ]?body|full\s+figure|entire\s+body|standing|walking|seated.*full", re.I),
    "VISUAL_HALFBODY": re.compile(
        r"half[- ]?body|upper\s+body|waist[- ]?up|torso|chest[- ]?up|bust", re.I),
    "VISUAL_CLOSEUP": re.compile(
        r"close[- ]?up|head\s+shot|face\s+(shot|only)|portrait|shoulders?\s+up", re.I),
}

ACTIVITY_PATTERNS = {
    "VISUAL_ACTIVE": re.compile(
        r"actively|energetic|animated|moving\s+around|walking|exercising|gesturing\s+(a\s+lot|actively|frequently)", re.I),
    "VISUAL_PASSIVE": re.compile(
        r"calmly|relaxed|seated|sitting\s+still|standing\s+still|minimal\s+movement", re.I),
    "VISUAL_STATIC": re.compile(
        r"static|stationary|not\s+moving|still|motionless|frozen", re.I),
}

EXPRESSION_PATTERNS = {
    "VISUAL_EXPR_SMILING": re.compile(
        r"smil(ing|e)|happy|cheerful|joyful|laughing|grinning|beaming", re.I),
    "VISUAL_EXPR_SERIOUS": re.compile(
        r"serious|stern|focused|concentrat|frown|concerned|worried|angry|upset|tense", re.I),
    "VISUAL_EXPR_ANIMATED": re.compile(
        r"animated|expressive|enthusiastic|excited|surprised|emotive|varied\s+expression", re.I),
    "VISUAL_EXPR_NEUTRAL": re.compile(
        r"neutral|calm|composed|blank|no\s+(particular|strong)\s+expression|expressionless|poker", re.I),
}

MOVEMENT_PATTERNS = {
    "VISUAL_GESTURING": re.compile(
        r"gestur(ing|e)|hand\s+mov|pointing|waving|signing|using\s+hands", re.I),
    "VISUAL_MOVING": re.compile(
        r"moving|shifting|swaying|rocking|bobbing|nodding|head\s+mov|leaning", re.I),
    "VISUAL_STILL": re.compile(
        r"still|stationary|not\s+moving|motionless|minimal", re.I),
}

# ── Scene classification patterns ──

SCENE_PATTERNS = {
    "SCENE_STUDIO": re.compile(
        r"studio|broadcast|news\s+(desk|room|anchor)|podcast|recording|backdrop|green\s*screen|professional\s+setup", re.I),
    "SCENE_MEETING": re.compile(
        r"meeting|conference\s+room|boardroom|office\s+meeting|video\s+call|zoom|teams|round\s*table", re.I),
    "SCENE_PRESENTATION": re.compile(
        r"present(ation|ing)|lecture|podium|stage|classroom|auditorium|slide|whiteboard|teaching|seminar|ted\s*talk", re.I),
    "SCENE_OUTDOOR": re.compile(
        r"outdoor|outside|street|park|garden|nature|sky|tree|building\s+exterior|sidewalk|field|beach|mountain", re.I),
    "SCENE_INDOOR": re.compile(
        r"indoor|inside|room|office|home|kitchen|living\s+room|bedroom|hallway|desk|chair|sofa|interior|wall|ceiling", re.I),
}

# ── Noise level thresholds (from l_score face clarity) ──
# l_score values: higher = clearer face detection
# We use face clarity as a proxy for overall quality
NOISE_CLARITY_THRESHOLDS = {
    "NOISE_CLEAN": 0.8,   # face clarity >= 0.8
    "NOISE_MILD": 0.5,    # face clarity >= 0.5
    "NOISE_HEAVY": 0.0,   # everything else
}

# ── Inline event patterns (extracted from MLLM captions) ──

INLINE_EVENT_PATTERNS = [
    (re.compile(r"\bnod(ding|s)?\b", re.I), "<NOD>"),
    (re.compile(r"\bshak(ing|e)\s*(head|his\s+head|her\s+head)\b", re.I), "<SHAKE>"),
    (re.compile(r"\bgestur(ing|e)\b|\bhand\s+(gesture|movement|motion)\b", re.I), "<GESTURE>"),
    (re.compile(r"\bsmil(ing|e)\b|\bgrin(ning)?\b", re.I), "<SMILE>"),
    (re.compile(r"\bfrown(ing)?\b|\bscowl(ing)?\b", re.I), "<FROWN>"),
    (re.compile(r"\blook(ing)?\s+(away|down|aside)\b|\bgaze\s+(away|averted)\b|\baverts?\b", re.I), "<GAZE_AWAY>"),
    (re.compile(r"\blean(ing|s)?\s*(forward|in|closer)\b", re.I), "<LEAN_IN>"),
    (re.compile(r"\blean(ing|s)?\s*(back|away)\b", re.I), "<LEAN_BACK>"),
    (re.compile(r"\blaugh(ing|s|ter)?\b", re.I), "<LAUGH>"),
    (re.compile(r"\bcough(ing|s)?\b", re.I), "<COUGH>"),
    (re.compile(r"\bapplau(se|ding)\b|\bclapping\b", re.I), "<APPLAUSE>"),
    (re.compile(r"\bmusic\b|\bsinging\b|\bplaying\s+(guitar|piano|instrument)\b", re.I), "<MUSIC>"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Parse SpeakerVid-5M annotations")
    parser.add_argument("--annotations-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--clips-dir", type=str, default=None,
        help="If set, derive clip names from files in this directory instead of "
             "globbing merged_anno/ (avoids NFS hang on million-file directories)",
    )
    return parser.parse_args()


def classify_from_caption(text: str, patterns: dict, default: str) -> str:
    if not text:
        return default
    for label, pattern in patterns.items():
        if pattern.search(text):
            return label
    return default


def extract_inline_events(text: str) -> list:
    """Extract inline event tokens from MLLM caption text."""
    events = []
    if not text:
        return events
    seen = set()
    for pattern, tag in INLINE_EVENT_PATTERNS:
        if tag not in seen and pattern.search(text):
            events.append(tag)
            seen.add(tag)
    return events


def estimate_noise_level(l_score_path: Path) -> str:
    """Estimate noise level from l_score face/hand clarity."""
    if not l_score_path.exists():
        return "NOISE_MILD"
    try:
        with open(l_score_path, "r", encoding="utf-8") as f:
            score = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "NOISE_MILD"

    face_clarity = 0.5
    if isinstance(score, dict):
        face_clarity = float(score.get("face_score", score.get("face_clarity",
                             score.get("score", 0.5))))
    elif isinstance(score, (int, float)):
        face_clarity = float(score)

    if face_clarity >= NOISE_CLARITY_THRESHOLDS["NOISE_CLEAN"]:
        return "NOISE_CLEAN"
    elif face_clarity >= NOISE_CLARITY_THRESHOLDS["NOISE_MILD"]:
        return "NOISE_MILD"
    return "NOISE_HEAVY"


def parse_mllm_caption(anno_path: Path) -> dict:
    if not anno_path.exists():
        return {}
    try:
        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}

    caption_parts = []
    for key in ["is_speaking", "is_moving", "number_of_people", "facing_direction",
                "body_view", "activity_level", "movement_caption", "expression_caption",
                "caption", "description", "scene_description", "environment",
                "background", "setting"]:
        val = anno.get(key)
        if isinstance(val, str) and val.strip():
            caption_parts.append(val.strip())

    full_caption = " ".join(caption_parts)

    inline_events = extract_inline_events(full_caption)

    return {
        "facing": classify_from_caption(full_caption, FACING_PATTERNS, "VISUAL_FACING_FRONT"),
        "body": classify_from_caption(full_caption, BODY_PATTERNS, "VISUAL_HALFBODY"),
        "activity": classify_from_caption(full_caption, ACTIVITY_PATTERNS, "VISUAL_PASSIVE"),
        "expression": classify_from_caption(full_caption, EXPRESSION_PATTERNS, "VISUAL_EXPR_NEUTRAL"),
        "movement": classify_from_caption(full_caption, MOVEMENT_PATTERNS, "VISUAL_STILL"),
        "scene_type": classify_from_caption(full_caption, SCENE_PATTERNS, "SCENE_INDOOR"),
        "inline_events": inline_events,
        "raw_caption": full_caption[:500] if full_caption else "",
    }


def parse_asr(asr_path: Path) -> dict:
    if not asr_path.exists():
        return {"transcript": "", "confidence": 0.0}
    try:
        with open(asr_path, "r", encoding="utf-8") as f:
            asr = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"transcript": "", "confidence": 0.0}

    transcript = ""
    confidence = 0.0

    if isinstance(asr, dict):
        transcript = asr.get("text", asr.get("transcript", ""))
        confidence = float(asr.get("confidence", asr.get("conf", 0.0)))
    elif isinstance(asr, str):
        transcript = asr

    if isinstance(transcript, list):
        transcript = " ".join(str(s) for s in transcript)

    return {"transcript": transcript.strip(), "confidence": confidence}


def parse_merge_anno(anno_path: Path) -> dict:
    if not anno_path.exists():
        return {}
    try:
        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}

    is_talking = anno.get("is_talking", 0)
    speaker_count = anno.get("clip_speaker_num", 1)
    start = anno.get("start_seconds", anno.get("start", 0))
    end = anno.get("end_seconds", start + anno.get("duration", 0))
    duration = end - start

    talking_tag = "VISUAL_TALKING" if is_talking else "VISUAL_NOT_TALKING"

    if speaker_count <= 1:
        speaker_tag = "VISUAL_SPEAKER_1"
    elif speaker_count == 2:
        speaker_tag = "VISUAL_SPEAKER_2"
    else:
        speaker_tag = "VISUAL_SPEAKER_MULTI"

    return {
        "video_name": anno.get("video_name", ""),
        "duration": round(duration, 3),
        "start": start,
        "end": end,
        "bbox": anno.get("bbox", []),
        "talking": talking_tag,
        "speakers": speaker_tag,
        "is_talking_raw": is_talking,
        "speaker_count_raw": speaker_count,
    }


def process_clip(clip_name: str, annotations_dir: Path) -> dict | None:
    # Support both HF layout (merged_anno, raw_labels/asr) and flat layout (merge_anno, asr)
    merge_path = annotations_dir / "merged_anno" / f"{clip_name}.json"
    if not merge_path.exists():
        merge_path = annotations_dir / "merge_anno" / f"{clip_name}.json"
    merge = parse_merge_anno(merge_path)
    if not merge:
        return None

    asr_name = "_".join(clip_name.split("_")[:3])
    asr_path = annotations_dir / "raw_labels" / "asr" / f"{asr_name}.json"
    if not asr_path.exists():
        asr_path = annotations_dir / "asr" / f"{asr_name}.json"
    asr = parse_asr(asr_path)

    anno_path = annotations_dir / "raw_labels" / "anno" / f"{clip_name}.json"
    if not anno_path.exists():
        anno_path = annotations_dir / "anno" / f"{clip_name}.json"
    visual = parse_mllm_caption(anno_path)

    l_score_path = annotations_dir / "raw_labels" / "l_score" / f"{clip_name}.json"
    if not l_score_path.exists():
        l_score_path = annotations_dir / "l_score" / f"{clip_name}.json"
    noise_level = estimate_noise_level(l_score_path)

    return {
        "clip_name": clip_name,
        "video_name": merge.get("video_name", ""),
        "duration": merge.get("duration", 0),
        "start": merge.get("start", 0),
        "end": merge.get("end", 0),
        "transcript": asr.get("transcript", ""),
        "asr_confidence": asr.get("confidence", 0),
        # Visual metadata tags
        "tag_VISUAL_TALKING": merge.get("talking", "VISUAL_NOT_TALKING"),
        "tag_VISUAL_SPEAKERS": merge.get("speakers", "VISUAL_SPEAKER_1"),
        "tag_VISUAL_FACING": visual.get("facing", "VISUAL_FACING_FRONT"),
        "tag_VISUAL_BODY": visual.get("body", "VISUAL_HALFBODY"),
        "tag_VISUAL_ACTIVITY": visual.get("activity", "VISUAL_PASSIVE"),
        "tag_VISUAL_EXPRESSION": visual.get("expression", "VISUAL_EXPR_NEUTRAL"),
        "tag_VISUAL_MOVEMENT": visual.get("movement", "VISUAL_STILL"),
        # Scene tags
        "tag_SCENE_TYPE": visual.get("scene_type", "SCENE_INDOOR"),
        "tag_NOISE_LEVEL": noise_level,
        # Inline events
        "inline_events": visual.get("inline_events", []),
        # Speaker count for scene classifier
        "speaker_count_raw": merge.get("speaker_count_raw", 1),
    }


def main():
    args = parse_args()
    annotations_dir = Path(args.annotations_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merge_dir = annotations_dir / "merged_anno"
    if not merge_dir.exists():
        merge_dir = annotations_dir / "merge_anno"
    if not merge_dir.exists():
        logger.error("Neither merged_anno/ nor merge_anno/ found in %s", annotations_dir)
        return

    if args.clips_dir:
        clips_path = Path(args.clips_dir)
        clip_names = sorted([f.stem for f in clips_path.iterdir() if f.suffix in (".mp4", ".wav")])
        logger.info("Deriving clip names from %s (%d files)", args.clips_dir, len(clip_names))
    else:
        logger.info("Globbing %s for clip names (may be slow on NFS)...", merge_dir)
        clip_names = sorted([f.stem for f in merge_dir.glob("*.json")])
    if args.limit > 0:
        clip_names = clip_names[:args.limit]

    logger.info("Parsing annotations for %d clips", len(clip_names))

    completed = 0
    written = 0
    t_start = time.time()

    # Collect tag distribution stats
    tag_stats = {}

    with open(output_path, "w", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_clip, name, annotations_dir): name for name in clip_names}
            for future in as_completed(futures):
                result = future.result()
                completed += 1

                if result is not None:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    written += 1

                    for key, val in result.items():
                        if key.startswith("tag_"):
                            tag_stats.setdefault(key, {})
                            tag_stats[key][val] = tag_stats[key].get(val, 0) + 1

                if completed % 50000 == 0 or completed == len(clip_names):
                    elapsed = time.time() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        "[%d/%d] written=%d, %.1f/s",
                        completed, len(clip_names), written, rate,
                    )

    logger.info("Done: %d parsed, %d written to %s", completed, written, output_path)

    # Log tag distributions
    for tag_name, dist in sorted(tag_stats.items()):
        logger.info("  %s: %s", tag_name,
                     ", ".join(f"{k}={v}" for k, v in sorted(dist.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
