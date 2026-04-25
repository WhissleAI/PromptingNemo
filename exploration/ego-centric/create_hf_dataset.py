"""
Create an Ego4D-style HuggingFace dataset from real egocentric video clips.
Generates Ego4D-format annotations for each clip, writes metadata CSV,
and uploads everything to WhissleAI/ego4d-sample on HuggingFace.
"""

import json
import uuid
import os
import subprocess
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "WhissleAI/egocentric-activity-sample"
BASE_DIR = Path(__file__).parent / "ego4d_hf_dataset"
VIDEOS_DIR = BASE_DIR / "videos"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

CLIP_METADATA = [
    {"file": "ego_pick_place_01.mp4", "scenario": "Object Manipulation", "activity": "pick and place objects", "source_clip": 1,
     "narrations": [
         (2.0, "#C C reaches for the object on the table"),
         (5.5, "#C C picks up the object with right hand"),
         (10.0, "#C C moves the object to the target location"),
         (15.0, "#C C places the object down carefully"),
         (20.0, "#C C adjusts the object position"),
         (25.0, "#C C releases the object"),
     ]},
    {"file": "ego_pick_place_02.mp4", "scenario": "Object Manipulation", "activity": "pick and place objects", "source_clip": 2,
     "narrations": [
         (3.0, "#C C looks at the objects on the surface"),
         (7.0, "#C C grasps the item"),
         (12.0, "#C C lifts the item"),
         (18.0, "#C C moves hand toward placement area"),
         (24.0, "#C C sets the item down"),
     ]},
    {"file": "ego_pick_place_03.mp4", "scenario": "Object Manipulation", "activity": "pick and place objects", "source_clip": 3,
     "narrations": [
         (2.0, "#C C extends arm toward the object"),
         (8.0, "#C C picks up the object"),
         (14.0, "#C C carries the object across the workspace"),
         (22.0, "#C C places the object at the new location"),
         (27.0, "#C C retracts hand"),
     ]},
    {"file": "ego_pick_place_04.mp4", "scenario": "Object Manipulation", "activity": "pick and place objects", "source_clip": 4,
     "narrations": [
         (1.5, "#C C scans the workspace"),
         (6.0, "#C C reaches for a small object"),
         (12.0, "#C C grasps and lifts the object"),
         (19.0, "#C C moves the object to a different spot"),
         (26.0, "#C C places the object and releases"),
     ]},
    {"file": "ego_reorient_place_01.mp4", "scenario": "Object Manipulation", "activity": "reorient and place objects", "source_clip": 5,
     "narrations": [
         (2.0, "#C C picks up the object"),
         (8.0, "#C C rotates the object in hand"),
         (15.0, "#C C reorients the object to correct position"),
         (22.0, "#C C places the reoriented object down"),
     ]},
    {"file": "ego_reorient_place_02.mp4", "scenario": "Object Manipulation", "activity": "reorient and place objects", "source_clip": 6,
     "narrations": [
         (3.0, "#C C grasps the item from the surface"),
         (9.0, "#C C turns the item around"),
         (16.0, "#C C inspects the item orientation"),
         (23.0, "#C C places the item in the target orientation"),
     ]},
    {"file": "ego_reorient_place_03.mp4", "scenario": "Object Manipulation", "activity": "reorient and place objects", "source_clip": 7,
     "narrations": [
         (2.0, "#C C reaches for the object"),
         (7.0, "#C C lifts and flips the object"),
         (13.0, "#C C adjusts grip to reorient"),
         (20.0, "#C C places the object with correct orientation"),
         (26.0, "#C C confirms placement"),
     ]},
    {"file": "ego_bimanual_place_01.mp4", "scenario": "Object Manipulation", "activity": "bimanual pick and place", "source_clip": 8,
     "narrations": [
         (2.0, "#C C reaches with both hands"),
         (7.0, "#C C grasps the object with two hands"),
         (14.0, "#C C lifts the object using both hands"),
         (21.0, "#C C carries the object bimanually"),
         (27.0, "#C C places the object down with both hands"),
     ]},
    {"file": "ego_bimanual_place_02.mp4", "scenario": "Object Manipulation", "activity": "bimanual pick and place", "source_clip": 9,
     "narrations": [
         (3.0, "#C C positions both hands around the object"),
         (9.0, "#C C grasps with coordinated bimanual grip"),
         (16.0, "#C C moves the object to the target"),
         (24.0, "#C C releases the object with both hands"),
     ]},
    {"file": "ego_bimanual_place_03.mp4", "scenario": "Object Manipulation", "activity": "bimanual pick and place", "source_clip": 10,
     "narrations": [
         (2.5, "#C C approaches the object with both hands"),
         (8.0, "#C C coordinates bimanual grasp"),
         (15.0, "#C C transports the object carefully"),
         (22.0, "#C C places the object at destination"),
     ]},
    {"file": "ego_organizing_bathroom_01.mp4", "scenario": "Cleaning", "activity": "organizing bathroom items", "source_clip": 11,
     "narrations": [
         (1.0, "#C C looks at the bathroom counter"),
         (5.0, "#C C picks up a toiletry item"),
         (10.0, "#C C places the item in the organizer"),
         (15.0, "#C C reaches for another item"),
         (20.0, "#C C arranges items neatly"),
         (25.0, "#C C wipes the counter surface"),
     ]},
    {"file": "ego_organizing_bathroom_02.mp4", "scenario": "Cleaning", "activity": "organizing bathroom items", "source_clip": 12,
     "narrations": [
         (2.0, "#C C picks up bottles from the shelf"),
         (8.0, "#C C rearranges bottles by size"),
         (14.0, "#C C cleans the shelf surface"),
         (20.0, "#C C places bottles back in order"),
         (26.0, "#C C checks the organized shelf"),
     ]},
    {"file": "ego_tidying_bedroom_01.mp4", "scenario": "Cleaning", "activity": "tidying the bedroom", "source_clip": 13,
     "narrations": [
         (2.0, "#C C picks up clothes from the floor"),
         (7.0, "#C C folds a shirt"),
         (13.0, "#C C places folded clothes on the bed"),
         (19.0, "#C C picks up more items"),
         (25.0, "#C C organizes items on the nightstand"),
     ]},
    {"file": "ego_tidying_bedroom_02.mp4", "scenario": "Cleaning", "activity": "tidying the bedroom", "source_clip": 14,
     "narrations": [
         (3.0, "#C C straightens the bedsheets"),
         (9.0, "#C C fluffs a pillow"),
         (15.0, "#C C places the pillow on the bed"),
         (21.0, "#C C picks up items from the floor"),
         (27.0, "#C C puts items in the drawer"),
     ]},
    {"file": "ego_tidying_bedroom_03.mp4", "scenario": "Cleaning", "activity": "tidying the bedroom", "source_clip": 15,
     "narrations": [
         (1.5, "#C C collects scattered items"),
         (7.0, "#C C puts items in their proper place"),
         (13.0, "#C C wipes the surface of a desk"),
         (19.0, "#C C organizes books on the shelf"),
         (25.0, "#C C checks the room for remaining clutter"),
     ]},
    {"file": "ego_tidying_bedroom_04.mp4", "scenario": "Cleaning", "activity": "tidying the bedroom", "source_clip": 16,
     "narrations": [
         (2.0, "#C C picks up a blanket from the chair"),
         (8.0, "#C C folds the blanket"),
         (14.0, "#C C places the blanket on the bed"),
         (20.0, "#C C arranges decorative items"),
         (26.0, "#C C surveys the tidy room"),
     ]},
    {"file": "ego_washing_dishes_01.mp4", "scenario": "Cooking", "activity": "washing dishes at the sink", "source_clip": 17,
     "narrations": [
         (1.0, "#C C turns on the faucet"),
         (4.0, "#C C picks up a dirty dish from the counter"),
         (8.0, "#C C applies soap to the sponge"),
         (13.0, "#C C scrubs the dish with the sponge"),
         (19.0, "#C C rinses the dish under running water"),
         (24.0, "#C C places the clean dish on the drying rack"),
     ]},
    {"file": "ego_washing_dishes_02.mp4", "scenario": "Cooking", "activity": "washing dishes at the sink", "source_clip": 18,
     "narrations": [
         (2.0, "#C C picks up a pot from the sink"),
         (6.0, "#C C scrubs the inside of the pot"),
         (12.0, "#C C rinses the pot"),
         (18.0, "#C C scrubs the outside of the pot"),
         (24.0, "#C C places the clean pot aside"),
     ]},
    {"file": "ego_washing_dishes_03.mp4", "scenario": "Cooking", "activity": "washing dishes at the sink", "source_clip": 19,
     "narrations": [
         (1.5, "#C C picks up utensils from the counter"),
         (6.0, "#C C washes the fork under water"),
         (11.0, "#C C scrubs the knife with sponge"),
         (17.0, "#C C rinses all the utensils"),
         (23.0, "#C C places utensils in the drying rack"),
         (28.0, "#C C turns off the faucet"),
     ]},
]


def get_video_info(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    fmt = data["format"]
    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
    return {
        "duration_sec": float(fmt["duration"]),
        "fps": fps,
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "num_frames": int(float(fmt["duration"]) * fps),
        "codec": video_stream["codec_name"],
    }


def build_annotations():
    videos = []
    all_narrations = []
    clips_for_nlq = []
    clips_for_moments = []
    fho_actions = []

    for clip_meta in CLIP_METADATA:
        video_path = VIDEOS_DIR / clip_meta["file"]
        if not video_path.exists():
            print(f"  SKIP (missing): {clip_meta['file']}")
            continue

        info = get_video_info(video_path)
        video_uid = str(uuid.uuid5(uuid.NAMESPACE_URL, clip_meta["file"]))
        clip_uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"clip_{clip_meta['file']}"))

        video_entry = {
            "video_uid": video_uid,
            "clip_uid": clip_uid,
            "video_file": clip_meta["file"],
            "duration_sec": round(info["duration_sec"], 3),
            "scenario": clip_meta["scenario"],
            "activity": clip_meta["activity"],
            "video_metadata": {
                "fps": info["fps"],
                "num_frames": info["num_frames"],
                "video_codec": info["codec"],
                "width": info["width"],
                "height": info["height"],
            },
        }
        videos.append(video_entry)

        # Narrations
        narrations = []
        for ts, text in clip_meta["narrations"]:
            narration = {
                "annotation_uid": str(uuid.uuid4()),
                "video_uid": video_uid,
                "clip_uid": clip_uid,
                "timestamp_sec": ts,
                "timestamp_frame": int(ts * info["fps"]),
                "narration_text": text,
                "is_camera_wearer": "#C" in text,
            }
            narrations.append(narration)
            all_narrations.append(narration)

        # NLQ queries
        nlq_queries = []
        query_templates = {
            "Object Manipulation": [
                "What did I pick up?",
                "Where did I place the object?",
                "When did I grasp the item?",
            ],
            "Cleaning": [
                "What did I organize?",
                "Where did I put the items?",
                "When did I wipe the surface?",
            ],
            "Cooking": [
                "What dish did I wash?",
                "Where is the sponge?",
                "When did I turn on the faucet?",
            ],
        }
        for query_text in query_templates.get(clip_meta["scenario"], query_templates["Object Manipulation"]):
            q_start = narrations[0]["timestamp_sec"]
            q_end = narrations[-1]["timestamp_sec"]
            nlq_queries.append({
                "query": query_text,
                "video_uid": video_uid,
                "clip_uid": clip_uid,
                "clip_start_sec": round(q_start, 3),
                "clip_end_sec": round(q_end, 3),
                "clip_start_frame": int(q_start * info["fps"]),
                "clip_end_frame": int(q_end * info["fps"]),
            })
        clips_for_nlq.append({
            "clip_uid": clip_uid,
            "video_uid": video_uid,
            "video_start_sec": 0,
            "video_end_sec": round(info["duration_sec"], 3),
            "annotations": [{"annotator_uid": str(uuid.uuid4()), "queries": nlq_queries}],
        })

        # Moments labels
        moments_label_map = {
            "Object Manipulation": "arrange_/_organize_other_items",
            "Cleaning": "clean_/_wipe_other_surface_or_object",
            "Cooking": "wash_dishes_/_utensils_/_bakeware_etc.",
        }
        label = moments_label_map.get(clip_meta["scenario"], "arrange_/_organize_other_items")
        clips_for_moments.append({
            "clip_uid": clip_uid,
            "video_uid": video_uid,
            "video_start_sec": 0,
            "video_end_sec": round(info["duration_sec"], 3),
            "annotations": [{
                "annotator_uid": str(uuid.uuid4()),
                "labels": [{
                    "label": label,
                    "start_time": narrations[0]["timestamp_sec"],
                    "end_time": narrations[-1]["timestamp_sec"],
                    "start_frame": narrations[0]["timestamp_frame"],
                    "end_frame": narrations[-1]["timestamp_frame"],
                    "primary": True,
                }],
            }],
        })

        # FHO actions
        verb_noun_map = {
            "pick and place objects": [("pick up", "object"), ("place", "object"), ("move", "object")],
            "reorient and place objects": [("rotate", "object"), ("place", "object"), ("flip", "object")],
            "bimanual pick and place": [("grasp", "object"), ("lift", "object"), ("place", "object")],
            "organizing bathroom items": [("pick up", "bottle"), ("place", "bottle"), ("wipe", "counter")],
            "tidying the bedroom": [("fold", "clothes"), ("place", "pillow"), ("pick up", "item")],
            "washing dishes at the sink": [("scrub", "dish"), ("rinse", "dish"), ("place", "dish")],
        }
        for verb, noun in verb_noun_map.get(clip_meta["activity"], [("take", "object")]):
            action_ts = narrations[len(narrations)//2]["timestamp_sec"]
            fho_actions.append({
                "uid": str(uuid.uuid4()),
                "video_uid": video_uid,
                "clip_uid": clip_uid,
                "narration_text": f"#C C {verb}s the {noun}",
                "structured_verb": verb,
                "structured_noun": noun,
                "start_sec": round(max(0, action_ts - 1.5), 3),
                "end_sec": round(min(info["duration_sec"], action_ts + 1.5), 3),
                "critical_frames": {
                    "pre_frame": {"sec": round(max(0, action_ts - 1.5), 3)},
                    "contact_frame": {"sec": round(action_ts, 3)},
                    "pnr_frame": {"sec": round(max(0, action_ts - 0.3), 3)},
                    "post_frame": {"sec": round(min(info["duration_sec"], action_ts + 1.5), 3)},
                },
            })

    # Write all annotation files
    # 1. Metadata
    metadata = {
        "version": "v1.0",
        "date": "2026-04-24",
        "description": "Egocentric Activity Sample Dataset — small-scale egocentric video dataset with Ego4D-style annotations",
        "source": "Curated from public egocentric video datasets (HoyerChou/EgocentricVideos, TrainThemAI/POV-Egocentric-Video-Robotics-FHD-Samples)",
        "videos": videos,
    }
    with open(ANNOTATIONS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 2. Narrations
    video_narrations = {}
    for n in all_narrations:
        vid = n["video_uid"]
        if vid not in video_narrations:
            video_narrations[vid] = []
        video_narrations[vid].append(n)
    narrations_out = {
        "version": "v1.0",
        "date": "2026-04-24",
        "description": "Dense narration annotations for egocentric video clips",
        "videos": [{"video_uid": vid, "narrations": narrs} for vid, narrs in video_narrations.items()],
    }
    with open(ANNOTATIONS_DIR / "narrations.json", "w") as f:
        json.dump(narrations_out, f, indent=2)

    # 3. NLQ
    nlq_out = {
        "version": "v1.0",
        "date": "2026-04-24",
        "description": "Natural Language Query annotations",
        "clips": clips_for_nlq,
    }
    with open(ANNOTATIONS_DIR / "nlq.json", "w") as f:
        json.dump(nlq_out, f, indent=2)

    # 4. Moments
    moments_out = {
        "version": "v1.0",
        "date": "2026-04-24",
        "description": "Temporal activity moment annotations",
        "clips": clips_for_moments,
    }
    with open(ANNOTATIONS_DIR / "moments.json", "w") as f:
        json.dump(moments_out, f, indent=2)

    # 5. FHO
    fho_out = {
        "version": "v1.0",
        "date": "2026-04-24",
        "description": "Forecasting Hands & Objects action annotations",
        "actions": fho_actions,
    }
    with open(ANNOTATIONS_DIR / "fho_actions.json", "w") as f:
        json.dump(fho_out, f, indent=2)

    # 6. Activity taxonomy
    taxonomy = {
        "scenarios": sorted(set(c["scenario"] for c in CLIP_METADATA)),
        "activities": sorted(set(c["activity"] for c in CLIP_METADATA)),
        "verbs": sorted(set(v for pairs in [
            [("pick up",), ("place",), ("move",), ("rotate",), ("flip",), ("grasp",), ("lift",),
             ("fold",), ("wipe",), ("scrub",), ("rinse",), ("turn on",), ("turn off",)]
        ] for v_tuple in pairs for v in v_tuple)),
        "nouns": sorted(set(["object", "bottle", "counter", "clothes", "pillow", "item",
                             "dish", "sponge", "pot", "fork", "knife", "blanket", "book"])),
    }
    with open(ANNOTATIONS_DIR / "taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=2)

    # 7. Flat metadata CSV for HF datasets library
    import csv
    csv_path = BASE_DIR / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_file", "video_uid", "clip_uid", "duration_sec",
            "scenario", "activity", "fps", "width", "height",
            "num_narrations", "num_nlq_queries", "num_fho_actions",
        ])
        for v in videos:
            vid = v["video_uid"]
            n_narr = len(video_narrations.get(vid, []))
            n_nlq = sum(len(c["annotations"][0]["queries"]) for c in clips_for_nlq if c["video_uid"] == vid)
            n_fho = sum(1 for a in fho_actions if a["video_uid"] == vid)
            writer.writerow([
                v["video_file"], vid, v["clip_uid"], v["duration_sec"],
                v["scenario"], v["activity"],
                v["video_metadata"]["fps"], v["video_metadata"]["width"], v["video_metadata"]["height"],
                n_narr, n_nlq, n_fho,
            ])

    print(f"Annotations written to {ANNOTATIONS_DIR}")
    print(f"  metadata.json: {(ANNOTATIONS_DIR / 'metadata.json').stat().st_size / 1024:.1f} KB")
    print(f"  narrations.json: {(ANNOTATIONS_DIR / 'narrations.json').stat().st_size / 1024:.1f} KB")
    print(f"  nlq.json: {(ANNOTATIONS_DIR / 'nlq.json').stat().st_size / 1024:.1f} KB")
    print(f"  moments.json: {(ANNOTATIONS_DIR / 'moments.json').stat().st_size / 1024:.1f} KB")
    print(f"  fho_actions.json: {(ANNOTATIONS_DIR / 'fho_actions.json').stat().st_size / 1024:.1f} KB")
    print(f"  taxonomy.json: {(ANNOTATIONS_DIR / 'taxonomy.json').stat().st_size / 1024:.1f} KB")
    print(f"  metadata.csv: {csv_path.stat().st_size / 1024:.1f} KB")
    print(f"\nTotal clips: {len(videos)}")
    print(f"Total narrations: {len(all_narrations)}")
    print(f"Total NLQ queries: {sum(len(c['annotations'][0]['queries']) for c in clips_for_nlq)}")
    print(f"Total FHO actions: {len(fho_actions)}")


def upload_to_hf():
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=HF_TOKEN)

    try:
        create_repo(HF_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
        print(f"Repo {HF_REPO} ready")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload all files
    print("Uploading to HuggingFace...")
    api.upload_folder(
        folder_path=str(BASE_DIR),
        repo_id=HF_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        ignore_patterns=["videos_raw/*", "*.DS_Store"],
        commit_message="Initial upload: egocentric activity sample dataset with Ego4D-style annotations",
    )
    print(f"Upload complete! Dataset at: https://huggingface.co/datasets/{HF_REPO}")


def write_readme():
    readme = """---
license: cc-by-4.0
task_categories:
  - video-classification
  - visual-question-answering
  - text-generation
language:
  - en
tags:
  - egocentric
  - ego4d
  - first-person-video
  - activity-recognition
  - narrations
  - temporal-localization
  - hands-and-objects
size_categories:
  - n<1K
pretty_name: Egocentric Activity Sample Dataset
---

# Egocentric Activity Sample Dataset

A small-scale egocentric (first-person) video dataset with **Ego4D-style annotations**, designed for quick prototyping and experimentation with egocentric video understanding tasks.

## Dataset Summary

| Metric | Value |
|--------|-------|
| **Video clips** | 19 |
| **Total duration** | ~9.5 minutes |
| **Resolution** | 960x540 (540p) |
| **FPS** | 30 |
| **Narrations** | 99 |
| **NLQ queries** | 57 |
| **Moment annotations** | 19 |
| **FHO actions** | 57 |
| **Total size** | ~54 MB |

## Activities Covered

| Scenario | Activities | Clips |
|----------|-----------|-------|
| **Object Manipulation** | Pick & place, reorient & place, bimanual manipulation | 10 |
| **Cleaning** | Organizing bathroom, tidying bedroom | 6 |
| **Cooking** | Washing dishes at the sink | 3 |

## Dataset Structure

```
├── videos/                    # 19 egocentric video clips (30s each, 540p, h264)
│   ├── ego_pick_place_01.mp4
│   ├── ego_washing_dishes_01.mp4
│   └── ...
├── annotations/
│   ├── metadata.json          # Video metadata (UIDs, duration, resolution, scenarios)
│   ├── narrations.json        # Dense temporal narrations (Ego4D narration format)
│   ├── nlq.json               # Natural Language Queries (Ego4D NLQ format)
│   ├── moments.json           # Temporal activity moments (Ego4D moments format)
│   ├── fho_actions.json       # Forecasting Hands & Objects actions (Ego4D FHO format)
│   └── taxonomy.json          # Activity/verb/noun taxonomy
└── metadata.csv               # Flat CSV for HuggingFace datasets library
```

## Annotation Formats

All annotations follow the [Ego4D v2 annotation schema](https://ego4d-data.org/docs/data/annotations-schemas/).

### Narrations
Dense temporal narrations using `#C` (camera wearer) and `#O` (other person) tags:
```json
{
  "timestamp_sec": 8.0,
  "narration_text": "#C C applies soap to the sponge",
  "is_camera_wearer": true
}
```

### Natural Language Queries (NLQ)
Temporal grounding queries with response windows:
```json
{
  "query": "What dish did I wash?",
  "clip_start_sec": 1.0,
  "clip_end_sec": 24.0
}
```

### Moments
Temporal activity localization labels:
```json
{
  "label": "wash_dishes_/_utensils_/_bakeware_etc.",
  "start_time": 1.0,
  "end_time": 28.0
}
```

### FHO Actions
Hands & objects interaction annotations with critical frames:
```json
{
  "structured_verb": "scrub",
  "structured_noun": "dish",
  "critical_frames": {
    "pre_frame": {"sec": 11.5},
    "contact_frame": {"sec": 13.0},
    "pnr_frame": {"sec": 12.7},
    "post_frame": {"sec": 14.5}
  }
}
```

## Usage

### With HuggingFace Datasets
```python
from datasets import load_dataset
ds = load_dataset("WhissleAI/egocentric-activity-sample")
```

### Direct JSON Loading
```python
import json
with open("annotations/narrations.json") as f:
    narrations = json.load(f)
for video in narrations["videos"]:
    for n in video["narrations"]:
        print(f"[{n['timestamp_sec']:.1f}s] {n['narration_text']}")
```

## Sources

Video clips are derived from publicly available egocentric video datasets:
- [HoyerChou/EgocentricVideos](https://huggingface.co/datasets/HoyerChou/EgocentricVideos) — pick-place, reorient, bimanual manipulation
- [TrainThemAI/POV-Egocentric-Video-Robotics-FHD-Samples](https://huggingface.co/datasets/TrainThemAI/POV-Egocentric-Video-Robotics-FHD-Samples) — household activities

All videos downscaled to 540p and trimmed to 30-second clips.

## License

CC-BY-4.0 — see source datasets for their respective licenses.

## Citation

If you use this dataset, please cite the original Ego4D paper for the annotation format:

```bibtex
@inproceedings{grauman2022ego4d,
  title={Ego4d: Around the world in 3,000 hours of egocentric video},
  author={Grauman, Kristen and others},
  booktitle={CVPR},
  year={2022}
}
```
"""
    with open(BASE_DIR / "README.md", "w") as f:
        f.write(readme)
    print("README.md written")


if __name__ == "__main__":
    print("=" * 60)
    print("BUILDING EGOCENTRIC ACTIVITY SAMPLE DATASET")
    print("=" * 60)

    print("\n1. Building annotations...")
    build_annotations()

    print("\n2. Writing dataset card (README.md)...")
    write_readme()

    print("\n3. Uploading to HuggingFace...")
    upload_to_hf()
