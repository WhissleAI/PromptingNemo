---
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
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/train-*.parquet"
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
