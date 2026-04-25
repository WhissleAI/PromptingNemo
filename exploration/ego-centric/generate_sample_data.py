"""
Generate realistic sample Ego4D annotations matching official schemas.
Creates sample JSON files for: metadata, narrations, moments, NLQ, FHO, and AV benchmarks.
"""

import json
import random
import uuid
import os
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path(__file__).parent / "ego4d_data" / "v2" / "annotations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "Cooking", "Cleaning", "Gardening", "Shopping", "Exercise",
    "Crafting", "Repair", "Social", "Pet Care", "Laundry"
]

UNIVERSITIES = [
    "bristol", "cmu", "georgiatech", "iiith", "indiana",
    "kaust", "minnesota", "nus", "unict", "utokyo"
]

DEVICES = ["GoPro Hero 8", "GoPro Hero 9", "Aria", "ZED Mini", "Pupil Invisible"]

VERBS = [
    "take", "put", "open", "close", "wash", "cut", "pour", "stir",
    "mix", "peel", "turn on", "turn off", "squeeze", "wipe", "fold",
    "pick up", "place", "move", "shake", "spread"
]

NOUNS = [
    "knife", "pan", "bowl", "cup", "plate", "spoon", "fork", "pot",
    "bottle", "lid", "towel", "sponge", "bag", "box", "cloth",
    "tomato", "onion", "pepper", "egg", "bread", "water", "oil"
]

MOMENT_LABELS = {
    "serve_food_onto_a_plate": 0,
    "converse_/_interact_with_someone": 1,
    "use_phone": 2,
    "clean_/_wipe_a_table_or_kitchen_counter": 3,
    "wash_hands": 12,
    "turn-on_/_light_the_stove_burner": 13,
    "cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat": 24,
    "stir_/_mix_food_while_cooking": 21,
    "fill_a_pot_/_bottle_/_container_with_water": 23,
    "drink_beverage": 36,
    "wash_dishes_/_utensils_/_bakeware_etc.": 47,
    "eat_a_snack": 52,
}


def gen_video_uid():
    return str(uuid.uuid4())


def gen_video(video_uid, duration_sec=None):
    if duration_sec is None:
        duration_sec = random.uniform(300, 3600)
    fps = random.choice([30.0, 29.97, 24.0, 60.0])
    num_frames = int(duration_sec * fps)
    w = random.choice([1920, 1280, 3840])
    h = random.choice([1080, 720, 2160])
    uni = random.choice(UNIVERSITIES)
    return {
        "video_uid": video_uid,
        "duration_sec": round(duration_sec, 3),
        "scenarios": random.sample(SCENARIOS, k=random.randint(1, 3)),
        "video_metadata": {
            "fps": fps,
            "num_frames": num_frames,
            "video_codec": "h264",
            "display_resolution_width": w,
            "display_resolution_height": h,
            "sample_resolution_width": w,
            "sample_resolution_height": h,
            "mp4_duration_sec": round(duration_sec, 3),
            "video_start_sec": 0.0,
            "video_duration_sec": round(duration_sec, 3),
            "audio_start_sec": 0.0,
            "audio_duration_sec": round(duration_sec, 3),
        },
        "split_em": random.choice(["train", "val", "test"]),
        "split_fho": random.choice(["train", "val", "test"]),
        "split_av": random.choice(["train", "val", "test"]),
        "s3_path": f"s3://ego4d-consortium-sharing/ego4d_data/v2/full_scale/{video_uid}.mp4",
        "origin_video_id": f"{uni}_{random.randint(1000, 9999)}",
        "video_source": uni,
        "device": random.choice(DEVICES),
        "physical_setting_name": f"setting_{random.randint(1, 50)}",
        "is_stereo": random.random() < 0.1,
        "has_imu": random.random() < 0.3,
        "has_gaze": random.random() < 0.2,
    }


def gen_narration(video_uid, timestamp_sec, idx):
    is_camera_wearer = random.random() < 0.8
    tag = "#C" if is_camera_wearer else "#O"
    verb = random.choice(VERBS)
    noun = random.choice(NOUNS)
    actor = "C" if is_camera_wearer else "someone"
    text = f"{tag} {actor} {verb}s the {noun}"
    return {
        "annotation_uid": str(uuid.uuid4()),
        "video_uid": video_uid,
        "timestamp_sec": round(timestamp_sec, 3),
        "timestamp_frame": int(timestamp_sec * 30),
        "narration_text": text,
        "is_camera_wearer": is_camera_wearer,
        "structured_verb": verb,
        "structured_noun": noun,
    }


def gen_narrations(videos, num_per_video=20):
    all_narrations = []
    for v in videos:
        dur = v["duration_sec"]
        timestamps = sorted(random.uniform(5, dur - 5) for _ in range(num_per_video))
        for i, ts in enumerate(timestamps):
            all_narrations.append(gen_narration(v["video_uid"], ts, i))
    return all_narrations


def gen_nlq_annotation(video_uid, clip_uid, clip_dur):
    templates = [
        "What did I put on the table?",
        "Where is the knife I was using?",
        "When did I last open the fridge?",
        "What did I pick up from the counter?",
        "Where did I place the bowl?",
        "When did I wash my hands?",
        "What was I cutting?",
        "Where is the towel?",
        "When did I turn on the stove?",
        "What did the other person hand me?",
    ]
    start = round(random.uniform(0, clip_dur * 0.7), 3)
    end = round(start + random.uniform(2, min(30, clip_dur - start)), 3)
    return {
        "clip_uid": clip_uid,
        "video_uid": video_uid,
        "query": random.choice(templates),
        "query_frame": int(end * 30 + random.randint(30, 300)),
        "clip_start_sec": round(start, 3),
        "clip_end_sec": round(end, 3),
        "clip_start_frame": int(start * 30),
        "clip_end_frame": int(end * 30),
    }


def gen_moments_annotation(video_uid, clip_uid, clip_dur):
    label_name, label_id = random.choice(list(MOMENT_LABELS.items()))
    start = round(random.uniform(0, clip_dur * 0.7), 3)
    end = round(start + random.uniform(2, min(60, clip_dur - start)), 3)
    return {
        "start_time": start,
        "end_time": end,
        "start_frame": int(start * 30),
        "end_frame": int(end * 30),
        "label": label_name,
        "label_id": label_id,
        "primary": True,
    }


def gen_fho_action(video_uid, clip_uid, timestamp_sec, fps=30.0):
    verb = random.choice(VERBS)
    noun = random.choice(NOUNS)
    pre_sec = max(0, timestamp_sec - random.uniform(0.5, 2.0))
    post_sec = timestamp_sec + random.uniform(0.5, 2.0)
    pnr_sec = timestamp_sec - random.uniform(0.1, 0.5)
    contact_sec = timestamp_sec

    def make_bbox():
        x = round(random.uniform(0.1, 0.7), 4)
        y = round(random.uniform(0.1, 0.7), 4)
        w = round(random.uniform(0.05, 0.3), 4)
        h = round(random.uniform(0.05, 0.3), 4)
        return {"x": x, "y": y, "width": w, "height": h}

    return {
        "uid": str(uuid.uuid4()),
        "video_uid": video_uid,
        "clip_uid": clip_uid,
        "narration_text": f"#C C {verb}s the {noun}",
        "structured_verb": verb,
        "structured_noun": noun,
        "verb_label": VERBS.index(verb),
        "noun_label": NOUNS.index(noun),
        "is_valid_action": True,
        "start_sec": round(pre_sec, 3),
        "end_sec": round(post_sec, 3),
        "start_frame": int(pre_sec * fps),
        "end_frame": int(post_sec * fps),
        "critical_frames": {
            "pre_frame": {"sec": round(pre_sec, 3), "frame": int(pre_sec * fps)},
            "contact_frame": {"sec": round(contact_sec, 3), "frame": int(contact_sec * fps)},
            "pnr_frame": {"sec": round(pnr_sec, 3), "frame": int(pnr_sec * fps)},
            "post_frame": {"sec": round(post_sec, 3), "frame": int(post_sec * fps)},
        },
        "boxes": {
            "pre_frame": {
                "right_hand": make_bbox(),
                "left_hand": make_bbox(),
                "object_of_change": make_bbox(),
            },
            "contact_frame": {
                "right_hand": make_bbox(),
                "left_hand": make_bbox(),
                "object_of_change": make_bbox(),
            },
        },
    }


def gen_av_clip(video_uid, clip_uid, clip_start, clip_end, fps=30.0):
    num_persons = random.randint(1, 4)
    persons = []
    for i in range(num_persons):
        pid = f"person_{i}"
        is_cw = i == 0
        num_voice_segs = random.randint(0, 5) if not is_cw else random.randint(2, 8)
        voice_segments = []
        t = clip_start + random.uniform(0, 5)
        for _ in range(num_voice_segs):
            seg_start = round(t, 3)
            seg_end = round(t + random.uniform(1, 8), 3)
            voice_segments.append({
                "start_time": seg_start,
                "end_time": min(seg_end, clip_end),
                "start_frame": int(seg_start * fps),
                "end_frame": int(min(seg_end, clip_end) * fps),
                "person": pid,
            })
            t = seg_end + random.uniform(2, 15)
            if t >= clip_end:
                break

        transcriptions = []
        phrases = [
            "Can you pass me that?", "Sure, here you go.",
            "What are we making today?", "Let me check the recipe.",
            "This looks good.", "Be careful with that knife.",
            "Is the oven preheated?", "Almost done.",
        ]
        for seg in voice_segments:
            transcriptions.append({
                "transcription": random.choice(phrases),
                "start_time_sec": seg["start_time"],
                "end_time_sec": seg["end_time"],
                "person_id": pid,
            })

        num_tracks = random.randint(1, 3) if not is_cw else 0
        tracking_paths = []
        for t_idx in range(num_tracks):
            track = []
            for frame in range(int(clip_start * fps), int(clip_end * fps), int(fps)):
                track.append({
                    "x": round(random.uniform(0.1, 0.8), 4),
                    "y": round(random.uniform(0.1, 0.8), 4),
                    "width": round(random.uniform(0.05, 0.2), 4),
                    "height": round(random.uniform(0.05, 0.3), 4),
                    "frame": frame,
                })
            tracking_paths.append({
                "track_id": f"{clip_uid}_track_{t_idx}",
                "track": track[:20],
            })

        persons.append({
            "person_id": pid,
            "camera_wearer": is_cw,
            "tracking_paths": tracking_paths,
            "voice_segments": voice_segments,
            "transcriptions": transcriptions,
        })

    return {
        "clip_uid": clip_uid,
        "video_uid": video_uid,
        "video_start_sec": clip_start,
        "video_end_sec": clip_end,
        "video_start_frame": int(clip_start * fps),
        "video_end_frame": int(clip_end * fps),
        "valid": True,
        "camera_wearer": {"person_id": "person_0", "camera_wearer": True},
        "persons": persons,
    }


def main():
    num_videos = 50
    video_uids = [gen_video_uid() for _ in range(num_videos)]
    videos = [gen_video(uid) for uid in video_uids]

    # 1. Metadata (ego4d.json)
    metadata = {
        "date": "2024-03-15",
        "version": "v2",
        "description": "Ego4D v2 Sample Metadata (synthetic for exploration)",
        "videos": videos,
    }
    with open(OUTPUT_DIR.parent / "ego4d.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 2. Narrations
    narrations_data = gen_narrations(videos, num_per_video=25)
    video_narrations = {}
    for n in narrations_data:
        vid = n["video_uid"]
        if vid not in video_narrations:
            video_narrations[vid] = []
        video_narrations[vid].append(n)
    narrations_output = {
        "version": "v2",
        "date": "2024-03-15",
        "description": "Narration annotations (synthetic sample)",
        "videos": [
            {"video_uid": vid, "narrations": narrs}
            for vid, narrs in video_narrations.items()
        ],
    }
    with open(OUTPUT_DIR / "narration.json", "w") as f:
        json.dump(narrations_output, f, indent=2)

    # 3. NLQ (Natural Language Queries)
    nlq_videos = []
    for v in videos[:30]:
        clips = []
        clip_start = 0
        while clip_start < v["duration_sec"] - 60:
            clip_dur = min(random.uniform(120, 480), v["duration_sec"] - clip_start)
            clip_uid = str(uuid.uuid4())
            queries = [gen_nlq_annotation(v["video_uid"], clip_uid, clip_dur) for _ in range(random.randint(2, 5))]
            clips.append({
                "clip_uid": clip_uid,
                "video_start_sec": round(clip_start, 3),
                "video_end_sec": round(clip_start + clip_dur, 3),
                "annotations": [{"annotator_uid": str(uuid.uuid4()), "queries": queries}],
            })
            clip_start += clip_dur
        nlq_videos.append({"video_uid": v["video_uid"], "split": v["split_em"], "clips": clips})

    for split in ["train", "val", "test"]:
        split_vids = [v for v in nlq_videos if v["split"] == split]
        nlq_output = {
            "version": "v2",
            "date": "2024-03-15",
            "description": f"NLQ {split} annotations (synthetic sample)",
            "videos": split_vids,
        }
        with open(OUTPUT_DIR / f"nlq_{split}.json", "w") as f:
            json.dump(nlq_output, f, indent=2)

    # 4. Moments
    moments_videos = []
    for v in videos[:35]:
        clips = []
        clip_start = 0
        while clip_start < v["duration_sec"] - 60:
            clip_dur = min(random.uniform(120, 600), v["duration_sec"] - clip_start)
            clip_uid = str(uuid.uuid4())
            num_annotations = random.randint(3, 10)
            labels = [gen_moments_annotation(v["video_uid"], clip_uid, clip_dur) for _ in range(num_annotations)]
            clips.append({
                "clip_uid": clip_uid,
                "video_start_sec": round(clip_start, 3),
                "video_end_sec": round(clip_start + clip_dur, 3),
                "video_start_frame": int(clip_start * 30),
                "video_end_frame": int((clip_start + clip_dur) * 30),
                "annotations": [{"annotator_uid": str(uuid.uuid4()), "labels": labels}],
            })
            clip_start += clip_dur
        moments_videos.append({"video_uid": v["video_uid"], "split": v["split_em"], "clips": clips})

    for split in ["train", "val", "test"]:
        split_vids = [v for v in moments_videos if v["split"] == split]
        moments_output = {
            "version": "v2",
            "date": "2024-03-15",
            "description": f"Moments {split} annotations (synthetic sample)",
            "videos": split_vids,
        }
        with open(OUTPUT_DIR / f"moments_{split}.json", "w") as f:
            json.dump(moments_output, f, indent=2)

    # 5. FHO (Forecasting Hands & Objects)
    fho_clips = []
    for v in videos[:25]:
        fps = v["video_metadata"]["fps"]
        dur = v["duration_sec"]
        num_actions = random.randint(10, 30)
        timestamps = sorted(random.uniform(5, dur - 5) for _ in range(num_actions))
        clip_uid = str(uuid.uuid4())
        for ts in timestamps:
            fho_clips.append(gen_fho_action(v["video_uid"], clip_uid, ts, fps))

    fho_output = {
        "version": "v2",
        "date": "2024-03-15",
        "description": "FHO annotations (synthetic sample)",
        "clips": fho_clips,
    }
    with open(OUTPUT_DIR / "fho_main.json", "w") as f:
        json.dump(fho_output, f, indent=2)

    # FHO LTA taxonomy
    lta_taxonomy = {"verbs": VERBS, "nouns": NOUNS}
    with open(OUTPUT_DIR / "fho_lta_taxonomy.json", "w") as f:
        json.dump(lta_taxonomy, f, indent=2)

    # 6. AV (Audio-Visual Diarization)
    av_videos = []
    for v in videos[:15]:
        clips = []
        clip_start = 0
        fps = v["video_metadata"]["fps"]
        while clip_start < v["duration_sec"] - 30:
            clip_dur = min(random.uniform(30, 120), v["duration_sec"] - clip_start)
            clip_uid = str(uuid.uuid4())
            clips.append(gen_av_clip(v["video_uid"], clip_uid, clip_start, clip_start + clip_dur, fps))
            clip_start += clip_dur
        av_videos.append({"video_uid": v["video_uid"], "split": v["split_av"], "clips": clips})

    for split in ["train", "val", "test"]:
        split_vids = [v for v in av_videos if v["split"] == split]
        av_output = {
            "version": "v2",
            "date": "2024-03-15",
            "description": f"AV {split} annotations (synthetic sample)",
            "videos": split_vids,
        }
        with open(OUTPUT_DIR / f"av_{split}.json", "w") as f:
            json.dump(av_output, f, indent=2)

    # Summary
    files = list(OUTPUT_DIR.glob("*.json")) + [OUTPUT_DIR.parent / "ego4d.json"]
    print(f"Generated {len(files)} annotation files in {OUTPUT_DIR}")
    for f in sorted(files):
        size = f.stat().st_size
        print(f"  {f.name}: {size / 1024:.1f} KB")

    total_narrations = sum(len(v["narrations"]) for v in narrations_output["videos"])
    total_nlq = sum(len(c["annotations"][0]["queries"]) for v in nlq_videos for c in v["clips"])
    total_moments = sum(len(c["annotations"][0]["labels"]) for v in moments_videos for c in v["clips"])
    print(f"\nDataset stats:")
    print(f"  Videos: {num_videos}")
    print(f"  Narrations: {total_narrations}")
    print(f"  NLQ queries: {total_nlq}")
    print(f"  Moments labels: {total_moments}")
    print(f"  FHO actions: {len(fho_clips)}")
    print(f"  AV clips: {sum(len(v['clips']) for v in av_videos)}")


if __name__ == "__main__":
    main()
