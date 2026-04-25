"""
Explore Ego4D benchmark annotations: NLQ, Moments, FHO, and AV.
"""

import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "ego4d_data" / "v2" / "annotations"


def explore_nlq():
    print(f"\n{'='*60}")
    print(f"NATURAL LANGUAGE QUERIES (NLQ)")
    print(f"{'='*60}")

    all_queries = []
    for split in ["train", "val", "test"]:
        fp = DATA_DIR / f"nlq_{split}.json"
        if not fp.exists():
            continue
        with open(fp) as f:
            data = json.load(f)
        videos = data["videos"]
        queries = []
        for v in videos:
            for c in v["clips"]:
                for ann in c["annotations"]:
                    queries.extend(ann["queries"])
        print(f"\n  {split}: {len(videos)} videos, {len(queries)} queries")
        all_queries.extend(queries)

    if not all_queries:
        return

    # Query template distribution
    query_counts = Counter(q["query"] for q in all_queries)
    print(f"\n--- Query Template Distribution ---")
    for query, count in query_counts.most_common(10):
        print(f"  {query:45s} {count:4d}")

    # Response window duration
    durations = [q["clip_end_sec"] - q["clip_start_sec"] for q in all_queries]
    print(f"\n--- Response Window Duration ---")
    print(f"  Mean: {sum(durations)/len(durations):.1f}s")
    print(f"  Min:  {min(durations):.1f}s")
    print(f"  Max:  {max(durations):.1f}s")

    # Distribution of durations
    bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, float('inf'))]
    print(f"\n--- Duration Distribution ---")
    for lo, hi in bins:
        count = sum(1 for d in durations if lo <= d < hi)
        bar = "█" * max(1, count // 3)
        label = f"{lo}-{hi}s" if hi != float('inf') else f"{lo}s+"
        print(f"  {label:10s} {count:4d} {bar}")


def explore_moments():
    print(f"\n{'='*60}")
    print(f"MOMENTS (Temporal Activity Localization)")
    print(f"{'='*60}")

    all_labels = []
    for split in ["train", "val", "test"]:
        fp = DATA_DIR / f"moments_{split}.json"
        if not fp.exists():
            continue
        with open(fp) as f:
            data = json.load(f)
        videos = data["videos"]
        labels = []
        for v in videos:
            for c in v["clips"]:
                for ann in c["annotations"]:
                    labels.extend(ann["labels"])
        print(f"\n  {split}: {len(videos)} videos, {len(labels)} moment labels")
        all_labels.extend(labels)

    if not all_labels:
        return

    # Label distribution
    label_counts = Counter(l["label"] for l in all_labels)
    print(f"\n--- Activity Label Distribution ---")
    for label, count in label_counts.most_common():
        short_label = label[:45]
        bar = "█" * max(1, count // 3)
        print(f"  {short_label:47s} {count:4d} {bar}")

    # Moment duration
    durations = [l["end_time"] - l["start_time"] for l in all_labels]
    print(f"\n--- Moment Duration ---")
    print(f"  Mean: {sum(durations)/len(durations):.1f}s")
    print(f"  Min:  {min(durations):.1f}s")
    print(f"  Max:  {max(durations):.1f}s")


def explore_fho():
    print(f"\n{'='*60}")
    print(f"FORECASTING HANDS & OBJECTS (FHO)")
    print(f"{'='*60}")

    fp = DATA_DIR / "fho_main.json"
    if not fp.exists():
        print("  fho_main.json not found")
        return
    with open(fp) as f:
        data = json.load(f)

    clips = data["clips"]
    print(f"  Total FHO actions: {len(clips)}")

    # Unique videos
    video_uids = set(c["video_uid"] for c in clips)
    print(f"  Unique videos: {len(video_uids)}")

    # Verb distribution
    verb_counts = Counter(c["structured_verb"] for c in clips)
    print(f"\n--- Top 15 Verbs ---")
    for verb, count in verb_counts.most_common(15):
        bar = "█" * max(1, count // 3)
        print(f"  {verb:15s} {count:4d} {bar}")

    # Noun distribution
    noun_counts = Counter(c["structured_noun"] for c in clips)
    print(f"\n--- Top 15 Nouns ---")
    for noun, count in noun_counts.most_common(15):
        bar = "█" * max(1, count // 3)
        print(f"  {noun:15s} {count:4d} {bar}")

    # Action duration (pre to post frame)
    durations = [c["end_sec"] - c["start_sec"] for c in clips]
    print(f"\n--- Action Duration (pre→post) ---")
    print(f"  Mean: {sum(durations)/len(durations):.2f}s")
    print(f"  Min:  {min(durations):.2f}s")
    print(f"  Max:  {max(durations):.2f}s")

    # Critical frame intervals
    pnr_to_contact = []
    for c in clips:
        cf = c["critical_frames"]
        interval = abs(cf["contact_frame"]["sec"] - cf["pnr_frame"]["sec"])
        pnr_to_contact.append(interval)
    print(f"\n--- PNR to Contact Interval ---")
    print(f"  Mean: {sum(pnr_to_contact)/len(pnr_to_contact):.3f}s")

    # Taxonomy
    tax_fp = DATA_DIR / "fho_lta_taxonomy.json"
    if tax_fp.exists():
        with open(tax_fp) as f:
            tax = json.load(f)
        print(f"\n--- LTA Taxonomy ---")
        print(f"  Verbs: {len(tax['verbs'])} ({', '.join(tax['verbs'][:8])}...)")
        print(f"  Nouns: {len(tax['nouns'])} ({', '.join(tax['nouns'][:8])}...)")


def explore_av():
    print(f"\n{'='*60}")
    print(f"AUDIO-VISUAL DIARIZATION (AV)")
    print(f"{'='*60}")

    all_clips = []
    for split in ["train", "val", "test"]:
        fp = DATA_DIR / f"av_{split}.json"
        if not fp.exists():
            continue
        with open(fp) as f:
            data = json.load(f)
        videos = data["videos"]
        clips = [c for v in videos for c in v["clips"]]
        print(f"\n  {split}: {len(videos)} videos, {len(clips)} clips")
        all_clips.extend(clips)

    if not all_clips:
        return

    # Persons per clip
    persons_per_clip = [len(c["persons"]) for c in all_clips]
    print(f"\n--- Persons Per Clip ---")
    print(f"  Mean: {sum(persons_per_clip)/len(persons_per_clip):.1f}")
    print(f"  Min:  {min(persons_per_clip)}")
    print(f"  Max:  {max(persons_per_clip)}")

    # Voice segment stats
    all_voice_segs = []
    for c in all_clips:
        for p in c["persons"]:
            all_voice_segs.extend(p.get("voice_segments", []))
    print(f"\n--- Voice Segments ---")
    print(f"  Total: {len(all_voice_segs)}")
    if all_voice_segs:
        seg_durations = [s["end_time"] - s["start_time"] for s in all_voice_segs]
        print(f"  Mean duration: {sum(seg_durations)/len(seg_durations):.1f}s")

    # Transcription stats
    all_transcriptions = []
    for c in all_clips:
        for p in c["persons"]:
            all_transcriptions.extend(p.get("transcriptions", []))
    print(f"\n--- Transcriptions ---")
    print(f"  Total: {len(all_transcriptions)}")

    # Sample transcriptions
    print(f"\n--- Sample Transcriptions ---")
    for t in all_transcriptions[:8]:
        print(f"  [{t['start_time_sec']:7.1f}s] [{t['person_id']}] {t['transcription']}")

    # Face tracking stats
    all_tracks = []
    for c in all_clips:
        for p in c["persons"]:
            all_tracks.extend(p.get("tracking_paths", []))
    print(f"\n--- Face Tracking ---")
    print(f"  Total tracks: {len(all_tracks)}")
    if all_tracks:
        track_lengths = [len(t["track"]) for t in all_tracks]
        print(f"  Mean track length: {sum(track_lengths)/len(track_lengths):.0f} frames")


def main():
    explore_nlq()
    explore_moments()
    explore_fho()
    explore_av()


if __name__ == "__main__":
    main()
