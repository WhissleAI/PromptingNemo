"""
Explore Ego4D metadata: video statistics, scenario distribution, device types, university sources.
"""

import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "ego4d_data" / "v2"


def main():
    with open(DATA_DIR / "ego4d.json") as f:
        meta = json.load(f)

    videos = meta["videos"]
    print(f"{'='*60}")
    print(f"EGO4D METADATA EXPLORATION")
    print(f"{'='*60}")
    print(f"Version: {meta['version']}")
    print(f"Date: {meta['date']}")
    print(f"Total videos: {len(videos)}")

    # Duration stats
    durations = [v["duration_sec"] for v in videos]
    total_hours = sum(durations) / 3600
    print(f"\n--- Duration Statistics ---")
    print(f"Total duration: {total_hours:.1f} hours")
    print(f"Mean duration: {sum(durations)/len(durations)/60:.1f} min")
    print(f"Min duration: {min(durations)/60:.1f} min")
    print(f"Max duration: {max(durations)/60:.1f} min")

    # Resolution distribution
    resolutions = Counter()
    for v in videos:
        vm = v["video_metadata"]
        resolutions[f"{vm['display_resolution_width']}x{vm['display_resolution_height']}"] += 1
    print(f"\n--- Resolution Distribution ---")
    for res, count in resolutions.most_common():
        print(f"  {res}: {count} videos ({100*count/len(videos):.0f}%)")

    # FPS distribution
    fps_counts = Counter(v["video_metadata"]["fps"] for v in videos)
    print(f"\n--- FPS Distribution ---")
    for fps, count in fps_counts.most_common():
        print(f"  {fps} fps: {count} videos")

    # Scenario distribution
    scenario_counts = Counter()
    for v in videos:
        for s in v["scenarios"]:
            scenario_counts[s] += 1
    print(f"\n--- Scenario Distribution ---")
    for scenario, count in scenario_counts.most_common():
        bar = "█" * (count * 2)
        print(f"  {scenario:20s} {count:3d} {bar}")

    # University/source distribution
    uni_counts = Counter(v["video_source"] for v in videos)
    print(f"\n--- University Source Distribution ---")
    for uni, count in uni_counts.most_common():
        bar = "█" * (count * 2)
        print(f"  {uni:15s} {count:3d} {bar}")

    # Device distribution
    device_counts = Counter(v["device"] for v in videos)
    print(f"\n--- Device Distribution ---")
    for device, count in device_counts.most_common():
        print(f"  {device:25s} {count:3d}")

    # Split distribution
    for split_key in ["split_em", "split_fho", "split_av"]:
        splits = Counter(v[split_key] for v in videos)
        benchmark = split_key.replace("split_", "").upper()
        print(f"\n--- {benchmark} Split Distribution ---")
        for split, count in sorted(splits.items()):
            print(f"  {split:8s} {count:3d} videos ({100*count/len(videos):.0f}%)")

    # Sensor availability
    imu_count = sum(1 for v in videos if v["has_imu"])
    gaze_count = sum(1 for v in videos if v["has_gaze"])
    stereo_count = sum(1 for v in videos if v["is_stereo"])
    print(f"\n--- Sensor Availability ---")
    print(f"  IMU:    {imu_count}/{len(videos)} ({100*imu_count/len(videos):.0f}%)")
    print(f"  Gaze:   {gaze_count}/{len(videos)} ({100*gaze_count/len(videos):.0f}%)")
    print(f"  Stereo: {stereo_count}/{len(videos)} ({100*stereo_count/len(videos):.0f}%)")


if __name__ == "__main__":
    main()
