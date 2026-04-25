"""
Explore Ego4D narration annotations: text analysis, verb/noun distributions,
temporal density, camera-wearer vs other actions.
"""

import json
import re
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "ego4d_data" / "v2" / "annotations"


def main():
    with open(DATA_DIR / "narration.json") as f:
        data = json.load(f)

    videos = data["videos"]
    all_narrations = [n for v in videos for n in v["narrations"]]

    print(f"{'='*60}")
    print(f"EGO4D NARRATION EXPLORATION")
    print(f"{'='*60}")
    print(f"Videos with narrations: {len(videos)}")
    print(f"Total narrations: {len(all_narrations)}")
    print(f"Avg narrations/video: {len(all_narrations)/len(videos):.1f}")

    # Camera wearer vs others
    cw_count = sum(1 for n in all_narrations if n["is_camera_wearer"])
    other_count = len(all_narrations) - cw_count
    print(f"\n--- Actor Distribution ---")
    print(f"  Camera wearer (#C): {cw_count} ({100*cw_count/len(all_narrations):.1f}%)")
    print(f"  Other person  (#O): {other_count} ({100*other_count/len(all_narrations):.1f}%)")

    # Verb distribution
    verb_counts = Counter(n["structured_verb"] for n in all_narrations)
    print(f"\n--- Top 15 Verbs ---")
    for verb, count in verb_counts.most_common(15):
        bar = "█" * max(1, count // 5)
        print(f"  {verb:15s} {count:4d} {bar}")

    # Noun distribution
    noun_counts = Counter(n["structured_noun"] for n in all_narrations)
    print(f"\n--- Top 15 Nouns ---")
    for noun, count in noun_counts.most_common(15):
        bar = "█" * max(1, count // 5)
        print(f"  {noun:15s} {count:4d} {bar}")

    # Verb-Noun co-occurrence (top action pairs)
    action_counts = Counter((n["structured_verb"], n["structured_noun"]) for n in all_narrations)
    print(f"\n--- Top 15 Verb-Noun Actions ---")
    for (verb, noun), count in action_counts.most_common(15):
        print(f"  {verb:12s} + {noun:12s} = {count:4d}")

    # Temporal density analysis
    print(f"\n--- Temporal Density (narrations per minute) ---")
    densities = []
    for v in videos:
        narrs = v["narrations"]
        if len(narrs) < 2:
            continue
        ts = [n["timestamp_sec"] for n in narrs]
        span = max(ts) - min(ts)
        if span > 0:
            density = len(narrs) / (span / 60)
            densities.append(density)

    if densities:
        print(f"  Mean density: {sum(densities)/len(densities):.2f} narr/min")
        print(f"  Min density:  {min(densities):.2f} narr/min")
        print(f"  Max density:  {max(densities):.2f} narr/min")

    # Sample narrations
    print(f"\n--- Sample Narrations (first 10) ---")
    for n in all_narrations[:10]:
        ts = n["timestamp_sec"]
        text = n["narration_text"]
        print(f"  [{ts:7.1f}s] {text}")

    # Word frequency in narration text
    all_words = []
    for n in all_narrations:
        words = re.findall(r'\b[a-z]+\b', n["narration_text"].lower())
        all_words.extend(words)
    word_counts = Counter(all_words)
    stop_words = {"the", "a", "an", "is", "in", "on", "to", "of", "and", "c", "o", "s"}
    print(f"\n--- Top 15 Content Words ---")
    for word, count in word_counts.most_common(30):
        if word not in stop_words:
            print(f"  {word:15s} {count:4d}")
            if len([w for w, c in word_counts.most_common(30) if w not in stop_words][:15]) <= 15:
                continue


if __name__ == "__main__":
    main()
