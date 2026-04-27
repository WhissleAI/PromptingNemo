#!/usr/bin/env python3
"""Create a smart 100K subset of Hindi training data.

Stratified sampling across tags, entities, and demographics to maximize
diversity per sample while maintaining coverage of all entity types.
"""

import json
import random
import collections
import argparse

random.seed(42)

TRAILING_TAG_PREFIXES = ('AGE_', 'GENDER_', 'EMOTION_')
ENTITY_PREFIX = 'ENTITY_'
GARBAGE_ENTITIES = {
    'ENTITY_EVENTEND', 'ENTITY_ENTITY', 'ENTITY_TECHNOLOGYCRISPR',
    'ENTITY_SOUND_BITEEND', 'ENTITY_WEBSITEWWW.',
}
TARGET_SIZE = 100000


def parse_sample(idx, entry):
    text = entry['text']
    tokens = text.split()
    age = gender = emotion = None
    entities = []
    for t in tokens:
        if t.startswith('AGE_'): age = t
        elif t.startswith('GENDER_'): gender = t
        elif t.startswith('EMOTION_'): emotion = t
        elif t.startswith('ENTITY_'): entities.append(t)
    return {
        'idx': idx,
        'duration': entry.get('duration', 0),
        'age': age,
        'gender': gender,
        'emotion': emotion,
        'entities': entities,
        'entity_set': set(entities),
        'n_entities': len(entities),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', type=int, default=TARGET_SIZE)
    args = parser.parse_args()

    with open(args.manifest) as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(entries)} samples")

    samples = [parse_sample(i, e) for i, e in enumerate(entries)]

    # Quality filter
    filtered = []
    removed_quality = 0
    for s in samples:
        if s['duration'] > 20:
            removed_quality += 1
            continue
        if any(e in GARBAGE_ENTITIES for e in s['entities']):
            removed_quality += 1
            continue
        filtered.append(s)
    print(f"Quality filter: removed {removed_quality}, remaining {len(filtered)}")

    selected = set()

    def add(indices, label=""):
        new = indices - selected
        selected.update(new)
        if label:
            print(f"  {label}: +{len(new)} (total {len(selected)})")

    # --- Priority 1: Rare tags ---
    print("\nPriority 1: Rare tag coverage")

    rare_age_0_18 = {s['idx'] for s in filtered if s['age'] == 'AGE_0_18'}
    add(rare_age_0_18, "AGE_0_18 (all)")

    age_18_30 = [s for s in filtered if s['age'] == 'AGE_18_30']
    add({s['idx'] for s in random.sample(age_18_30, min(3000, len(age_18_30)))}, "AGE_18_30 (3K)")

    rare_gender = {s['idx'] for s in filtered if s['gender'] == 'GENDER_OTHER'}
    add(rare_gender, "GENDER_OTHER (all)")

    sad = [s for s in filtered if s['emotion'] == 'EMOTION_SAD']
    add({s['idx'] for s in random.sample(sad, min(8000, len(sad)))}, "EMOTION_SAD (8K)")

    angry = {s['idx'] for s in filtered if s['emotion'] == 'EMOTION_ANGRY'}
    add(angry, "EMOTION_ANGRY (all)")

    surprise = {s['idx'] for s in filtered if s['emotion'] == 'EMOTION_SURPRISE'}
    add(surprise, "EMOTION_SURPRISE (all)")

    fear = {s['idx'] for s in filtered if s['emotion'] == 'EMOTION_FEAR'}
    add(fear, "EMOTION_FEAR (all)")

    # --- Priority 2: Rare entity type coverage ---
    print("\nPriority 2: Rare entity coverage")
    entity_type_counts = collections.Counter()
    entity_type_samples = collections.defaultdict(list)
    for s in filtered:
        for e in s['entity_set']:
            entity_type_counts[e] += 1
            entity_type_samples[e].append(s['idx'])

    rare_entity_types = [e for e, c in entity_type_counts.items() if c < 500]
    rare_entity_indices = set()
    for et in rare_entity_types:
        rare_entity_indices.update(entity_type_samples[et])
    add(rare_entity_indices, f"Rare entity types ({len(rare_entity_types)} types)")

    # --- Priority 3: Balanced common entity sampling ---
    print("\nPriority 3: Balanced common entity sampling")
    common_entity_types = [e for e, c in entity_type_counts.items() if c >= 500]
    common_entity_types.sort(key=lambda e: entity_type_counts[e])
    per_type = 1200
    for et in common_entity_types:
        available = [idx for idx in entity_type_samples[et] if idx not in selected]
        n = min(per_type, len(available))
        if n > 0:
            add({idx for idx in random.sample(available, n)}, f"{et} ({n})")

    # --- Priority 4: No-entity samples ---
    print("\nPriority 4: No-entity samples")
    no_entity = [s for s in filtered if s['n_entities'] == 0 and s['idx'] not in selected]
    n_no_ent = min(15000, len(no_entity))
    add({s['idx'] for s in random.sample(no_entity, n_no_ent)}, f"No-entity ({n_no_ent})")

    # --- Priority 5: Stratified AGE × GENDER fill ---
    print(f"\nPriority 5: Stratified fill (need {args.target - len(selected)} more)")
    remaining = [s for s in filtered if s['idx'] not in selected]

    age_gender_bins = collections.defaultdict(list)
    for s in remaining:
        key = (s['age'] or 'NONE', s['gender'] or 'NONE')
        age_gender_bins[key].append(s)

    slots_left = args.target - len(selected)
    if slots_left > 0 and age_gender_bins:
        per_bin = max(1, slots_left // len(age_gender_bins))
        for key in sorted(age_gender_bins.keys()):
            bin_samples = age_gender_bins[key]
            n = min(per_bin, len(bin_samples))
            if n > 0:
                chosen = random.sample(bin_samples, n)
                add({s['idx'] for s in chosen})

        # If still short, fill randomly
        if len(selected) < args.target:
            still_remaining = [s for s in filtered if s['idx'] not in selected]
            n = min(args.target - len(selected), len(still_remaining))
            if n > 0:
                add({s['idx'] for s in random.sample(still_remaining, n)}, f"Random fill ({n})")

    # Trim if overshot
    if len(selected) > args.target:
        selected_list = list(selected)
        random.shuffle(selected_list)
        selected = set(selected_list[:args.target])

    print(f"\n{'='*60}")
    print(f"FINAL: {len(selected)} samples selected")

    # Write output
    subset_entries = [entries[i] for i in sorted(selected)]
    random.shuffle(subset_entries)
    with open(args.output, 'w') as f:
        for entry in subset_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Print distribution summary
    subset_samples = [parse_sample(0, e) for e in subset_entries]
    total_dur = sum(s['duration'] for s in subset_samples)
    print(f"Total hours: {total_dur/3600:.1f}h")

    print(f"\nAGE distribution:")
    age_dist = collections.Counter(s['age'] for s in subset_samples)
    for k, v in age_dist.most_common():
        pct = v / len(subset_samples) * 100
        print(f"  {k}: {v} ({pct:.1f}%)")

    print(f"\nGENDER distribution:")
    gender_dist = collections.Counter(s['gender'] for s in subset_samples)
    for k, v in gender_dist.most_common():
        pct = v / len(subset_samples) * 100
        print(f"  {k}: {v} ({pct:.1f}%)")

    print(f"\nEMOTION distribution:")
    emotion_dist = collections.Counter(s['emotion'] for s in subset_samples)
    for k, v in emotion_dist.most_common():
        pct = v / len(subset_samples) * 100
        print(f"  {k}: {v} ({pct:.1f}%)")

    print(f"\nEntity density:")
    ent_density = collections.Counter(min(s['n_entities'], 5) for s in subset_samples)
    for k in range(6):
        label = f"{k}" if k < 5 else "5+"
        print(f"  {label} entities: {ent_density.get(k, 0)}")

    # Entity type coverage
    subset_entity_types = set()
    for s in subset_samples:
        subset_entity_types.update(s['entity_set'])
    all_entity_types = set(entity_type_counts.keys())
    coverage = len(subset_entity_types) / len(all_entity_types) * 100
    print(f"\nEntity type coverage: {len(subset_entity_types)}/{len(all_entity_types)} ({coverage:.1f}%)")

    print(f"\nDuration distribution:")
    dur_buckets = collections.Counter()
    for s in subset_samples:
        d = s['duration']
        if d < 2: dur_buckets["0-2s"] += 1
        elif d < 5: dur_buckets["2-5s"] += 1
        elif d < 10: dur_buckets["5-10s"] += 1
        elif d < 15: dur_buckets["10-15s"] += 1
        else: dur_buckets["15-20s"] += 1
    for k in ["0-2s", "2-5s", "5-10s", "10-15s", "15-20s"]:
        print(f"  {k}: {dur_buckets.get(k, 0)}")

    print(f"\nWritten to {args.output}")


if __name__ == '__main__':
    main()
