#!/usr/bin/env python3
"""Compute distribution stats across all NeMo manifests."""
import json, os, re, sys
from collections import defaultdict, Counter

data_root = sys.argv[1]

# Patterns
AGE_RE = re.compile(r'\bAGE_(\S+)')
GENDER_RE = re.compile(r'\bGENDER_(\S+)')
EMOTION_RE = re.compile(r'\bEMOTION_(\S+)')
INTENT_RE = re.compile(r'\bINTENT_(\S+)')
ENTITY_RE = re.compile(r'\bENTITY_(\S+)')
LANG_TAG_RE = re.compile(r'\bLANG_(\S+)')

# Counters
lang_family_counts = Counter()
lang_counts = Counter()
dataset_counts = Counter()
age_counts = Counter()
gender_counts = Counter()
emotion_counts = Counter()
intent_counts = Counter()
entity_counts = Counter()
lang_tag_counts = Counter()

total_samples = 0
total_duration_hrs = 0.0
samples_with_tags = 0
samples_with_intent = 0
samples_with_entity = 0
samples_no_tags = 0

dataset_details = defaultdict(lambda: {"samples": 0, "duration_hrs": 0.0, "lang_family": "", "langs": set()})

for root, dirs, files in os.walk(data_root):
    dirs[:] = [d for d in dirs if d != 'audio']
    for f in sorted(files):
        if not f.endswith('.json') or f == 'dataset_info.json' or '.bak' in f or '.progress' in f or '.pre_annotate' in f:
            continue
        filepath = os.path.join(root, f)
        rel = os.path.relpath(filepath, data_root)
        dataset = rel.split('/')[0]

        with open(filepath, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except:
                    continue

                total_samples += 1
                dur = s.get("duration", 0)
                total_duration_hrs += dur / 3600.0

                lf = s.get("lang_family", "UNKNOWN")
                lang = s.get("lang", "UNK")
                lang_family_counts[lf] += 1
                lang_counts[lang] += 1
                dataset_counts[dataset] += 1
                dataset_details[dataset]["samples"] += 1
                dataset_details[dataset]["duration_hrs"] += dur / 3600.0
                dataset_details[dataset]["lang_family"] = lf
                dataset_details[dataset]["langs"].add(lang)

                text = s.get("text", "")
                has_any_tag = False

                for m in AGE_RE.finditer(text):
                    age_counts[m.group(1)] += 1
                    has_any_tag = True
                for m in GENDER_RE.finditer(text):
                    gender_counts[m.group(1)] += 1
                    has_any_tag = True
                for m in EMOTION_RE.finditer(text):
                    emotion_counts[m.group(1)] += 1
                    has_any_tag = True
                for m in INTENT_RE.finditer(text):
                    intent_counts[m.group(1)] += 1
                    has_any_tag = True
                    samples_with_intent += 1
                for m in ENTITY_RE.finditer(text):
                    entity_counts[m.group(1)] += 1
                    has_any_tag = True
                    samples_with_entity += 1
                for m in LANG_TAG_RE.finditer(text):
                    lang_tag_counts[m.group(1)] += 1

                if has_any_tag:
                    samples_with_tags += 1
                else:
                    samples_no_tags += 1

print(f"{'='*70}")
print(f"TOTAL: {total_samples:,} samples, {total_duration_hrs:,.1f} hours")
print(f"With META tags: {samples_with_tags:,} ({samples_with_tags*100/total_samples:.1f}%)")
print(f"With INTENT: {samples_with_intent:,} ({samples_with_intent*100/total_samples:.1f}%)")
print(f"With ENTITY: {samples_with_entity:,} ({samples_with_entity*100/total_samples:.1f}%)")
print(f"No tags: {samples_no_tags:,} ({samples_no_tags*100/total_samples:.1f}%)")

print(f"\n{'='*70}")
print("LANGUAGE FAMILY DISTRIBUTION")
print(f"{'='*70}")
for lf, c in sorted(lang_family_counts.items(), key=lambda x: -x[1]):
    pct = c * 100 / total_samples
    hrs = sum(dataset_details[d]["duration_hrs"] for d in dataset_details if dataset_details[d]["lang_family"] == lf)
    print(f"  {lf:<20} {c:>10,} samples ({pct:5.1f}%)  ~{hrs:,.0f} hrs")

print(f"\n{'='*70}")
print("DATASET DISTRIBUTION")
print(f"{'='*70}")
for ds, info in sorted(dataset_details.items(), key=lambda x: -x[1]["samples"]):
    langs = ", ".join(sorted(info["langs"]))[:40]
    print(f"  {ds:<30} {info['samples']:>10,} samples  {info['duration_hrs']:>8,.1f} hrs  [{langs}]")

print(f"\n{'='*70}")
print("LANGUAGE DISTRIBUTION (top 20)")
print(f"{'='*70}")
for lang, c in lang_counts.most_common(20):
    print(f"  {lang:<10} {c:>10,} ({c*100/total_samples:5.1f}%)")

print(f"\n{'='*70}")
print("AGE DISTRIBUTION")
print(f"{'='*70}")
for tag, c in sorted(age_counts.items(), key=lambda x: -x[1]):
    print(f"  AGE_{tag:<15} {c:>10,} ({c*100/total_samples:5.1f}%)")

print(f"\n{'='*70}")
print("GENDER DISTRIBUTION")
print(f"{'='*70}")
for tag, c in sorted(gender_counts.items(), key=lambda x: -x[1]):
    print(f"  GENDER_{tag:<12} {c:>10,} ({c*100/total_samples:5.1f}%)")

print(f"\n{'='*70}")
print("EMOTION DISTRIBUTION")
print(f"{'='*70}")
for tag, c in sorted(emotion_counts.items(), key=lambda x: -x[1]):
    print(f"  EMOTION_{tag:<12} {c:>10,} ({c*100/total_samples:5.1f}%)")

print(f"\n{'='*70}")
print("INTENT DISTRIBUTION")
print(f"{'='*70}")
for tag, c in sorted(intent_counts.items(), key=lambda x: -x[1]):
    print(f"  INTENT_{tag:<12} {c:>10,} ({c*100/total_samples:5.1f}%)")

print(f"\n{'='*70}")
print("ENTITY DISTRIBUTION")
print(f"{'='*70}")
for tag, c in sorted(entity_counts.items(), key=lambda x: -x[1]):
    print(f"  ENTITY_{tag:<15} {c:>10,} ({c*100/total_samples:5.1f}%)")

if lang_tag_counts:
    print(f"\n{'='*70}")
    print("LANG TAG DISTRIBUTION")
    print(f"{'='*70}")
    for tag, c in sorted(lang_tag_counts.items(), key=lambda x: -x[1]):
        print(f"  LANG_{tag:<15} {c:>10,} ({c*100/total_samples:5.1f}%)")
