#!/usr/bin/env python3
"""
Prepare multilingual data for TDT fine-tuning.

Steps:
  1. Discover all raw manifest files under --data-root
  2. Normalize annotations (GER→GENDER, AGE buckets, missing INTENT, etc.)
  3. Normalize lang_family labels (EAST_ASIAN→MANDARIN, fix DRAVIDIAN, etc.)
  4. Validate audio files exist
  5. Create balanced train/valid/test manifests per language family

Usage:
  # Scan and report stats (dry-run)
  python prepare_multilingual.py --data-root /mnt/nfs/data/multilingual_v1/raw --dry-run

  # Full preparation
  python prepare_multilingual.py \
    --data-root /mnt/nfs/data/multilingual_v1/raw \
    --extra-data /mnt/nfs/data/meta_stt_hi_set1 /mnt/nfs/data/slavic_cv_full \
    --output-dir /mnt/nfs/data/multilingual_v1/manifests \
    --validate-audio
"""
import argparse
import json
import logging
import os
import re
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lang code → family mapping
# ---------------------------------------------------------------------------
LANG_TO_FAMILY = {
    # English
    "ENGLISH": "ENGLISH", "EN": "ENGLISH", "EN-US": "ENGLISH", "EN-GB": "ENGLISH",
    "EN-IN": "ENGLISH",
    # European
    "EUROPEAN": "EUROPEAN", "DE": "EUROPEAN", "FR": "EUROPEAN", "ES": "EUROPEAN",
    "IT": "EUROPEAN", "PT": "EUROPEAN", "NL": "EUROPEAN", "DA": "EUROPEAN",
    "SV": "EUROPEAN", "SV-SE": "EUROPEAN", "FI": "EUROPEAN", "ET": "EUROPEAN",
    "RO": "EUROPEAN", "GL": "EUROPEAN", "CA": "EUROPEAN", "EU": "EUROPEAN",
    # Slavic
    "SLAVIC": "SLAVIC", "RU": "SLAVIC", "UK": "SLAVIC", "PL": "SLAVIC",
    "CS": "SLAVIC", "SK": "SLAVIC", "BE": "SLAVIC", "BG": "SLAVIC",
    "SR": "SLAVIC", "SL": "SLAVIC", "MK": "SLAVIC", "KA": "SLAVIC",
    # Indo-Aryan
    "INDO_ARYAN": "INDO_ARYAN", "HI": "INDO_ARYAN", "HI-EN": "INDO_ARYAN",
    "BH": "INDO_ARYAN", "BN": "INDO_ARYAN", "CH": "INDO_ARYAN",
    "MG": "INDO_ARYAN", "MR": "INDO_ARYAN", "MT": "INDO_ARYAN",
    "GU": "INDO_ARYAN", "PA": "INDO_ARYAN", "OR": "INDO_ARYAN",
    "AS": "INDO_ARYAN", "NE": "INDO_ARYAN", "SD": "INDO_ARYAN",
    "UR": "INDO_ARYAN",
    # Mandarin
    "MANDARIN": "MANDARIN", "ZH": "MANDARIN", "EAST_ASIAN": "MANDARIN",
    # Dravidian
    "DRAVIDIAN": "DRAVIDIAN", "KN": "DRAVIDIAN", "TE": "DRAVIDIAN",
    "TA": "DRAVIDIAN", "ML": "DRAVIDIAN",
    # Indian low resource (catch-all for Indian languages not in above)
    "INDIAN_LOW_RESOURCE": "INDIAN_LOW_RESOURCE",
    # Multi / code-switched
    "MULTI": None,  # needs context-based resolution
}

# Canonical families
ALL_FAMILIES = [
    "ENGLISH", "EUROPEAN", "SLAVIC", "INDO_ARYAN",
    "MANDARIN", "DRAVIDIAN", "INDIAN_LOW_RESOURCE",
]

# ---------------------------------------------------------------------------
# Annotation normalization
# ---------------------------------------------------------------------------

# Regex patterns
RE_GER = re.compile(r'\bGER_(MALE|FEMALE|OTHER|NA)\b')
GER_MAP = {'MALE': 'GENDER_MALE', 'FEMALE': 'GENDER_FEMALE', 'OTHER': 'GENDER_OTHER', 'NA': 'GENDER_OTHER'}
RE_AGE_14_25 = re.compile(r'\bAGE_14_25\b')
RE_AGE_25_35 = re.compile(r'\bAGE_25_35\b')
RE_AGE_35_50 = re.compile(r'\bAGE_35_50\b')
RE_AGE_50_PLUS = re.compile(r'\bAGE_50\+?\b')
RE_DIALECT = re.compile(r'\bDIALECT_\S+')
RE_ENTITY_GLUED = re.compile(r'\b(ENTITY_\w+?)END\b')
RE_ENTITY_GENDER = re.compile(r'\bENTITY_GENDER\w*\b')
RE_HAS_INTENT = re.compile(r'\bINTENT_\S+')
RE_HAS_AGE = re.compile(r'\bAGE_\S+')
RE_HAS_GENDER = re.compile(r'\bGENDER_\S+')
RE_HAS_EMOTION = re.compile(r'\bEMOTION_\S+')
RE_MULTI_SPACE = re.compile(r'\s+')
RE_SPEAKER_ID_ONLY = re.compile(r'^\d+_\d+$')
RE_EMOTION_DISGUST = re.compile(r'\bEMOTION_DISGUST\b')
RE_META_TAG = re.compile(r'\b(AGE_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+)\b')


def _dedup_meta_tags(text: str) -> str:
    """Remove duplicate meta tags from concatenated annotation passes."""
    parts = text.split()
    seen_categories = {}
    result = []
    for token in parts:
        m = RE_META_TAG.match(token)
        if m:
            category = token.split('_')[0]
            if category in seen_categories:
                continue
            seen_categories[category] = token
        result.append(token)
    return ' '.join(result)


def normalize_text(text: str) -> str:
    """Normalize annotations in transcript text."""
    # Fix GER_* → GENDER_*
    text = RE_GER.sub(lambda m: GER_MAP[m.group(1)], text)

    # Remap non-standard AGE buckets to our canonical set
    text = RE_AGE_14_25.sub('AGE_18_30', text)
    text = RE_AGE_25_35.sub('AGE_30_45', text)
    text = RE_AGE_35_50.sub('AGE_30_45', text)
    text = RE_AGE_50_PLUS.sub('AGE_45_60', text)

    # Remove DIALECT_ tags
    text = RE_DIALECT.sub('', text)

    # Fix glued entity tags: ENTITY_CITYEND → ENTITY_CITY END
    text = RE_ENTITY_GLUED.sub(r'\1 END', text)

    # Remove ENTITY_GENDER* false positives
    text = RE_ENTITY_GENDER.sub('', text)

    # Normalize EMOTION_DISGUST → EMOTION_SAD
    text = RE_EMOTION_DISGUST.sub('EMOTION_SAD', text)

    # Deduplicate meta tags from concatenated annotation passes
    text = _dedup_meta_tags(text)

    # Add INTENT_INFORM if meta tags present but INTENT missing
    has_meta = RE_HAS_AGE.search(text) or RE_HAS_EMOTION.search(text)
    if has_meta and not RE_HAS_INTENT.search(text):
        text = text.rstrip() + ' INTENT_INFORM'

    # Clean up whitespace
    text = RE_MULTI_SPACE.sub(' ', text).strip()

    return text


def resolve_family(sample: dict, hint_family: str = None) -> Optional[str]:
    """Determine the canonical language family for a sample.

    Priority: hint_family (from source registry) > lang code mapping > lang_family field.
    The hint overrides because some datasets have incorrect lang_family
    (e.g., MADASR labels Kannada/Telugu as INDO_ARYAN instead of DRAVIDIAN).
    """
    # If we have a source-level hint, use it
    if hint_family:
        return hint_family

    # Try lang code — more reliable than lang_family for sub-classification
    lang = sample.get('lang', '').upper().strip()
    if lang in LANG_TO_FAMILY and LANG_TO_FAMILY[lang] is not None:
        return LANG_TO_FAMILY[lang]

    # Try lang_family field
    lf = sample.get('lang_family', '').upper().strip()
    if lf in LANG_TO_FAMILY and LANG_TO_FAMILY[lf] is not None:
        return LANG_TO_FAMILY[lf]

    # MULTI needs source context
    if lang == "MULTI" or lf == "MULTI":
        return None

    return None


def is_valid_transcript(text: str) -> bool:
    """Check if transcript text is a real transcription (not just speaker ID)."""
    if not text or len(text.strip()) < 2:
        return False
    if RE_SPEAKER_ID_ONLY.match(text.strip()):
        return False
    return True


# ---------------------------------------------------------------------------
# Data source discovery
# ---------------------------------------------------------------------------

def discover_manifests(data_root: str, extra_data: list[str] = None) -> list[dict]:
    """Find all train/valid/test manifest files.

    Uses an explicit registry of known data sources to avoid walking
    through massive audio directories.

    Returns list of dicts: {path, split, source, hint_family}
    """
    manifests = []
    root = Path(data_root)

    # Explicit data source registry: (subpath, manifest_name, split, family)
    SOURCES = [
        # English
        ('en_people', 'train.json', 'train', 'ENGLISH'),
        ('en_people', 'valid.json', 'valid', 'ENGLISH'),
        ('en_in_tech', 'train.json', 'train', 'ENGLISH'),
        ('en_in_tech', 'valid.json', 'valid', 'ENGLISH'),
        ('commonvoice_17/english', 'train.json', 'train', 'ENGLISH'),
        ('commonvoice_17/english', 'valid.json', 'valid', 'ENGLISH'),
        # European
        ('commonvoice_17/euro', 'train.json', 'train', 'EUROPEAN'),
        ('commonvoice_17/euro', 'valid.json', 'valid', 'EUROPEAN'),
        ('commonvoice_17/euro', 'test.json', 'test', 'EUROPEAN'),
        # Slavic
        ('commonvoice_17/slavic', 'train.json', 'train', 'SLAVIC'),
        ('commonvoice_17/slavic', 'valid.json', 'valid', 'SLAVIC'),
        ('commonvoice_17/slavic', 'test.json', 'test', 'SLAVIC'),
        ('global', 'train.json', 'train', 'SLAVIC'),
        ('global', 'valid.json', 'valid', 'SLAVIC'),
        # Indo-Aryan
        ('hindi_set1', 'train.json', 'train', 'INDO_ARYAN'),
        ('hinglish_indicvoices', 'train.json', 'train', 'INDO_ARYAN'),
        ('hinglish_indicvoices', 'valid.json', 'valid', 'INDO_ARYAN'),
        ('hinglish_indicvoices', 'test.json', 'test', 'INDO_ARYAN'),
        ('hinglish_mucs', 'train.json', 'train', 'INDO_ARYAN'),
        ('betrac', 'train.json', 'train', 'INDO_ARYAN'),
        ('betrac', 'valid.json', 'valid', 'INDO_ARYAN'),
        # Mandarin
        ('aishell1', 'train.json', 'train', 'MANDARIN'),
        ('aishell1', 'valid.json', 'valid', 'MANDARIN'),
        ('aishell1', 'test.json', 'test', 'MANDARIN'),
        ('aishell3', 'train.json', 'train', 'MANDARIN'),
        ('aishell3', 'valid.json', 'valid', 'MANDARIN'),
        ('aishell3', 'test.json', 'test', 'MANDARIN'),
        ('magicdata', 'train.json', 'train', 'MANDARIN'),
        ('magicdata', 'valid.json', 'valid', 'MANDARIN'),
        ('magicdata', 'test.json', 'test', 'MANDARIN'),
    ]

    for subpath, manifest_name, split, family in SOURCES:
        json_path = root / subpath / manifest_name
        if json_path.exists():
            manifests.append({
                'path': str(json_path),
                'split': split,
                'source': subpath,
                'hint_family': family,
            })
        else:
            log.debug("Manifest not found: %s", json_path)

    # MADASR sub-language directories
    MADASR_FAMILY = {
        'bn': 'INDO_ARYAN', 'mr': 'INDO_ARYAN',
        'bh': 'INDIAN_LOW_RESOURCE', 'ch': 'INDIAN_LOW_RESOURCE',
        'mg': 'INDIAN_LOW_RESOURCE', 'mt': 'INDIAN_LOW_RESOURCE',
        'kn': 'DRAVIDIAN', 'te': 'DRAVIDIAN',
    }
    madasr_root = root / 'madasr'
    if madasr_root.exists():
        for lang_code, family in sorted(MADASR_FAMILY.items()):
            lang_dir = madasr_root / lang_code
            if not lang_dir.is_dir():
                continue
            for manifest_name in ['train.json', 'valid.json', 'test.json']:
                json_path = lang_dir / manifest_name
                if json_path.exists():
                    split = manifest_name.replace('.json', '')
                    manifests.append({
                        'path': str(json_path),
                        'split': split,
                        'source': f'madasr/{lang_code}',
                        'hint_family': family,
                    })

    # Extra data directories (e.g., pre-validated manifests)
    if extra_data:
        for extra_dir in extra_data:
            extra_path = Path(extra_dir)
            if not extra_path.exists():
                log.warning("Extra data dir not found: %s", extra_dir)
                continue
            # Only look at top-level JSON files, not recursively
            for json_path in sorted(extra_path.glob('*.json')):
                name = json_path.name
                if name == 'dataset_info.json' or name.endswith('.progress'):
                    continue
                if not any(x in name for x in ['train', 'valid', 'test']):
                    continue

                split = 'train' if 'train' in name else ('valid' if 'valid' in name else 'test')
                source = extra_path.name
                hint = None
                if 'hi_set1' in source or 'hindi' in source:
                    hint = 'INDO_ARYAN'
                elif 'slavic' in source:
                    hint = 'SLAVIC'
                elif 'zh' in source:
                    hint = 'MANDARIN'
                manifests.append({
                    'path': str(json_path),
                    'split': split,
                    'source': source,
                    'hint_family': hint,
                })

    return manifests


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_manifest(
    manifest_info: dict,
    validate_audio: bool = False,
    audio_path_prefix: str = None,
) -> tuple[list[dict], dict]:
    """Process a single manifest file.

    Returns (processed_samples, stats_dict)
    """
    path = manifest_info['path']
    hint_family = manifest_info['hint_family']
    source = manifest_info['source']

    stats = Counter()
    samples = []

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    stats['json_error'] += 1
                    continue

                text = sample.get('text', '')
                stats['total'] += 1

                # Skip invalid transcripts (speaker IDs, empty)
                if not is_valid_transcript(text):
                    stats['invalid_transcript'] += 1
                    continue

                # Normalize text annotations
                orig_text = text
                text = normalize_text(text)
                if text != orig_text:
                    stats['text_normalized'] += 1
                sample['text'] = text

                # Resolve language family (hint takes priority over data fields)
                family = resolve_family(sample, hint_family)
                if family is None:
                    stats['no_family'] += 1
                    continue
                sample['lang_family'] = family

                # Ensure lang field exists
                if 'lang' not in sample:
                    sample['lang'] = family

                # Add LANG_ prefix token for controllable inference
                lang_code = sample['lang'].upper().strip()
                lang_token = f"LANG_{LANG_TO_FAMILY.get(lang_code, family)}"
                sample['text'] = f"{lang_token} {sample['text']}"

                # Duration filter
                duration = sample.get('duration', 0)
                if duration < 0.5 or duration > 30.0:
                    stats['duration_filtered'] += 1
                    continue

                # Audio validation
                if validate_audio:
                    audio_path = sample.get('audio_filepath', '')
                    if audio_path_prefix and not os.path.isabs(audio_path):
                        audio_path = os.path.join(audio_path_prefix, audio_path)
                    if not os.path.exists(audio_path):
                        stats['audio_missing'] += 1
                        continue

                sample['source'] = source
                samples.append(sample)
                stats[f'family:{family}'] += 1

    except Exception as e:
        log.error("Error processing %s: %s", path, e)
        stats['file_error'] += 1

    return samples, dict(stats)


def create_balanced_splits(
    all_samples: list[dict],
    family_caps: dict[str, int],
    valid_per_family: int = 2000,
    test_per_family: int = 1000,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Create balanced train/valid/test splits."""
    rng = random.Random(seed)

    # Group by family
    by_family = defaultdict(list)
    for s in all_samples:
        by_family[s['lang_family']].append(s)

    train_out, valid_out, test_out = [], [], []

    for family in ALL_FAMILIES:
        samples = by_family.get(family, [])
        if not samples:
            log.warning("No data for family %s", family)
            continue

        rng.shuffle(samples)

        # Split: test first, then valid, then train (capped)
        n_test = min(test_per_family, len(samples) // 10, len(samples))
        test_split = samples[:n_test]
        remaining = samples[n_test:]

        n_valid = min(valid_per_family, len(remaining) // 10, len(remaining))
        valid_split = remaining[:n_valid]
        train_pool = remaining[n_valid:]

        # Cap training samples
        cap = family_caps.get(family, len(train_pool))
        if len(train_pool) > cap:
            train_split = rng.sample(train_pool, cap)
        else:
            train_split = train_pool

        train_out.extend(train_split)
        valid_out.extend(valid_split)
        test_out.extend(test_split)

        log.info(
            "Family %-20s: total=%7d train=%7d valid=%5d test=%5d (cap=%s)",
            family, len(samples), len(train_split), len(valid_split), len(test_split),
            cap if cap < len(train_pool) else 'all',
        )

    # Shuffle final outputs
    rng.shuffle(train_out)
    rng.shuffle(valid_out)
    rng.shuffle(test_out)

    return train_out, valid_out, test_out


def write_manifest(samples: list[dict], output_path: str, strip_source: bool = True):
    """Write samples as JSONL manifest."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in samples:
            out = dict(s)
            if strip_source:
                out.pop('source', None)
            f.write(json.dumps(out, ensure_ascii=False) + '\n')
    log.info("Wrote %d samples to %s", len(samples), output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare multilingual ASR data")
    parser.add_argument('--data-root', required=True, help='Root of raw data directories')
    parser.add_argument('--extra-data', nargs='*', default=[], help='Extra data directories with validated manifests')
    parser.add_argument('--output-dir', default=None, help='Output directory for balanced manifests')
    parser.add_argument('--validate-audio', action='store_true', help='Check audio files exist')
    parser.add_argument('--dry-run', action='store_true', help='Report stats only, no output')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers for audio validation')
    parser.add_argument('--seed', type=int, default=42)
    # Per-family training caps
    parser.add_argument('--cap-english', type=int, default=400000)
    parser.add_argument('--cap-european', type=int, default=400000)
    parser.add_argument('--cap-slavic', type=int, default=300000)
    parser.add_argument('--cap-indo-aryan', type=int, default=350000)
    parser.add_argument('--cap-mandarin', type=int, default=250000)
    parser.add_argument('--cap-dravidian', type=int, default=200000)
    parser.add_argument('--cap-indian-low-resource', type=int, default=200000)
    parser.add_argument('--valid-per-family', type=int, default=2000)
    parser.add_argument('--test-per-family', type=int, default=1000)
    args = parser.parse_args()

    # Discover manifests
    log.info("Discovering manifests under %s ...", args.data_root)
    manifests = discover_manifests(args.data_root, args.extra_data)
    log.info("Found %d manifest files", len(manifests))

    for m in manifests:
        log.info("  %s [%s] family_hint=%s", m['source'], m['split'], m['hint_family'])

    # Process all manifests
    all_samples = []
    total_stats = Counter()

    for m in manifests:
        samples, stats = process_manifest(
            m,
            validate_audio=args.validate_audio,
        )
        all_samples.extend(samples)
        for k, v in stats.items():
            total_stats[k] += v
        log.info(
            "  Processed %-40s: %d samples kept (total=%s)",
            os.path.basename(m['path']),
            len(samples),
            stats.get('total', 0),
        )

    # Report stats
    log.info("=" * 70)
    log.info("TOTAL STATS:")
    for k in sorted(total_stats.keys()):
        log.info("  %-30s %d", k, total_stats[k])

    log.info("\nPER-FAMILY COUNTS:")
    family_counts = Counter()
    for s in all_samples:
        family_counts[s['lang_family']] += 1
    for fam in ALL_FAMILIES:
        log.info("  %-25s %7d", fam, family_counts.get(fam, 0))
    log.info("  %-25s %7d", "TOTAL", len(all_samples))

    if args.dry_run:
        log.info("Dry run complete. No files written.")
        return

    if not args.output_dir:
        log.error("--output-dir required when not in --dry-run mode")
        sys.exit(1)

    # Create balanced splits
    family_caps = {
        'ENGLISH': args.cap_english,
        'EUROPEAN': args.cap_european,
        'SLAVIC': args.cap_slavic,
        'INDO_ARYAN': args.cap_indo_aryan,
        'MANDARIN': args.cap_mandarin,
        'DRAVIDIAN': args.cap_dravidian,
        'INDIAN_LOW_RESOURCE': args.cap_indian_low_resource,
    }

    train, valid, test = create_balanced_splits(
        all_samples,
        family_caps=family_caps,
        valid_per_family=args.valid_per_family,
        test_per_family=args.test_per_family,
        seed=args.seed,
    )

    # Write output manifests
    write_manifest(train, os.path.join(args.output_dir, 'train_balanced.json'))
    write_manifest(valid, os.path.join(args.output_dir, 'valid_balanced.json'))
    write_manifest(test, os.path.join(args.output_dir, 'test_balanced.json'))

    log.info("=" * 70)
    log.info("DONE: train=%d valid=%d test=%d", len(train), len(valid), len(test))


if __name__ == '__main__':
    main()
