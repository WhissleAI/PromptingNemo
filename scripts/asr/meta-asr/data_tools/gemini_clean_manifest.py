#!/usr/bin/env python3
"""Clean any manifest using Gemini transcription.

Sends ALL samples to Gemini for independent transcription, compares with
ground truth labels using CER, and produces a filtered manifest keeping
only samples where Gemini agrees (CER < threshold).
"""

import argparse
import json
import os
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed


TRAILING_TAG_PREFIXES = ('AGE_', 'GENDER_', 'EMOTION_', 'INTENT_', 'DIALECT_')
ENTITY_PREFIX = 'ENTITY_'


def strip_all_tags(text):
    tokens = text.split()
    clean = []
    for t in tokens:
        if any(t.startswith(p) for p in TRAILING_TAG_PREFIXES):
            continue
        if t.startswith(ENTITY_PREFIX) or t == 'END':
            continue
        clean.append(t)
    return ' '.join(clean)


def text_to_chars(text):
    return [ch for ch in text if not ch.isspace()]


def compute_cer(hypothesis, reference):
    ref_chars = text_to_chars(reference)
    hyp_chars = text_to_chars(hypothesis)
    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


def verify_single(idx, audio_path, ref_clean, model_name, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    try:
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        response = model.generate_content([
            {
                "mime_type": "audio/wav",
                "data": base64.b64encode(audio_bytes).decode('utf-8'),
            },
            "Transcribe this Chinese audio exactly as spoken. "
            "Output ONLY the Chinese transcription text, nothing else. "
            "Do not add any explanation or formatting."
        ])
        gemini_text = response.text.strip()
        cer = compute_cer(gemini_text, ref_clean)
        return (idx, gemini_text, cer, None)

    except Exception as e:
        return (idx, '', -1, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Manifest to clean')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--api_key', required=True, help='Gemini API key')
    parser.add_argument('--model', default='gemini-2.5-flash', help='Gemini model')
    parser.add_argument('--cer_threshold', type=float, default=0.4,
                        help='CER threshold: keep if Gemini CER < this')
    parser.add_argument('--workers', type=int, default=16, help='Concurrent workers')
    parser.add_argument('--rpm_limit', type=int, default=500, help='RPM limit')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.manifest) as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(entries)} samples from {args.manifest}")
    print(f"Model: {args.model}, CER threshold: {args.cer_threshold}")
    print(f"Workers: {args.workers}, RPM limit: {args.rpm_limit}")

    work_items = []
    for i, entry in enumerate(entries):
        audio_path = entry['audio_filepath']
        ref_clean = strip_all_tags(entry['text'])
        work_items.append((i, audio_path, ref_clean))

    min_interval = 60.0 / args.rpm_limit
    results = {}
    keeps = 0
    removes = 0
    errors = 0
    completed = 0

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        submitted = 0
        for idx, audio_path, ref_clean in work_items:
            if submitted > 0:
                elapsed_since_start = time.time() - start_time
                expected_time = submitted * min_interval
                if elapsed_since_start < expected_time:
                    time.sleep(expected_time - elapsed_since_start)
            fut = executor.submit(verify_single, idx, audio_path, ref_clean,
                                  args.model, args.api_key)
            futures[fut] = idx
            submitted += 1

        for fut in as_completed(futures):
            idx, gemini_text, cer, error = fut.result()
            results[idx] = {
                'gemini_text': gemini_text,
                'cer': cer,
                'error': error,
            }
            if error:
                errors += 1
                keeps += 1  # default keep on error
            elif cer < args.cer_threshold:
                keeps += 1
            else:
                removes += 1
            completed += 1

            if completed % 100 == 0:
                elapsed_total = time.time() - start_time
                rate = completed / elapsed_total * 60
                eta = (len(work_items) - completed) / (rate / 60) if rate > 0 else 0
                print(f"  [{completed}/{len(work_items)}] keep={keeps} remove={removes} "
                      f"errors={errors} rate={rate:.0f}/min ETA={eta/60:.1f}min")

    print(f"\nComplete: keep={keeps}, remove={removes}, errors={errors}")

    # Write filtered and removed manifests
    filtered_path = os.path.join(args.output_dir, 'test_filtered.json')
    removed_path = os.path.join(args.output_dir, 'test_removed.json')
    results_path = os.path.join(args.output_dir, 'gemini_results.json')

    keep_count = 0
    remove_count = 0
    with open(filtered_path, 'w') as fk, open(removed_path, 'w') as fr:
        for i, entry in enumerate(entries):
            r = results.get(i)
            if r is None or r.get('error') or r['cer'] < args.cer_threshold:
                fk.write(json.dumps(entry, ensure_ascii=False) + '\n')
                keep_count += 1
            else:
                fr.write(json.dumps(entry, ensure_ascii=False) + '\n')
                remove_count += 1

    with open(results_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    keep_hours = sum(entries[i].get('duration', 0) for i in range(len(entries))
                     if i not in results or results[i].get('error')
                     or results[i]['cer'] < args.cer_threshold) / 3600
    remove_hours = sum(entries[i].get('duration', 0) for i in range(len(entries))
                       if i in results and not results[i].get('error')
                       and results[i]['cer'] >= args.cer_threshold) / 3600

    summary = {
        'total': len(entries),
        'kept': keep_count,
        'removed': remove_count,
        'kept_hours': round(keep_hours, 1),
        'removed_hours': round(remove_hours, 1),
        'errors': errors,
        'cer_threshold': args.cer_threshold,
    }
    with open(os.path.join(args.output_dir, 'filter_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nFINAL: KEEP={keep_count} ({keep_hours:.1f}h), REMOVE={remove_count} ({remove_hours:.1f}h)")
    print(f"Output: {args.output_dir}/")


if __name__ == '__main__':
    main()
