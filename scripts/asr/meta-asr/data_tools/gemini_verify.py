#!/usr/bin/env python3
"""Gemini verification for borderline training samples.

Reads scores.json from the scoring step, sends borderline samples to Gemini
for independent transcription, and produces updated filtered manifests.
"""

import argparse
import json
import os
import sys
import time
import base64
try:
    import numpy as np
except ImportError:
    class _np:
        @staticmethod
        def percentile(arr, p):
            s = sorted(arr)
            k = int(len(s) * p / 100)
            return s[min(k, len(s)-1)]
        @staticmethod
        def array(x):
            return list(x)
    np = _np()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def text_to_chars(text):
    chars = []
    for ch in text:
        if ch.isspace():
            continue
        chars.append(ch)
    return chars


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


def verify_single(idx, audio_path, ref_clean, model_name, api_key):
    """Verify a single sample with Gemini. Returns (idx, verdict, gemini_text, cer)."""
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
        cer_vs_ref = compute_cer(gemini_text, ref_clean)

        if cer_vs_ref < 0.4:
            return (idx, 'keep', gemini_text, cer_vs_ref, None)
        else:
            return (idx, 'remove', gemini_text, cer_vs_ref, None)

    except Exception as e:
        return (idx, 'keep', '', 0, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_file', required=True, help='scores.json from scoring step')
    parser.add_argument('--manifest', required=True, help='Original training manifest')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--api_key', required=True, help='Gemini API key')
    parser.add_argument('--model', default='gemini-2.0-flash-lite', help='Gemini model name')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to verify')
    parser.add_argument('--workers', type=int, default=8, help='Concurrent workers')
    parser.add_argument('--rpm_limit', type=int, default=500, help='Requests per minute limit')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load scores
    print(f"Loading scores from {args.scores_file}...")
    with open(args.scores_file) as f:
        all_scores = json.load(f)

    # Load manifest
    with open(args.manifest) as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    # Re-classify samples to find borderline ones
    # (scoring script set all borderline to 'keep' before saving)
    loss_per_secs = np.array([s['loss_per_sec'] for s in all_scores])
    loss_p75 = np.percentile(loss_per_secs, 75)
    print(f"Loss/sec 75th percentile: {loss_p75:.4f}")

    borderline = []
    auto_keep = 0
    auto_remove = 0
    for s in all_scores:
        cer = s['cer']
        dur = s['duration']
        lps = s['loss_per_sec']
        if cer < 0.3:
            s['_class'] = 'keep'
            auto_keep += 1
        elif dur < 2.0 and cer < 0.8:
            s['_class'] = 'keep'
            auto_keep += 1
        elif dur < 2.0 and cer >= 0.8:
            s['_class'] = 'remove'
            auto_remove += 1
        elif cer > 0.7 and lps > loss_p75:
            s['_class'] = 'remove'
            auto_remove += 1
        else:
            s['_class'] = 'borderline'
            borderline.append(s)

    print(f"Re-classified: auto_keep={auto_keep}, auto_remove={auto_remove}, borderline={len(borderline)}")
    # Sort by CER descending (worst first)
    borderline.sort(key=lambda x: -x['cer'])

    if args.max_samples:
        borderline = borderline[:args.max_samples]

    print(f"Verifying {len(borderline)} borderline samples with {args.model}")
    print(f"Workers: {args.workers}, RPM limit: {args.rpm_limit}")

    min_interval = 60.0 / args.rpm_limit
    results = {}
    errors = 0
    keeps = 0
    removes = 0
    completed = 0

    # Prepare work items
    work_items = []
    for sample in borderline:
        idx = sample['idx']
        entry = entries[idx]
        audio_path = entry['audio_filepath']
        ref_clean = strip_all_tags(entry['text'])
        work_items.append((idx, audio_path, ref_clean))

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        submitted = 0
        for idx, audio_path, ref_clean in work_items:
            # Simple rate limiting: wait if submitting too fast
            if submitted > 0:
                elapsed_since_start = time.time() - start_time
                expected_time = submitted * min_interval
                if elapsed_since_start < expected_time:
                    time.sleep(expected_time - elapsed_since_start)
            fut = executor.submit(verify_single, idx, audio_path, ref_clean, args.model, args.api_key)
            futures[fut] = idx
            submitted += 1

        for fut in as_completed(futures):
            idx_result, verdict, gemini_text, cer, error = fut.result()
            results[idx_result] = {
                'verdict': verdict,
                'gemini_text': gemini_text,
                'gemini_cer_vs_ref': cer,
                'error': error,
            }
            if verdict == 'keep':
                keeps += 1
            else:
                removes += 1
            if error:
                errors += 1
            completed += 1

            if completed % 100 == 0:
                elapsed_total = time.time() - start_time
                rate = completed / elapsed_total * 60
                eta = (len(borderline) - completed) / (rate / 60) if rate > 0 else 0
                print(f"  [{completed}/{len(borderline)}] keep={keeps} remove={removes} "
                      f"errors={errors} rate={rate:.0f}/min ETA={eta/60:.1f}min")

    print(f"\nGemini verification complete:")
    print(f"  Keep:   {keeps}")
    print(f"  Remove: {removes}")
    print(f"  Errors: {errors}")

    # Rebuild filtered manifests using auto-classification + Gemini verdicts
    keep_indices = set()
    remove_indices = set()

    for s in all_scores:
        idx = s['idx']
        if idx in results:
            verdict = results[idx]['verdict']
            s['final_verdict'] = verdict
            s['gemini_text'] = results[idx].get('gemini_text', '')
            s['gemini_cer'] = results[idx].get('gemini_cer_vs_ref', 0)
        else:
            verdict = s.get('_class', 'keep')
            s['final_verdict'] = verdict

        if verdict == 'remove':
            remove_indices.add(idx)
        else:
            keep_indices.add(idx)

    # Write outputs
    filtered_path = os.path.join(args.output_dir, 'train_filtered.json')
    removed_path = os.path.join(args.output_dir, 'train_removed.json')
    scores_path = os.path.join(args.output_dir, 'scores_with_gemini.json')
    gemini_results_path = os.path.join(args.output_dir, 'gemini_results.json')

    with open(filtered_path, 'w') as f:
        for i, entry in enumerate(entries):
            if i in keep_indices:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open(removed_path, 'w') as f:
        for i, entry in enumerate(entries):
            if i in remove_indices:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open(scores_path, 'w') as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    with open(gemini_results_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    keep_hours = sum(entries[i].get('duration', 0) for i in keep_indices) / 3600
    remove_hours = sum(entries[i].get('duration', 0) for i in remove_indices) / 3600

    summary = {
        'total': len(entries),
        'kept': len(keep_indices),
        'removed': len(remove_indices),
        'kept_hours': round(keep_hours, 1),
        'removed_hours': round(remove_hours, 1),
        'gemini_verified': len(results),
        'gemini_keeps': keeps,
        'gemini_removes': removes,
        'gemini_errors': errors,
    }
    with open(os.path.join(args.output_dir, 'filter_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nFINAL RESULT:")
    print(f"  KEEP:   {len(keep_indices)} samples ({keep_hours:.1f} hours)")
    print(f"  REMOVE: {len(remove_indices)} samples ({remove_hours:.1f} hours)")
    print(f"\nOutput files in {args.output_dir}/")

    # Show some Gemini-removed samples
    gemini_removed = [(idx, r) for idx, r in results.items() if r['verdict'] == 'remove']
    if gemini_removed:
        print(f"\nSample Gemini-removed (first 10):")
        for idx, r in gemini_removed[:10]:
            ref = strip_all_tags(entries[idx]['text'])
            print(f"  ref: {ref[:50]}")
            print(f"  gem: {r['gemini_text'][:50]}  cer={r['gemini_cer_vs_ref']:.2f}")
            print()


if __name__ == '__main__':
    main()
