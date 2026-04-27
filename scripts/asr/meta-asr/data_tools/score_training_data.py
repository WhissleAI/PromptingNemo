#!/usr/bin/env python3
"""Multi-signal confidence-based training data filter for Meta-ASR.

Scores training data using CTC loss + CER, then applies multi-signal
filtering that protects short utterances and only removes samples where
multiple signals agree the annotation is bad.

Step 1: Score all samples (this script)
Step 2: Optionally verify borderline samples with Gemini (--gemini_verify)

Usage:
  # Score only (produces scores + auto-filtered manifests)
  python score_training_data.py \
    --nemo_file /path/to/model.nemo \
    --manifest /path/to/train.json \
    --output_dir /path/to/output/ \
    --categories AGE GENDER DIALECT

  # Score + Gemini verification for borderline samples
  python score_training_data.py \
    --nemo_file /path/to/model.nemo \
    --manifest /path/to/train.json \
    --output_dir /path/to/output/ \
    --categories AGE GENDER DIALECT \
    --gemini_verify --gemini_api_key YOUR_KEY
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import Counter


TRAILING_TAG_PREFIXES = ('AGE_', 'GENDER_', 'EMOTION_', 'INTENT_', 'DIALECT_')
ENTITY_PREFIX = 'ENTITY_'


def strip_all_tags_from_text(text):
    """Remove ALL tag tokens (entity + trailing) from text for clean comparison."""
    tokens = text.split()
    clean = []
    for t in tokens:
        if any(t.startswith(p) for p in TRAILING_TAG_PREFIXES):
            continue
        if t.startswith(ENTITY_PREFIX) or t == 'END':
            continue
        clean.append(t)
    return ' '.join(clean)


def strip_trailing_tags_from_text(text, categories):
    """Remove trailing tag tokens from end of text (mirrors training code)."""
    active_prefixes = tuple(f"{cat}_" for cat in categories)
    all_tag_prefixes = TRAILING_TAG_PREFIXES + (ENTITY_PREFIX,)
    tokens = text.split()
    while tokens:
        t = tokens[-1]
        if any(t.startswith(p) for p in all_tag_prefixes) or t == 'END':
            tokens.pop()
        else:
            break
    return ' '.join(tokens)


def text_to_chars(text):
    """Convert text to character list for CER computation.
    Handles Chinese (no spaces between chars) and mixed CJK/Latin text.
    Removes spaces and punctuation for fair comparison."""
    chars = []
    for ch in text:
        if ch.isspace():
            continue
        chars.append(ch)
    return chars


def compute_cer(hypothesis, reference):
    """Compute character error rate between hypothesis and reference."""
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


def classify_sample(cer, loss_per_sec, duration, loss_p75):
    """Classify a sample into keep/remove/borderline using multi-signal logic.

    Returns:
        'keep'       - confidently good sample
        'remove'     - confidently bad sample
        'borderline' - uncertain, needs Gemini verification
    """
    # Perfect or near-perfect transcription → always keep
    if cer < 0.3:
        return 'keep'

    # Short utterance protection: don't remove unless clearly wrong
    if duration < 2.0:
        if cer < 0.8:
            return 'keep'
        else:
            return 'remove'

    # Both signals agree it's bad → remove
    if cer > 0.7 and loss_per_sec > loss_p75:
        return 'remove'

    # High CER but low loss → model is confident about something different
    # Could be bad annotation → borderline
    if cer > 0.7:
        return 'borderline'

    # Moderate CER (0.3-0.7) → borderline, worth verifying
    if cer >= 0.3:
        return 'borderline'

    return 'keep'


def gemini_verify_batch(samples, entries, api_key, batch_size=20):
    """Use Gemini to transcribe audio and verify annotations.

    For each borderline sample, ask Gemini to transcribe the audio.
    Compare Gemini's transcription with ground truth:
      - If Gemini agrees with ground truth → keep (hard but correct)
      - If Gemini disagrees → remove (likely bad annotation)

    Returns dict of {idx: 'keep' or 'remove'}
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: google-generativeai package not installed. Skipping Gemini verification.")
        return {}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    results = {}
    print(f"\nVerifying {len(samples)} borderline samples with Gemini...")

    for i, sample in enumerate(tqdm(samples)):
        idx = sample['idx']
        entry = entries[idx]
        audio_path = entry['audio_filepath']
        ref_clean = strip_all_tags_from_text(entry['text'])

        try:
            audio_file = genai.upload_file(audio_path)
            response = model.generate_content([
                audio_file,
                "Transcribe this Chinese audio exactly as spoken. "
                "Output ONLY the transcription, nothing else."
            ])
            gemini_text = response.text.strip()

            gemini_cer = compute_cer(gemini_text, ref_clean)

            if gemini_cer < 0.4:
                # Gemini agrees with ground truth → keep (model just hasn't learned it)
                results[idx] = 'keep'
                sample['gemini_hyp'] = gemini_text
                sample['gemini_cer_vs_ref'] = gemini_cer
                sample['gemini_verdict'] = 'keep'
            else:
                # Gemini also disagrees with ground truth → bad annotation
                results[idx] = 'remove'
                sample['gemini_hyp'] = gemini_text
                sample['gemini_cer_vs_ref'] = gemini_cer
                sample['gemini_verdict'] = 'remove'

        except Exception as e:
            # On error, keep the sample (conservative)
            results[idx] = 'keep'
            sample['gemini_error'] = str(e)
            sample['gemini_verdict'] = 'keep (error)'

        if (i + 1) % 100 == 0:
            keeps = sum(1 for v in results.values() if v == 'keep')
            removes = sum(1 for v in results.values() if v == 'remove')
            print(f"  Progress: {i+1}/{len(samples)} — keep={keeps}, remove={removes}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-signal training data filter')
    parser.add_argument('--nemo_file', required=True, help='Path to .nemo model file')
    parser.add_argument('--manifest', required=True, help='Path to training manifest JSON')
    parser.add_argument('--output_dir', required=True, help='Output directory for filtered manifests')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--categories', nargs='+', default=['AGE', 'GENDER', 'DIALECT'],
                        help='Trailing tag categories to strip from CTC targets')
    parser.add_argument('--device', default=None, help='Device (default: auto)')
    # Gemini verification options
    parser.add_argument('--gemini_verify', action='store_true',
                        help='Run Gemini verification on borderline samples')
    parser.add_argument('--gemini_api_key', default=None,
                        help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--gemini_max_samples', type=int, default=5000,
                        help='Max borderline samples to verify with Gemini')
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load model ───
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf, open_dict

    print(f"Loading model from {args.nemo_file}...")
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        args.nemo_file,
        map_location=args.device,
        strict=False,
    )
    model.eval()
    model.freeze()

    vocab = list(model.decoder.vocabulary)
    num_classes = len(vocab)
    blank_id = num_classes
    print(f"Vocabulary: {num_classes} tokens, blank_id={blank_id}")

    # ─── Build tag token ID sets ───
    all_tag_prefixes = TRAILING_TAG_PREFIXES + (ENTITY_PREFIX,)
    all_special_ids = set()
    for idx, token in enumerate(vocab):
        clean = token.lstrip('▁')
        for prefix in all_tag_prefixes:
            if clean.startswith(prefix):
                all_special_ids.add(idx)
                break
        if clean == 'END':
            all_special_ids.add(idx)
    print(f"All special token IDs: {len(all_special_ids)}")

    # ─── Read manifest ───
    with open(args.manifest) as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(entries)} manifest entries")

    # ─── Setup dataloader with trailing tags stripped ───
    import tempfile
    clean_manifest_path = tempfile.mktemp(suffix='.json')
    with open(clean_manifest_path, 'w') as f:
        for entry in entries:
            clean_entry = dict(entry)
            clean_entry['text'] = strip_trailing_tags_from_text(entry['text'], args.categories)
            f.write(json.dumps(clean_entry, ensure_ascii=False) + '\n')

    with open_dict(model.cfg):
        model.cfg.test_ds = OmegaConf.create({
            'manifest_filepath': clean_manifest_path,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'shuffle': False,
            'sample_rate': model.cfg.get('sample_rate', 16000),
        })
    model.setup_test_data(model.cfg.test_ds)

    # ─── Score all samples ───
    ctc_loss_fn = torch.nn.CTCLoss(blank=blank_id, reduction='none', zero_infinity=True)
    all_results = []
    entry_idx = 0

    print("Computing per-utterance CTC loss + CER...")
    with torch.no_grad():
        for batch in tqdm(model._test_dl):
            signal, signal_len, tokens, tokens_len = batch[:4]
            signal = signal.to(args.device)
            signal_len = signal_len.to(args.device)
            tokens = tokens.to(args.device)
            tokens_len = tokens_len.to(args.device)

            log_probs, encoded_len, greedy_predictions = model(
                input_signal=signal, input_signal_length=signal_len
            )

            # CTC loss per utterance
            log_probs_t = log_probs.transpose(0, 1)
            losses = ctc_loss_fn(log_probs_t, tokens, encoded_len, tokens_len)

            # Greedy decode
            greedy_texts = []
            for pred in greedy_predictions:
                decoded = []
                prev = -1
                for t in pred:
                    t = t.item()
                    if t != blank_id and t != prev:
                        if t < num_classes:
                            decoded.append(vocab[t])
                    prev = t
                greedy_texts.append(''.join(decoded).replace('▁', ' ').strip())

            for i in range(len(losses)):
                if entry_idx >= len(entries):
                    break
                entry = entries[entry_idx]
                duration = entry.get('duration', 1.0)
                ref_clean = strip_all_tags_from_text(entry['text'])
                hyp_clean = strip_all_tags_from_text(greedy_texts[i])
                cer = compute_cer(hyp_clean, ref_clean)
                loss_raw = losses[i].item()

                all_results.append({
                    'idx': entry_idx,
                    'audio_filepath': entry['audio_filepath'],
                    'duration': duration,
                    'ctc_loss': loss_raw,
                    'loss_per_sec': loss_raw / max(duration, 0.1),
                    'cer': cer,
                    'ref': ref_clean,
                    'hyp': hyp_clean,
                })
                entry_idx += 1

    os.unlink(clean_manifest_path)
    print(f"\nScored {len(all_results)} utterances")

    # ─── Statistics ───
    cers = np.array([r['cer'] for r in all_results])
    losses_per_sec = np.array([r['loss_per_sec'] for r in all_results])
    durations = np.array([r['duration'] for r in all_results])

    print(f"\n{'='*60}")
    print(f"CER:          mean={cers.mean():.4f}  median={np.median(cers):.4f}  "
          f"std={cers.std():.4f}  min={cers.min():.4f}  max={cers.max():.4f}")
    print(f"Loss/sec:     mean={losses_per_sec.mean():.4f}  median={np.median(losses_per_sec):.4f}  "
          f"std={losses_per_sec.std():.4f}")
    print(f"Duration:     mean={durations.mean():.2f}s  median={np.median(durations):.2f}s  "
          f"<2s: {np.sum(durations < 2)}")

    # CER distribution
    print(f"\nCER distribution:")
    for lo, hi, label in [(0, 0.1, '0-0.1'), (0.1, 0.2, '0.1-0.2'), (0.2, 0.3, '0.2-0.3'),
                          (0.3, 0.5, '0.3-0.5'), (0.5, 0.7, '0.5-0.7'), (0.7, 1.0, '0.7-1.0'),
                          (1.0, 99, '>1.0')]:
        count = np.sum((cers >= lo) & (cers < hi))
        print(f"  CER {label:>8}: {count:>6} ({count/len(cers)*100:>5.1f}%)")

    # ─── Multi-signal classification ───
    loss_p75 = np.percentile(losses_per_sec, 75)
    print(f"\nLoss/sec 75th percentile: {loss_p75:.4f}")

    keep_list = []
    remove_list = []
    borderline_list = []

    for r in all_results:
        verdict = classify_sample(r['cer'], r['loss_per_sec'], r['duration'], loss_p75)
        r['verdict'] = verdict
        if verdict == 'keep':
            keep_list.append(r)
        elif verdict == 'remove':
            remove_list.append(r)
        else:
            borderline_list.append(r)

    print(f"\n{'='*60}")
    print(f"Initial classification:")
    print(f"  KEEP:       {len(keep_list):>6} ({len(keep_list)/len(all_results)*100:.1f}%)")
    print(f"  REMOVE:     {len(remove_list):>6} ({len(remove_list)/len(all_results)*100:.1f}%)")
    print(f"  BORDERLINE: {len(borderline_list):>6} ({len(borderline_list)/len(all_results)*100:.1f}%)")

    # Stats for each group
    for label, group in [('KEEP', keep_list), ('REMOVE', remove_list), ('BORDERLINE', borderline_list)]:
        if group:
            g_cers = [r['cer'] for r in group]
            g_durs = [r['duration'] for r in group]
            print(f"  {label:>10}: mean_cer={np.mean(g_cers):.3f}  mean_dur={np.mean(g_durs):.2f}s  "
                  f"hours={sum(g_durs)/3600:.1f}h")

    # ─── Gemini verification for borderline samples ───
    if args.gemini_verify and borderline_list:
        api_key = args.gemini_api_key or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print("\nWARNING: --gemini_verify set but no API key provided. "
                  "Set --gemini_api_key or GEMINI_API_KEY env var.")
            print("Treating all borderline samples as KEEP (conservative).")
            for r in borderline_list:
                r['verdict'] = 'keep'
                keep_list.append(r)
            borderline_list = []
        else:
            # Sort borderline by CER descending (verify worst first)
            borderline_list.sort(key=lambda x: -x['cer'])
            verify_samples = borderline_list[:args.gemini_max_samples]

            gemini_results = gemini_verify_batch(verify_samples, entries, api_key)

            new_keep = []
            new_remove = []
            for r in borderline_list:
                if r['idx'] in gemini_results:
                    r['verdict'] = gemini_results[r['idx']]
                    if r['verdict'] == 'keep':
                        new_keep.append(r)
                    else:
                        new_remove.append(r)
                else:
                    # Not verified → keep (conservative)
                    r['verdict'] = 'keep'
                    new_keep.append(r)

            keep_list.extend(new_keep)
            remove_list.extend(new_remove)
            print(f"\nAfter Gemini verification:")
            print(f"  Borderline → KEEP:   {len(new_keep)}")
            print(f"  Borderline → REMOVE: {len(new_remove)}")
            borderline_list = []
    elif borderline_list:
        # No Gemini verification → keep all borderline (conservative)
        print(f"\nNo Gemini verification — keeping all {len(borderline_list)} borderline samples.")
        for r in borderline_list:
            r['verdict'] = 'keep'
        keep_list.extend(borderline_list)
        borderline_list = []

    # ─── Final counts ───
    keep_indices = set(r['idx'] for r in keep_list)
    remove_indices = set(r['idx'] for r in remove_list)

    keep_hours = sum(entries[i].get('duration', 0) for i in keep_indices) / 3600
    remove_hours = sum(entries[i].get('duration', 0) for i in remove_indices) / 3600

    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"  KEEP:   {len(keep_indices):>6} samples  ({keep_hours:.1f} hours)")
    print(f"  REMOVE: {len(remove_indices):>6} samples  ({remove_hours:.1f} hours)")

    # Show removed samples
    remove_list.sort(key=lambda x: -x['cer'])
    print(f"\nTop 15 removed samples (by CER):")
    for r in remove_list[:15]:
        print(f"  cer={r['cer']:.2f}  loss/s={r['loss_per_sec']:.1f}  dur={r['duration']:.1f}s  "
              f"| ref: {r['ref'][:50]}")
        print(f"  {'':>50}  | hyp: {r['hyp'][:50]}")

    # ─── Write output files ───
    # Filtered manifest (kept samples)
    filtered_path = os.path.join(args.output_dir, 'train_filtered.json')
    with open(filtered_path, 'w') as f:
        for i, entry in enumerate(entries):
            if i in keep_indices:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Removed manifest
    removed_path = os.path.join(args.output_dir, 'train_removed.json')
    with open(removed_path, 'w') as f:
        for i, entry in enumerate(entries):
            if i in remove_indices:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Full scores
    scores_path = os.path.join(args.output_dir, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Summary
    summary = {
        'total_samples': len(entries),
        'kept': len(keep_indices),
        'removed': len(remove_indices),
        'kept_hours': round(keep_hours, 1),
        'removed_hours': round(remove_hours, 1),
        'mean_cer_kept': round(float(np.mean([r['cer'] for r in keep_list])), 4),
        'mean_cer_removed': round(float(np.mean([r['cer'] for r in remove_list])), 4) if remove_list else 0,
        'loss_p75_threshold': round(float(loss_p75), 4),
        'gemini_verified': args.gemini_verify,
    }
    summary_path = os.path.join(args.output_dir, 'filter_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutput files in {args.output_dir}:")
    print(f"  train_filtered.json  — {len(keep_indices)} samples to train on")
    print(f"  train_removed.json   — {len(remove_indices)} samples removed")
    print(f"  scores.json          — full per-utterance scores")
    print(f"  filter_summary.json  — summary statistics")


if __name__ == '__main__':
    main()
