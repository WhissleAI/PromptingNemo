"""Quick standalone WER check — bypasses NeMo's WER metric entirely.

Loads a .nemo checkpoint, decodes N samples with greedy CTC, strips tags,
and computes WER on the clean text. Use this to verify whether val_wer=1.0
is a metric bug or a real model problem.

Usage:
    python scripts/asr/meta-asr/quick_wer_check.py \
        --nemo /path/to/model.nemo \
        --manifest /path/to/valid.json \
        --n 20
"""

import argparse
import json
import os
import re
import sys

import torch
import soundfile as sf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

TAG_RE = re.compile(
    r'\s*(AGE_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+|ENTITY_\S+|DIALECT_\S+|'
    r'KEYWORD_\S+|LANG_\S+|END)\s*'
)


def strip_tags(text):
    return ' '.join(TAG_RE.sub(' ', text).split()).strip()


def edit_distance(ref_words, hyp_words):
    r, h = len(ref_words), len(hyp_words)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[r][h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--n', type=int, default=20)
    args = parser.parse_args()

    device = 'cpu'  # Always use CPU — GPU may be busy with training

    print(f"Loading model from {args.nemo} on CPU ...")
    from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE
    model = CustomEncDecCTCModelBPE.restore_from(args.nemo, map_location=device, strict=False)
    model = model.to(device)
    model.eval()

    vocab = list(model.decoder.vocabulary)
    blank_id = len(vocab)
    print(f"Vocab size: {len(vocab)}, blank_id: {blank_id}")

    entries = []
    with open(args.manifest) as f:
        for line in f:
            e = json.loads(line.strip())
            if 0.5 <= e.get('duration', 0) <= 20.0:
                entries.append(e)
            if len(entries) >= args.n:
                break

    print(f"Evaluating {len(entries)} samples\n")

    total_errors = 0
    total_words = 0
    total_errors_with_tags = 0
    total_words_with_tags = 0

    for i, entry in enumerate(entries):
        audio_path = entry['audio_filepath']
        ref_raw = entry.get('text', '')
        ref_clean = strip_tags(ref_raw)

        try:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
        except Exception as e:
            print(f"  SKIP {audio_path}: {e}")
            continue

        audio_t = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
        audio_len = torch.tensor([len(audio)], dtype=torch.long, device=device)

        with torch.no_grad():
            processed, processed_len = model.preprocessor(
                input_signal=audio_t, length=audio_len
            )
            encoded, encoded_len = model.encoder(
                audio_signal=processed, length=processed_len
            )
            log_probs = model.decoder(encoder_output=encoded)
            preds = log_probs.argmax(dim=-1)[0, :encoded_len[0]].cpu().tolist()

        # Greedy CTC decode
        decoded = []
        prev = None
        for tid in preds:
            if tid != blank_id and tid != prev and tid < len(vocab):
                decoded.append(vocab[tid])
            prev = tid
        hyp_raw = ''.join(decoded).replace('▁', ' ').strip()
        hyp_clean = strip_tags(hyp_raw)

        # WER on clean text
        ref_words = ref_clean.split()
        hyp_words = hyp_clean.split()
        errors = edit_distance(ref_words, hyp_words)
        total_errors += errors
        total_words += len(ref_words)

        # WER with tags (the buggy metric)
        ref_tag_words = ref_raw.replace('▁', ' ').split()
        hyp_tag_words = hyp_raw.split()
        errors_t = edit_distance(ref_tag_words, hyp_tag_words)
        total_errors_with_tags += errors_t
        total_words_with_tags += len(ref_tag_words)

        wer_sample = errors / max(len(ref_words), 1)
        print(f"[{i:3d}] WER={wer_sample:.2f} ({errors}/{len(ref_words)})")
        print(f"      REF: {ref_clean[:100]}")
        print(f"      HYP: {hyp_clean[:100]}")
        if i < 3:
            print(f"      RAW HYP: {hyp_raw[:120]}")
        print()

    wer_clean = total_errors / max(total_words, 1)
    wer_tags = total_errors_with_tags / max(total_words_with_tags, 1)

    print("=" * 60)
    print(f"RESULTS ({len(entries)} samples)")
    print(f"  WER (clean, tags stripped): {wer_clean:.4f}  ({total_errors}/{total_words})")
    print(f"  WER (with tags, buggy):     {wer_tags:.4f}  ({total_errors_with_tags}/{total_words_with_tags})")
    print("=" * 60)


if __name__ == '__main__':
    main()
