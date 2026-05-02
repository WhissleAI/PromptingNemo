"""Compare fine-tuned Hindi model WER against Whissle gateway (STT-Meta-1B) baseline.

Usage:
    python scripts/asr/meta-asr/eval_compare.py \
        --model /path/to/finetuned.nemo \
        --manifest /path/to/valid.cleaned.json \
        --n 500 \
        --api-token wh_...
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sdk_root = os.path.abspath(os.path.join(project_root, '..', 'whissle_python_api'))
if os.path.isdir(sdk_root) and sdk_root not in sys.path:
    sys.path.insert(0, sdk_root)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s %(asctime)s] %(message)s')


def load_local_model(nemo_path: str, device: str = 'cuda'):
    from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE
    model = CustomEncDecCTCModelBPE.restore_from(nemo_path, map_location=device, strict=False)
    model.eval()
    return model.to(device)


def transcribe_local(model, audio_path: str) -> str:
    result = model.transcribe([audio_path])
    t = result[0] if isinstance(result, list) else result
    if isinstance(t, list):
        t = t[0]
    if hasattr(t, 'text'):
        t = t.text
    return str(t)


def transcribe_api(client, audio_path: str) -> str:
    try:
        result = client.transcribe_file(audio_path)
        if isinstance(result, dict):
            return result.get('transcript', result.get('text', str(result)))
        return str(result)
    except Exception as e:
        logging.warning("API error for %s: %s", audio_path, e)
        return ""


def strip_meta_tags(text: str) -> str:
    """Remove meta-tags (AGE_*, GENDER_*, EMOTION_*, INTENT_*, ENTITY_*, END, LANG_*) for text-only WER."""
    import re
    tags = re.compile(
        r'\b(AGE_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+|ENTITY_\S+|KEYWORD_\S+|LANG_\S+|END)\b'
    )
    cleaned = tags.sub('', text)
    return ' '.join(cleaned.split())


def compute_wer(hyps, refs):
    from nemo.collections.asr.metrics.wer import word_error_rate_detail
    result = word_error_rate_detail(hyps, refs)
    return result[0], result[1]


def main():
    parser = argparse.ArgumentParser(description="Compare fine-tuned model vs Whissle API baseline")
    parser.add_argument("--model", required=True, help="Path to fine-tuned .nemo model")
    parser.add_argument("--manifest", required=True, help="Path to evaluation manifest (JSONL)")
    parser.add_argument("--n", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--api-token", default=None, help="Whissle API token (or set WHISSLE_API_TOKEN)")
    parser.add_argument("--api-url", default="https://api.whissle.ai", help="Whissle API base URL")
    parser.add_argument("--skip-api", action="store_true", help="Skip API comparison, only eval local model")
    parser.add_argument("--text-only-wer", action="store_true", help="Also compute WER with meta-tags stripped")
    parser.add_argument("--output", default=None, help="Save detailed results to JSON file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    samples = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    logging.info("Loaded %d samples from %s", len(samples), args.manifest)

    if args.n < len(samples):
        samples = random.sample(samples, args.n)
    logging.info("Evaluating on %d samples", len(samples))

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info("Loading local model from %s", args.model)
    model = load_local_model(args.model, device)

    client = None
    if not args.skip_api:
        try:
            from whissle.gateway_client import WhissleGatewayClient
            token = args.api_token or os.getenv("WHISSLE_API_TOKEN") or os.getenv("WHISSLE_AUTH_TOKEN")
            if token:
                client = WhissleGatewayClient(api_token=token, base_url=args.api_url, timeout=180.0)
                logging.info("Connected to Whissle API at %s", args.api_url)
            else:
                logging.warning("No API token — skipping API comparison")
        except ImportError:
            logging.warning("whissle SDK not found — skipping API comparison")

    refs = []
    local_hyps = []
    api_hyps = []
    results = []

    for i, s in enumerate(samples):
        audio_path = s['audio_filepath']
        ref = s['text']

        local_hyp = transcribe_local(model, audio_path)
        local_hyps.append(local_hyp)
        refs.append(ref)

        api_hyp = ""
        if client:
            api_hyp = transcribe_api(client, audio_path)
            api_hyps.append(api_hyp)
            time.sleep(0.1)

        results.append({
            'audio': audio_path,
            'ref': ref,
            'local_hyp': local_hyp,
            'api_hyp': api_hyp,
        })

        if (i + 1) % 50 == 0:
            logging.info("Processed %d/%d samples", i + 1, len(samples))

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    local_wer, local_words = compute_wer(local_hyps, refs)
    print(f"\nFine-tuned model ({Path(args.model).stem}):")
    print(f"  WER:  {local_wer:.4f} ({(1-local_wer)*100:.1f}% accuracy)")
    print(f"  Words: {local_words}")

    if api_hyps:
        api_wer, api_words = compute_wer(api_hyps, refs)
        print(f"\nWhissle API (STT-Meta-1B baseline):")
        print(f"  WER:  {api_wer:.4f} ({(1-api_wer)*100:.1f}% accuracy)")
        print(f"  Words: {api_words}")

        delta = api_wer - local_wer
        print(f"\nDelta (API - Local): {delta:+.4f} {'(local is better)' if delta > 0 else '(API is better)'}")

    if args.text_only_wer:
        print("\n--- Text-only WER (meta-tags stripped) ---")
        refs_clean = [strip_meta_tags(r) for r in refs]
        local_clean = [strip_meta_tags(h) for h in local_hyps]
        local_text_wer, local_text_words = compute_wer(local_clean, refs_clean)
        print(f"  Fine-tuned (text only): WER={local_text_wer:.4f} ({(1-local_text_wer)*100:.1f}%)")

        if api_hyps:
            api_clean = [strip_meta_tags(h) for h in api_hyps]
            api_text_wer, api_text_words = compute_wer(api_clean, refs_clean)
            print(f"  API (text only):        WER={api_text_wer:.4f} ({(1-api_text_wer)*100:.1f}%)")

    print("\n--- Sample predictions ---")
    for r in results[:5]:
        print(f"  REF: {r['ref'][:100]}")
        print(f"  LOC: {r['local_hyp'][:100]}")
        if r['api_hyp']:
            print(f"  API: {r['api_hyp'][:100]}")
        print()

    if args.output:
        summary = {
            'model': args.model,
            'manifest': args.manifest,
            'n_samples': len(samples),
            'local_wer': local_wer,
            'local_words': local_words,
        }
        if api_hyps:
            summary['api_wer'] = api_wer
            summary['api_words'] = api_words
        summary['results'] = results

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logging.info("Saved detailed results to %s", args.output)

    if client:
        client.close()


if __name__ == "__main__":
    main()
