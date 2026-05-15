#!/usr/bin/env python3
"""
eval_full.py — Comprehensive ASR + tag evaluation for Meta-ASR models.

Phase A: Run inference on a JSONL manifest → write predictions.jsonl
Phase B: Compute WER, CER, per-tag accuracy, confusion matrices → write Excel + JSON summary

Usage:
    python eval_full.py \
        --model /path/to/model.nemo \
        --manifest /path/to/valid.cleaned.json \
        --output-dir /path/to/eval_results/ \
        [--batch-size 32] [--cpu] [--skip-inference]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio

project_root = os.environ.get('PROMPTINGNEMO_ROOT', '/mnt/nfs/code/PromptingNemo')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE

sys.path.insert(0, os.path.join(project_root, 'scripts', 'asr'))
from meta_asr.tag_classifier import (
    build_trailing_tag_maps,
    build_all_special_token_ids,
    masked_mean_pool,
    TrailingTagClassifier,
)

# ---------------------------------------------------------------------------
# Tag regex patterns
# ---------------------------------------------------------------------------
TAG_PATTERNS = {
    "AGE":     re.compile(r"\b(AGE_\S+)\b"),
    "GENDER":  re.compile(r"\b(GENDER_\S+)\b"),
    "EMOTION": re.compile(r"\b(EMOTION_\S+)\b"),
    "INTENT":  re.compile(r"\b(INTENT_\S+)\b"),
}

ENTITY_RE = re.compile(r"\b(ENTITY_\S+)\b")
END_RE = re.compile(r"\bEND\b")

ALL_TAG_RE = re.compile(
    r"\b(?:AGE_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+|ENTITY_\S+|KEYWORD_\S+|LANG_\S+|DIALECT_\S+|END)\b"
)


def strip_tags(text: str) -> str:
    return " ".join(ALL_TAG_RE.sub("", text).split())


# ---------------------------------------------------------------------------
# WER / CER (pure Python, no jiwer dependency needed)
# ---------------------------------------------------------------------------
def _edit_distance(ref_tokens, hyp_tokens):
    n, m = len(ref_tokens), len(hyp_tokens)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m]


def compute_wer(refs, hyps):
    total_errors = total_words = 0
    for ref, hyp in zip(refs, hyps):
        ref_w = ref.split()
        total_words += len(ref_w)
        total_errors += _edit_distance(ref_w, hyp.split())
    return total_errors / max(total_words, 1)


def compute_cer(refs, hyps):
    total_errors = total_chars = 0
    for ref, hyp in zip(refs, hyps):
        ref_c = list(ref.replace(" ", ""))
        hyp_c = list(hyp.replace(" ", ""))
        total_chars += len(ref_c)
        total_errors += _edit_distance(ref_c, hyp_c)
    return total_errors / max(total_chars, 1)


# ---------------------------------------------------------------------------
# Classification metrics (no sklearn dependency)
# ---------------------------------------------------------------------------
def _classification_metrics(y_true, y_pred, labels):
    """Compute per-class precision, recall, F1 and confusion matrix."""
    label_set = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(label_set)}
    n = len(label_set)
    conf = [[0] * n for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(t)
        pi = label_to_idx.get(p)
        if ti is not None and pi is not None:
            conf[ti][pi] += 1

    per_class = {}
    for i, label in enumerate(label_set):
        tp = conf[i][i]
        fp = sum(conf[j][i] for j in range(n)) - tp
        fn = sum(conf[i][j] for j in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(conf[i])
        per_class[label] = {"precision": prec, "recall": rec, "f1": f1, "support": support}

    total_correct = sum(conf[i][i] for i in range(n))
    total_samples = sum(sum(row) for row in conf)
    accuracy = total_correct / max(total_samples, 1)

    macro_f1 = sum(v["f1"] for v in per_class.values()) / max(len(per_class), 1)
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / max(total_samples, 1)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": conf,
        "labels": label_set,
        "total_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# Entity span extraction
# ---------------------------------------------------------------------------
def extract_entity_spans(text):
    """Extract (ENTITY_TYPE, surface_text) tuples from inline entity annotations.
    Format: ENTITY_TYPE surface text END
    """
    spans = []
    tokens = text.split()
    i = 0
    while i < len(tokens):
        if ENTITY_RE.match(tokens[i]):
            etype = tokens[i]
            surface = []
            i += 1
            while i < len(tokens) and tokens[i] != "END" and not ENTITY_RE.match(tokens[i]):
                if not any(tokens[i].startswith(p) for p in ["AGE_", "GENDER_", "EMOTION_", "INTENT_"]):
                    surface.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i] == "END":
                i += 1
            spans.append((etype, " ".join(surface)))
        else:
            i += 1
    return spans


# ---------------------------------------------------------------------------
# Phase A: Inference
# ---------------------------------------------------------------------------
def _setup_tag_classifier(model, nemo_path):
    """Set up tag classifier from .nemo checkpoint weights.

    The tag classifier is not restored by restore_from() — we need to
    manually create it and load the weights from the checkpoint.
    """
    import tarfile, io

    # Extract tag_classifier weights from .nemo archive
    tc_state = {}
    with tarfile.open(nemo_path) as t:
        for m in t.getmembers():
            if 'model_weights' in m.name:
                f = t.extractfile(m)
                state = torch.load(io.BytesIO(f.read()), map_location='cpu')
                tc_state = {k: v for k, v in state.items() if 'tag_classifier' in k}
                break

    if not tc_state:
        return None

    # Determine categories and sizes from weight shapes
    category_sizes = {}
    for key, tensor in tc_state.items():
        if key.endswith('.weight'):
            cat_name = key.split('.')[2]  # tag_classifier.heads.AGE.weight
            category_sizes[cat_name] = tensor.shape[0]  # num_classes

    vocab = model.decoder.vocabulary
    encoder_dim = model.encoder.d_model

    # Build the classifier
    classifier = TrailingTagClassifier(encoder_dim, category_sizes)
    classifier.load_state_dict({k.replace('tag_classifier.', ''): v for k, v in tc_state.items()})
    classifier.eval()

    device = next(model.parameters()).device
    classifier = classifier.to(device)
    model.tag_classifier = classifier

    # Register forward hook on encoder to capture output
    def _capture_encoder_output(module, input, output):
        if isinstance(output, tuple):
            model._last_encoder_output = output[0]
        else:
            model._last_encoder_output = output
    model._encoder_hook = model.encoder.register_forward_hook(_capture_encoder_output)
    model._last_encoder_output = None

    # Build reverse mappings
    _, category_to_id, _ = build_trailing_tag_maps(vocab, categories=sorted(category_sizes.keys()))

    id_to_label = {}
    for cat_name in sorted(category_sizes.keys()):
        cat_map = category_to_id.get(cat_name, {})
        reverse = {0: "NONE"}
        for vocab_id, class_idx in cat_map.items():
            token = vocab[vocab_id].lstrip('▁')
            reverse[class_idx] = token
        id_to_label[cat_name] = reverse

    category_names = sorted(category_sizes.keys())
    print(f"  Tag classifier set up: {category_names}")
    for cat in category_names:
        labels = [id_to_label[cat][i] for i in sorted(id_to_label[cat].keys())]
        print(f"    {cat}: {labels}")

    return {"category_names": category_names, "id_to_label": id_to_label}


def run_inference(model_path, manifest_path, output_path, device, batch_size):
    print(f"Loading model from {model_path}... (device={device})")
    model = CustomEncDecCTCModelBPE.restore_from(model_path, map_location='cpu', strict=False)
    if device == 'cuda':
        model = model.cuda()
    model.eval()

    tag_maps = _setup_tag_classifier(model, model_path)
    has_tag_cls = tag_maps is not None

    entries = []
    with open(manifest_path) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Running inference on {len(entries)} samples...")
    results = []

    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        waveforms = []
        lengths = []

        for entry in batch:
            waveform, sr = torchaudio.load(entry['audio_filepath'])
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform.mean(dim=0)
            waveforms.append(waveform)
            lengths.append(waveform.shape[0])

        max_len = max(lengths)
        padded = torch.zeros(len(batch), max_len)
        for j, w in enumerate(waveforms):
            padded[j, :w.shape[0]] = w

        input_signal = padded.to(device)
        input_length = torch.tensor(lengths, device=device)

        with torch.no_grad():
            log_probs, encoded_len, _ = model.forward(
                input_signal=input_signal,
                input_signal_length=input_length,
            )
            decode_result = model.decoding.ctc_decoder_predictions_tensor(
                log_probs, decoder_lengths=encoded_len, return_hypotheses=False,
            )
            best_hyps = decode_result[0] if isinstance(decode_result, tuple) else decode_result

            tag_preds_batch = {}
            if has_tag_cls and hasattr(model, '_last_encoder_output') and model._last_encoder_output is not None:
                encoder_out = model._last_encoder_output  # [B, D, T] from NeMo
                B_enc, D_enc, T_enc = encoder_out.shape
                # Manual masked mean pool — bypass masked_mean_pool's broken heuristic
                time_mask = torch.arange(T_enc, device=encoder_out.device).unsqueeze(0) < encoded_len.unsqueeze(1)  # [B, T]
                enc_btd = encoder_out.transpose(1, 2)  # [B, T, D]
                mask_3d = time_mask.unsqueeze(-1).float()  # [B, T, 1]
                pooled = (enc_btd * mask_3d).sum(dim=1) / mask_3d.sum(dim=1).clamp(min=1)  # [B, D]
                tag_logits = model.tag_classifier(pooled)
                for cat_name in tag_maps["category_names"]:
                    class_ids = tag_logits[cat_name].argmax(dim=-1).cpu().tolist()
                    tag_preds_batch[cat_name] = [
                        tag_maps["id_to_label"][cat_name].get(cid, "NONE") for cid in class_ids
                    ]

        for j, entry in enumerate(batch):
            raw = best_hyps[j] if j < len(best_hyps) else ""
            pred_text = raw.text if hasattr(raw, 'text') else str(raw)
            result = {
                "audio_filepath": entry['audio_filepath'],
                "text": entry.get('text', ''),
                "predicted_text": pred_text,
                "duration": entry.get('duration', 0),
            }
            if tag_preds_batch:
                result["tag_preds"] = {cat: tag_preds_batch[cat][j] for cat in tag_preds_batch}
            results.append(result)

        done = min(i + batch_size, len(entries))
        if done % 500 < batch_size or done == len(entries):
            print(f"  [{done}/{len(entries)}]")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"Wrote {len(results)} predictions to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Phase B: Metrics
# ---------------------------------------------------------------------------
def compute_all_metrics(predictions):
    refs_clean = []
    preds_clean = []

    tag_true = defaultdict(list)
    tag_pred = defaultdict(list)

    entity_true_spans = []
    entity_pred_spans = []

    has_tag_cls = any("tag_preds" in entry for entry in predictions[:1])

    for entry in predictions:
        ref = entry['text']
        pred = entry['predicted_text']

        refs_clean.append(strip_tags(ref))
        preds_clean.append(strip_tags(pred))

        for tag_name, pattern in TAG_PATTERNS.items():
            m_ref = pattern.search(ref)
            ref_label = m_ref.group(1) if m_ref else "NONE"

            if has_tag_cls and "tag_preds" in entry and tag_name in entry["tag_preds"]:
                pred_label = entry["tag_preds"][tag_name]
            else:
                m_pred = pattern.search(pred)
                pred_label = m_pred.group(1) if m_pred else "NONE"

            tag_true[tag_name].append(ref_label)
            tag_pred[tag_name].append(pred_label)

        entity_true_spans.append(set(extract_entity_spans(ref)))
        entity_pred_spans.append(set(extract_entity_spans(pred)))

    # ASR metrics
    wer_val = compute_wer(refs_clean, preds_clean)
    cer_val = compute_cer(refs_clean, preds_clean)

    print(f"\n{'=' * 60}")
    print(f"Clean WER (tags stripped): {wer_val:.4f}")
    print(f"Clean CER (tags stripped): {cer_val:.4f}")
    print(f"Samples: {len(predictions)}")
    print(f"{'=' * 60}")

    # Tag classification metrics
    tag_results = {}
    for tag_name in TAG_PATTERNS:
        y_true = tag_true[tag_name]
        y_pred = tag_pred[tag_name]

        pairs_with_label = [(t, p) for t, p in zip(y_true, y_pred) if t != "NONE" or p != "NONE"]
        if not pairs_with_label:
            continue

        yt, yp = zip(*pairs_with_label)
        all_labels = sorted(set(yt) | set(yp))
        metrics = _classification_metrics(list(yt), list(yp), all_labels)
        tag_results[tag_name] = metrics

        print(f"\n--- {tag_name} ---")
        print(f"  Accuracy: {metrics['accuracy']:.4f}  |  Macro F1: {metrics['macro_f1']:.4f}  |  Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"  Samples with label: {metrics['total_samples']}")
        print(f"  {'Label':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
        for label in metrics['labels']:
            pc = metrics['per_class'][label]
            print(f"  {label:<25} {pc['precision']:>6.3f} {pc['recall']:>6.3f} {pc['f1']:>6.3f} {pc['support']:>8}")

        # Print confusion matrix
        print(f"\n  Confusion Matrix ({tag_name}):")
        labels = metrics['labels']
        header = "  " + " " * 20 + "".join(f"{l[-12:]:>13}" for l in labels)
        print(header)
        for ri, row_label in enumerate(labels):
            row_str = f"  {row_label:<20}" + "".join(f"{metrics['confusion_matrix'][ri][ci]:>13}" for ci in range(len(labels)))
            print(row_str)

    # Entity detection metrics
    entity_tp = entity_fp = entity_fn = 0
    entity_type_tp = defaultdict(int)
    entity_type_fp = defaultdict(int)
    entity_type_fn = defaultdict(int)

    for true_set, pred_set in zip(entity_true_spans, entity_pred_spans):
        true_types = {etype for etype, _ in true_set}
        pred_types = {etype for etype, _ in pred_set}
        for etype in true_types & pred_types:
            entity_type_tp[etype] += 1
            entity_tp += 1
        for etype in pred_types - true_types:
            entity_type_fp[etype] += 1
            entity_fp += 1
        for etype in true_types - pred_types:
            entity_type_fn[etype] += 1
            entity_fn += 1

    ent_prec = entity_tp / max(entity_tp + entity_fp, 1)
    ent_rec = entity_tp / max(entity_tp + entity_fn, 1)
    ent_f1 = 2 * ent_prec * ent_rec / max(ent_prec + ent_rec, 1e-9)

    print(f"\n--- ENTITY (type detection) ---")
    print(f"  Precision: {ent_prec:.4f}  |  Recall: {ent_rec:.4f}  |  F1: {ent_f1:.4f}")

    top_entities = sorted(
        set(entity_type_tp) | set(entity_type_fn),
        key=lambda e: entity_type_tp.get(e, 0) + entity_type_fn.get(e, 0),
        reverse=True,
    )[:20]
    if top_entities:
        print(f"\n  Top entity types:")
        print(f"  {'Type':<30} {'TP':>6} {'FP':>6} {'FN':>6} {'F1':>8}")
        for etype in top_entities:
            tp = entity_type_tp.get(etype, 0)
            fp = entity_type_fp.get(etype, 0)
            fn = entity_type_fn.get(etype, 0)
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f = 2 * p * r / max(p + r, 1e-9)
            print(f"  {etype:<30} {tp:>6} {fp:>6} {fn:>6} {f:>8.3f}")

    # Build summary
    summary = {
        "wer": round(wer_val, 4),
        "cer": round(cer_val, 4),
        "samples": len(predictions),
        "tags": {},
        "entity": {
            "precision": round(ent_prec, 4),
            "recall": round(ent_rec, 4),
            "f1": round(ent_f1, 4),
        },
    }
    for tag_name, metrics in tag_results.items():
        summary["tags"][tag_name] = {
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "per_class": {
                label: {k: round(v, 4) for k, v in vals.items()}
                for label, vals in metrics["per_class"].items()
            },
            "confusion_matrix": metrics["confusion_matrix"],
            "labels": metrics["labels"],
        }

    return summary, tag_results


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------
def export_excel(output_path, summary, tag_results):
    try:
        import xlsxwriter
    except ImportError:
        print("xlsxwriter not available, skipping Excel export")
        return

    wb = xlsxwriter.Workbook(str(output_path))
    bold = wb.add_format({"bold": True})
    pct = wb.add_format({"num_format": "0.00%"})

    # Summary sheet
    ws = wb.add_worksheet("Summary")
    ws.write(0, 0, "Metric", bold)
    ws.write(0, 1, "Value", bold)
    ws.write(1, 0, "WER")
    ws.write(1, 1, summary["wer"], pct)
    ws.write(2, 0, "CER")
    ws.write(2, 1, summary["cer"], pct)
    ws.write(3, 0, "Samples")
    ws.write(3, 1, summary["samples"])

    row = 5
    ws.write(row, 0, "Tag Category", bold)
    ws.write(row, 1, "Accuracy", bold)
    ws.write(row, 2, "Macro F1", bold)
    ws.write(row, 3, "Weighted F1", bold)
    row += 1
    for tag_name, tag_data in sorted(summary["tags"].items()):
        ws.write(row, 0, tag_name)
        ws.write(row, 1, tag_data["accuracy"], pct)
        ws.write(row, 2, tag_data["macro_f1"], pct)
        ws.write(row, 3, tag_data["weighted_f1"], pct)
        row += 1

    row += 1
    ws.write(row, 0, "Entity Detection", bold)
    row += 1
    ws.write(row, 0, "Precision")
    ws.write(row, 1, summary["entity"]["precision"], pct)
    row += 1
    ws.write(row, 0, "Recall")
    ws.write(row, 1, summary["entity"]["recall"], pct)
    row += 1
    ws.write(row, 0, "F1")
    ws.write(row, 1, summary["entity"]["f1"], pct)

    # Per-tag sheets
    for tag_name, metrics in sorted(tag_results.items()):
        labels = metrics["labels"]
        conf = metrics["confusion_matrix"]

        # Confusion matrix sheet
        ws = wb.add_worksheet(f"{tag_name}_matrix")
        ws.write(0, 0, f"{tag_name} Confusion Matrix", bold)
        ws.write(1, 0, "Actual \\ Predicted", bold)
        for ci, label in enumerate(labels):
            ws.write(1, ci + 1, label, bold)
        for ri, row_label in enumerate(labels):
            ws.write(ri + 2, 0, row_label, bold)
            for ci in range(len(labels)):
                ws.write(ri + 2, ci + 1, conf[ri][ci])

        # Classification report sheet
        ws = wb.add_worksheet(f"{tag_name}_report")
        ws.write(0, 0, "Label", bold)
        ws.write(0, 1, "Precision", bold)
        ws.write(0, 2, "Recall", bold)
        ws.write(0, 3, "F1", bold)
        ws.write(0, 4, "Support", bold)
        for ri, label in enumerate(labels):
            pc = metrics["per_class"][label]
            ws.write(ri + 1, 0, label)
            ws.write(ri + 1, 1, pc["precision"], pct)
            ws.write(ri + 1, 2, pc["recall"], pct)
            ws.write(ri + 1, 3, pc["f1"], pct)
            ws.write(ri + 1, 4, pc["support"])

    wb.close()
    print(f"\nExcel workbook saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full ASR + tag evaluation")
    parser.add_argument("--model", required=True, help="Path to .nemo model")
    parser.add_argument("--manifest", required=True, help="JSONL manifest with text + audio_filepath")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference, read existing predictions.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    # Phase A: Inference
    if args.skip_inference and predictions_path.exists():
        print(f"Skipping inference, loading {predictions_path}")
        predictions = []
        with open(predictions_path) as f:
            for line in f:
                predictions.append(json.loads(line))
    else:
        device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
        predictions = run_inference(
            args.model, args.manifest, str(predictions_path), device, args.batch_size,
        )

    # Phase B: Metrics
    summary, tag_results = compute_all_metrics(predictions)

    # Write JSON summary
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")

    # Write Excel
    excel_path = output_dir / "eval_results.xlsx"
    export_excel(excel_path, summary, tag_results)

    # Print sample predictions
    print("\nSample predictions (first 5):")
    for entry in predictions[:5]:
        ref_clean = strip_tags(entry['text'])
        pred_clean = strip_tags(entry['predicted_text'])
        print(f"\n  REF:  {ref_clean[:150]}")
        print(f"  PRED: {pred_clean[:150]}")
        if entry['predicted_text'] != pred_clean:
            print(f"  TAGS: {entry['predicted_text'][:150]}")


if __name__ == "__main__":
    main()
