#!/usr/bin/env python3
"""
create_tag_eval_xlsx.py
=======================

Evaluate inline‑tag quality *and* ASR quality from a JSONL manifest.

INPUT  (per line)
------
{
  "text":           "... AGE_18_30 GER_FEMALE EMOTION_NEU INTENT_INFORM",
  "predicted_text": "... AGE_18_30 GER_FEMALE EMOTION_NEU INTENT_INFORM",
  ...
}

OUTPUT
------
Excel workbook (.xlsx) with these sheets
  • <TAG>_matrix   : square confusion matrix for AGE / GER / EMOTION / INTENT
  • <TAG>_report   : full sklearn classification report
  • Summary        : Accuracy, Macro‑F1, Micro‑F1, Weighted‑F1 for every tag
                     + overall WER & CER after stripping tags

Usage
-----
python create_tag_eval_xlsx.py \
       --input  /path/to/valid_withpredictions.jsonl \
       --output tag_evaluation_results.xlsx

Dependencies
------------
pip install pandas openpyxl xlsxwriter scikit-learn jiwer
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from jiwer import wer, cer
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# --------------------------------------------------------------------------- #
# 1. Command‑line args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate tag & ASR quality -> XLSX")
    p.add_argument("--input", type=Path, required=True, help="Input .jsonl")
    p.add_argument("--output", type=Path, required=True, help="Output .xlsx")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 2. Regex patterns
# --------------------------------------------------------------------------- #
PATTERNS = {
    "AGE":     re.compile(r"\b(AGE_\d+_\d+)\b"),
    "GER":     re.compile(r"\b(GER_[A-Z]+)\b"),
    "EMOTION": re.compile(r"\b(EMOTION_[A-Z]+)\b"),
    "INTENT":  re.compile(r"\b(INTENT_[A-Z]+)\b"),
}

TAG_REMOVE_REGEX = re.compile(
    r"\b(?:AGE_\d+_\d+|GER_[A-Z]+|EMOTION_[A-Z]+|INTENT_[A-Z]+)\b"
)


def strip_tags(text: str) -> str:
    """Remove all tag tokens and normalise whitespace."""
    return " ".join(TAG_REMOVE_REGEX.sub("", text).split())


# --------------------------------------------------------------------------- #
# 3. Read JSONL and collect data
# --------------------------------------------------------------------------- #
def extract(jsonl_path: Path):
    orig_tags = defaultdict(list)     # ground‑truth tags per family
    pred_tags = defaultdict(list)     # predicted tags per family
    gt_clean, pr_clean = [], []       # tag‑stripped transcripts

    with jsonl_path.open("r", encoding="utf‑8") as fh:
        for line in fh:
            obj = json.loads(line)
            gt_raw = obj.get("text", "")
            pr_raw = obj.get("predicted_text", "")

            # tag extraction
            for tag, pat in PATTERNS.items():
                m_gt = pat.search(gt_raw)
                m_pr = pat.search(pr_raw)
                orig_tags[tag].append(m_gt.group(1) if m_gt else None)
                pred_tags[tag].append(m_pr.group(1) if m_pr else None)

            # tag‑stripped text
            gt_clean.append(strip_tags(gt_raw.lower()))
            pr_clean.append(strip_tags(pr_raw.lower()))

    return orig_tags, pred_tags, gt_clean, pr_clean


# --------------------------------------------------------------------------- #
# 4. Build matrices & reports
# --------------------------------------------------------------------------- #
def build_results(orig_tags, pred_tags):
    dfs_conf, dfs_report = {}, {}
    summary_rows = []

    for tag_family in PATTERNS.keys():
        y_true_all = orig_tags[tag_family]
        y_pred_all = pred_tags[tag_family]

        # complete label set (union from both sides)
        labels = sorted(
            {l for l in y_true_all if l is not None}
            | {l for l in y_pred_all if l is not None}
        )
        if not labels:      # nothing to evaluate
            continue

        # keep rows where we have *both* actual & predicted labels
        pairs = [(a, p) for a, p in zip(y_true_all, y_pred_all) if a and p]
        if not pairs:
            continue
        y_true, y_pred = zip(*pairs)

        # 4a. confusion matrix, forced square
        df_conf = (
            pd.crosstab(
                pd.Series(y_true, name="Actual"),
                pd.Series(y_pred, name="Predicted"),
                dropna=False,
            )
            .reindex(index=labels, columns=labels, fill_value=0)
        )
        dfs_conf[tag_family] = df_conf

        # 4b. classification report (remove 'accuracy' row → re‑insert cleanly)
        rpt_dict = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        acc_val = rpt_dict.pop("accuracy")  # scalar

        df_rpt = pd.DataFrame(rpt_dict).T
        # insert accuracy as its own row with value only in 'accuracy' column
        df_rpt.loc["accuracy", :] = 0.0
        df_rpt.at["accuracy", "accuracy"] = acc_val
        dfs_report[tag_family] = df_rpt

        # 4c. summary metrics
        summary_rows.append(
            {
                "Tag": tag_family,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Macro F1": f1_score(y_true, y_pred, average="macro"),
                "Micro F1": f1_score(y_true, y_pred, average="micro"),
                "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
            }
        )

    return dfs_conf, dfs_report, summary_rows


# --------------------------------------------------------------------------- #
# 5. Excel writer
# --------------------------------------------------------------------------- #
def export_excel(
    dfs_conf, dfs_report, summary_rows, wer_val, cer_val, outfile: Path
):
    # add ASR row
    summary_rows.append(
        {
            "Tag": "ASR",
            "Accuracy": None,
            "Macro F1": None,
            "Micro F1": None,
            "Weighted F1": None,
            "WER": wer_val,
            "CER": cer_val,
        }
    )

    summary_df = pd.DataFrame(summary_rows).set_index("Tag")

    with pd.ExcelWriter(outfile, engine="xlsxwriter") as xls:
        for tag, df in dfs_conf.items():
            df.to_excel(xls, sheet_name=f"{tag}_matrix")

        for tag, df in dfs_report.items():
            df.to_excel(xls, sheet_name=f"{tag}_report")

        summary_df.to_excel(xls, sheet_name="Summary")

    print(f"✅  Workbook saved to {outfile.resolve()}")
    print(f"   • Global WER: {wer_val:.3%}")
    print(f"   • Global CER: {cer_val:.3%}")


# --------------------------------------------------------------------------- #
# 6. Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    orig_tags, pred_tags, gt_clean, pr_clean = extract(args.input)
    dfs_conf, dfs_report, summary_rows = build_results(orig_tags, pred_tags)

    # global ASR metrics
    wer_val = wer(gt_clean, pr_clean)
    cer_val = cer(gt_clean, pr_clean)

    export_excel(dfs_conf, dfs_report, summary_rows, wer_val, cer_val, args.output)


if __name__ == "__main__":
    main()
