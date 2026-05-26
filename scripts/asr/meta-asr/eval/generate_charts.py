#!/usr/bin/env python3
"""
generate_charts.py — Generate benchmark comparison charts from benchmark_summary.json.

Usage:
    python generate_charts.py \
        --summary /path/to/benchmark_summary.json \
        --output-dir /path/to/charts/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# Colors
BG_COLOR = '#1a1a2e'
CARD_COLOR = '#16213e'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#2a2a4e'
WHISSLE_RED = '#e63946'
DEEPGRAM_GREEN = '#2ec4b6'
GEMINI_BLUE = '#457b9d'

SYSTEM_COLORS = {
    'whissle': WHISSLE_RED,
    'deepgram': DEEPGRAM_GREEN,
    'gemini': GEMINI_BLUE,
}

SYSTEM_LABELS = {
    'whissle': 'Whissle v18',
    'deepgram': 'Deepgram Nova-2',
    'gemini': 'Gemini 2.5 Flash',
}


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': CARD_COLOR,
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'axes.edgecolor': GRID_COLOR,
        'grid.color': GRID_COLOR,
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 12,
    })


def chart_wer_comparison(summary, output_dir):
    """Grouped bar chart: WER by system, grouped by test set."""
    fig, ax = plt.subplots(figsize=(10, 6))

    test_sets = list(summary.keys())
    systems = ['whissle', 'deepgram', 'gemini']
    n_sets = len(test_sets)
    n_sys = len(systems)
    bar_width = 0.22
    x = np.arange(n_sets)

    for i, sys_key in enumerate(systems):
        wers = []
        for ts in test_sets:
            wers.append(summary[ts][sys_key]['wer'] * 100)
        bars = ax.bar(
            x + i * bar_width - bar_width,
            wers,
            bar_width,
            label=SYSTEM_LABELS[sys_key],
            color=SYSTEM_COLORS[sys_key],
            edgecolor='none',
            zorder=3,
        )
        for bar, val in zip(bars, wers):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.1f}%',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color=SYSTEM_COLORS[sys_key],
            )

    test_labels = {
        'in_house': 'In-House (5K)',
        'fleurs': 'FLEURS Hindi',
    }
    ax.set_xticks(x)
    ax.set_xticklabels([test_labels.get(t, t) for t in test_sets], fontsize=13)
    ax.set_ylabel('Word Error Rate (%)', fontsize=13)
    ax.set_title('Hindi ASR: WER Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.8, facecolor=CARD_COLOR, edgecolor=GRID_COLOR)
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / 'wer_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def chart_tag_accuracy(summary, output_dir):
    """Bar chart: per-tag accuracy for Whissle (in-house only)."""
    in_house = summary.get('in_house', {}).get('whissle', {})
    tags = in_house.get('tags', {})
    if not tags:
        print("  No tag data for tag accuracy chart, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = sorted(tags.keys())
    accuracies = [tags[c]['accuracy'] * 100 for c in categories]
    macro_f1s = [tags[c]['macro_f1'] * 100 for c in categories]

    x = np.arange(len(categories))
    bar_width = 0.35

    bars1 = ax.bar(x - bar_width / 2, accuracies, bar_width, label='Accuracy',
                   color=WHISSLE_RED, edgecolor='none', zorder=3)
    bars2 = ax.bar(x + bar_width / 2, macro_f1s, bar_width, label='Macro F1',
                   color='#f4845f', edgecolor='none', zorder=3)

    for bar, val in zip(bars1, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=WHISSLE_RED)
    for bar, val in zip(bars2, macro_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='#f4845f')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Whissle v18: Tag Classification Performance', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.8, facecolor=CARD_COLOR, edgecolor=GRID_COLOR)
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    path = output_dir / 'tag_accuracy.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def chart_latency(summary, output_dir):
    """Bar chart: average latency per sample per system."""
    fig, ax = plt.subplots(figsize=(10, 6))

    systems = ['whissle', 'deepgram', 'gemini']
    test_sets = list(summary.keys())

    x = np.arange(len(test_sets))
    bar_width = 0.22

    for i, sys_key in enumerate(systems):
        latencies = []
        for ts in test_sets:
            latencies.append(summary[ts][sys_key]['avg_latency_sec'])

        bars = ax.bar(
            x + i * bar_width - bar_width,
            latencies,
            bar_width,
            label=SYSTEM_LABELS[sys_key],
            color=SYSTEM_COLORS[sys_key],
            edgecolor='none',
            zorder=3,
        )
        for bar, val in zip(bars, latencies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{val:.2f}s',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                color=SYSTEM_COLORS[sys_key],
            )

    test_labels = {'in_house': 'In-House (5K)', 'fleurs': 'FLEURS Hindi'}
    ax.set_xticks(x)
    ax.set_xticklabels([test_labels.get(t, t) for t in test_sets], fontsize=13)
    ax.set_ylabel('Avg Latency per Sample (seconds)', fontsize=13)
    ax.set_title('Inference Latency Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.8, facecolor=CARD_COLOR, edgecolor=GRID_COLOR)
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / 'latency_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def chart_training_progression(output_dir):
    """WER across epochs showing early stopping."""
    epochs = [0, 1, 2]
    wers = [17.73, 18.23, 20.67]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, wers, 'o-', color=WHISSLE_RED, linewidth=2.5, markersize=10, zorder=5)

    ax.axvline(x=0, color='#2ec4b6', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
    ax.annotate('Best checkpoint\n(epoch 0)', xy=(0, 17.73), xytext=(0.5, 16.5),
                fontsize=11, color='#2ec4b6', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ec4b6', lw=1.5))

    for e, w in zip(epochs, wers):
        ax.text(e, w + 0.3, f'{w:.2f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=TEXT_COLOR)

    ax.fill_between([0.5, 2.5], 15, 22, alpha=0.08, color='#e63946', zorder=1)
    ax.text(1.5, 21.3, 'Overfitting Zone', ha='center', fontsize=10,
            color='#e63946', fontstyle='italic', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation WER (%)', fontsize=13)
    ax.set_title('v18 Training: Early Stopping Decision', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(epochs)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(16, 22)
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / 'training_progression.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def chart_architecture(output_dir):
    """Architecture overview diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    box_style = dict(boxstyle='round,pad=0.5', facecolor=CARD_COLOR, edgecolor=WHISSLE_RED, linewidth=2)
    box_style2 = dict(boxstyle='round,pad=0.5', facecolor=CARD_COLOR, edgecolor=GEMINI_BLUE, linewidth=2)

    ax.text(6, 6.3, 'META-ASR Architecture', ha='center', fontsize=18, fontweight='bold', color=TEXT_COLOR)

    ax.text(2, 5, 'Audio Input\n(16kHz WAV)', ha='center', fontsize=11, color=TEXT_COLOR, bbox=box_style2)
    ax.annotate('', xy=(4, 5), xytext=(3.2, 5),
                arrowprops=dict(arrowstyle='->', color=DEEPGRAM_GREEN, lw=2))

    ax.text(6, 5, 'FastConformer\nEncoder (17 layers)', ha='center', fontsize=11,
            color=TEXT_COLOR, bbox=box_style)
    ax.annotate('', xy=(8.5, 5.3), xytext=(7.5, 5.1),
                arrowprops=dict(arrowstyle='->', color=WHISSLE_RED, lw=2))
    ax.annotate('', xy=(8.5, 4.7), xytext=(7.5, 4.9),
                arrowprops=dict(arrowstyle='->', color=WHISSLE_RED, lw=2))

    ax.text(10, 5.5, 'CTC Decoder\n(transcript + entities)', ha='center', fontsize=10,
            color=TEXT_COLOR, bbox=box_style)

    ax.text(10, 4.2, 'Tag Classifier\n(AGE, GENDER,\nEMOTION, INTENT)', ha='center', fontsize=10,
            color=TEXT_COLOR, bbox=box_style)

    ax.text(6, 2.5, 'Single-Pass Output', ha='center', fontsize=13, fontweight='bold', color=WHISSLE_RED)
    ax.text(6, 1.8, '"ENTITY_PERSON राहुल END मुझे कल दिल्ली जाना है"\n'
            'AGE_ADULT  GENDER_MALE  EMOTION_NEUTRAL  INTENT_TRAVEL',
            ha='center', fontsize=10, color=DEEPGRAM_GREEN, family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#0d1117', edgecolor=GRID_COLOR, linewidth=1))

    plt.tight_layout()
    path = output_dir / 'architecture.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to benchmark_summary.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for chart PNGs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.summary) as f:
        summary = json.load(f)

    setup_style()

    print("Generating charts...")
    chart_wer_comparison(summary, output_dir)
    chart_tag_accuracy(summary, output_dir)
    chart_latency(summary, output_dir)
    chart_training_progression(output_dir)
    chart_architecture(output_dir)
    print("All charts generated.")


if __name__ == "__main__":
    main()
