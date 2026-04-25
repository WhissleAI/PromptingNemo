"""
Master runner: generate sample data, then run all exploration scripts.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

SCRIPTS = [
    ("Generating sample Ego4D annotations...", "generate_sample_data.py"),
    ("Exploring metadata...", "explore_metadata.py"),
    ("Exploring narrations...", "explore_narrations.py"),
    ("Exploring benchmarks (NLQ, Moments, FHO, AV)...", "explore_benchmarks.py"),
    ("Cross-benchmark analysis...", "explore_cross_benchmark.py"),
]


def main():
    print("=" * 70)
    print("  EGO4D DATASET EXPLORATION PIPELINE")
    print("=" * 70)

    for desc, script in SCRIPTS:
        print(f"\n{'─'*70}")
        print(f"▶ {desc}")
        print(f"{'─'*70}")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / script)],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  ✗ {script} failed with exit code {result.returncode}")
            sys.exit(1)

    print(f"\n{'='*70}")
    print("  ALL EXPLORATIONS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
