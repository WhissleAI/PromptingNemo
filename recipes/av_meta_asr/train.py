#!/usr/bin/env python3
"""Training entry point for AV-Meta-ASR next-gen model.

Wraps the PromptingNemo training CLI with AV-Meta-specific defaults.

Usage:
    python train.py --config conf/av_meta_nextgen.yaml
    python train.py --config conf/av_meta_nextgen.yaml --mode train --resume_from /path/to/checkpoint.ckpt
"""
import sys
from pathlib import Path

# Add PromptingNemo root to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from promptingnemo.training.cli import main

if __name__ == "__main__":
    main()
