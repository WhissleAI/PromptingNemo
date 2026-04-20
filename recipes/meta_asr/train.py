#!/usr/bin/env python3
"""Meta-ASR training entry point. Imports from promptingnemo library."""
import sys
from pathlib import Path

if __name__ == "__main__":
    try:
        from promptingnemo.training.cli import main
    except ImportError:
        # If promptingnemo is not installed, add the repo root to sys.path
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from promptingnemo.training.cli import main

    main()
