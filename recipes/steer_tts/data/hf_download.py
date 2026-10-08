"""Authenticated, parallel, resumable HF dataset downloads for steer_tts (run from $STEER_TTS_ROOT/data; fallback for hf_curl.sh)."""
from huggingface_hub import snapshot_download

JOBS = [
    ("ylacombe/expresso", "read/*", "raw/expresso"),
    ("mythicinfinity/libritts_r", "data/train.clean.100/*", "raw/libritts_r"),
    ("mythicinfinity/libritts_r", "data/dev.clean/*", "raw/libritts_r"),
    ("parler-tts/libritts-r-filtered-speaker-descriptions", "clean/train.clean.100*", "raw/libritts_r"),
    ("ajd12342/paraspeechcaps", "data/*", "raw/paraspeechcaps"),
]
for repo, pattern, dest in JOBS:
    snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=[pattern], local_dir=dest, max_workers=6)
    print("DONE", repo, pattern, flush=True)
