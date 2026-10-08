"""Corpus -> trimmed 24 kHz mono wavs + manifest rows.

python -m steer.prep ears|expresso|libritts [nproc]
Writes data/wav24/<corpus>/<id>.wav and data/meta/manifest_<corpus>.jsonl with
{id, corpus, speaker, text, raw_style, tag, wav, dur}. Re-runnable: existing wavs are kept.
"""
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .config import RAW, META, ROOT, SR, canonical_tag

WAV24 = ROOT / "data" / "wav24"


def trim_resample(y, sr, top_db=40, pad=0.12):
    if y.ndim > 1:
        y = y.mean(1)
    y = soxr.resample(y.astype(np.float32), sr, SR) if sr != SR else y.astype(np.float32)
    # energy-based trim of leading/trailing silence, keeping a short pad
    hop = 240
    n = len(y) // hop
    if n < 3:
        return y
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    on = np.where(db > db.max() - top_db)[0]
    s = max(0, on[0] * hop - int(pad * SR))
    e = min(len(y), (on[-1] + 1) * hop + int(pad * SR))
    y = y[s:e]
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y


def write(job):
    uid, corpus, src, kind = job
    out = WAV24 / corpus / f"{uid}.wav"
    if out.exists():
        return uid, sf.info(out).duration
    if kind == "path":
        y, sr = sf.read(src, dtype="float32")
    else:  # raw bytes
        y, sr = sf.read(io.BytesIO(src), dtype="float32")
    y = trim_resample(y, sr)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, SR, subtype="PCM_16")
    return uid, len(y) / SR


def ears_jobs():
    root = RAW / "ears"
    tr = json.load(open(root / "transcripts.json"))
    rows, jobs = [], []
    for spk_dir in sorted(root.glob("p[0-9][0-9][0-9]")):
        if not spk_dir.is_dir():
            continue
        for key, text in tr.items():
            f = spk_dir / f"{key}.wav"
            if not f.exists():
                continue
            if key.startswith("emo_"):
                raw = key.split("_")[1]
            else:
                raw = key.split("_")[-1]
            uid = f"ears_{spk_dir.name}_{key}"
            rows.append(dict(id=uid, corpus="ears", speaker=f"ears_{spk_dir.name}", text=text, raw_style=raw,
                             tag=canonical_tag("ears", raw), group=key.rsplit("_", 1)[0] if not key.startswith("emo_") else key))
            jobs.append((uid, "ears", str(f), "path"))
    return rows, jobs


def parquet_jobs(corpus, files, text_col, spk_col, style_col, id_col):
    import pyarrow.parquet as pq
    rows, jobs = [], []
    for fp in files:
        t = pq.read_table(fp)
        cols = t.column_names
        d = t.to_pydict()
        for i in range(t.num_rows):
            a = d["audio"][i]
            uid = f"{corpus}_{d[id_col][i]}" if id_col in cols else f"{corpus}_{fp.stem}_{i}"
            uid = uid.replace("/", "_")
            raw = d[style_col][i] if style_col and style_col in cols else "none"
            rows.append(dict(id=uid, corpus=corpus, speaker=f"{corpus}_{d[spk_col][i]}", text=d[text_col][i],
                             raw_style=str(raw), tag=canonical_tag(corpus, str(raw)), group=None))
            jobs.append((uid, corpus, a["bytes"], "bytes"))
    return rows, jobs


def main():
    corpus = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    if corpus == "ears":
        rows, jobs = ears_jobs()
    elif corpus == "expresso":
        files = sorted((RAW / "expresso" / "read").glob("*.parquet"))
        rows, jobs = parquet_jobs("expresso", files, "text", "speaker_id", "style", "id")
    elif corpus == "libritts":
        files = sorted((RAW / "libritts_r" / "data" / "train.clean.100").glob("*.parquet"))
        rows, jobs = parquet_jobs("libritts", files, "text_normalized", "speaker_id", None, "id")
    else:
        raise SystemExit(corpus)
    print(corpus, len(rows), "utterances")
    with ProcessPoolExecutor(nproc) as ex:
        durs = dict(ex.map(write, jobs, chunksize=8))
    out = META / f"manifest_{corpus}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            r["wav"] = str(WAV24 / corpus / f"{r['id']}.wav")
            r["dur"] = round(durs[r["id"]], 3)
            f.write(json.dumps(r) + "\n")
    tot = sum(durs.values()) / 3600
    print(f"wrote {out}: {len(rows)} utts, {tot:.1f} h")


if __name__ == "__main__":
    main()
