"""Corpus -> trimmed 24 kHz mono wavs + manifest rows.

python -m steer.prep ears|expresso|libritts|libritts360|libritts500 [nproc]
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


# NonverbalTTS (deepvk, VoxCeleb + Expresso speech with inline nonverbal annotations, 17 h): emoji markers -> our
# inline event tokens; markers we have no token for are dropped. Corpus "nvtts"; VoxCeleb part is 16 kHz audio.
NVTTS_EMOJI = {"🌬️": "breath", "🌬": "breath", "🤣": "laugh", "😂": "laugh", "😷": "cough", "🤧": "sneeze",
               "😮‍💨": "sigh", "👃": "sniff", "🗣️": "throat", "🗣": "throat", "😭": "cry"}
NVTTS_EMO = {"angry": "anger", "happy": "happiness", "sad": "sadness", "fearful": "fear", "disgusted": "disgust",
             "surprised": "amazement", "neutral": "neutral"}


def nvtts_text(t):
    import re
    for e, tok in sorted(NVTTS_EMOJI.items(), key=lambda x: -len(x[0])):
        t = t.replace(e, f" [{tok}] ")
    t = re.sub(r"[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F\u200D]", " ", t)  # unmapped markers
    return re.sub(r"\s+", " ", t).strip()


def nvtts_jobs():
    import pyarrow.parquet as pq
    root = Path("/external1/karan/steer_tts/noise_raw/nonverbaltts/default")
    rows, jobs = [], []
    for fp in sorted(root.glob("*/*.parquet")):
        if fp.parent.name == "test":  # held out for evaluation
            continue
        for r in pq.read_table(fp).to_pylist():
            if not r.get("Result") or r["Result"] == "None":
                continue
            uid = "nvtts_" + r["index"].replace("/", "_")
            raw = (r.get("Emotion") or "neutral").lower()
            rows.append(dict(id=uid, corpus="nvtts", speaker=f"nvtts_{r['speaker_id']}", text=nvtts_text(r["Result"]),
                             raw_style=raw, tag=NVTTS_EMO.get(raw, "none"), group=r.get("data_name")))
            jobs.append((uid, "nvtts", r["audio"]["bytes"], "bytes"))
    return rows, jobs


def main():
    corpus = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    if corpus == "ears":
        rows, jobs = ears_jobs()
    elif corpus == "expresso":
        files = sorted((RAW / "expresso" / "read").glob("*.parquet"))
        rows, jobs = parquet_jobs("expresso", files, "text", "speaker_id", "style", "id")
    elif corpus in ("libritts", "libritts360", "libritts500"):  # 360/500 = train.clean.360 / train.other.500:
        # same corpus label, ids and wav dir as train.clean.100, own manifest (speaker coverage for spk_enc)
        split = {"libritts": "train.clean.100", "libritts360": "train.clean.360", "libritts500": "train.other.500"}[corpus]
        files = sorted((RAW / "libritts_r" / "data" / split).glob("*.parquet"))
        rows, jobs = parquet_jobs("libritts", files, "text_normalized", "speaker_id", None, "id")
    elif corpus == "nvtts":
        rows, jobs = nvtts_jobs()
    else:
        raise SystemExit(corpus)
    print(corpus, len(rows), "utterances")
    with ProcessPoolExecutor(nproc) as ex:
        durs = dict(ex.map(write, jobs, chunksize=8))
    out = META / f"manifest_{corpus}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            r["wav"] = str(WAV24 / r["corpus"] / f"{r['id']}.wav")  # libritts360/500 write to wav24/libritts
            r["dur"] = round(durs[r["id"]], 3)
            f.write(json.dumps(r) + "\n")
    tot = sum(durs.values()) / 3600
    print(f"wrote {out}: {len(rows)} utts, {tot:.1f} h")


if __name__ == "__main__":
    main()
