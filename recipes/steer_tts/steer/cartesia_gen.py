"""Synthetic prompt-steering corpus from Cartesia Sonic (user-approved 2026-10-08).

Why: SteerTTS follows descriptions poorly (ParaSpeechCaps pitch/rate accuracy ~0.6 vs ~0.9 for real speech) and the
prompt adapter was trained on template captions only. Cartesia renders the SAME text in the SAME voice under exact,
labelled emotion / speed / volume settings, so every clip comes with (a) its neutral reference render, so the
speaker-relative controls are measured against that voice's own neutral speech exactly as for EARS, and (b) a
ground-truth instruction. Its ~430 English voices also widen speaker coverage for the spk_enc run.

python -m steer.cartesia_gen voices  [--n 120]          -> data/raw/cartesia/voices.json  (gender/accent-balanced)
python -m steer.cartesia_gen plan    [--texts 40 --conds 8 --seed 0]   -> data/raw/cartesia/plan.jsonl
python -m steer.cartesia_gen plan_coverage [--texts 20 --conds 2]  -> APPENDS every other English voice (cloning)
python -m steer.cartesia_gen run     [--conc 4 --max_chars 3e6 --limit N]
    -> data/wav24/cartesia/<id>.wav (trimmed, 24 kHz) + data/meta/manifest_cartesia.jsonl (prep-format rows)
    resumable: finished ids are skipped; stops at --max_chars (logged cumulative in data/raw/cartesia/usage.json)
python -m steer.cartesia_gen check                       -> measured control deltas per requested setting (sanity)

Then the usual pipeline: steer.extract cartesia -> steer.build_controls ... cartesia -> steer.xvec cartesia,
and steer.prompt train picks up the instruction pairs (manifest field `instruction`).
API key: ~/.config/cartesia/api_key (never in the repo).
"""
import argparse
import asyncio
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import META, RAW, ROOT, SR

API = "https://api.cartesia.ai"
VERSION = "2026-08-14"
MODEL = "sonic-3.6"
DIR = RAW / "cartesia"
WAV = ROOT / "data" / "wav24" / "cartesia"
MANIFEST = META / "manifest_cartesia.jsonl"
KEY_FILE = Path.home() / ".config" / "cartesia" / "api_key"

# Cartesia emotion -> SteerTTS style tag. Synonyms merge into existing tags where the corpora already have them
# (EARS / Expresso), new styles get their own tag. Chosen to cover what users ask for in prompts.
EMOTIONS = {
    "neutral": "neutral", "happy": "happiness", "excited": "excited", "enthusiastic": "excited",
    # angry/outraged render at ~neutral loudness and diluted EARS anger (spk_enc@130k) -> their own tag
    "angry": "irritation", "outraged": "irritation", "frustrated": "frustration", "sad": "sadness", "melancholic": "sadness",
    "disappointed": "disappointment", "scared": "fear", "anxious": "anxiety", "panicked": "panic",
    "surprised": "amazement", "amazed": "amazement", "curious": "interest", "confused": "confusion",
    "calm": "serenity", "peaceful": "serenity", "content": "contentment", "affectionate": "adoration",
    "sympathetic": "sympathy", "sarcastic": "sarcasm", "bored": "bored", "tired": "tired", "disgusted": "disgust",
    "contempt": "contempt", "proud": "pride", "confident": "confidence", "hesitant": "hesitant",
    "apologetic": "apologetic", "mysterious": "mysterious", "determined": "determined", "skeptical": "skeptical",
    "nostalgic": "nostalgia", "guilty": "guilt",
}
SPEEDS = [0.6, 0.75, 0.9, 1.15, 1.3, 1.5]
VOLUMES = [0.5, 0.7, 1.4, 2.0]

# natural-language instruction phrasing for the labels (adapter training sees many paraphrases per label)
EMO_WORDS = {
    "happy": ["happily", "in a happy voice", "cheerfully", "sounding pleased"],
    "excited": ["excitedly", "full of excitement", "like you can't contain your excitement"],
    "enthusiastic": ["enthusiastically", "with real enthusiasm", "upbeat and eager"],
    # angry/outraged clips are tagged `irritation` (they render at ~neutral loudness), so they are described as
    # irritation - "angry"/"furious" must keep pointing at EARS anger (loud), which is what users mean by it
    "angry": ["irritated", "annoyed", "testily", "with irritation"], "outraged": ["indignantly", "exasperated", "affronted"],
    "frustrated": ["frustrated", "sounding fed up", "with frustration"], "sad": ["sadly", "in a sad voice", "sorrowfully"],
    "melancholic": ["melancholy", "wistfully sad", "in a melancholic tone"],
    "disappointed": ["disappointed", "sounding let down", "with disappointment"],
    "scared": ["scared", "fearfully", "like you're frightened"], "anxious": ["anxiously", "nervous and worried", "with anxiety"],
    "panicked": ["in a panic", "panicked", "frantically"], "surprised": ["surprised", "with surprise", "caught off guard"],
    "amazed": ["amazed", "in awe", "astonished"], "curious": ["curiously", "intrigued", "with curiosity"],
    "confused": ["confused", "puzzled", "like you don't quite understand"], "calm": ["calmly", "in a calm voice", "relaxed and even"],
    "peaceful": ["peacefully", "serene and gentle", "soothingly"], "content": ["contentedly", "satisfied", "at ease"],
    "affectionate": ["affectionately", "warmly, with love", "tenderly"], "sympathetic": ["sympathetically", "with compassion", "kindly, consoling"],
    "sarcastic": ["sarcastically", "dripping with sarcasm", "with an ironic edge"], "bored": ["bored", "uninterested", "flat and bored"],
    "tired": ["tired", "exhausted", "sleepily"], "disgusted": ["disgusted", "with disgust", "grossed out"],
    "contempt": ["contemptuously", "with disdain", "scornfully"], "proud": ["proudly", "with pride", "beaming with pride"],
    "confident": ["confidently", "self-assured", "with confidence"], "hesitant": ["hesitantly", "unsure", "tentatively"],
    "apologetic": ["apologetically", "sorry", "regretful"], "mysterious": ["mysteriously", "secretive", "with an air of mystery"],
    "determined": ["determined", "resolute", "with determination"], "skeptical": ["skeptically", "doubtful", "unconvinced"],
    "nostalgic": ["nostalgically", "fondly remembering", "wistful"], "guilty": ["guiltily", "with guilt", "sheepishly"],
}
SPEED_WORDS = {0.6: ["very slowly", "extremely slowly"], 0.75: ["slowly", "at a slow pace"], 0.9: ["a bit slowly", "unhurried"],
               1.15: ["a little faster", "briskly"], 1.3: ["quickly", "fast"], 1.5: ["very fast", "rapidly"]}
VOL_WORDS = {0.5: ["very quietly", "softly, barely audible"], 0.7: ["quietly", "softly"], 1.4: ["loudly", "with a raised voice"],
             2.0: ["very loudly", "projecting loudly"]}


def instruction(emotion, speed, volume, rng):
    parts = []
    if emotion != "neutral":
        parts.append(rng.choice(EMO_WORDS[emotion]))
    if speed != 1.0:
        parts.append(rng.choice(SPEED_WORDS[speed]))
    if volume != 1.0:
        parts.append(rng.choice(VOL_WORDS[volume]))
    if not parts:
        return rng.choice(["say it normally", "in a neutral voice", "plainly, no particular emotion", "just read it"])
    lead = rng.choice(["", "say it ", "speak ", "read this ", "deliver it "])
    return (lead + ", ".join(parts)).strip()


# ---------------------------------------------------------------------------------------------------- API
def key():
    return KEY_FILE.read_text().strip()


def headers():
    return {"Authorization": f"Bearer {key()}", "Cartesia-Version": VERSION, "Content-Type": "application/json"}


def list_voices():
    import urllib.request
    vs, after = [], None
    while True:
        url = f"{API}/voices?limit=100" + (f"&starting_after={after}" if after else "")
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers())))
        vs += d["data"]
        if not d.get("has_more"):
            return vs
        after = d["data"][-1]["id"]


def select_voices(n, seed=0):
    """English, gender-balanced, spread over countries/accents; deterministic."""
    rng = random.Random(seed)
    vs = [v for v in list_voices() if v.get("language") == "en" and v.get("gender") in ("feminine", "masculine")]
    out = []
    for g in ("feminine", "masculine"):
        pool = [v for v in vs if v["gender"] == g]
        by_c = {}
        for v in sorted(pool, key=lambda v: v["id"]):
            by_c.setdefault(v.get("country") or "?", []).append(v)
        for lst in by_c.values():
            rng.shuffle(lst)
        picked = []
        while len(picked) < n // 2 and any(by_c.values()):  # round-robin over countries
            for c in sorted(by_c):
                if by_c[c] and len(picked) < n // 2:
                    picked.append(by_c[c].pop())
        out += picked
    return [dict(id=v["id"], name=v["name"], gender=v["gender"], country=v.get("country"),
                 description=v.get("description")) for v in out]


# ---------------------------------------------------------------------------------------------------- plan
def text_pool(n, seed):
    """Mostly neutral read sentences (6-28 words) from LibriTTS-R / Expresso transcripts, so style must come from the
    settings rather than the words (CosyVoice 3 and our CREMA-D study both found text sentiment carries emotion)."""
    rng = random.Random(seed)
    texts = []
    for c in ("libritts", "expresso"):
        for l in open(META / f"manifest_{c}.jsonl"):
            t = json.loads(l)["text"].strip()
            if 6 <= len(t.split()) <= 28 and t.isascii():
                texts.append(t)
    texts = sorted(set(texts))
    rng.shuffle(texts)
    return texts[:n]


def plan_coverage(n_texts, n_conds, seed):
    """Speaker-coverage pass for reference-conditioned / cloning training: EVERY other English voice (the steering
    plan's 120 are excluded), few clips each - mostly the neutral default read, which is what the speaker vector is
    computed from, plus a couple of varied settings so these voices are steerable too. Appends to plan.jsonl."""
    have = {v["id"] for v in json.load(open(DIR / "voices.json"))}
    vs = [v for v in list_voices() if v.get("language") == "en" and v["id"] not in have]
    extra = [dict(id=v["id"], name=v["name"], gender=v.get("gender"), country=v.get("country"),
                  description=v.get("description")) for v in sorted(vs, key=lambda v: v["id"])]
    json.dump(extra, open(DIR / "voices_coverage.json", "w"), indent=1)
    rows = make_rows(extra, n_texts, n_conds, seed + 1, prefix="cartcov", p_neutral_only=0.5)
    with open(DIR / "plan.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"coverage plan: {len(extra)} voices, {len(rows)} clips, {sum(len(r['text']) for r in rows):,} chars appended")


def plan(n_texts, n_conds, seed):
    voices = json.load(open(DIR / "voices.json"))
    rows = make_rows(voices, n_texts, n_conds, seed)
    with open(DIR / "plan.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    chars = sum(len(r["text"]) for r in rows)
    print(f"plan: {len(voices)} voices x {n_texts} texts x {n_conds}+1 conds = {len(rows)} clips, {chars:,} chars")


def make_rows(voices, n_texts, n_conds, seed, prefix="cartesia", p_neutral_only=0.0):
    """p_neutral_only: share of (voice, text) groups rendered ONLY at the neutral default (speaker-coverage pass)."""
    rng = random.Random(seed)
    texts = text_pool(n_texts * len(voices), seed)
    emos = [e for e in EMOTIONS if e != "neutral"]
    rows = []
    for vi, v in enumerate(voices):
        for ti in range(n_texts):
            text = texts[vi * n_texts + ti]
            group = f"{prefix}_{v['id'][:8]}_{vi:03d}_{ti:03d}"
            conds = {("neutral", 1.0, 1.0)}  # the per-group neutral reference render
            target = 1 if rng.random() < p_neutral_only else n_conds + 1
            while len(conds) < target:
                e = rng.choice(emos) if rng.random() < 0.75 else "neutral"
                s = rng.choice(SPEEDS) if rng.random() < 0.4 else 1.0
                vol = rng.choice(VOLUMES) if rng.random() < 0.3 else 1.0
                conds.add((e, s, vol))
            for e, s, vol in sorted(conds):
                cid = f"{group}_{e}_s{int(s * 100)}_v{int(vol * 100)}"
                rows.append(dict(id=cid, group=group, voice_id=v["id"], voice=v["name"], gender=v["gender"],
                                 text=text, emotion=e, speed=s, volume=vol,
                                 instruction=instruction(e, s, vol, rng)))
    return rows


# ---------------------------------------------------------------------------------------------------- run
async def synth_one(session, r, sem, usage, max_chars):
    import aiohttp
    from .prep import trim_resample
    body = dict(model_id=MODEL, transcript=r["text"], voice=dict(id=r["voice_id"]), language="en",
                output_format=dict(container="wav", encoding="pcm_s16le", sample_rate=SR),
                generation_config=dict(speed=r["speed"], volume=r["volume"],
                                       **({"emotion": r["emotion"]} if r["emotion"] != "neutral" else {})))
    async with sem:
        if usage["chars"] >= max_chars:
            return None
        for attempt in range(6):
            try:
                async with session.post(f"{API}/tts/bytes", json=body, headers=headers(),
                                        timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        break
                    msg = (await resp.text())[:200]
                    if resp.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(2 ** attempt + random.random())
                        continue
                    print(f"FAIL {r['id']} {resp.status} {msg}", flush=True)
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                await asyncio.sleep(2 ** attempt + random.random())
        else:
            print(f"GAVE UP {r['id']}", flush=True)
            return None
    usage["chars"] += len(r["text"])
    y, sr = sf.read(io.BytesIO(data), dtype="float32")
    y = trim_resample(y, sr)
    sf.write(WAV / f"{r['id']}.wav", y, SR)
    return r, len(y) / SR


async def run_async(limit, conc, max_chars):
    import aiohttp
    WAV.mkdir(parents=True, exist_ok=True)
    done = set()
    if MANIFEST.exists():
        done = {json.loads(l)["id"] for l in open(MANIFEST)}
    rows = [json.loads(l) for l in open(DIR / "plan.jsonl")]
    # breadth first over voices so a partial run still has every voice with neutral references
    rows.sort(key=lambda r: (r["group"].rsplit("_", 1)[-1], r["group"], r["emotion"] != "neutral"))
    todo = [r for r in rows if r["id"] not in done][: limit or None]
    uf = DIR / "usage.json"
    usage = json.load(open(uf)) if uf.exists() else dict(chars=0, clips=0, seconds=0.0)
    print(f"run: {len(todo)} to render ({len(done)} done), used {usage['chars']:,}/{int(max_chars):,} chars", flush=True)
    sem = asyncio.Semaphore(conc)
    t0, n = time.time(), 0
    async with aiohttp.ClientSession() as session:
        with open(MANIFEST, "a") as man:
            for i in range(0, len(todo), 200):
                res = await asyncio.gather(*[synth_one(session, r, sem, usage, max_chars) for r in todo[i:i + 200]])
                for out in res:
                    if out is None:
                        continue
                    r, dur = out
                    man.write(json.dumps(dict(id=r["id"], corpus="cartesia", speaker=f"cartesia_{r['voice_id'][:8]}",
                                              text=r["text"], raw_style=r["emotion"], tag=EMOTIONS[r["emotion"]],
                                              group=r["group"], wav=str(WAV / f"{r['id']}.wav"), dur=round(dur, 3),
                                              speed=r["speed"], volume=r["volume"], emotion=r["emotion"],
                                              instruction=r["instruction"], voice=r["voice"], gender=r["gender"])) + "\n")
                    usage["clips"] += 1; usage["seconds"] += dur; n += 1
                man.flush()
                json.dump(usage, open(uf, "w"))
                el = time.time() - t0
                print(f"  {n} clips, {usage['seconds'] / 3600:.2f} h audio total, {usage['chars']:,} chars, "
                      f"{n / el:.2f} clips/s", flush=True)
                if usage["chars"] >= max_chars:
                    print("max_chars reached; stopping", flush=True)
                    break


def check():
    """Does each requested setting move the measured controls the right way? Delta vs the group's neutral render,
    in raw units (rate phones/s, loud dB, f0_med st)."""
    import pandas as pd
    from .features import measure
    from .text import to_phonemes
    from .config import CONTROL_DIMS
    rows = [json.loads(l) for l in open(MANIFEST)]
    meas = {}
    for r in rows:
        y, _ = sf.read(r["wav"], dtype="float32")
        c = measure(y, n_phones=len(to_phonemes(r["text"])[0]), want_mel=False)["controls"]
        meas[r["id"]] = dict(zip(CONTROL_DIMS, c))
    df = pd.DataFrame([{**r, **meas[r["id"]]} for r in rows])
    ref = df[(df.emotion == "neutral") & (df.speed == 1.0) & (df.volume == 1.0)].set_index("group")
    for k in ("rate", "loud", "f0_med", "f0_iqr", "voiced"):
        df[f"d_{k}"] = df[k] - df.group.map(ref[k])
    print("speed  ->", df[df.emotion == "neutral"].groupby("speed")[["d_rate"]].mean().round(2).T.to_string())
    print("volume ->", df[df.emotion == "neutral"].groupby("volume")[["d_loud"]].mean().round(2).T.to_string())
    e = df[(df.speed == 1.0) & (df.volume == 1.0)].groupby("emotion")[["d_rate", "d_loud", "d_f0_med", "d_f0_iqr"]]
    print("emotion (speed=vol=1) ->\n", e.mean().round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["voices", "plan", "plan_coverage", "run", "check"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--texts", type=int, default=40)
    ap.add_argument("--conds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_chars", type=float, default=3e6)
    a = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    if a.cmd == "voices":
        v = select_voices(a.n, a.seed)
        json.dump(v, open(DIR / "voices.json", "w"), indent=1)
        print(len(v), "voices;", {g: sum(x["gender"] == g for x in v) for g in ("feminine", "masculine")},
              "countries", len({x["country"] for x in v}))
    elif a.cmd == "plan":
        plan(a.texts, a.conds, a.seed)
    elif a.cmd == "plan_coverage":
        plan_coverage(a.texts, a.conds, a.seed)
    elif a.cmd == "run":
        asyncio.run(run_async(a.limit, a.conc, a.max_chars))
    else:
        check()


if __name__ == "__main__":
    main()
