"""LLM baseline for the prompt -> control mapping: a local instruction LLM (Ollama, CPU only: num_gpu=0) is given
the control schema, the tag list and 4 fixed few-shot examples, and must emit JSON
{"controls": {dim: z}, "tags": [up to 3]}. Greedy decoding, JSON-constrained output. Predictions are cached.

python -m steer.paper.adapter_llm [--model qwen2.5:3b]  -> runs/paper/llm_<model>.json (cache) 
"""
import argparse
import json
import time
import urllib.request

from ..config import CONTROL_DIMS
from .common import RES, WORK

DESC = {"rate": "speaking rate (+ faster, - slower)", "loud": "loudness (+ louder, - quieter)",
        "f0_med": "pitch level (+ higher, - lower/deeper)", "f0_iqr": "pitch variation / intonation range (+ lively, - monotone)",
        "f0_end": "pitch movement at the end of the sentence (+ rising, - falling)", "pause": "amount of pausing (+ more pauses, - fluent, no pauses)",
        "tilt": "spectral tilt (+ sharp/bright/edgy timbre, - mellow/dark)", "voiced": "voicing (+ full voice, - whispered/breathy)",
        "flat": "spectral flatness (+ airy/breathy/noisy, - clean/clear tone)", "centroid": "brightness/articulation (+ crisp/bright, - muffled/dull)",
        "cpps": "periodicity/steadiness (+ steady, resonant, - shaky, rough, trembling)", "h1h2": "phonation (+ breathy/soft, - pressed/tense)"}
TAGS = ['none', 'adoration', 'amazement', 'amusement', 'anger', 'anxiety', 'apologetic', 'bored', 'confidence', 'confusion',
        'contempt', 'contentment', 'cuteness', 'desire', 'determined', 'disappointment', 'disgust', 'distress', 'ecstasy',
        'embarrassment', 'emphasis', 'enunciated', 'excited', 'fast', 'fear', 'frustration', 'guilt', 'happiness', 'hesitant',
        'high_pitch', 'interest', 'irritation', 'laughing', 'loud', 'low_pitch', 'mysterious', 'neutral', 'nostalgia', 'pain',
        'panic', 'pride', 'realization', 'relief', 'sadness', 'sarcasm', 'serenity', 'skeptical', 'slow', 'sympathy', 'tired',
        'whisper']
SYSTEM = (
    "You convert a voice direction for a text-to-speech system into JSON. The system has 12 controls, each a "
    "speaker-relative z-score (0 = the speaker's normal delivery; about 0.5 = slight, 1 = clear, 2 = strong, 3 = extreme "
    "change; negative = the opposite direction):\n" + "\n".join(f"- {k}: {v}" for k, v in DESC.items()) +
    "\nInclude ONLY the controls the direction asks for or clearly implies; leave everything else out. "
    "Also choose up to 3 style tags, most likely first, from this list: " + ", ".join(TAGS) +
    ". Use \"none\" when no style or emotion is requested.\n"
    'Answer with JSON only: {"controls": {"<control>": <number>, ...}, "tags": ["<tag>", ...]}')
SHOTS = [("Read it a bit more slowly and softly.", {"controls": {"rate": -0.7, "loud": -0.7}, "tags": ["none"]}),
         ("Sound excited, with lots of pitch movement.", {"controls": {"f0_iqr": 1.5}, "tags": ["excited", "happiness"]}),
         ("In a whisper.", {"controls": {"voiced": -3.0}, "tags": ["whisper"]}),
         ("Very high-pitched.", {"controls": {"f0_med": 2.5}, "tags": ["high_pitch"]})]


def ask(model, prompt, host="http://localhost:11434"):
    msgs = [{"role": "system", "content": SYSTEM}]
    for p, a in SHOTS:
        msgs += [{"role": "user", "content": p}, {"role": "assistant", "content": json.dumps(a)}]
    msgs.append({"role": "user", "content": prompt})
    body = dict(model=model, messages=msgs, stream=False, format="json", keep_alive="30m",
                options=dict(temperature=0, num_gpu=0, num_thread=8, num_ctx=4096, num_predict=200, seed=0))
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))["message"]["content"]
    return out


def parse(raw):
    try:
        d = json.loads(raw)
    except Exception:
        return dict(controls={}, tags=["none"], parse_error=True)
    c = d.get("controls", {}) if isinstance(d, dict) else {}
    ctrl = {}
    for k, v in (c.items() if isinstance(c, dict) else []):
        if k in CONTROL_DIMS:
            try:
                ctrl[k] = float(v)
            except Exception:
                pass
    tags = [t for t in (d.get("tags") or []) if isinstance(t, str) and t in TAGS][:3] if isinstance(d, dict) else []
    return dict(controls=ctrl, tags=tags or ["none"])


def all_prompts():
    """v2 items first (end-to-end prompts first), then v1 items, then the v1 intensity probes."""
    g2 = json.load(open(RES / "adapter_handwritten_gold_v2.json"))["items"]
    g1 = json.load(open(RES / "adapter_handwritten_gold.json"))
    first = [it for it in g2 if it["cat"] in ("dim", "combo", "distractor")]
    ps = [it["prompt"] for it in first] + [it["prompt"] for it in g2] + [it["prompt"] for it in g1["items"]] + \
        [p for pr in g1["probes"] for p in pr["prompts"]]
    return list(dict.fromkeys(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b")
    a = ap.parse_args()
    f = WORK / f"llm_{a.model.replace(':', '_')}.json"
    cache = json.load(open(f)) if f.exists() else {}
    ps = [p for p in all_prompts() if p not in cache]
    t0 = time.time()
    for i, p in enumerate(ps):
        raw = ask(a.model, p)
        cache[p] = dict(raw=raw, **parse(raw))
        json.dump(cache, open(f, "w"), indent=1)
        if i % 10 == 0:
            print(f"{i}/{len(ps)} {time.time() - t0:.0f}s  {p!r} -> {cache[p]['controls']} {cache[p]['tags']}", flush=True)
    print("done", len(cache))


if __name__ == "__main__":
    main()
