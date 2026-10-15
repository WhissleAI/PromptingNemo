"""Acoustic scene, room, channel and sound events: vocabulary, noise library, training-time augmentation.

The TTS learns to render speech INSIDE an acoustic environment, steered like everything else:
  background scene  (rain, cafe, traffic ...) at an SNR    -> environment vector -> g (model.env_proj)
  room              (small / medium / large reverberant)   -> environment vector
  channel           (telephone band + mu-law)              -> environment vector
  sound events      ([dog], [knock] ...) and in-voice nonverbals ([laugh], [cough] ...)  -> inline text tokens
Every clean (un-augmented) item gets the all-zero environment vector, so "nothing requested" = clean speech, which is
the default behaviour that must not regress.

Where things go (data simulation as in DNS-challenge / VoiceDiT / Audiobox-style environment captions):
  - room: RIR convolved with the speech (and edge events), direct path kept at t=0, level preserved
  - background: whole clip, SNR ~ U(0, 25) dB, plus 0.2-0.8 s of noise-only lead-in / tail so the clip opens in the scene
  - events / nonverbals: at the utterance edges ("[dog] text" or "text [dog]") with a 0.1-0.4 s gap, so they never
    mask words and the aligner gives the event token exactly the event's frames; nonverbals are the SAME speaker's
    (EARS records 11 nonverbal types for each of its 107 speakers)
  - channel: last (phone line over the whole mix)
The aligner trains on the dry mel (speech + events + room, before background and channel).

python -m steer.acoustic build      # noise library: data/noise/<kind>/*.wav (24 kHz) + data/meta/noise_lib.jsonl
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

from .config import HOP, META, RAW, ROOT, SR
from .text import EVENTS, NONVERBAL

NOISE = ROOT / "data" / "noise"            # processed 24 kHz library (SSD: read during training)
NOISE_RAW = Path("/external1/karan/steer_tts/noise_raw")  # raw archives (HDD)

SCENES = ["rain", "wind", "water", "thunderstorm", "traffic", "street", "crowd", "cafe", "restaurant", "office",
          "kitchen", "home", "park", "birds", "station", "train", "bus", "car", "music", "fan", "fire", "static",
          "machine", "nature"]
ROOMS = ["small", "medium", "large"]
ENV_DIM = len(SCENES) + len(ROOMS) + 1 + 2  # scene one-hot, room one-hot, phone, (snr, snr given)

# DEMAND environments (Thiemann et al. 2013) -> scene
DEMAND = {"DKITCHEN": "kitchen", "DLIVING": "home", "DWASHING": "machine", "NFIELD": "nature", "NPARK": "park",
          "NRIVER": "water", "OHALLWAY": "office", "OMEETING": "office", "OOFFICE": "office", "PCAFETER": "cafe",
          "PRESTO": "restaurant", "PSTATION": "station", "SPSQUARE": "street", "STRAFFIC": "traffic", "TBUS": "bus",
          "TCAR": "car", "TMETRO": "train"}
# AudioSet class name (regex, matched against AST's top labels) -> ("bg", scene) or ("event", token)
AUDIOSET = [
    (r"^(Rain|Raindrop|Rain on surface)$", ("bg", "rain")), (r"^(Thunderstorm|Thunder)$", ("bg", "thunderstorm")),
    (r"^(Wind|Rustling leaves|Wind noise \(microphone\))$", ("bg", "wind")),
    (r"^(Stream|Waterfall|Ocean|Waves, surf|Water|Gurgling)$", ("bg", "water")),
    (r"^(Traffic noise, roadway noise|Car passing by|Motor vehicle \(road\)|Vehicle)$", ("bg", "traffic")),
    (r"^(Hubbub, speech noise, speech babble|Crowd|Chatter)$", ("bg", "crowd")),
    (r"^(Music|Musical instrument|Piano|Guitar|Orchestra)$", ("bg", "music")),
    (r"^(Mechanical fan|Air conditioning|Hum|Mains hum)$", ("bg", "fan")), (r"^(Fire|Crackle)$", ("bg", "fire")),
    (r"^(White noise|Pink noise|Static|Noise)$", ("bg", "static")),
    (r"^(Bird|Bird vocalization, bird call, bird song|Chirp, tweet)$", ("bg", "birds")),
    (r"^(Engine|Idling|Machine|Mechanisms|Vacuum cleaner|Washing machine)$", ("bg", "machine")),
    (r"^(Train|Rail transport|Subway, metro, underground)$", ("bg", "train")),
    (r"^(Dog|Bark|Bow-wow|Yip)$", ("event", "dog")), (r"^(Cat|Meow)$", ("event", "cat")),
    (r"^(Knock)$", ("event", "knock")), (r"^(Door|Slam|Sliding door|Cupboard open or close)$", ("event", "door")),
    (r"^(Telephone|Telephone bell ringing|Ringtone)$", ("event", "phone")),
    (r"^(Bell|Doorbell|Church bell|Bicycle bell|Jingle bell)$", ("event", "bell")),
    (r"^(Vehicle horn, car horn, honking|Air horn, truck horn)$", ("event", "horn")),
    (r"^(Applause|Clapping)$", ("event", "applause")), (r"^(Glass|Shatter|Chink, clink)$", ("event", "glass")),
    (r"^(Siren|Police car \(siren\)|Ambulance \(siren\)|Fire engine, fire truck \(siren\))$", ("event", "siren")),
    (r"^(Walk, footsteps|Run)$", ("event", "footsteps")),
    (r"^(Typing|Computer keyboard|Typewriter)$", ("event", "typing")), (r"^(Tick-tock|Tick|Clock)$", ("event", "clock")),
]
# FSD50K vocabulary -> ("bg", scene) / ("event", token) / ("nv", token); a clip takes its first mapped label
FSD50K = {
    "Rain": ("bg", "rain"), "Raindrop": ("bg", "rain"), "Thunder": ("event", "thunder"), "Thunderstorm": ("bg", "thunderstorm"),
    "Wind": ("bg", "wind"), "Rustling_leaves": ("bg", "wind"), "Stream": ("bg", "water"), "Ocean": ("bg", "water"),
    "Waves_and_surf": ("bg", "water"), "Traffic_noise_and_roadway_noise": ("bg", "traffic"), "Car_passing_by": ("bg", "traffic"),
    "Crowd": ("bg", "crowd"), "Chatter": ("bg", "crowd"), "Fire": ("bg", "fire"), "Crackle": ("bg", "fire"),
    "Bird_vocalization_and_bird_call_and_bird_song": ("bg", "birds"), "Chirp_and_tweet": ("bg", "birds"), "Bird": ("bg", "birds"),
    "Mechanical_fan": ("bg", "fan"), "Engine": ("bg", "machine"), "Idling": ("bg", "car"), "Train": ("bg", "train"),
    "Subway_and_metro_and_underground": ("bg", "train"), "Bus": ("bg", "bus"),
    "Bark": ("event", "dog"), "Dog": ("event", "dog"), "Meow": ("event", "cat"), "Cat": ("event", "cat"),
    "Knock": ("event", "knock"), "Slam": ("event", "door"), "Door": ("event", "door"), "Doorbell": ("event", "bell"),
    "Ringtone": ("event", "phone"), "Telephone": ("event", "phone"), "Church_bell": ("event", "bell"),
    "Bicycle_bell": ("event", "bell"), "Bell": ("event", "bell"), "Vehicle_horn_and_car_horn_and_honking": ("event", "horn"),
    "Applause": ("event", "applause"), "Clapping": ("event", "applause"), "Shatter": ("event", "glass"),
    "Glass": ("event", "glass"), "Siren": ("event", "siren"), "Walk_and_footsteps": ("event", "footsteps"),
    "Typing": ("event", "typing"), "Computer_keyboard": ("event", "typing"), "Typewriter": ("event", "typing"),
    "Tick-tock": ("event", "clock"), "Tick": ("event", "clock"),
    # human nonverbals from strangers: used as environmental events (other people), never as the speaker's own voice
    "Laughter": ("event_h", "laugh"), "Cough": ("event_h", "cough"), "Sneeze": ("event_h", "sneeze"),
    "Cheering": ("event_h", "cheer"), "Screaming": ("event_h", "scream"),
}
# EARS nonverbal / vegetative recordings -> in-voice token
EARS_NV = {"nonverbal_laughter_open": "laugh", "nonverbal_laughter_closed": "laugh", "nonverbal_crying": "cry",
           "nonverbal_screaming": "scream", "nonverbal_yelling": "yell", "nonverbal_cheering": "cheer",
           "vegetative_coughing": "cough", "vegetative_sneezing": "sneeze", "vegetative_yawning": "yawn",
           "vegetative_throat": "throat"}


def env_vector(scene=None, room=None, phone=False, snr=None):
    """-> float32 [ENV_DIM]; all zeros = clean speech (nothing requested)."""
    v = np.zeros(ENV_DIM, np.float32)
    if scene in SCENES:
        v[SCENES.index(scene)] = 1
    if room in ROOMS:
        v[len(SCENES) + ROOMS.index(room)] = 1
    if phone:
        v[len(SCENES) + len(ROOMS)] = 1
    if snr is not None and scene in SCENES:
        v[-2], v[-1] = (float(snr) - 15.0) / 10.0, 1.0
    return v


def describe(v):
    """Inverse of env_vector, for logs."""
    out = {}
    if v[: len(SCENES)].max() > 0.5:
        out["scene"] = SCENES[int(v[: len(SCENES)].argmax())]
    r = v[len(SCENES): len(SCENES) + len(ROOMS)]
    if r.max() > 0.5:
        out["room"] = ROOMS[int(r.argmax())]
    if v[len(SCENES) + len(ROOMS)] > 0.5:
        out["phone"] = True
    if v[-1] > 0.5:
        out["snr"] = round(float(v[-2]) * 10 + 15, 1)
    return out


# --------------------------------------------------------------------------------------------- signal helpers
def rms(y):
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + 1e-12))


def active_rms(y, frame=480):
    """RMS over the louder half of 20 ms frames (speech level without its pauses)."""
    n = len(y) // frame
    if n < 2:
        return rms(y)
    e = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, 1) + 1e-12)
    return float(np.sqrt(np.mean(np.sort(e)[n // 2:] ** 2)))


def t60(rir, sr=SR):
    """Reverberation time from Schroeder backward integration (T20 fit, extrapolated to 60 dB)."""
    e = np.cumsum(rir[::-1] ** 2)[::-1]
    e = 10 * np.log10(e / (e[0] + 1e-12) + 1e-12)
    i5, i25 = np.argmax(e < -5), np.argmax(e < -25)
    if i25 <= i5:
        return 0.0
    return float(3.0 * (i25 - i5) / sr)


def apply_rir(y, rir):
    from scipy.signal import fftconvolve
    k = int(np.argmax(np.abs(rir)))
    out = fftconvolve(y, rir[k:])[: len(y)]  # direct path at t=0: no added delay, same length
    return (out * (rms(y) / (rms(out) + 1e-9))).astype(np.float32)


def phone_channel(y):
    import soxr
    from scipy.signal import butter, sosfilt
    x = soxr.resample(y, SR, 8000)
    x = sosfilt(butter(4, [300, 3400], btype="band", fs=8000, output="sos"), x)
    mu = 255.0
    q = np.sign(x) * np.log1p(mu * np.clip(np.abs(x), 0, 1)) / np.log1p(mu)
    q = np.round(q * 127) / 127  # 8-bit mu-law
    x = np.sign(q) * (np.power(1 + mu, np.abs(q)) - 1) / mu
    return soxr.resample(x.astype(np.float32), 8000, SR)[: len(y)].astype(np.float32)


# --------------------------------------------------------------------------------------------- augmentation
class Augmenter:
    """Training-time environment augmentation of one utterance (waveform level, inside DataLoader workers).

    __call__(y, text, speaker, rng) -> None (leave the item clean) or dict(
        y      final waveform (what the model must produce),
        dry    speech + events + room (what the aligner sees),
        text   text with event tokens inserted,
        off    sample offset of the original speech inside y (pitch targets are shifted by it),
        env    environment vector, desc  short description for logging)
    """

    def __init__(self, lib_path=META / "noise_lib.jsonl", p=0.35, p_bg=0.7, p_room=0.3, p_phone=0.1, p_event=0.25,
                 p_nv=0.35):
        rows = [json.loads(l) for l in open(lib_path)]
        self.bg = {}
        self.ev = {}
        self.nv = {}  # speaker -> token -> [paths]
        self.rir = {r: [] for r in ROOMS}
        for r in rows:
            if r["kind"] == "bg":
                self.bg.setdefault(r["label"], []).append(r)
            elif r["kind"] == "event":
                self.ev.setdefault(r["label"], []).append(r["path"])
            elif r["kind"] == "nv":
                self.nv.setdefault(r["speaker"], {}).setdefault(r["label"], []).append(r["path"])
            elif r["kind"] == "rir" and r["label"] in self.rir:
                self.rir[r["label"]].append(r["path"])
        self.p, self.p_bg, self.p_room, self.p_phone, self.p_event, self.p_nv = p, p_bg, p_room, p_phone, p_event, p_nv

    @staticmethod
    def _read(path, n=None, rng=None):
        import soundfile as sf
        if n is None:
            return sf.read(path, dtype="float32")[0]
        info = sf.info(path)
        total = info.frames
        if total <= n:
            y = sf.read(path, dtype="float32")[0]
            return np.tile(y, int(np.ceil(n / max(len(y), 1))))[:n]
        s = int(rng.integers(0, total - n))
        return sf.read(path, start=s, frames=n, dtype="float32")[0]

    def __call__(self, y, text, speaker, rng):
        if rng.random() >= self.p:
            return None
        y = y.astype(np.float32)
        lvl = active_rms(y)
        desc, room, scene, snr, phone = [], None, None, None, False
        # 1. edge events: in-voice nonverbal of this speaker, or an environmental event
        pre, post = [], []
        if self.nv.get(speaker) and rng.random() < self.p_nv:
            tok = rng.choice(sorted(self.nv[speaker]))
            ev, tok_kind = self._read(rng.choice(self.nv[speaker][tok])), "nv"  # same session level: no rescale
        elif self.ev and self.p_event > 0 and rng.random() < self.p_event:
            tok = rng.choice(sorted(self.ev))
            ev, tok_kind = self._read(rng.choice(self.ev[tok])), "event"
            ev = ev * (lvl * 10 ** (rng.uniform(-6, 6) / 20) / (active_rms(ev) + 1e-9))
        else:
            tok = None
        ev_span = None
        if tok is not None:
            gap = np.zeros(int(rng.uniform(0.1, 0.4) * SR), np.float32)
            if rng.random() < 0.5:
                pre, text = [ev, gap], f"[{tok}] {text}"
                ev_span = (0, len(ev))
            else:
                post, text = [gap, ev], f"{text} [{tok}]"
                ev_span = (len(y) + len(gap), len(y) + len(gap) + len(ev))
            desc.append(f"{tok_kind}:{tok}")
        dry = np.concatenate(pre + [y] + post).astype(np.float32)
        off = sum(len(a) for a in pre)
        if ev_span is not None and post:
            ev_span = (ev_span[0] + off, ev_span[1] + off)
        # 2. room
        if self.p_room > 0 and rng.random() < self.p_room:
            room = rng.choice([r for r in ROOMS if self.rir[r]] or [None])
            if room:
                dry = apply_rir(dry, self._read(rng.choice(self.rir[room])))
                desc.append(f"room:{room}")
        out = dry
        # 3. background scene over the whole clip, with a noise-only lead-in / tail
        if self.bg and self.p_bg > 0 and (rng.random() < self.p_bg or not desc):
            scene = rng.choice(sorted(self.bg))
            lead, tail = int(rng.uniform(0.2, 0.8) * SR), int(rng.uniform(0.1, 0.5) * SR)
            out = np.concatenate([np.zeros(lead, np.float32), dry, np.zeros(tail, np.float32)])
            dry = out.copy()
            off += lead
            ev_span = (ev_span[0] + lead, ev_span[1] + lead) if ev_span else None
            clip = self.bg[scene][int(rng.integers(len(self.bg[scene])))]
            nz = self._read(clip["path"], len(out), rng)
            snr = float(rng.uniform(0, 25))
            out = out + nz * (lvl * 10 ** (-snr / 20) / (rms(nz) + 1e-9))
            desc.append(f"bg:{scene}@{snr:.0f}dB")
        # 4. channel
        if self.p_phone > 0 and rng.random() < self.p_phone:
            out, phone = phone_channel(out), True
            desc.append("phone")
        # digital-silence padding (gaps, lead-in) would give -100 dB energy targets: add a -90 dBFS floor
        floor = rng.normal(0, 3e-5, len(dry)).astype(np.float32)
        dry = dry + floor
        out = out + floor[: len(out)] if len(out) == len(dry) else out
        if not desc:  # nothing applied (e.g. in-voice-only mode for a speaker without nonverbal recordings)
            return None
        peak = np.abs(out).max()
        if peak > 0.99:  # keep relative levels; the dry copy gets the same gain
            out, dry = out * (0.99 / peak), dry * (0.99 / peak)
        return dict(y=out.astype(np.float32), dry=dry.astype(np.float32), text=text, off=off,
                    env=env_vector(scene, room, phone, snr), desc=" ".join(desc), ev_span=ev_span,
                    ev_tok=tok if ev_span else None)


# --------------------------------------------------------------------------------------------- library build
def _seg_events(y, min_s=0.3, max_s=3.0, top_db=30):
    """Cut a recording of repeated sounds (laughs, coughs ...) into single events by energy."""
    frame = 240
    n = len(y) // frame
    e = 20 * np.log10(np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, 1) + 1e-9) + 1e-9)
    on = e > e.max() - top_db
    on = np.convolve(on, np.ones(15), "same") > 0  # bridge gaps < ~0.15 s
    out, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            s, t = max(0, i * frame - 1200), min(len(y), j * frame + 1200)
            if min_s * SR <= t - s <= max_s * SR:
                out.append(y[s:t])
            elif t - s > max_s * SR:  # long burst: keep its first max_s with a fade
                seg = y[s:s + int(max_s * SR)].copy(); seg[-1200:] *= np.linspace(1, 0, 1200)
                out.append(seg)
            i = j
        else:
            i += 1
    return out


def _write(y, kind, name):
    import soundfile as sf
    p = NOISE / kind / f"{name}.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, y.astype(np.float32), SR, subtype="PCM_16")
    return str(p)


def _load(path, ch=None):
    import soundfile as sf
    import soxr
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y[:, ch] if ch is not None else y.mean(1)
    return soxr.resample(y, sr, SR).astype(np.float32) if sr != SR else y


def build():
    rows = []
    # EARS nonverbals: per speaker, in-voice
    for f in sorted((RAW / "ears").glob("p[0-9][0-9][0-9]/*.wav")):
        tok = EARS_NV.get(f.stem)
        if not tok:
            continue
        spk = f"ears_{f.parent.name}"
        for k, seg in enumerate(_seg_events(_load(f))):
            rows.append(dict(kind="nv", label=tok, speaker=spk, src="ears", dur=len(seg) / SR,
                             path=_write(seg, "nv", f"{spk}_{f.stem}_{k:02d}")))
    print("nonverbal", sum(r["kind"] == "nv" for r in rows), flush=True)
    # DEMAND: channel 1 of each 5-min environment, 15 s chunks
    for env, scene in DEMAND.items():
        d = NOISE_RAW / "demand" / env
        if not d.exists():
            continue
        y = _load(d / "ch01.wav")
        for k in range(0, len(y) - 5 * SR, 15 * SR):
            rows.append(dict(kind="bg", label=scene, src=f"demand/{env}",
                             dur=min(15.0, (len(y) - k) / SR), path=_write(y[k:k + 15 * SR], "bg", f"demand_{env}_{k // SR:04d}")))
    print("demand", sum(r["src"].startswith("demand") for r in rows), flush=True)
    # RIRs: simulated rooms (and real RIRs) binned by measured T60
    rir_root = NOISE_RAW / "rirs" / "RIRS_NOISES"
    for f in sorted(rir_root.glob("simulated_rirs/*/Room*/*.wav"))[::10] + sorted(rir_root.glob("real_rirs_isotropic_noises/*rir*.wav")):
        if "noise" in f.name:
            continue
        h = _load(f, ch=0)
        t = t60(h)
        room = "small" if t < 0.35 else "medium" if t < 0.8 else "large"
        rows.append(dict(kind="rir", label=room, t60=round(t, 3), src="rirs_noises",
                         dur=len(h) / SR, path=_write(h[: int(1.5 * SR)], "rir", f.parent.name + "_" + f.stem)))
    print("rir", {r: sum(x["kind"] == "rir" and x["label"] == r for x in rows) for r in ROOMS}, flush=True)
    # unlabelled public-domain / CC sources: MUSAN noise + music, RIRS point-source noises -> AST AudioSet tagging
    cand = []
    for f in sorted((NOISE_RAW / "musan_x" / "musan" / "noise").rglob("*.wav")):
        cand.append((f, "musan/noise", ""))
    for f in sorted((NOISE_RAW / "musan_x" / "musan" / "music").rglob("*.wav"))[::4]:
        cand.append((f, "musan/music", ""))
    for f in sorted(rir_root.glob("pointsource_noises/*.wav")):
        cand.append((f, "rirs/pointsource", ""))
    rows += tag_and_cut(cand)
    rows += fsd50k_rows()
    with open(META / "noise_lib.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    import collections
    print("wrote", META / "noise_lib.jsonl", len(rows), dict(collections.Counter((r["kind"], r["label"]) for r in rows)))


def _cut_event(y):
    """Loudest <= 3 s of an event recording, faded."""
    n = len(y) // 240
    if n < 4:
        return y
    e = np.convolve(np.sqrt(np.mean(y[: n * 240].reshape(n, 240) ** 2, 1)), np.ones(25), "same")
    s = max(0, int(np.argmax(e)) * 240 - SR)
    seg = y[s:s + 3 * SR].copy()
    fade = min(1200, len(seg) // 4)
    seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade)
    return seg


def fsd50k_rows():
    """FSD50K (parquet mirror: labels + audio bytes per row) -> library rows for clips whose labels we map.
    Only the mapped label and our own file name are kept (no uploader / title / Freesound id)."""
    import io
    import glob
    import os
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr
    shards = [f for f in sorted(glob.glob(str(NOISE_RAW / "fsd50k_pq" / "data" / "*.parquet"))) if os.path.exists(f + ".done")]
    rows, n = [], 0
    for si, f in enumerate(shards):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=["labels", "audio"]).to_pylist():
                n += 1
                hit = next((FSD50K[lab] for lab in r["labels"].split(",") if lab in FSD50K), None)
                if hit is None:
                    continue
                try:
                    y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
                except Exception:  # noqa: BLE001
                    continue
                y = y.mean(1)
                y = soxr.resample(y, sr, SR).astype(np.float32) if sr != SR else y
                if len(y) < 0.3 * SR:
                    continue
                kind, lab = hit
                kind = "event" if kind == "event_h" else kind  # other people's laughs/coughs: background events
                base = dict(kind=kind, label=lab, src="fsd50k")
                name = f"fsd_{si:02d}_{n:06d}"
                if kind == "bg":
                    if len(y) < 3 * SR:
                        continue
                    for k in range(0, max(1, len(y) - 3 * SR), 15 * SR):
                        rows.append(dict(base, dur=min(15.0, (len(y) - k) / SR), path=_write(y[k:k + 15 * SR], "bg", f"{name}_{k // SR:03d}")))
                else:
                    seg = _cut_event(y)
                    rows.append(dict(base, dur=len(seg) / SR, path=_write(seg, "event", name)))
        print(f"  fsd50k shard {si + 1}/{len(shards)}: {len(rows)} rows from {n} clips", flush=True)
    return rows


def tag_and_cut(cand, dev="cuda"):
    """AST (AudioSet) top labels -> our scene/event vocabulary; backgrounds cut to 15 s, events to their loudest 3 s."""
    import torch
    import soxr
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe, m = ASTFeatureExtractor.from_pretrained(name), ASTForAudioClassification.from_pretrained(name).to(dev).eval()
    labels = m.config.id2label
    rules = [(re.compile(p), v) for p, v in AUDIOSET]
    rows = []
    for i, (f, src, lic) in enumerate(cand):
        try:
            y = _load(f)
        except Exception as e:  # noqa: BLE001
            print("skip", f, e); continue
        if len(y) < 0.3 * SR:
            continue
        x = soxr.resample(y[: 10 * SR], SR, 16000)
        with torch.no_grad():
            p = torch.sigmoid(m(**fe(x, sampling_rate=16000, return_tensors="pt").to(dev)).logits[0]).cpu().numpy()
        top = [(labels[j], float(p[j])) for j in np.argsort(-p)[:5]]
        hit = None
        for lab, sc in top:
            if sc < 0.15 or lab in ("Speech", "Male speech, man speaking", "Female speech, woman speaking"):
                continue
            hit = next((v for rx, v in rules if rx.match(lab)), None)
            if hit:
                break
        if src == "musan/music":
            hit = ("bg", "music")
        if not hit:
            continue
        kind, lab = hit
        base = dict(kind=kind, label=lab, src=src, ast=top[:3])
        stem = f"{src.replace('/', '_')}_{f.stem}"
        if kind == "bg":
            for k in range(0, max(1, len(y) - 3 * SR), 15 * SR):
                seg = y[k:k + 15 * SR]
                rows.append(dict(base, dur=len(seg) / SR, path=_write(seg, "bg", f"{stem}_{k // SR:04d}")))
        else:
            seg = _cut_event(y)
            rows.append(dict(base, dur=len(seg) / SR, path=_write(seg, "event", stem)))
        if i % 200 == 0:
            print(f"  tagged {i}/{len(cand)}: {f.name} -> {hit} {top[:2]}", flush=True)
    return rows


if __name__ == "__main__":
    {"build": build}[sys.argv[1]]()
