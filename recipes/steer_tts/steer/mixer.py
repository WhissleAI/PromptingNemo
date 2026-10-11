"""Inference-time acoustic environment: real recordings mixed around the synthesized speech.

The L1 mel decoder cannot learn stochastic textures (v2, 2026-10-09: the AudioSet tagger found 0/24 trained events
and 0/13 scenes in its renders), so scenes / rooms / phone lines / environmental events are rendered here from the
noise library (steer.acoustic: DEMAND, MUSAN, FSD50K, RIRS_NOISES) instead of by the model:

  scene    a background clip of that scene, continuous across streamed chunks (pauses carry the ambience),
           mixed at the requested SNR against the running speech level (default 12 dB)
  room     a real room impulse response of that size class, convolved with each chunk (direct path kept at t=0)
  phone    300-3400 Hz band-limit + 8-bit mu-law, applied last
  events   environmental sound tokens ([dog] [knock] [door] [phone] ...) become their own chunks: a real clip of
           that class at a level relative to the speech (default -3 dB)
In-voice nonverbals ([laugh] [cough] [sigh] ...): voice_nv() inserts a REAL nonverbal of the EARS speaker whose
x-vector is closest to the current voice (the speaker's own for EARS voices) - used until the TTS model renders
them itself (v3 at 303k: +0.2 s, AST p(laughter) 0.00).

    mx = EnvMixer()
    st = mx.start(dict(scene="cafe", room="medium", phone=False, snr=10), seed=0)
    y = mx.process(speech_chunk, st)        # per streamed chunk, in order
    y = mx.event("dog", st)                 # an event chunk
"""
import json

import numpy as np

from .acoustic import ROOMS, SCENES, active_rms, apply_rir, phone_channel, rms
from .config import META, SR
from .text import EVENTS

ENV_EVENTS = set(EVENTS)
NV_MIN_S = {"laugh": 0.8, "cry": 0.8, "cheer": 0.8, "scream": 0.6, "yell": 0.6, "yawn": 0.8}  # rendered by the mixer; NONVERBAL tokens stay in the TTS text


class EnvMixer:
    def __init__(self, lib_path=META / "noise_lib.jsonl"):
        self.bg, self.ev, self.rir, self.nv = {}, {}, {r: [] for r in ROOMS}, {}
        for l in open(lib_path):
            r = json.loads(l)
            if r["kind"] == "bg" and r.get("dur", 0) >= 5:
                self.bg.setdefault(r["label"], []).append(r["path"])
            elif r["kind"] == "event":
                self.ev.setdefault(r["label"], []).append(r["path"])
            elif r["kind"] == "rir" and r["label"] in self.rir:
                self.rir[r["label"]].append(r["path"])
            elif r["kind"] == "nv":
                self.nv.setdefault(r["label"], {}).setdefault(r["speaker"], []).append((r["path"], r.get("dur", 0.0)))
        self._spk_vec = None

    @staticmethod
    def _read(p):
        import soundfile as sf
        return sf.read(p, dtype="float32")[0]

    def start(self, env, seed=0):
        """Per-render state: the chosen background clip + its cursor, RIR, levels."""
        env = env or {}
        rng = np.random.default_rng(seed)
        st = dict(rng=rng, env=env, cursor=0, level=0.05, bg=None, rir=None)
        sc = env.get("scene")
        if sc in self.bg:
            st["bg"] = self._read(self.bg[sc][int(rng.integers(len(self.bg[sc])))])
            st["cursor"] = int(rng.integers(0, max(1, len(st["bg"]) // 2)))
        room = env.get("room")
        if room in self.rir and self.rir[room]:
            st["rir"] = self._read(self.rir[room][int(rng.integers(len(self.rir[room])))])
        return st

    def _bg(self, n, st):
        bg = st["bg"]
        out = np.empty(n, np.float32)
        k = 0
        while k < n:  # loop the clip with a short crossfade-free wrap (clips are >= 5 s)
            take = min(n - k, len(bg) - st["cursor"])
            out[k:k + take] = bg[st["cursor"]:st["cursor"] + take]
            k += take
            st["cursor"] = (st["cursor"] + take) % len(bg)
        return out

    def process(self, y, st, speech=True):
        """One streamed chunk (speech or pause) -> the chunk inside the environment."""
        y = np.asarray(y, np.float32)
        if speech and len(y) and rms(y) > 1e-4:
            st["level"] = 0.7 * st["level"] + 0.3 * active_rms(y) if st.get("seen") else active_rms(y)
            st["seen"] = True
        if st["rir"] is not None and len(y):
            y = apply_rir(y, st["rir"]) if rms(y) > 1e-5 else y
        if st["bg"] is not None and len(y):
            snr = float(st["env"].get("snr", 12.0))
            nz = self._bg(len(y), st)
            y = y + nz * (st["level"] * 10 ** (-snr / 20) / (rms(st["bg"]) + 1e-9))
        if st["env"].get("phone") and len(y):
            y = phone_channel(y)
        peak = np.abs(y).max() if len(y) else 0
        return (y * (0.99 / peak) if peak > 0.99 else y).astype(np.float32)

    def event(self, label, st, rel_db=-3.0):
        """An environmental event chunk (a real clip), then through the same room / background / phone."""
        if label not in self.ev:
            return np.zeros(0, np.float32)
        e = self._read(self.ev[label][int(st["rng"].integers(len(self.ev[label])))])
        e = e * (st["level"] * 10 ** (rel_db / 20) / (active_rms(e) + 1e-9))
        return self.process(e, st, speech=False)

    def nv_speakers(self, spk_ref):
        """spk_ref: {speaker: x-vector} (Synth.spk_ref); keeps the EARS speakers that have nonverbal clips."""
        have = {s for d in self.nv.values() for s in d}
        names = [s for s in spk_ref if s in have]
        self._spk_vec = (names, np.stack([np.asarray(spk_ref[s], np.float32) for s in names])) if names else None

    def voice_nv(self, label, st, spk_vec):
        """A real nonverbal (laugh / cough / cry ...) of the closest-voiced EARS speaker who recorded one."""
        if label not in self.nv or self._spk_vec is None:
            return np.zeros(0, np.float32)
        names, V = self._spk_vec
        v = np.asarray(spk_vec, np.float32).reshape(-1)
        sims = V @ v / (np.linalg.norm(V, axis=1) * np.linalg.norm(v) + 1e-9)
        for i in np.argsort(-sims):
            if names[i] in self.nv[label]:
                clips = self.nv[label][names[i]]
                # the segmenter also keeps single "ha" fragments (0.3 s): prefer whole events
                long = [p for p, d in clips if d >= NV_MIN_S.get(label, 0.4)]
                pick = long or [max(clips, key=lambda c: c[1])[0]]
                y = self._read(pick[int(st["rng"].integers(len(pick)))])
                y = y * (st["level"] / (active_rms(y) + 1e-9))  # at the speech level
                return self.process(y, st, speech=False)
        return np.zeros(0, np.float32)
