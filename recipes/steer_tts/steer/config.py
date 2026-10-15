"""Paths and constants for steer_tts."""
import os
from pathlib import Path

# data/ and runs/ live under STEER_TTS_ROOT (default: this recipe directory)
ROOT = Path(os.environ.get("STEER_TTS_ROOT", Path(__file__).resolve().parents[1]))
RAW = ROOT / "data" / "raw"
FEATS = ROOT / "data" / "feats"  # per-utterance npz
META = ROOT / "data" / "meta"  # manifests, stats, controls
RUNS = ROOT / "runs"
for d in (FEATS, META, RUNS):
    d.mkdir(parents=True, exist_ok=True)

# audio / mel: matches charactr/vocos-mel-24khz so its pretrained vocoder can be used directly
SR = 24000
N_FFT = 1024
HOP = 256
N_MELS = 100
FPS = SR / HOP  # 93.75 mel frames per second

# utterance-level control dimensions, all expressed as speaker-relative z-scores
# (vs the same speaker's neutral speech); the same quantities are read back for evaluation
CONTROL_DIMS = [
    "rate",      # articulation rate, phones per second of speech (pauses excluded)
    "loud",      # mean RMS level of speech frames, dB
    "f0_med",    # median F0, semitones
    "f0_iqr",    # F0 interquartile range, semitones
    "f0_end",    # F0 slope over the final 300 ms of voicing, st/s
    "pause",     # share of the speech span that is silent pauses >= 100 ms
    "tilt",      # alpha ratio: energy 1-5 kHz vs 50 Hz-1 kHz, dB
    "voiced",    # voiced share of speech frames (whisper -> low)
    "flat",      # spectral flatness of speech frames, log (breathy/whisper -> high)
    "centroid",  # spectral centroid of speech frames, log Hz (brightness / articulation)
    "cpps",      # smoothed cepstral peak prominence, dB (periodicity; fear/breathiness -> low)
    "h1h2",      # H1-H2, dB (breathy -> high)
]

# the timing / level / pitch subset (prosody_only ablation); the rest are voice-quality dims
PROSODY_DIMS = CONTROL_DIMS[:6]

# unified style tag vocabulary across corpora ("none" = unknown / unspecified)
EARS_STYLES = ["regular", "loud", "whisper", "highpitch", "lowpitch", "fast", "slow"]
EARS_EMOTIONS = ["adoration", "amazement", "amusement", "anger", "confusion", "contentment", "cuteness", "desire",
                 "disappointment", "disgust", "distress", "embarassment", "extasy", "fear", "guilt", "interest",
                 "neutral", "pain", "pride", "realization", "relief", "sadness", "serenity"]
EXPRESSO_STYLES = ["default", "confused", "enunciated", "happy", "laughing", "sad", "whisper", "emphasis",
                   "essentials", "longform", "singing", "narration"]


def canonical_tag(corpus: str, raw: str) -> str:
    raw = raw.lower()
    if corpus == "libritts":
        return "none"
    if raw in ("regular", "default", "neutral", "essentials", "longform", "narration"):
        return "neutral"
    alias = {"happy": "happiness", "sad": "sadness", "confused": "confusion", "embarassment": "embarrassment",
             "extasy": "ecstasy", "laughing": "laughing", "highpitch": "high_pitch", "lowpitch": "low_pitch"}
    return alias.get(raw, raw)
