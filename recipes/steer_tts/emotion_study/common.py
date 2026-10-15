"""Shared paths and constants for the CREMA-D emotion study."""
import os
from pathlib import Path

CREMA = Path(os.environ.get("CREMA_D_DIR", "CREMA-D"))  # dir holding AudioWAV/ and processedResults/
WAV = CREMA / "AudioWAV"
OUT = Path(os.environ.get("EMOTION_STUDY_OUT", Path(__file__).resolve().parent / "out"))
FRAMES = OUT / "frames"  # per-clip frame-level npz
FIG = OUT / "fig"
for d in (OUT, FRAMES, FIG):
    d.mkdir(parents=True, exist_ok=True)

SR = 16000
HOP = 0.010  # 10 ms frame hop for all frame-level features

EMOS = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]
EMO_LETTER = {"N": "NEU", "H": "HAP", "S": "SAD", "A": "ANG", "F": "FEA", "D": "DIS"}
# validated (dataviz validator, adjacent pairs, light+dark) in this order: ANG SAD HAP FEA DIS
EMO_COLOR = {"ANG": "#e34948", "SAD": "#2a78d6", "HAP": "#eda100", "FEA": "#4a3aa7",
             "DIS": "#008300", "NEU": "#898781"}
EMO_COLOR_DARK = {"ANG": "#e66767", "SAD": "#3987e5", "HAP": "#c98500", "FEA": "#9085e9",
                  "DIS": "#008300", "NEU": "#898781"}

SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}
