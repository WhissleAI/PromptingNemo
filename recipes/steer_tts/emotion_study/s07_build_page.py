"""Inline report data + example audio (data: URIs) into viz/template.html -> viz/emotion_in_the_voice.html."""
import base64
import json
import re
from pathlib import Path

VIZ = Path(__file__).resolve().parent / "viz"
data = (VIZ / "viz_data.js").read_text()
data = re.sub(r"^window\.VIZ = |;\s*$", "", data.strip())
audio = {p.stem: "data:audio/wav;base64," + base64.b64encode(p.read_bytes()).decode() for p in sorted((VIZ / "audio").glob("*.wav"))}
html = (VIZ / "template.html").read_text()
html = html.replace("__DATA__", data).replace("__AUDIO__", json.dumps(audio))
out = VIZ / "emotion_in_the_voice.html"
out.write_text(html)
print(out, round(out.stat().st_size / 1e6, 2), "MB")
