"""Intelligibility guard: Whisper transcription + WER."""
import re

import numpy as np
import torch

_m = None


def _norm(s):
    import inflect
    eng = inflect.engine()
    s = re.sub(r"\d+", lambda m: " " + eng.number_to_words(m.group(0)) + " ", s)
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return s.split()


def wer(ref, hyp):
    r, h = _norm(ref), _norm(hyp)
    d = np.arange(len(h) + 1)
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[-1] / max(1, len(r))


class ASR:
    def __init__(self, name="openai/whisper-base.en", device="cuda"):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.p = WhisperProcessor.from_pretrained(name)
        self.m = WhisperForConditionalGeneration.from_pretrained(name).to(device).eval()
        self.dev = device

    @torch.no_grad()
    def __call__(self, wav24):
        import soxr
        y = soxr.resample(wav24.astype(np.float32), 24000, 16000)
        f = self.p(y, sampling_rate=16000, return_tensors="pt").input_features.to(self.dev)
        out = self.m.generate(f, max_new_tokens=200, max_length=None)
        return self.p.batch_decode(out, skip_special_tokens=True)[0].strip()
