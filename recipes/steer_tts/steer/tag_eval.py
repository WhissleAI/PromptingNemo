"""Tag accuracy beyond the harness's 7 tags. Every request is paired with the same voice + text rendered with no
request, as in steer.harness.

  emotion  each style tag the emotion probe ("ear": NEU HAP SAD ANG FEA DIS, steer.ear) can judge
           (rewards.EMO_TARGET: 13 tags); hit = probe(request - reference) picks the target class
  events   inline tokens [laugh] [cough] [dog] ... (checkpoints trained with --noise); hit = AudioSet (AST) score of
           the token's classes rises by > 0.1 and ends above 0.2
  scenes   environment requests (--noise checkpoints); same AST criterion on the scene's classes
  clean    plain renders with no request: share where no noise / event class scores above 0.2

python -m steer.tag_eval --ckpt runs/X/last.pt --out runs/X/tag_eval.json [--guidance 1.5]
"""
import argparse
import json

import numpy as np
import torch

from .rewards import AUDIOSET_NV, EMO_TARGET

TEXTS = ["The meeting has been moved to Thursday afternoon at three.",
         "Please remember to bring the blue folder when you come in.",
         "I put the keys on the table next to the kitchen window.",
         "She said the package should arrive sometime early next week.",
         "We will need two more chairs for the people arriving later.",
         "Turn left at the second light and park behind the building."]


class Judges:
    def __init__(self, dev="cuda"):
        import re
        from transformers import ASTForAudioClassification, WavLMModel
        from .acoustic import AUDIOSET
        from .config import META
        from .ear import Ear, EMOS, LAYER
        ck = torch.load(META / "ear.pt", map_location="cpu")
        self.ear = Ear(); self.ear.load_state_dict(ck["state"]); self.ear = self.ear.to(dev).eval()
        self.emos, self.layer = EMOS, LAYER
        self.wl = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(dev).eval()
        self.ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
        rules = [(re.compile(p), v) for p, v in AUDIOSET] + [(re.compile(p), ("event", t)) for p, t in AUDIOSET_NV]
        self.cls = {}
        for i, lab in self.ast.config.id2label.items():
            for rx, v in rules:
                if rx.match(lab):
                    self.cls.setdefault(v, []).append(int(i))
        self.noise_ids = sorted({i for v in self.cls.values() for i in v})
        self.dev = dev

    @staticmethod
    def _16k(y):
        import soxr
        return torch.from_numpy(soxr.resample(np.asarray(y, np.float32), 24000, 16000))

    @torch.no_grad()
    def emo_emb(self, y):
        x = self._16k(y)[None].to(self.dev)
        h = self.wl(x, output_hidden_states=True).hidden_states[self.layer]
        return h.mean(1)

    @torch.no_grad()
    def ast_p(self, y):
        import torchaudio.compliance.kaldi as K
        x = self._16k(y)
        fb = K.fbank(x[None], htk_compat=True, sample_frequency=16000, use_energy=False, window_type="hanning",
                     num_mel_bins=128, dither=0.0, frame_shift=10)
        fb = torch.nn.functional.pad(fb, (0, 0, 0, max(0, 1024 - fb.shape[0])))[:1024]
        fb = (fb - (-4.2677393)) / (4.5689974 * 2)
        return torch.sigmoid(self.ast(input_values=fb[None].to(self.dev)).logits[0]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--voices", type=int, default=4)
    a = ap.parse_args()
    from .acoustic import SCENES
    from .synth import Synth
    from .text import EVENT_TOKENS
    s = Synth(a.ckpt, guidance=a.guidance)
    J = Judges()
    voices = [v for v in s.stats["speakers"] if v.startswith("ears")][: a.voices]
    res = {"emotion": {}, "events": {}, "scenes": {}}
    ref = {(v, t): s(t, v) for v in voices for t in TEXTS}
    ref_emb = {k: J.emo_emb(y) for k, y in ref.items()}
    ref_ast = {k: J.ast_p(y) for k, y in ref.items()}
    res["clean"] = float(np.mean([p[J.noise_ids].max() < 0.2 for p in ref_ast.values()]))
    for tag, emo in EMO_TARGET.items():
        if tag not in s.tag2i:
            continue
        hits = []
        for (v, t), e0 in ref_emb.items():
            p = torch.softmax(J.ear(J.emo_emb(s(t, v, tag=tag)) - e0), -1)[0]
            hits.append(int(p.argmax()) == J.emos.index(emo))
        res["emotion"][tag] = round(float(np.mean(hits)), 3)
    noise_model = getattr(s.model, "env_proj", None) is not None
    if noise_model:
        for tok in EVENT_TOKENS:
            if tok.startswith("reserved") or ("event", tok) not in J.cls:
                continue
            hits = []
            for (v, t), p0 in ref_ast.items():
                p = J.ast_p(s(f"{t} [{tok}]", v))
                c = J.cls[("event", tok)]
                hits.append(p[c].max() > 0.2 and p[c].max() - p0[c].max() > 0.1)
            res["events"][tok] = round(float(np.mean(hits)), 3)
        for sc in SCENES:
            if ("bg", sc) not in J.cls:
                continue
            hits = []
            for (v, t), p0 in list(ref_ast.items())[:8]:
                p = J.ast_p(s(t, v, env=dict(scene=sc, snr=10)))
                c = J.cls[("bg", sc)]
                hits.append(p[c].max() > 0.2 and p[c].max() - p0[c].max() > 0.1)
            res["scenes"][sc] = round(float(np.mean(hits)), 3)
    summ = {k: round(float(np.mean(list(v.values()))), 3) for k, v in res.items() if isinstance(v, dict) and v}
    res["summary"] = dict(summ, clean=round(res["clean"], 3))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["summary"]), "\n", json.dumps(res["emotion"]), "\n", json.dumps(res["events"]),
          "\n", json.dumps(res["scenes"]))


if __name__ == "__main__":
    main()
