"""Load a SteerTTS checkpoint and synthesize with explicit controls.

python -m steer.synth --ckpt runs/base/last.pt --text "..." --speaker voice_0042 --tag anger \
    --ctrl "loud=2,f0_med=1.5" --out out.wav
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import META, SR, CONTROL_DIMS
from .data import PITCH_SCALE
from .model import SteerTTS
from .text import to_phonemes


class Synth:
    def __init__(self, ckpt, device="cuda", guidance=1.0, flow=None, flow_steps=10, flow_temp=0.4, vocoder=None,
                 flow_cfg=0.0):
        """flow: optional steer.flow checkpoint (flow-matching refiner of this base's mels); vocoder: optional
        fine-tuned Vocos state (steer.vocoder_ft) in place of the pretrained charactr/vocos-mel-24khz."""
        self.dev, self.guidance = device, guidance  # guidance: default steering scale (see SteerTTS.infer)
        ckpt = str(ckpt)
        pflow = None
        if ",pflow=" in ckpt:  # "...,pflow=p.pt": a prompted flow, used only when a reference clip is given
            ckpt, pflow = ckpt.split(",pflow=", 1)
        if ",flow=" in ckpt:  # "base.pt,flow=flow.pt": any tool that takes a checkpoint path can run the flow path
            ckpt, flow = ckpt.split(",flow=", 1)
        ck = torch.load(ckpt, map_location=device)
        # speaker/tag order of THIS checkpoint: its run dir's copy (META's is rewritten when the corpus grows)
        run_stats = Path(ckpt).parent / "control_stats.json"
        self.stats = json.load(open(run_stats if run_stats.exists() else META / "control_stats.json"))
        assert len(self.stats["speakers"]) == ck["n_speakers"], (ckpt, "speaker table / control_stats mismatch")
        self.model = SteerTTS(ck["n_speakers"], ck["n_tags"], **ck.get("arch", {})).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.spk2i = {s: i for i, s in enumerate(self.stats["speakers"])}
        # abs_z ablation checkpoints take corpus-wide z-scores: requests (speaker-relative z, like every other model)
        # are mapped through the voice's absolute neutral level (_abs_anchor)
        self.gstats, self._anchor = None, {}
        if ck.get("arch", {}).get("ctrl_norm") == "global":
            self.gstats = json.load(open(Path(ckpt).parent / "abs_stats.json"))
        self.tag2i = {t: i for i, t in enumerate(self.stats["tags"])}
        # reference-conditioned checkpoints: known voices use their stored mean neutral vector (spk_ref.npz);
        # any other voice can be cloned from a reference clip via ref_wav=
        self.spk_ref, self._xv = None, None
        if ck.get("arch", {}).get("spk_enc_dim"):
            z = np.load(Path(ckpt).parent / "spk_ref.npz")
            self.spk_ref = dict(zip(z["speakers"].tolist(), z["vec"]))
        from vocos import Vocos
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        if vocoder:
            self.vocos.load_state_dict(torch.load(vocoder, map_location="cpu", weights_only=False)["vocos"])
        self.vocos = self.vocos.to(device).eval()
        self.flow, self.flow_steps, self.flow_temp, self.flow_cfg = None, flow_steps, flow_temp, flow_cfg
        if flow:
            from .flow import Flow
            self.flow = Flow.load(flow, device)
        self.pflow = None
        if pflow:
            from .flow import Flow
            self.pflow = Flow.load(pflow, device)

    @torch.no_grad()
    def prompt_mel(self, ref_wav, mel, max_s=8.0):
        """Log-mel of the reference clip (first max_s seconds), level-matched to this render: training mels are
        level-normalised per speaker, so the prompt is shifted to put its loud half of frames at the same mean as
        the base mel's (log-mel + c == the waveform scaled by e^c)."""
        from .features import mel_extractor
        sr = SR
        if isinstance(ref_wav, (str, Path)):
            import soxr
            y, sr = sf.read(str(ref_wav), dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            y = soxr.resample(y, sr, SR) if sr != SR else y
        else:
            y = np.asarray(ref_wav, np.float32)
        y = y[: int(max_s * SR)]
        if len(y) < SR // 2:
            return None
        pm = mel_extractor()(torch.from_numpy(y)[None]).to(self.dev).float()  # [1,100,T]
        mel = mel.float()
        # band-limited references (CV3-Eval prompts are 16 kHz: nothing above 8 kHz, the top ~15 of 100 bands): the
        # prompted flow would copy the empty highs as part of the voice, and the empty bands would skew the level match.
        # Match levels on the bands the clip has, and fill the missing ones from this render's own spectral shape,
        # anchored to the clip's top valid band.
        nb = pm.shape[1]
        if sr < SR:
            hz = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + (SR / 2) / 700), nb + 2) / 2595) - 1)  # HTK mel edges
            kb = int(np.searchsorted(hz[2:], 0.95 * sr / 2))  # bands whose upper edge is inside the clip's bandwidth
            nb = max(8, min(kb, pm.shape[1]))
        loud = lambda m: m[:, :nb].mean(1)[0].sort().values[m.shape[-1] // 2:].mean()  # louder half, valid bands
        pm = pm + (loud(mel) - loud(pm))
        if nb < pm.shape[1]:
            shape = mel.mean(-1, keepdim=True)  # [1,100,1] the render's mean spectrum
            pm[:, nb:] = pm[:, nb - 1:nb] + (shape[:, nb:] - shape[:, nb - 1:nb])
        return pm

    def control_tensors(self, ctrl: dict, speaker=None, ref_wav=None):
        c = torch.zeros(1, len(CONTROL_DIMS), device=self.dev)
        m = torch.zeros(1, len(CONTROL_DIMS), device=self.dev)
        anchor = self._abs_anchor(speaker, ref_wav) if (self.gstats is not None and ctrl) else None
        for k, v in (ctrl or {}).items():
            i = CONTROL_DIMS.index(k)
            v = float(v)
            if anchor is not None:  # speaker-relative z -> raw -> corpus-wide z (what the abs_z model was trained on)
                v = (anchor[k] + v * self.stats["pooled_sd"][k] - self.gstats["mu"][k]) / self.gstats["sd"][k]
                v = float(np.clip(v, -5, 5))
            c[0, i], m[0, i] = v, 1.0
        return c, m

    def _abs_anchor(self, speaker, ref_wav):
        """abs_z: raw neutral level of this voice per dim. Training voices: their stored neutral mean; a cloned voice:
        measured on its reference clip (steer.features.measure, no transcript: rate falls back to the mean neutral
        rate over training speakers, as does any dim the clip leaves undefined)."""
        key = str(ref_wav) if isinstance(ref_wav, (str, Path)) else (None if ref_wav is not None else speaker)
        if key is not None and key in self._anchor:
            return self._anchor[key]
        fb = {k: float(np.nanmean([v[k] for v in self.stats["spk_mean"].values()])) for k in CONTROL_DIMS}
        if ref_wav is None and speaker in self.stats.get("spk_mean", {}):
            a = {k: (fb[k] if v is None or not np.isfinite(v) else float(v)) for k, v in self.stats["spk_mean"][speaker].items()}
        elif ref_wav is not None:
            from .features import measure
            y = ref_wav
            if isinstance(ref_wav, (str, Path)):
                import soxr
                y, sr = sf.read(str(ref_wav), dtype="float32")
                y = y.mean(1) if y.ndim > 1 else y
                y = soxr.resample(y, sr, SR) if sr != SR else y
            ctl = measure(np.asarray(y, np.float32), n_phones=None, want_mel=False)["controls"]
            a = {k: (float(x) if np.isfinite(x) else fb[k]) for k, x in zip(CONTROL_DIMS, ctl)}
        else:
            a = fb
        if key is not None:
            self._anchor[key] = a
        return a

    @torch.no_grad()
    def speaker_vector(self, speaker=None, ref_wav=None):
        """[1,512] tensor from a reference clip (float32 array at 24 kHz, or a path), else the stored known-voice vector."""
        if ref_wav is not None:
            if self._xv is None:
                from .xvec import XVec
                self._xv = XVec(self.dev)
            if isinstance(ref_wav, (str, Path)):
                import soxr
                y, sr = sf.read(str(ref_wav), dtype="float32")
                y = y.mean(1) if y.ndim > 1 else y
                ref_wav = soxr.resample(y, sr, SR) if sr != SR else y
            v = self._xv([np.asarray(ref_wav, np.float32)])[0]
        else:
            if speaker not in self.spk_ref:  # public name (voice_0042)
                from .voices import resolve
                speaker = resolve(speaker)
            v = self.spk_ref[speaker]
        return torch.from_numpy(np.asarray(v, np.float32))[None].to(self.dev)

    def __call__(self, text, speaker, tag="none", ctrl=None, pace=1.0, word_shift=None, return_mel=False, ref_wav=None,
                 env=None, guidance=None):
        """env: optional acoustic environment dict(scene=, room=, phone=, snr=) (steer.acoustic) for checkpoints
        trained with --noise; ignored by older ones, which also skip inline sound-event tokens they never learned."""
        if speaker not in self.spk2i:  # public name (voice_0042) of a training speaker -> its internal id
            from .voices import resolve
            speaker = resolve(speaker)
        ids, widx, words = to_phonemes(text)
        n_sym = self.model.emb.num_embeddings
        if max(ids) >= n_sym:
            keep = [k for k, i in enumerate(ids) if i < n_sym]
            ids, widx = [ids[k] for k in keep], [widx[k] for k in keep]
        ids_t = torch.tensor([ids], device=self.dev)
        widx_t = torch.tensor([widx], device=self.dev)
        c, m = self.control_tensors(ctrl, speaker, ref_wav)
        spk = torch.tensor([self.spk2i.get(speaker, len(self.spk2i))], device=self.dev)  # unknown -> last row
        spk_vec = self.speaker_vector(speaker, ref_wav) if self.spk_ref is not None else None
        tg = torch.tensor([self.tag2i.get(tag, 0)], device=self.dev)
        if word_shift:  # user units: pitch in semitones, energy in dB, dur factor
            word_shift = {w: dict(pitch=s.get("pitch", 0) / PITCH_SCALE, energy=s.get("energy", 0) / 10.0,
                                  dur=s.get("dur", 1.0)) for w, s in word_shift.items()}
        env_t = None
        if env and self.model.env_proj is not None:
            from .acoustic import env_vector
            env_t = torch.from_numpy(env_vector(**env))[None].to(self.dev)
        kw = dict(env=env_t) if env_t is not None else {}  # only --noise models take env
        s = self.guidance if guidance is None else guidance
        if s != 1.0 and (ctrl or tag not in (None, "none")):  # guide from the no-request render of this voice/text
            kw.update(guidance=s, base=(torch.zeros_like(tg), torch.zeros_like(c), torch.zeros_like(m)))
        mel, aux = self.model.infer(ids_t, spk, tg, c, m, pace=pace, word_idx=widx_t, word_shift=word_shift,
                                    spk_vec=spk_vec, **kw)
        fl = self.pflow if (ref_wav is not None and self.pflow is not None) else self.flow
        if fl is not None:
            pm = self.prompt_mel(ref_wav, mel) if (ref_wav is not None and getattr(fl.net, "prompt", False)) else None
            if pm is None:
                mel = fl.sample(mel.float(), aux["g"].float(), steps=self.flow_steps, temp=self.flow_temp, seed=0)
            else:  # prompted flow: the reference clip's own mel as context before the target (Voicebox-style infill)
                P = pm.shape[-1]
                full = torch.cat([pm, mel.float()], -1)
                cm = torch.zeros_like(full[:, :1]); cm[..., :P] = 1
                mel = fl.sample(full, aux["g"].float(), steps=self.flow_steps, temp=self.flow_temp, seed=0,
                                ctx=full, cm=cm, cfg=self.flow_cfg)[..., P:]
        wav = self.vocos.decode(mel.float()).squeeze().cpu().numpy()
        return (wav, mel) if return_mel else wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--tag", default="none")
    ap.add_argument("--ctrl", default="")
    ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--out", default="out.wav")
    a = ap.parse_args()
    ctrl = dict(kv.split("=") for kv in a.ctrl.split(",") if kv)
    s = Synth(a.ckpt)
    wav = s(a.text, a.speaker, a.tag, {k: float(v) for k, v in ctrl.items()}, a.pace)
    sf.write(a.out, wav, SR)
    print(a.out, len(wav) / SR, "s")


if __name__ == "__main__":
    main()
