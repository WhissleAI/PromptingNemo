"""Reward fine-tuning losses (differentiable, ReFL/DRaFT-style): steer -> free-run -> vocode -> frozen scorers.

Every reward step takes K items of the supervised batch, gives each a STEERING REQUEST (a style tag drawn from the
emotion set, and/or 1-2 perturbed controls, as in the stage-2 consistency loss), free-runs the model on it, vocodes
the mel with the frozen Vocos (differentiable iSTFT), and backpropagates frozen-scorer losses into the TTS model:

  control   readback (acoustic dims) + soft-duration timing of the requested controls    steer.consistency.score
  leak      for pure control requests (tag unchanged): every NON-requested dim of the steered render must measure
            as the same voice/text rendered with the original settings (no-grad pair, the harness's leak). Without
            it the control reward is gamed by moving every dim at once (attempt 1, 2026-10-09: leak 0.29 -> 0.62)
  emotion   ear probe on WavLM-base+ L6, paired against the same voice/text rendered with no request
            (exactly the harness's measurement), -log p(target emotion)                    only for emotion requests
  ctc       intelligibility guard: CTC loss of the transcript under frozen wav2vec2-base-960h
  utmos     naturalness guard: -UTMOS22-strong
  spk       voice guard: 1 - cos(WavLM-base-plus-sv(generated), the item's reference speaker vector)   (spk_enc)
  anchor    trust region: the batch's own (unsteered) conditions free-run by the model and by a frozen copy of the
            starting model (self.ref, set by train.py) must agree per phoneme (log-duration, pitch, energy) and in mean
            spectrum - the deterministic-model analogue of the KL term in RLHF; stops quality drift on plain renders
  env       acoustic environment (models trained with --noise): a frozen AudioSet tagger (AST) on differentiable
            fbank features. Requests sample a background scene and/or an inline sound event ([dog], [laugh] at the
            start or end of the text); loss = -log p(requested scene classes) - log p(requested event classes), and
            for items with NO environment request + log(1 - p(any noise/event class)): output stays clean unless asked

The supervised losses on the same batch stay in the objective (the anchor against reward hacking); weights via
--reward_w in train.py. Durations are rounded in free_run (no gradient through them): timing rewards use the soft
predicted durations, everything else flows through pitch/energy/decoder. Uses level-normalised readback files
(readback_ln.pt / spk_ref_mel_ln.npy; python -m steer.readback train --level_norm) for --spk_enc runs.
"""
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F

from .config import CONTROL_DIMS

# style tag -> ear class (ear: NEU HAP SAD ANG FEA DIS, trained on CREMA-D crowd labels)
EMO_TARGET = {"anger": "ANG", "irritation": "ANG", "frustration": "ANG", "sadness": "SAD", "disappointment": "SAD",
              "happiness": "HAP", "excited": "HAP", "amusement": "HAP", "fear": "FEA", "anxiety": "FEA", "panic": "FEA",
              "disgust": "DIS", "contempt": "DIS"}
REQUEST_TAGS = ["anger", "sadness", "happiness", "fear", "disgust", "excited", "anxiety"]
# AudioSet classes for in-voice nonverbals (the scene / environmental-event rules are steer.acoustic.AUDIOSET)
AUDIOSET_NV = [(r"^(Laughter|Giggle|Chuckle, chortle|Belly laugh|Snicker)$", "laugh"), (r"^(Cough)$", "cough"),
               (r"^(Sneeze)$", "sneeze"), (r"^(Sigh)$", "sigh"), (r"^(Breathing|Gasp|Pant)$", "breath"),
               (r"^(Crying, sobbing|Baby cry, infant cry|Whimper)$", "cry"), (r"^(Screaming)$", "scream"),
               (r"^(Cheering)$", "cheer"), (r"^(Yell|Shout)$", "yell"), (r"^(Throat clearing)$", "throat"),
               (r"^(Sniff)$", "sniff")]


class RewardLoss:
    def __init__(self, stats, device, weights, level_norm=True, k=4):
        from vocos import Vocos
        from transformers import AutoFeatureExtractor, WavLMModel, WavLMForXVector, Wav2Vec2ForCTC, Wav2Vec2Processor
        from .consistency import Consistency
        from .ear import Ear, EMOS, LAYER
        from .config import META
        self.w, self.k, self.dev = weights, k, device
        self.tags = stats["tags"]
        self.req_tags = [t for t in REQUEST_TAGS if t in self.tags]
        self.cons = Consistency(stats, device, level_norm=level_norm)
        # voicing / periodicity are not requested by the control reward: whisper is a categorical mode that saturates
        # at full devoicing (CPPS -> 0; Hillenbrand et al. 1994, 1996), but the learned readback keeps asking for
        # more, and v2 overshot the whisper control to 1.84x. Paired real data coaches whisper instead.
        from .consistency import PERT_DIMS
        self.cons.pert_dims = [d for d in PERT_DIMS if d not in ("voiced", "cpps")]
        frozen = lambda m: m.to(device).eval().requires_grad_(False)
        self.vocos = frozen(Vocos.from_pretrained("charactr/vocos-mel-24khz"))
        if weights.get("emotion"):
            ck = torch.load(META / "ear.pt", map_location="cpu")
            self.ear = Ear(); self.ear.load_state_dict(ck["state"]); self.ear = frozen(self.ear)
            self.emos, self.layer = EMOS, LAYER
            self.wl = frozen(WavLMModel.from_pretrained("microsoft/wavlm-base-plus"))
        if weights.get("ctc"):
            self.asr_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.asr = frozen(Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h"))
        if weights.get("utmos"):
            self.utmos = frozen(torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True))
        if weights.get("spk"):
            self.sv = frozen(WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv"))
        if weights.get("env"):
            from transformers import ASTForAudioClassification
            from .acoustic import AUDIOSET, SCENES
            self.ast = frozen(ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593"))
            labels = self.ast.config.id2label
            rules = [(re.compile(p), v) for p, v in AUDIOSET] + [(re.compile(p), ("event", t)) for p, t in AUDIOSET_NV]
            self.cls = {}  # ("bg", scene) / ("event", token) -> [AudioSet class ids]
            for i, lab in labels.items():
                for rx, v in rules:
                    if rx.match(lab):
                        self.cls.setdefault(v, []).append(int(i))
            self.noise_ids = sorted({i for v in self.cls.values() for i in v})
            self.env_scenes = [s for s in SCENES if ("bg", s) in self.cls]
            self.env_events = [t for (k, t) in self.cls if k == "event"]

    # ------------------------------------------------------------------ helpers
    def _wav16(self, mel, lens):
        import torchaudio.functional as AF
        # free-run renders can run far longer than their source ("slower" requests): those OOMed the WavLM emotion judge
        # at 306.6k even with sources <= 500 frames. Judges see at most REWARD_MAX_RENDER frames (~8 s); for the CTC
        # reward a truncated render reads as missing words, a mild push against runaway durations on the rare long ones
        cap = int(os.environ.get("REWARD_MAX_RENDER", 750))
        mel, lens = mel[..., :cap], lens.clamp_max(cap)
        y = self.vocos.decode(mel.float())                                       # [B, T*256] @ 24 kHz
        n = (lens * 256).clamp_max(y.shape[1])
        y = y * (torch.arange(y.shape[1], device=y.device)[None] < n[:, None])
        return AF.resample(y, 24000, 16000), (n * 2 // 3)

    @staticmethod
    def _norm16(y, n):  # zero-mean / unit-variance over the valid samples (what HF feature extractors do)
        m = (torch.arange(y.shape[1], device=y.device)[None] < n[:, None]).float()
        mu = (y * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
        sd = (((y - mu) * m) ** 2).sum(1, keepdim=True).div(m.sum(1, keepdim=True)).sqrt().clamp_min(1e-5)
        return (y - mu) / sd * m, m

    @staticmethod
    def _mask16(y, n):
        return (torch.arange(y.shape[1], device=y.device)[None] < n[:, None]).float()

    # input conventions match each model's HF feature extractor: WavLM (base-plus, base-plus-sv) take RAW audio with
    # an attention mask; wav2vec2-base-960h takes zero-mean/unit-var audio and NO attention mask (padding = zeros)
    def _ear_emb(self, y16, n):
        m = self._mask16(y16, n)
        h = self.wl(y16 * m, attention_mask=m.long(), output_hidden_states=True).hidden_states[self.layer]
        fm = F.interpolate(m[:, None], size=h.shape[1], mode="nearest")[:, 0]
        return (h * fm[..., None]).sum(1) / fm.sum(1, keepdim=True).clamp_min(1)

    def _ast_logp(self, y16, n16):
        """AST (AudioSet) log-sigmoid class scores [B, 527] of 16 kHz audio, differentiable w.r.t. the audio:
        the same kaldi fbank (128 bins, 10 ms) and normalisation as ASTFeatureExtractor, computed in torch."""
        import torchaudio.compliance.kaldi as K
        feats = []
        for b in range(y16.shape[0]):
            x = y16[b, : max(int(n16[b]), 400)].float()
            fb = K.fbank(x[None], htk_compat=True, sample_frequency=16000,
                         use_energy=False, window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
            fb = F.pad(fb, (0, 0, 0, max(0, 1024 - fb.shape[0])))[:1024]
            feats.append((fb - (-4.2677393)) / (4.5689974 * 2))
        return F.logsigmoid(self.ast(input_values=torch.stack(feats)).logits.float())

    def env_request(self, batch, idx, ids, tl):
        """Per selected item: maybe a background scene (+ SNR) and/or an edge sound event inserted into the text.
        Returns (env [K, ENV_DIM], ids, text_lens, targets list of (kind, label) per item; [] = clean request)."""
        from .acoustic import ENV_DIM, env_vector
        from .text import to_phonemes
        K = len(idx)
        env = torch.zeros(K, ENV_DIM, device=ids.device)
        targets, new_ids = [], []
        for j, i in enumerate(idx.tolist()):
            t, tg = batch["text"][i], []
            r = random.random()
            if r < 0.6:  # 60% environment requests, 40% clean (the leak guard)
                if random.random() < 0.7 and self.env_scenes:
                    sc = random.choice(self.env_scenes)
                    env[j] = torch.from_numpy(env_vector(sc, snr=random.uniform(3, 20)))
                    tg.append(("bg", sc))
                if (random.random() < 0.5 or not tg) and self.env_events:
                    ev = random.choice(self.env_events)
                    t = f"[{ev}] {t}" if random.random() < 0.5 else f"{t} [{ev}]"
                    tg.append(("event", ev))
            targets.append(tg)
            new_ids.append(to_phonemes(t)[0])
        L = max(len(x) for x in new_ids)
        ids2 = torch.zeros(K, L, dtype=torch.long, device=ids.device)
        for j, x in enumerate(new_ids):
            ids2[j, : len(x)] = torch.tensor(x, device=ids.device)
        return env, ids2, torch.tensor([len(x) for x in new_ids], device=ids.device), targets

    def request(self, batch, idx):
        """Steering request per selected item: (tag, ctrl, cmask, pert, emotion target or None)."""
        c, m, pert = self.cons.perturb(batch["ctrl"][idx], batch["cmask"][idx])
        tag = batch["tag"][idx].clone()
        target = []
        for j in range(len(idx)):
            if self.req_tags and random.random() < 0.5:
                t = random.choice(self.req_tags)
                tag[j] = self.tags.index(t)
                target.append(self.emos.index(EMO_TARGET[t]) if hasattr(self, "emos") else None)
                if random.random() < 0.5:  # pure tag request half of the time: no control perturbation
                    pert[j] = 0; m[j] = batch["cmask"][idx][j]; c[j] = batch["ctrl"][idx][j]
            else:
                target.append(None)
        return tag, c, m, pert, target

    # ------------------------------------------------------------------ loss
    def __call__(self, model, batch):
        B = batch["ids"].shape[0]
        # reward items come from utterances <= ~7 s: the frozen judges (wav2vec2 / WavLM / AST) run WITH gradients on
        # free-run renders, and four long ones OOMed the 12 GB GPU (v3, 2026-10-09 17:30)
        # free-run renders of longer items OOM-crashed the judges (WavLM attention) at 305.1k: cap (env REWARD_MAX_FRAMES)
        short = torch.nonzero(batch["mel_lens"].to(batch["ids"].device) <= int(os.environ.get("REWARD_MAX_FRAMES", 650))).flatten()
        # no fallback to the batch's shortest items: batches are length-bucketed, so in a long bucket "the shortest 4"
        # were ~1000-frame clips whose renders the judge cap (REWARD_MAX_RENDER) then cut off - the CTC reward read the
        # cut as missing words (r_ctc 0.5 -> 5 at 306.6k). Fewer than k short items: use those; none: no reward step
        if len(short) == 0:
            return None
        idx = short[torch.randperm(len(short), device=short.device)[: self.k]]
        ids, tl, spk = batch["ids"][idx], batch["text_lens"][idx], batch["spk"][idx]
        sv = batch["spk_vec"][idx] if "spk_vec" in batch else None
        tag, c, m, pert, target = self.request(batch, idx)
        env, env_t = None, None
        if self.w.get("env") and getattr(model, "env_proj", None) is not None:
            env, ids, tl, env_t = self.env_request(batch, idx, ids, tl)
        g = model.condition(spk, tag, c, m, sv, env)
        mel, lens, aux = model.free_run(ids, tl, g)
        out = {}
        if self.w.get("control") or self.w.get("leak"):
            meas = self.cons.measure(ids, spk, mel, lens, aux)
        if self.w.get("control"):
            cl = self.cons.score(ids, spk, c, m, pert, mel, lens, aux, meas=meas)
            out["r_control"] = cl["cons_ac"] + cl["cons_t"]
        if self.w.get("leak") or self.w.get("anchor"):
            # the same voice/text under the batch's own conditions (no request): leak reference and trust region
            c0, m0, t0 = batch["ctrl"][idx], batch["cmask"][idx], batch["tag"][idx]
            with torch.set_grad_enabled(bool(self.w.get("anchor"))):
                mel0, lens0, aux0 = model.free_run(ids, tl, model.condition(spk, t0, c0, m0, sv, env))
            if self.w.get("anchor") and getattr(self, "ref", None) is not None:
                with torch.no_grad():
                    melr, lensr, auxr = self.ref.free_run(ids, tl, self.ref.condition(spk, t0, c0, m0, sv, env))
                tm = aux0["tmask"].float()
                an = sum((((aux0[k].float() - auxr[k].float()) ** 2) * tm).sum() / tm.sum().clamp_min(1)
                         for k in ("logdur", "pitch", "energy"))
                mmean = lambda x, n: (x.float() * (torch.arange(x.shape[2], device=x.device)[None] < n[:, None])[:, None]).sum(2) / n[:, None].clamp_min(1)
                out["r_anchor"] = an + ((mmean(mel0, lens0) - mmean(melr, lensr)) ** 2).mean()
            if self.w.get("leak"):
                keep = (tag == t0) & (pert.sum(1) > 0)  # pure control requests
                if keep.any():
                    with torch.no_grad():
                        meas0, _ = self.cons.measure(ids, spk, mel0.detach(), lens0, {k: v.detach() for k, v in aux0.items()})
                    w = (1 - pert) * meas[1][None] * keep[:, None].float()
                    out["r_leak"] = (F.smooth_l1_loss(meas[0], meas0, reduction="none") * w).sum() / w.sum().clamp_min(1)
        need_wav = any(self.w.get(k) for k in ("emotion", "ctc", "utmos", "spk"))
        if need_wav:
            y16, n16 = self._wav16(mel, lens)
        if self.w.get("emotion") and any(t is not None for t in target):
            sel = [j for j, t in enumerate(target) if t is not None]
            with torch.no_grad():  # reference: same voice + text, no request (the harness's pairing)
                g0 = model.condition(spk[sel], torch.full_like(tag[sel], self.tags.index("none")),
                                     torch.zeros_like(c[sel]), torch.zeros_like(m[sel]), sv[sel] if sv is not None else None)
                mel0, lens0, _ = model.free_run(ids[sel], tl[sel], g0)
                y0, n0 = self._wav16(mel0, lens0)
                e0 = self._ear_emb(y0, n0)
            e = self._ear_emb(y16[sel], n16[sel])
            logp = torch.log_softmax(self.ear(e - e0), -1)
            out["r_emotion"] = -logp[torch.arange(len(sel)), torch.tensor([target[j] for j in sel])].mean()
        # CTC only on renders the judge cap did not cut (a cut render reads as missing words: r_ctc 0.5 -> 1.0)
        whole = (lens <= int(os.environ.get("REWARD_MAX_RENDER", 750))).nonzero().flatten() if need_wav else None
        if self.w.get("ctc") and whole is not None and len(whole):
            x, _ = self._norm16(y16[whole], n16[whole])
            logits = self.asr(x).logits.float()
            texts = [re.sub(r"[^A-Z' ]", "", batch["text"][i].upper()).strip() for i in idx[whole].tolist()]
            lab = [self.asr_proc.tokenizer(t.replace(" ", "|")).input_ids for t in texts]
            in_len = (n16[whole].float() / y16.shape[1] * logits.shape[1]).long().clamp_min(1).cpu()
            tgt = torch.tensor([t for l in lab for t in l], dtype=torch.long)
            out["r_ctc"] = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), tgt, in_len,
                                      torch.tensor([len(l) for l in lab]), blank=self.asr_proc.tokenizer.pad_token_id,
                                      zero_infinity=True, reduction="mean")
        if self.w.get("utmos"):
            out["r_utmos"] = -self.utmos(y16, 16000).mean()
        if self.w.get("spk") and sv is not None:
            msk = self._mask16(y16, n16)
            e = F.normalize(self.sv(input_values=y16 * msk, attention_mask=msk.long()).embeddings, dim=-1)
            out["r_spk"] = (1 - (e * F.normalize(sv, dim=-1)).sum(-1)).mean()
        if env_t is not None:
            lp = self._ast_logp(y16, n16)  # [K, 527]
            terms = []
            for j, tg in enumerate(env_t):
                if tg:  # requested scene / event present (max over the label's AudioSet classes)
                    for v in tg:
                        terms.append(-lp[j, self.cls[v]].max())
                else:  # clean request: no noise/event class may fire
                    p_any = lp[j, self.noise_ids].exp().max()
                    terms.append(-torch.log1p(-p_any.clamp(max=1 - 1e-4)))
            out["r_env"] = torch.stack(terms).mean()
        out["reward_total"] = sum(self.w.get(k[2:], 0.0) * v for k, v in out.items() if k.startswith("r_"))
        return out
