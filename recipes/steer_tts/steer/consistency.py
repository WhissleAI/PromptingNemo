"""Stage-2 control-consistency (cycle) loss.

Perturb 1-2 control dims per utterance, free-run the model, and require the output to *measure* as asked:
  - acoustic dims (RB_DIMS) via the frozen readback net on the generated mel
  - rate / pause from the predicted (soft) durations, f0_end from the predicted phone pitch
The independent DSP harness (steer.harness) is the evaluator; the readback net is only a training signal.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from .config import META, CONTROL_DIMS, FPS
from .data import PITCH_SCALE
from .readback import Readback, RB_DIMS, RB_IDX
from .text import SYMBOLS, SP

PERT_DIMS = RB_DIMS + ["rate", "pause", "f0_end"]
_PUNCT = torch.tensor([i for i, s in enumerate(SYMBOLS) if len(s) == 1 and s not in ("<pad>",)])


class Consistency:
    def __init__(self, stats, device, level_norm=False):
        sfx = "_ln" if level_norm else ""  # spk_enc runs generate level-normalised mels (data.LEVEL_DB)
        ck = torch.load(META / f"readback{sfx}.pt", map_location=device)
        self.rb = Readback().to(device).eval()
        self.rb.load_state_dict(ck["state"])
        for p in self.rb.parameters():
            p.requires_grad_(False)
        refs = np.load(META / f"spk_ref_mel{sfx}.npy", allow_pickle=True).item()
        spk = stats["speakers"]
        self.ref = torch.tensor(np.stack([refs.get(s, np.full(100, -6.0, np.float32)) for s in spk]), device=device)
        mu = lambda k: torch.tensor([stats["spk_mean"][s].get(k, np.nan) for s in spk], device=device).nan_to_num(0)
        self.mu = {k: mu(k) for k in ("rate", "pause", "f0_end")}
        self.sd = {k: float(stats["pooled_sd"][k]) for k in CONTROL_DIMS}
        self.dev = device

    def perturb(self, ctrl, cmask):
        B = ctrl.shape[0]
        c, m = ctrl.clone(), cmask.clone()
        pert = torch.zeros_like(m)
        dims = getattr(self, "pert_dims", PERT_DIMS)
        for b in range(B):
            for d in np.random.choice(len(dims), min(len(dims), np.random.randint(1, 3)), replace=False):
                i = CONTROL_DIMS.index(dims[d])
                base = c[b, i] if m[b, i] > 0 else torch.zeros((), device=c.device)
                step = float(np.random.uniform(1.0, 2.5) * np.random.choice([-1, 1]))
                c[b, i] = (base + step).clamp(-4, 4)
                m[b, i] = 1.0
                pert[b, i] = 1.0
        return c, m, pert

    def __call__(self, model, batch):
        ids, tl, spk = batch["ids"], batch["text_lens"], batch["spk"]
        c, m, pert = self.perturb(batch["ctrl"], batch["cmask"])
        g = model.condition(spk, batch["tag"], c, m, batch.get("spk_vec"))
        mel, lens, aux = model.free_run(ids, tl, g)
        return self.score(ids, spk, c, m, pert, mel, lens, aux)

    def measure(self, ids, spk, mel, lens, aux):
        """What a free-run output measures as, in control units: [B, N_CTRL] (z for every RB dim and for rate /
        pause / f0_end; other dims 0) and a [N_CTRL] mask of the dims measured. Differentiable."""
        out = torch.zeros(mel.shape[0], len(CONTROL_DIMS), device=mel.device)
        have = torch.zeros(len(CONTROL_DIMS), device=mel.device)
        out[:, RB_IDX] = self.rb(mel.float(), lens, self.ref[spk]).float()
        have[RB_IDX] = 1
        # timing from soft predicted durations
        tm = aux["tmask"].float()
        dur = (torch.exp(aux["logdur"].float()) - 1).clamp_min(0) * tm
        is_ph = tm * (ids != SP).float() * (~torch.isin(ids, _PUNCT.to(ids.device))).float()
        # inner span: from first to last phone token
        pos = torch.arange(ids.shape[1], device=ids.device)[None].float()
        first = torch.where(is_ph > 0, pos, torch.full_like(pos, 1e9)).amin(1, keepdim=True)
        last = torch.where(is_ph > 0, pos, torch.full_like(pos, -1)).amax(1, keepdim=True)
        inner = ((pos >= first) & (pos <= last)).float() * tm
        span = (dur * inner).sum(1).clamp_min(5)
        gap = inner * (1 - is_ph)
        pause = (dur * gap * torch.sigmoid((dur - 9.0) / 2.0)).sum(1)
        rate = is_ph.sum(1) / ((span - pause).clamp_min(5) / FPS)
        # end contour: duration-weighted slope over the last ~3 phones (semitones / s)
        p = aux["pitch"].float() * PITCH_SCALE
        idx_last = torch.argsort(torch.where(is_ph > 0, pos, torch.full_like(pos, -1)), dim=1, descending=True)[:, :4]
        pl = torch.gather(p, 1, idx_last)  # last, 2nd last, ...
        dl = torch.gather(dur, 1, idx_last)
        t_span = (dl[:, :3].sum(1) / FPS).clamp_min(0.05)
        slope = (pl[:, 0] - pl[:, 3]) / t_span
        for k, v in (("rate", (rate - self.mu["rate"][spk]) / self.sd["rate"]),
                     ("pause", (pause / span - self.mu["pause"][spk]) / self.sd["pause"]),
                     ("f0_end", (slope - self.mu["f0_end"][spk]) / self.sd["f0_end"])):
            out[:, CONTROL_DIMS.index(k)] = v.float()
            have[CONTROL_DIMS.index(k)] = 1
        return out, have

    def score(self, ids, spk, c, m, pert, mel, lens, aux, meas=None):
        """Control-consistency losses for an already free-run batch (shared with steer.rewards)."""
        meas, have = meas if meas is not None else self.measure(ids, spk, mel, lens, aux)
        tidx = [CONTROL_DIMS.index(k) for k in ("rate", "pause", "f0_end")]
        w = m * (1 + 2 * pert) * have  # perturbed dims weigh 3x
        err = F.smooth_l1_loss(meas, c, reduction="none") * w
        l_ac = err[:, RB_IDX].sum() / w[:, RB_IDX].sum().clamp_min(1)
        l_t = err[:, tidx].sum() / w[:, tidx].sum().clamp_min(1)
        return dict(cons_ac=l_ac, cons_t=l_t)
