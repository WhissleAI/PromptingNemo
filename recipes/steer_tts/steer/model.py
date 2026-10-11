"""SteerTTS: a small explicit-variance TTS rendered from an interpretable control vector.

text (ARPAbet) -> encoder -> per-phone duration / pitch / energy predictors -> length regulator
-> decoder -> 100-band log-mel (Vocos-compatible). Every block is conditioned through adaptive
LayerNorm on g = MLP([speaker, style tag, control z-vector * mask, mask]).
Durations come from a jointly trained RAD-TTS-style aligner (forward-sum + binarisation, MAS).
"""
import math
import sys
from pathlib import Path

import numba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_MELS, CONTROL_DIMS
from .text import SYMBOLS, PAD, N_BASE

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # PromptingNemo repo root, for promptingnemo.*
from promptingnemo.models.mamba3 import Mamba3Layer, scan_forward_ssd  # noqa: E402

N_CTRL = len(CONTROL_DIMS)


# ----------------------------------------------------------------------------- helpers
def seq_mask(lens, max_len=None):
    max_len = max_len or int(lens.max())
    return torch.arange(max_len, device=lens.device)[None] < lens[:, None]


class SinusoidalPos(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, T, device):
        pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.d, 2, device=device).float() * (-math.log(10000.0) / self.d))
        pe = torch.zeros(T, self.d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe


class AdaLN(nn.Module):
    """LayerNorm whose scale/shift are predicted from the conditioning vector (zero-init = plain LN)."""

    def __init__(self, d, g_dim):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.proj = nn.Linear(g_dim, 2 * d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, g):  # x [B,T,d], g [B,g]
        s, b = self.proj(g)[:, None].chunk(2, -1)
        return self.ln(x) * (1 + s) + b


class BiMamba3(nn.Module):
    """Bidirectional Mamba-3 token mixer: a forward and a backward Mamba3Layer, summed.

    TTS sees the whole utterance, so the causal cell runs both ways. The backward pass reverses
    each sequence within its OWN length, so padding comes after the valid frames in both
    directions and never enters the state. Runs the SSD fast path in fp32 (the chunk decays are
    exp of log-cumsums, which bf16 autocast would quantise). zero_init zeroes both out_proj so the
    block starts as an exact no-op - used when adding it to an already-trained model.
    """

    def __init__(self, d, d_state=32, head_dim=32, rank=1, chunk=64, zero_init=False):
        super().__init__()
        self.fwd = Mamba3Layer(d, d_state=d_state, rank=rank, head_dim=head_dim)
        self.bwd = Mamba3Layer(d, d_state=d_state, rank=rank, head_dim=head_dim)
        self.chunk = chunk
        if zero_init:
            for l in (self.fwd, self.bwd):
                nn.init.zeros_(l.out_proj.weight)
                nn.init.zeros_(l.out_proj.bias)

    compiled = None  # set by enable_compile(): the same body under torch.compile, shared by every instance

    def forward(self, x, mask):  # x [B,T,d], mask [B,T] True=valid
        # eager for inference (no grad): batch-1 free-running synthesis trips an Inductor cumsum codegen bug
        # (torch 2.11, "list indices must be integers or slices, not NoneType"), and validation is rare anyway
        if BiMamba3.compiled is None or torch.compiler.is_compiling() or not torch.is_grad_enabled():
            return BiMamba3.body(self, x, mask)
        # pad T to a chunk multiple (padding is masked, so the result is unchanged) and mark B/T dynamic,
        # so one compiled graph covers every batch shape instead of recompiling per length
        T = x.shape[1]
        pad = (-T) % self.chunk
        if pad:
            x, mask = F.pad(x, (0, 0, 0, pad)), F.pad(mask, (0, pad))
        for t in (x, mask):  # maybe_: x and mask share B and T, which a strict mark would call a violation
            torch._dynamo.maybe_mark_dynamic(t, 0); torch._dynamo.maybe_mark_dynamic(t, 1)
        return BiMamba3.compiled(self, x, mask)[:, :T]

    @staticmethod
    def enable_compile():
        """Fuse the Mamba-3 elementwise/einsum chain (~12k kernel launches per training step eager, so the
        step was CPU launch-bound). dynamic=True: T changes every batch; inlined nn.Module params mean one
        graph serves all layers instead of one compile per instance."""
        BiMamba3.compiled = torch.compile(BiMamba3.body, dynamic=True)

    def body(self, x, mask):
        T = x.shape[1]
        t = torch.arange(T, device=x.device)[None]
        lens = mask.sum(1, keepdim=True)
        rev = torch.where(t < lens, lens - 1 - t, t)[..., None].expand(-1, -1, x.shape[2])  # an involution
        with torch.autocast(x.device.type, enabled=False):
            x = (x * mask[..., None]).float()
            y = scan_forward_ssd(self.fwd, x, self.chunk, vectorized=True)
            y = y + torch.gather(scan_forward_ssd(self.bwd, torch.gather(x, 1, rev), self.chunk, vectorized=True), 1, rev)
        return y * mask[..., None]


class FFTBlock(nn.Module):
    """Pre-norm block: token mixer + conv FFN, all norms adaptive on g.

    mixer: "attn"   multi-head self-attention (the original FastPitch block)
           "mamba3" bidirectional Mamba-3 in place of attention
           "hybrid" attention, then a zero-initialised bidirectional Mamba-3 branch (fine-tuning: the
                    block is exactly the attention block at init and learns the SSM path on top)
    """

    def __init__(self, d, heads, ffn, k, g_dim, p=0.1, mixer="attn", ssm_state=32):
        super().__init__()
        assert mixer in ("attn", "mamba3", "hybrid"), mixer
        self.mixer = mixer
        self.n1 = AdaLN(d, g_dim)
        if mixer == "mamba3":
            self.mix = BiMamba3(d, d_state=ssm_state)
        else:
            self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        if mixer == "hybrid":
            self.n_ssm = AdaLN(d, g_dim)
            self.ssm = BiMamba3(d, d_state=ssm_state, zero_init=True)
        self.n2 = AdaLN(d, g_dim)
        self.c1 = nn.Conv1d(d, ffn, k, padding=k // 2)
        self.c2 = nn.Conv1d(ffn, d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x, mask, g):  # mask [B,T] True=valid
        h = self.n1(x, g)
        if self.mixer == "mamba3":
            h = self.mix(h, mask)
        else:
            h, _ = self.att(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + self.drop(h)
        if self.mixer == "hybrid":
            x = x + self.drop(self.ssm(self.n_ssm(x, g), mask))
        h = self.n2(x, g).transpose(1, 2)
        h = self.c2(F.gelu(self.c1(h))).transpose(1, 2)
        x = x + self.drop(h)
        return x * mask[..., None]


class Stack(nn.Module):
    def __init__(self, n, d, heads, ffn, k, g_dim, p, mixer="attn", ssm_state=32):
        super().__init__()
        self.pos = SinusoidalPos(d)
        self.layers = nn.ModuleList([FFTBlock(d, heads, ffn, k, g_dim, p, mixer, ssm_state) for _ in range(n)])
        self.out_norm = AdaLN(d, g_dim)

    compiled = None  # set by enable_compile(); one graph for every encoder/decoder Stack

    @staticmethod
    def enable_compile():
        """Compile whole stacks (attention, AdaLN, conv FFN and the Mamba-3 branch together): with only the
        Mamba-3 branch compiled the step is still launch-bound on the eager blocks around it."""
        Stack.compiled = torch.compile(Stack.body, dynamic=True)

    def forward(self, x, mask, g):
        if Stack.compiled is None:
            return self.body(x, mask, g)
        T = x.shape[1]  # pad to a 64 multiple (masked, so the valid rows are unchanged): bounds recompiles
        pad = (-T) % 64
        if pad:
            x, mask = F.pad(x, (0, 0, 0, pad)), F.pad(mask, (0, pad))
        for t in (x, mask):
            torch._dynamo.maybe_mark_dynamic(t, 0); torch._dynamo.maybe_mark_dynamic(t, 1)
        return Stack.compiled(self, x, mask, g)[:, :T]

    def body(self, x, mask, g):
        x = x + self.pos(x.shape[1], x.device)[None]
        for l in self.layers:
            x = l(x, mask, g)
        return self.out_norm(x, g) * mask[..., None]


class VariancePredictor(nn.Module):
    def __init__(self, d, g_dim, k=3, p=0.15):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, k, padding=k // 2)
        self.n1 = AdaLN(d, g_dim)
        self.c2 = nn.Conv1d(d, d, k, padding=k // 2)
        self.n2 = AdaLN(d, g_dim)
        self.out = nn.Linear(d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x, mask, g):
        h = F.relu(self.c1(x.transpose(1, 2))).transpose(1, 2)
        h = self.drop(self.n1(h, g))
        h = F.relu(self.c2(h.transpose(1, 2))).transpose(1, 2)
        h = self.drop(self.n2(h, g))
        return self.out(h).squeeze(-1) * mask


# ----------------------------------------------------------------------------- aligner
class Aligner(nn.Module):
    """RAD-TTS / FastPitch-1.1 style unsupervised aligner (soft attention mel->text with a prior)."""

    def __init__(self, d_text, n_mel=N_MELS, n_att=80, temperature=0.0005):
        super().__init__()
        self.t = temperature
        self.key = nn.Sequential(nn.Conv1d(d_text, d_text, 3, padding=1), nn.ReLU(), nn.Conv1d(d_text, n_att, 1))
        self.query = nn.Sequential(nn.Conv1d(n_mel, 2 * n_mel, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(2 * n_mel, n_mel, 1), nn.ReLU(), nn.Conv1d(n_mel, n_att, 1))

    def forward(self, text_emb, mel, text_mask, prior):
        # text_emb [B,N,d], mel [B,100,T], text_mask [B,N], prior [B,T,N]
        k = self.key(text_emb.transpose(1, 2)).float()  # [B,att,N]
        q = self.query(mel).float()  # [B,att,T]
        # squared distance via ||q||^2 + ||k||^2 - 2 q.k (no [B,att,T,N] blow-up)
        dist = (q.pow(2).sum(1)[:, :, None] + k.pow(2).sum(1)[:, None, :] - 2 * torch.bmm(q.transpose(1, 2), k))
        logp = -self.t * dist
        logp = F.log_softmax(logp.masked_fill(~text_mask[:, None], -1e4), dim=-1) + torch.log(prior + 1e-8)
        logp = logp.masked_fill(~text_mask[:, None], -1e4)
        soft = torch.softmax(logp, dim=-1)
        return soft, logp  # [B,T,N]


@numba.njit(cache=True)
def _mas(logp):
    T, N = logp.shape
    Q = np.full((T, N), -1e30)
    Q[0, 0] = logp[0, 0]
    for t in range(1, T):
        for n in range(0, min(N, t + 1)):
            stay = Q[t - 1, n]
            move = Q[t - 1, n - 1] if n > 0 else -1e30
            Q[t, n] = max(stay, move) + logp[t, n]
    dur = np.zeros(N, np.int64)
    n = N - 1
    for t in range(T - 1, -1, -1):
        dur[n] += 1
        if t > 0 and n > 0 and Q[t - 1, n - 1] >= Q[t - 1, n]:
            n -= 1
    return dur


@numba.njit(cache=True, parallel=True)
def _mas_batch(logp, text_lens, mel_lens):
    out = np.zeros((logp.shape[0], logp.shape[2]), np.int64)
    for b in numba.prange(logp.shape[0]):
        T, N = mel_lens[b], text_lens[b]
        out[b, :N] = _mas(np.ascontiguousarray(logp[b, :T, :N]))
    return out


def mas_durations(soft, text_lens, mel_lens):
    logp = torch.log(soft.detach().float() + 1e-8).cpu().numpy()
    out = _mas_batch(logp, text_lens.cpu().numpy().astype(np.int64), mel_lens.cpu().numpy().astype(np.int64))
    return torch.from_numpy(out).to(soft.device, non_blocking=True)


def forward_sum_loss(logp, text_lens, mel_lens, blank_logprob=-1.0):
    """One batched CTC call (was a per-utterance loop: ~10 kernels x B per step). Columns past each text's
    length are already -1e4 in logp, so after the log-softmax they carry exp(-1e4) = 0 probability and the
    result equals the per-utterance loss over lp[b, :T, :N+1]; reduction="mean" divides by N_b then averages."""
    lp = F.pad(logp, (1, 0), value=blank_logprob).float().log_softmax(-1)  # blank at index 0, [B,T,N+1]
    N = logp.shape[2]
    tgt = torch.arange(1, N + 1, device=lp.device)[None].expand(lp.shape[0], -1)
    return F.ctc_loss(lp.transpose(0, 1), tgt, mel_lens, text_lens, blank=0, zero_infinity=True)


def bin_loss(hard, soft, mask_TN):
    return -(torch.log(soft.clamp_min(1e-12)) * hard)[mask_TN].sum() / hard.sum().clamp_min(1)


def length_regulate(x, dur, max_len=None):
    """x [B,N,d], dur [B,N] int -> [B,T,d], mel_lens."""
    lens = dur.sum(1)
    T = int(max_len or lens.max())
    idx = torch.cumsum(dur, 1)  # [B,N]
    frames = torch.arange(T, device=x.device)[None]
    # token index for each frame = number of cumulative boundaries <= frame
    tok = (frames[:, :, None] >= idx[:, None, :]).sum(-1).clamp(max=x.shape[1] - 1)
    out = torch.gather(x, 1, tok[..., None].expand(-1, -1, x.shape[2]))
    return out * seq_mask(lens, T)[..., None], lens


def average_over_durations(frame_vals, dur, voiced_only=None):
    """frame_vals [B,T], dur [B,N] -> per-token mean [B,N] (over voiced frames if mask given)."""
    B, N = dur.shape
    T = frame_vals.shape[1]
    ends = torch.cumsum(dur, 1)
    starts = ends - dur
    frames = torch.arange(T, device=dur.device)[None, None]
    m = (frames >= starts[..., None]) & (frames < ends[..., None])  # [B,N,T]
    if voiced_only is not None:
        m = m & voiced_only[:, None, :]
    m = m.float()
    s = torch.bmm(m, frame_vals[..., None].float()).squeeze(-1)
    n = m.sum(-1)
    return torch.where(n > 0, s / n.clamp_min(1), torch.zeros_like(s))


# ----------------------------------------------------------------------------- model
class SteerTTS(nn.Module):
    def __init__(self, n_speakers, n_tags, d=192, heads=2, ffn=384, k=3, enc_layers=4, dec_layers=4, g_dim=128,
                 spk_dim=64, tag_dim=32, p=0.1, enc_mixer="attn", dec_mixer="attn", ssm_state=32, spk_enc_dim=0,
                 n_symbols=None, n_env=0):
        super().__init__()
        # n_symbols: N_BASE (phonemes + punctuation) for every checkpoint before sound events; len(SYMBOLS) adds the
        # inline event tokens ([laugh], [dog], ...). n_env: size of the acoustic-environment vector (steer.acoustic),
        # 0 = no environment conditioning. Both default to the pre-noise architecture so old checkpoints load.
        n_symbols = n_symbols or N_BASE
        # saved in checkpoints so synthesis rebuilds the same architecture
        self.arch = dict(enc_mixer=enc_mixer, dec_mixer=dec_mixer, ssm_state=ssm_state, spk_enc_dim=spk_enc_dim,
                         n_symbols=n_symbols, n_env=n_env, d=d, heads=heads, ffn=ffn)
        self.emb = nn.Embedding(n_symbols, d, padding_idx=PAD)
        self.spk = nn.Embedding(n_speakers + 1, spk_dim)  # last index = unknown speaker
        # reference-conditioned voice (spk_enc_dim > 0): a frozen speaker-verification vector of a reference clip
        # projected into the same slot as the table row. Linear, so it can be initialised by least squares to
        # reproduce a trained table (train.fit_spk_proj); the table is then only a fallback.
        self.spk_proj = nn.Linear(spk_enc_dim, spk_dim) if spk_enc_dim else None
        self.tag = nn.Embedding(n_tags, tag_dim)  # index 0 = 'none'
        self.cond = nn.Sequential(nn.Linear(spk_dim + tag_dim + 2 * N_CTRL, 256), nn.SiLU(), nn.Linear(256, g_dim))
        # acoustic environment (background scene, room, phone channel, SNR) added to g; zero-initialised so a model
        # extended from a clean checkpoint is unchanged until it learns the environment
        self.env_proj = None
        if n_env:
            self.env_proj = nn.Sequential(nn.Linear(n_env, 128), nn.SiLU(), nn.Linear(128, g_dim))
            nn.init.zeros_(self.env_proj[2].weight); nn.init.zeros_(self.env_proj[2].bias)
        self.encoder = Stack(enc_layers, d, heads, ffn, k, g_dim, p, enc_mixer, ssm_state)
        self.decoder = Stack(dec_layers, d, heads, ffn, k, g_dim, p, dec_mixer, ssm_state)
        self.dur_pred = VariancePredictor(d, g_dim)
        self.pitch_pred = VariancePredictor(d, g_dim)
        self.energy_pred = VariancePredictor(d, g_dim)
        self.pitch_emb = nn.Conv1d(1, d, 3, padding=1)
        self.energy_emb = nn.Conv1d(1, d, 3, padding=1)
        self.aligner = Aligner(d)
        self.to_mel = nn.Linear(d, N_MELS)

    def condition(self, spk, tag, ctrl, cmask, spk_vec=None, env=None):
        c = torch.nan_to_num(ctrl) * cmask
        s = self.spk_proj(spk_vec) if (self.spk_proj is not None and spk_vec is not None) else self.spk(spk)
        g = self.cond(torch.cat([s, self.tag(tag), c, cmask], -1))
        if self.env_proj is not None and env is not None:
            g = g + self.env_proj(env)
        return g

    def encode(self, ids, text_lens, g):
        tmask = seq_mask(text_lens, ids.shape[1])
        e = self.emb(ids)
        return e, self.encoder(e, tmask, g), tmask

    def forward(self, batch, step=0):
        ids, text_lens, mel, mel_lens = batch["ids"], batch["text_lens"], batch["mel"], batch["mel_lens"]
        g = self.condition(batch["spk"], batch["tag"], batch["ctrl"], batch["cmask"], batch.get("spk_vec"),
                           batch.get("env"))
        e, h, tmask = self.encode(ids, text_lens, g)
        mmask = seq_mask(mel_lens, mel.shape[2])

        # noise-augmented items align on the dry mel (speech + events + room, before background noise / channel)
        soft, logp = self.aligner(e, batch.get("mel_align", mel), tmask, batch["prior"])
        hard_dur = mas_durations(soft, text_lens, mel_lens)
        hard = torch.zeros_like(soft)
        # rebuild the hard attention matrix [B,T,N] from durations
        ends = torch.cumsum(hard_dur, 1)
        starts = ends - hard_dur
        fr = torch.arange(soft.shape[1], device=soft.device)[None, :, None]
        hard = ((fr >= starts[:, None, :]) & (fr < ends[:, None, :])).float()

        # per-token targets (speaker-normalised pitch in st, energy in dB/10)
        pitch_t = average_over_durations(batch["pitch"], hard_dur, voiced_only=batch["voiced"])
        energy_t = average_over_durations(batch["energy"], hard_dur)
        logdur_t = torch.log(hard_dur.float() + 1)

        dur_p = self.dur_pred(h, tmask, g)
        pitch_p = self.pitch_pred(h, tmask, g)
        energy_p = self.energy_pred(h, tmask, g)

        h = h + self.pitch_emb(pitch_t[:, None]).transpose(1, 2) + self.energy_emb(energy_t[:, None]).transpose(1, 2)
        x, _ = length_regulate(h, hard_dur, mel.shape[2])
        y = self.decoder(x, mmask, g)
        mel_p = self.to_mel(y).transpose(1, 2)  # [B,100,T]

        tm = tmask.float()
        n_tok = tm.sum()
        m3 = mmask[:, None].float()
        losses = dict(
            mel=((mel_p - mel).abs() * m3).sum() / (m3.sum() * N_MELS),
            dur=(((dur_p - logdur_t) ** 2) * tm).sum() / n_tok,
            pitch=(((pitch_p - pitch_t) ** 2) * tm).sum() / n_tok,
            energy=(((energy_p - energy_t) ** 2) * tm).sum() / n_tok,
            align=forward_sum_loss(logp, text_lens, mel_lens),
            bin=bin_loss(hard, soft, mmask[:, :, None] & tmask[:, None, :]),
        )
        return losses, dict(mel_p=mel_p, hard_dur=hard_dur, soft=soft, tmask=tmask, mmask=mmask, dur_p=dur_p,
                            pitch_p=pitch_p, energy_p=energy_p, logdur_t=logdur_t, pitch_t=pitch_t, energy_t=energy_t)

    def free_run(self, ids, text_lens, g):
        """Batched synthesis from predicted durations/pitch/energy, differentiable w.r.t. everything except
        the (rounded) durations. Returns mel [B,100,T], mel_lens, and the token-level predictions."""
        tmask = seq_mask(text_lens, ids.shape[1])
        e = self.emb(ids)
        h = self.encoder(e, tmask, g)
        logdur = self.dur_pred(h, tmask, g)
        pitch = self.pitch_pred(h, tmask, g)
        energy = self.energy_pred(h, tmask, g)
        dur = torch.round((torch.exp(logdur.detach()) - 1).clamp_min(0)).long() * tmask
        dur[:, 0] = dur[:, 0].clamp_min(1)
        h = h + self.pitch_emb(pitch[:, None]).transpose(1, 2) + self.energy_emb(energy[:, None]).transpose(1, 2)
        x, lens = length_regulate(h, dur)
        y = self.decoder(x, seq_mask(lens, x.shape[1]), g)
        return self.to_mel(y).transpose(1, 2), lens, dict(logdur=logdur, pitch=pitch, energy=energy, tmask=tmask)

    @torch.no_grad()
    def infer(self, ids, spk, tag, ctrl, cmask, pace=1.0, word_idx=None, word_shift=None, spk_vec=None, env=None,
              guidance=1.0, base=None):
        """ids [1,N]. word_shift: optional dict word_index -> dict(pitch=st, energy=dB/10, dur=factor).
        spk_vec [1,spk_enc_dim]: reference speaker vector (reference-conditioned models); else the table row.
        guidance s != 1 with base=(tag0, ctrl0, cmask0): classifier-free-style guidance on every stage,
        out = out(base) + s * (out(request) - out(base)). Control labels are noisy measurements, so the learned effect
        is attenuated (regression dilution, ratio ~0.5 in the harness); s > 1 undoes the attenuation. Durations are
        guided first so both passes share one timeline."""
        g = self.condition(spk, tag, ctrl, cmask, spk_vec, env)
        g0 = self.condition(spk, *base, spk_vec, env) if (base is not None and guidance != 1.0) else None
        text_lens = torch.tensor([ids.shape[1]], device=ids.device)
        _, h, tmask = self.encode(ids, text_lens, g)
        logdur, pitch, energy = self.dur_pred(h, tmask, g), self.pitch_pred(h, tmask, g), self.energy_pred(h, tmask, g)
        if g0 is not None:
            _, h0, _ = self.encode(ids, text_lens, g0)
            mix = lambda a, a0: a0 + guidance * (a - a0)
            logdur = mix(logdur, self.dur_pred(h0, tmask, g0))
            pitch = mix(pitch, self.pitch_pred(h0, tmask, g0))
            energy = mix(energy, self.energy_pred(h0, tmask, g0))
        dur = (torch.exp(logdur) - 1).clamp_min(0) / pace
        if word_shift and word_idx is not None:
            for w, sh in word_shift.items():
                sel = word_idx[0] == w
                pitch[0, sel] += sh.get("pitch", 0.0)
                energy[0, sel] += sh.get("energy", 0.0)
                dur[0, sel] *= sh.get("dur", 1.0)
        dur = torch.round(dur).long().clamp_min(0)
        dur[:, 0] = dur[:, 0].clamp_min(1)
        pe = self.pitch_emb(pitch[:, None]).transpose(1, 2) + self.energy_emb(energy[:, None]).transpose(1, 2)
        x, lens = length_regulate(h + pe, dur)
        mel = self.to_mel(self.decoder(x, seq_mask(lens, x.shape[1]), g)).transpose(1, 2)
        if g0 is not None:
            x0, _ = length_regulate(h0 + pe, dur)
            mel0 = self.to_mel(self.decoder(x0, seq_mask(lens, x0.shape[1]), g0)).transpose(1, 2)
            mel = mel0 + guidance * (mel - mel0)
        return mel, dict(dur=dur, pitch=pitch, energy=energy, g=g)
