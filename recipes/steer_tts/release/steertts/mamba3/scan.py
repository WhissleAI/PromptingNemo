"""Chunked parallel scan for the Mamba-3 recurrence.

WHY. The recurrent path is a Python loop of T sequential steps. Measured on
this laptop: d_model 256 / 4 layers / N 32 runs at 3,670 tok/s, while
384 / 6 / 64 runs at 179 tok/s - 20x slower for ~4x the FLOPs. The gap is not
arithmetic, it is kernel-launch overhead: 6 layers x 128 timesteps is 768
sequential iterations each firing a handful of tiny kernels, so DEPTH
MULTIPLIES LAUNCHES. That is what blocks scaling, not parameters.

THE TRICK, and it is the paper's own (Proposition 3). The recurrence

    h_t = alpha_t R(theta_t) h_{t-1} + beta_t S_{t-1} + gamma_t S_t

has a data-dependent rotation inside it, which a scan cannot factor. But
rotations compose: define the CUMULATIVE angle Phi_t = sum_{s<=t} theta_s and
change basis by R(-Phi_t). In that frame the rotation disappears and what is
left is a plain first-order linear recurrence

    g_t = alpha_t g_{t-1} + w_t          w_t = R(-Phi_t) (beta_t S_{t-1} + gamma_t S_t)

with h_t = R(Phi_t) g_t. The paper states this as applying data-dependent
RoPE to B and C; the same identity in state space is what makes the scan
possible at all.

THE SCAN. g_t = alpha_t g_{t-1} + w_t has the closed form

    g_t = A_t * sum_{s<=t} w_s / A_s     A_t = prod_{r<=t} alpha_r

which is numerically hopeless directly - A_s underflows for long sequences
and the division explodes. So this chunks: within a chunk of length L the sum
is materialised as an L x L lower-triangular decay matrix (no division, decay
is always relative to the chunk start), and chunk states are carried
recurrently. T/L sequential steps instead of T. The intra-chunk work becomes
one batched matmul, which is what GPUs are for.

CORRECTNESS IS CHECKED AGAINST THE RECURRENCE, NOT AGAINST A LOSS CURVE.
`verify()` compares this to Mamba3Layer.forward at float64. A fast scan that
computes something slightly different would show up as a slightly worse loss
curve and be indistinguishable from a hyperparameter difference - which is
exactly how this project has lost time before.

    python -m steertts.mamba3.scan
"""
from __future__ import annotations

import math
import torch


def cumulative_angles(theta: torch.Tensor) -> torch.Tensor:
    """Phi_t = sum_{s<=t} theta_s.  theta: (B,T,H,N/2)."""
    return torch.cumsum(theta, dim=1)


def rotate_by(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of the N axis. v: (...,N,P), ang: (...,N/2) broadcastable.

    Built with stack+reshape rather than writing into an empty tensor. The
    first version used `out = torch.empty_like(v)` with strided assignment,
    which is correct in the forward direction and BREAKS AUTOGRAD: in-place
    writes bump the version counter of a tensor the graph still needs. The
    forward equivalence check passed anyway, because it ran under no_grad.
    """
    a, b = v[..., 0::2, :], v[..., 1::2, :]
    c, s = torch.cos(ang).unsqueeze(-1), torch.sin(ang).unsqueeze(-1)
    even, odd = c * a - s * b, s * a + c * b
    return torch.stack((even, odd), dim=-2).flatten(-3, -2)


def chunked_scan(alpha: torch.Tensor, w: torch.Tensor, chunk: int = 64,
                 state=None):
    """g_t = alpha_t g_{t-1} + w_t, computed in chunks.

    alpha: (B,T,H)          per-step decay, in (0,1)
    w:     (B,T,H,N,P)      per-step input, already in the rotated frame
    returns g: (B,T,H,N,P) and the final state.

    Within a chunk, g_t = d_t * g_0 + sum_{s<=t} (d_t/d_s) w_s where
    d_t = prod_{r<=t} alpha_r RELATIVE TO THE CHUNK START. Both d_t and d_t/d_s
    stay bounded because the product never runs longer than `chunk` steps -
    that is the whole reason for chunking rather than a global cumprod.
    """
    B, T, H = alpha.shape
    N, P = w.shape[-2], w.shape[-1]
    # Accumulate chunks in a list and cat once. Writing into a preallocated
    # g_out and then reading g_out[:, e-1] back as the next chunk's state is
    # an in-place write into a tensor autograd still needs.
    pieces = []
    g = torch.zeros(B, H, N, P, device=w.device, dtype=w.dtype) \
        if state is None else state
    la = torch.log(alpha.clamp_min(1e-30))                 # (B,T,H)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        L = e - s
        cl = torch.cumsum(la[:, s:e], dim=1)               # (B,L,H) log d_t
        d = torch.exp(cl)                                  # decay from chunk start
        # (d_t / d_s) for s<=t, lower triangular, exp of a difference of logs
        rel = cl.unsqueeze(2) - cl.unsqueeze(1)            # (B,L,L,H)  log d_t - log d_s
        # mask before exp (see ssd.ssd_pytorch): exp of the positive upper triangle overflows and where()
        # of an inf back-propagates NaN
        mask = torch.tril(torch.ones(L, L, device=w.device, dtype=torch.bool))
        M = torch.exp(rel.masked_fill(~mask.view(1, L, L, 1), float("-inf")))
        # intra-chunk: sum_s M[t,s] * w_s  -> one batched matmul over (L,L) x (L, N*P)
        wf = w[:, s:e].reshape(B, L, H, N * P).permute(0, 2, 1, 3)   # (B,H,L,NP)
        Mp = M.permute(0, 3, 1, 2)                                   # (B,H,L,L)
        acc = (Mp @ wf).permute(0, 2, 1, 3).reshape(B, L, H, N, P)
        # carry the incoming chunk state, decayed
        piece = acc + d.unsqueeze(-1).unsqueeze(-1) * g.unsqueeze(1)
        pieces.append(piece)
        g = piece[:, -1]
    return torch.cat(pieces, dim=1), g


def scan_forward(layer, u: torch.Tensor, chunk: int = 64):
    """Mamba3Layer.forward computed with the chunked scan instead of the loop."""
    Bsz, T, _ = u.shape
    H, N, R, P = layer.H, layer.N, layer.R, layer.P
    nu = layer.norm(u)
    alpha, beta, gamma, Bm, Cm, x, theta = layer._coeffs(nu)
    S = Bm @ x.transpose(-1, -2)                       # (B,T,H,N,P)
    # trapezoidal: beta carries the PREVIOUS step's write. Built by cat, not
    # by assigning into a zeros tensor, for the same autograd reason.
    Sprev = torch.cat([torch.zeros_like(S[:, :1]), S[:, :-1]], dim=1)
    Phi = cumulative_angles(theta)                     # (B,T,H,N/2) = Phi_s
    Phi_prev = torch.cat([torch.zeros_like(Phi[:, :1]), Phi[:, :-1]], dim=1)
    # Paper eq (11): the gamma (current) write rotates by R(-Phi_s), the beta
    # (previous) write by R(-Phi_{s-1}) - one step less. Rotate the two terms
    # separately into the global frame, then scan.
    w = (gamma[..., None, None] * rotate_by(S, -Phi)
         + beta[..., None, None] * rotate_by(Sprev, -Phi_prev))
    g, _ = chunked_scan(alpha, w, chunk)
    h = rotate_by(g, Phi)                              # back out
    y = (Cm.transpose(-1, -2) @ h).reshape(Bsz, T, H * R * P)
    return layer.gate_and_proj(y, nu)


def verify(chunks=(8, 16, 64), verbose: bool = True) -> dict:
    """Against the recurrence, at float64 - FORWARD AND BACKWARD.

    The backward half is not optional. The first version of this file passed
    every forward check and could not backpropagate at all: it built outputs
    with in-place strided writes, which autograd rejects. The forward checks
    ran under no_grad and saw nothing. A scan that cannot train is useless,
    and one that trains with subtly wrong gradients is worse than useless -
    it would surface as a slightly worse loss curve, indistinguishable from a
    hyperparameter difference."""
    from .layer import Mamba3Layer

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    out = {}
    lay = Mamba3Layer(d_model=32, d_state=16, rank=2, head_dim=8).eval()
    u = torch.randn(3, 40, 32)
    with torch.no_grad():
        ref = lay(u)
        for c in chunks:
            got = scan_forward(lay, u, chunk=c)
            out[f"chunk{c}"] = float((got - ref).abs().max())
        # a length that is not a multiple of the chunk
        u2 = torch.randn(2, 37, 32)
        r2 = lay(u2)
        out["ragged"] = float((scan_forward(lay, u2, chunk=16) - r2).abs().max())
        # causality of the scan path itself
        u3 = torch.cat([u, torch.randn(3, 9, 32) * 50], 1)
        out["causality"] = float(
            (scan_forward(lay, u3, chunk=16)[:, :40]
             - scan_forward(lay, u, chunk=16)).abs().max())

    # ---- BACKWARD ----
    ug = torch.randn(3, 24, 32, requires_grad=True)
    lay.zero_grad()
    lay(ug).pow(2).sum().backward()
    gu_rec = ug.grad.clone()
    gp_rec = {n: p.grad.clone() for n, p in lay.named_parameters()
              if p.grad is not None}
    ug.grad = None
    lay.zero_grad()
    scan_forward(lay, ug, chunk=8).pow(2).sum().backward()
    out["grad_input"] = float((gu_rec - ug.grad).abs().max())
    out["grad_params"] = max(
        float((gp_rec[n] - p.grad).abs().max())
        for n, p in lay.named_parameters() if p.grad is not None)
    torch.set_default_dtype(torch.float32)
    if verbose:
        ok = lambda c: "OK  " if c else "FAIL"
        print("chunked scan vs recurrence  (float64)")
        for k, v in out.items():
            print(f"  {k:<12}{v:.3e}   {ok(v < 1e-9)}")
    return out


if __name__ == "__main__":
    verify()
