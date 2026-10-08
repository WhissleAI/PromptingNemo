"""Fused Mamba-3 scan: the SSD reduction, verified, then a Triton kernel.

WHY THIS FILE. scan.py's chunked scan materialises (B,T,H,N,P) intermediates
(S, inp, w, g, h) in HBM and fires many kernels; on the RTX 5070 it is
memory-bandwidth bound and runs at ~3.4k tok/s for a 12M-param model at ctx
1024. The Mamba-3 paper's speed comes from a fused chunked kernel that keeps the
chunk state in SRAM and never writes the expanded state to HBM. This file builds
that, test-first, against the verified recurrence in layer.py.

THE REDUCTION (paper Sec 3.2-3.3, "data-dependent RoPE on B and C"). The
recurrence

    h_t = alpha_t R(theta_t) h_{t-1} + gamma_t B_t x_t^T + beta_t B_{t-1} x_{t-1}^T
    y_t = C_t^T h_t

becomes, with Phi_t = cumsum(theta) and the rotations absorbed into B and C,

    y_t = Ctil_t^T sum_{s<=t} decay(s,t) (Beff_s xeff_s^T),   decay(s,t)=prod_{s<r<=t} alpha_r
    Ctil_t = R(-Phi_t) C_t
    Beff_s = [ gamma_s R(-Phi_s)B_s , beta_s R(-Phi_s)B_{s-1} ]   (N, 2R)
    xeff_s = [ x_s , x_{s-1} ]                                    (P, 2R)

i.e. a standard Mamba-2 SSD scan on RoPE'd B/C with a rank-2R write that folds
in the trapezoidal two-tap. SSD is the fusible, backward-known form.

STAGE 1 (this file, first): a PyTorch chunked-SSD reference that implements
exactly the arithmetic the Triton kernel will, and is checked fwd+bwd against
Mamba3Layer at float64. Only once this matches does the Triton port begin.
"""
from __future__ import annotations

import torch

from .scan import rotate_by  # rotate (...,N,P) pairs of the N axis by ang (...,N/2)


def ssd_inputs(layer, u: torch.Tensor):
    """Reduce Mamba3Layer's coefficients to the SSD form.

    Returns alpha (B,T,H), Beff (B,T,H,N,2R), xeff (B,T,H,P,2R), Ctil (B,T,H,N,R).
    All the rotation and the trapezoidal fold happen here, in cheap elementwise
    ops on tensors no larger than (.,.,.,N,2R) - never the (N,P) expanded state.
    """
    alpha, beta, gamma, Bm, Cm, x, theta = layer._coeffs(layer.norm(u))
    Phi = torch.cumsum(theta, dim=1)                       # (B,T,H,N/2) = Phi_s
    Phi_prev = torch.cat([torch.zeros_like(Phi[:, :1]), Phi[:, :-1]], dim=1)  # Phi_{s-1}
    Btil = rotate_by(Bm, -Phi)                             # R(-Phi_s) B_s      (paper gamma term)
    Ctil = rotate_by(Cm, -Phi)                             # R(-Phi_t) C_t
    # Trapezoidal previous-step write, rotated by R(-Phi_{s-1}) per paper eq (11)
    # (NOT Phi_s): B_{s-1} lands rotated by prod_{i<=s-1} R_i in the global frame.
    Bm_prev = torch.cat([torch.zeros_like(Bm[:, :1]), Bm[:, :-1]], dim=1)
    x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    Btil_prev = rotate_by(Bm_prev, -Phi_prev)             # R(-Phi_{s-1}) B_{s-1}
    # rank-2R effective write: coefficients ride on B, so w_s = Beff_s xeff_s^T.
    Beff = torch.cat([gamma[..., None, None] * Btil,
                      beta[..., None, None] * Btil_prev], dim=-1)   # (B,T,H,N,2R)
    xeff = torch.cat([x, x_prev], dim=-1)                          # (B,T,H,P,2R)
    return alpha, Beff, xeff, Ctil


def ssd_pytorch(alpha, Beff, xeff, Ctil, chunk: int = 64):
    """Chunked SSD forward in PyTorch - the reference the Triton kernel matches.

    alpha (B,T,H) ; Beff (B,T,H,N,K) ; xeff (B,T,H,P,K) ; Ctil (B,T,H,N,R).
    Returns y (B,T,H,R,P). State g (N,P) is carried per chunk (kept small).
    """
    B, T, H, N, K = Beff.shape
    P = xeff.shape[-2]
    R = Ctil.shape[-1]
    la = torch.log(alpha.clamp_min(1e-30))                 # (B,T,H)
    g = torch.zeros(B, H, N, P, dtype=Beff.dtype, device=Beff.device)
    ys = []
    for s0 in range(0, T, chunk):
        e = min(s0 + chunk, T)
        L = e - s0
        Bc = Beff[:, s0:e]          # (B,L,H,N,K)
        xc = xeff[:, s0:e]          # (B,L,H,P,K)
        Cc = Ctil[:, s0:e]          # (B,L,H,N,R)
        cl = torch.cumsum(la[:, s0:e], dim=1)              # (B,L,H) = log prod_{s0..t}
        d = torch.exp(cl)                                  # (B,L,H)  prod_{s0..t} alpha
        # intra-chunk decay L[t,s] = exp(cl_t - cl_s) for s<=t else 0
        rel = cl.unsqueeze(2) - cl.unsqueeze(1)            # (B,L(t),L(s),H)
        tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=Beff.device))
        Ldec = torch.where(tril.view(1, L, L, 1), torch.exp(rel),
                           torch.zeros((), dtype=Beff.dtype, device=Beff.device))
        # scores G[t,s] = Ctil_t^T Beff_s over N -> (B,H,L,L,R,K)
        G = torch.einsum('blhnr,bmhnk->bhlmrk', Cc, Bc)
        # intra y_t = sum_{s<=t} Ldec[t,s] G[t,s] xeff_s^T
        y_intra = torch.einsum('blmh,bhlmrk,bmhpk->blhrp', Ldec, G, xc)
        # inter y_t = d_t * (Ctil_t^T g)
        y_inter = torch.einsum('blh,blhnr,bhnp->blhrp', d, Cc, g)
        ys.append(y_intra + y_inter)
        # state update: g <- d_{L-1} g + sum_s (d_{L-1}/d_s) Beff_s xeff_s^T
        coef = torch.exp(cl[:, -1:, :] - cl)               # (B,L,H)
        g = d[:, -1].unsqueeze(-1).unsqueeze(-1) * g \
            + torch.einsum('blh,blhnk,blhpk->bhnp', coef, Bc, xc)
    return torch.cat(ys, dim=1)                            # (B,T,H,R,P)


def ssd_vectorized(alpha, Beff, xeff, Ctil, chunk: int = 64):
    """ssd_pytorch with NO Python loop over chunks - same arithmetic, batched.

    ssd_pytorch walks the chunks sequentially, firing ~10 kernels per chunk; at T ~ 2000 that loop,
    not the FLOPs, is the cost. Here every chunk's intra-chunk term is one batched einsum, and the
    chunk-to-chunk state pass - g_c = D_{c-1} g_{c-1} + U_{c-1} - is unrolled into a (C x C)
    lower-triangular decay matrix over chunks, the formulation of Mamba-2's `ssd_minimal`. C = T/chunk
    is small (tens), so that matrix is cheap.

    Exponents are masked to -inf BEFORE exp, never exp-then-where: above the diagonal cl_t - cl_s is
    positive and can overflow, and where() of an inf still back-propagates NaN.
    Same signature and output as ssd_pytorch; verify() checks the two agree fwd+bwd at float64.
    """
    B, T, H, N, K = Beff.shape
    P, R = xeff.shape[-2], Ctil.shape[-1]
    pad = (-T) % chunk
    if pad:  # alpha=1 (log 0) and zero writes/readouts: padded steps change nothing
        alpha = torch.cat([alpha, alpha.new_ones(B, pad, H)], 1)
        Beff, xeff, Ctil = (torch.cat([t, t.new_zeros(B, pad, *t.shape[2:])], 1) for t in (Beff, xeff, Ctil))
    C, L = (T + pad) // chunk, chunk
    la = torch.log(alpha.clamp_min(1e-30)).view(B, C, L, H)
    Bc, xc, Cc = Beff.view(B, C, L, H, N, K), xeff.view(B, C, L, H, P, K), Ctil.view(B, C, L, H, N, R)
    cl = torch.cumsum(la, 2)                                         # (B,C,L,H) log decay from chunk start
    tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=Beff.device)).view(1, 1, L, L, 1)
    Ldec = torch.exp((cl.unsqueeze(3) - cl.unsqueeze(2)).masked_fill(~tril, float("-inf")))  # (B,C,L,L,H)
    G = torch.einsum('bclhnr,bcmhnk->bchlmrk', Cc, Bc)
    y = torch.einsum('bclmh,bchlmrk,bcmhpk->bclhrp', Ldec, G, xc)   # intra-chunk
    # each chunk's own contribution to the state at its end, then the state at each chunk's START
    U = torch.einsum('bclh,bclhnk,bclhpk->bchnp', torch.exp(cl[:, :, -1:] - cl), Bc, xc)  # (B,C,H,N,P)
    tot = cl[:, :, -1]                                               # (B,C,H) log decay across chunk c
    ct = torch.cumsum(tot, 1)
    # start state of chunk c = sum_{j<c} exp(ct_{c-1} - ct_j) U_j    (strictly lower triangular in c, j)
    stril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=Beff.device), -1).view(1, C, C, 1)
    ctm1 = ct - tot                                                  # ct_{c-1} (0 for c=0)
    Dc = torch.exp((ctm1.unsqueeze(2) - ct.unsqueeze(1)).masked_fill(~stril, float("-inf")))  # (B,C,C,H)
    g0 = torch.einsum('bcjh,bjhnp->bchnp', Dc, U)
    y = y + torch.einsum('bclh,bclhnr,bchnp->bclhrp', torch.exp(cl), Cc, g0)  # inter-chunk
    return y.reshape(B, C * L, H, R, P)[:, :T]


def scan_forward_ssd(layer, u: torch.Tensor, chunk: int = 64, vectorized: bool = False):
    """Drop-in for scan.scan_forward, via the SSD reduction (PyTorch stage).

    vectorized=True uses ssd_vectorized (no per-chunk Python loop; much faster at long T)."""
    Bsz, T, _ = u.shape
    H, R, P = layer.H, layer.R, layer.P
    nu = layer.norm(u)
    alpha, Beff, xeff, Ctil = ssd_inputs(layer, u)
    y = (ssd_vectorized if vectorized else ssd_pytorch)(alpha, Beff, xeff, Ctil, chunk)  # (B,T,H,R,P)
    return layer.gate_and_proj(y.reshape(Bsz, T, H * R * P), nu)


def verify(chunks=(8, 16, 64), verbose: bool = True) -> dict:
    """SSD reduction vs the recurrence, fwd + bwd, at float64 - loop and vectorized paths."""
    from .layer import Mamba3Layer
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    out = {}
    lay = Mamba3Layer(d_model=32, d_state=16, rank=2, head_dim=8).eval()
    u = torch.randn(3, 40, 32)
    u2 = torch.randn(2, 37, 32)
    u3 = torch.cat([u, torch.randn(3, 9, 32) * 50], 1)
    ug = torch.randn(3, 24, 32, requires_grad=True)
    with torch.no_grad():
        ref, ref2 = lay(u), lay(u2)
    lay.zero_grad(); lay(ug).pow(2).sum().backward()
    gu = ug.grad.clone(); gp = {n: p.grad.clone() for n, p in lay.named_parameters()
                                if p.grad is not None}
    for vec in (False, True):
        pre = "vec_" if vec else ""
        f = lambda x, c: scan_forward_ssd(lay, x, c, vectorized=vec)
        with torch.no_grad():
            for c in chunks:
                out[f"{pre}chunk{c}"] = float((f(u, c) - ref).abs().max())
            out[f"{pre}ragged"] = float((f(u2, 16) - ref2).abs().max())
            out[f"{pre}causality"] = float((f(u3, 16)[:, :40] - f(u, 16)).abs().max())
        ug.grad = None; lay.zero_grad()
        f(ug, 8).pow(2).sum().backward()
        out[f"{pre}grad_input"] = float((gu - ug.grad).abs().max())
        out[f"{pre}grad_params"] = max(float((gp[n] - p.grad).abs().max())
                                       for n, p in lay.named_parameters() if p.grad is not None)
    # long sequence with strong decay: the vectorized path's masked exponents must stay finite fwd+bwd
    a = torch.full((1, 3000, 2), 1e-3)                     # log alpha ~ -6.9 per step
    Bf, xf, Cf = (torch.randn(1, 3000, 2, 4, k, requires_grad=True) for k in (2, 2, 1))
    yv = ssd_vectorized(a, Bf, xf, Cf, 64)
    yv.sum().backward()
    out["vec_long_finite"] = 0.0 if all(torch.isfinite(t).all() for t in (yv, Bf.grad, xf.grad, Cf.grad)) else 1.0
    with torch.no_grad():
        out["vec_long_vs_loop"] = float((yv - ssd_pytorch(a, Bf, xf, Cf, 64)).abs().max())
    torch.set_default_dtype(torch.float32)
    if verbose:
        ok = lambda c: "OK  " if c else "FAIL"
        print("SSD reduction vs recurrence  (float64, fwd+bwd)")
        for k, v in out.items():
            print(f"  {k:<18}{v:.3e}   {ok(v < 1e-9)}")
    return out


if __name__ == "__main__":
    verify()
