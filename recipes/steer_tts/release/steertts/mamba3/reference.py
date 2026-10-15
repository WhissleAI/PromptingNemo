"""Independent Mamba-3 references, written from the paper rather than from this package.

Nothing here calls Mamba3Layer._coeffs, ssm_scan or rotate_by: every coefficient is recomputed from
the raw parameters with the paper's definitions, so a coefficient bug in layer.py cannot hide by being
shared with the thing that checks it (tests.paper_eq11_check reuses _coeffs and so cannot catch one).

    ref_complex   the complex-valued SSM itself (Prop 2; App B.3 eq 24), in complex128, MIMO as an
                  explicit sum over ranks (eq 12-14). A sequential loop, the most literal form.
    ref_parallel  the SSD parallel form, eq (7): L = (1-semiseparable decay) @ (2-band beta/gamma),
                  applied to cumulative-RoPE'd B and C (Prop 3/4). Quadratic in T.

The two are derived independently (complex arithmetic vs real rotations + masks) and agree to ~1e-15,
so each layer path is checked against two references that do not share code with it or each other.
Both are float64 and slow; they are for tests only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _rms(x, w, eps):
    eps = torch.finfo(x.dtype).eps if eps is None else eps
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def raw_coeffs(L, u):
    """Every per-step quantity of a Mamba3Layer, from raw parameters (float64)."""
    u = u.double()
    Bsz, T, _ = u.shape
    H, N, R, P = L.H, L.N, L.R, L.P
    nu = _rms(u, L.norm.weight, L.norm.eps)
    dt = F.softplus(F.linear(nu, L.dt_proj.weight, L.dt_proj.bias))              # (B,T,H)
    A = -torch.exp(L.A_log)                                                      # (H,)
    lam = torch.sigmoid(F.linear(nu, L.lam_proj.weight, L.lam_proj.bias))       # trapezoid weight
    Bn = _rms(F.linear(nu, L.B_proj.weight, L.B_proj.bias).view(Bsz, T, H, R, N),
              L.B_norm.weight, L.B_norm.eps) + L.B_bias.permute(0, 2, 1)         # (B,T,H,R,N)
    Cn = _rms(F.linear(nu, L.C_proj.weight, L.C_proj.bias).view(Bsz, T, H, R, N),
              L.C_norm.weight, L.C_norm.eps) + L.C_bias.permute(0, 2, 1)
    x = F.linear(nu, L.x_proj.weight, L.x_proj.bias).view(Bsz, T, H, P, R)
    th = F.linear(nu, L.theta_proj.weight, L.theta_proj.bias).view(Bsz, T, H, N // 2)
    z = F.linear(nu, L.in_proj.weight, L.in_proj.bias)                          # gate pre-activation
    return dt, A, lam, Bn, Cn, x, th, z


def _finish(L, y, z):  # y (B,T,H,R,P) -> gate, out_proj
    Bsz, T = y.shape[:2]
    return F.linear(y.reshape(Bsz, T, -1) * F.silu(z), L.out_proj.weight, L.out_proj.bias)


def ref_complex(L, u):
    """h_t = e^{dt A} e^{i dt theta} h_{t-1} + beta_t e^{i dt theta} W_{t-1} + gamma_t W_t,
    W_t = sum_r B_t^(r) x_t^(r)T,  y_t^(r) = Re(conj(C_t^(r)) . h_t),  state of N/2 complex numbers."""
    dt, A, lam, Bn, Cn, x, th, z = raw_coeffs(L, u)
    Bsz, T, H, R, N = Bn.shape
    Bc = torch.complex(Bn[..., 0::2], Bn[..., 1::2])                             # (B,T,H,R,N/2)
    Cc = torch.complex(Cn[..., 0::2], Cn[..., 1::2])
    h = torch.zeros(Bsz, H, N // 2, L.P, dtype=torch.complex128, device=u.device)
    Wprev = torch.zeros_like(h)
    ys = []
    for t in range(T):
        d = dt[:, t]
        alpha = torch.exp(d * A)
        rot = torch.exp(1j * (d[..., None] * th[:, t]))[..., None]               # (B,H,N/2,1)
        W = torch.einsum('bhrk,bhpr->bhkp', Bc[:, t], x[:, t].to(Bc.dtype))
        beta = ((1 - lam[:, t]) * d * alpha)[..., None, None]
        gamma = (lam[:, t] * d)[..., None, None]
        h = alpha[..., None, None] * rot * h + beta * rot * Wprev + gamma * W
        Wprev = W
        ys.append(torch.einsum('bhrk,bhkp->bhrp', Cc[:, t].conj(), h).real)
    return _finish(L, torch.stack(ys, 1), z)


def ref_parallel(L, u):
    """eq (7): y = (decay @ band) * (C~ B~^T) x with C~, B~ rotated by the cumulative angle."""
    dt, A, lam, Bn, Cn, x, th, z = raw_coeffs(L, u)
    Bsz, T, H, R, N = Bn.shape
    ang = torch.cumsum(dt[..., None] * th, 1)[:, :, :, None]                    # Phi_t (B,T,H,1,N/2)
    c, s = torch.cos(ang), torch.sin(ang)

    def rope_T(v):  # (prod_{i<=t} R_i)^T v, i.e. rotate each pair by -Phi_t
        a, b = v[..., 0::2], v[..., 1::2]
        return torch.stack((c * a + s * b, -s * a + c * b), -1).flatten(-2)
    Bb, Cb = rope_T(Bn), rope_T(Cn)
    la = dt * A
    alpha = torch.exp(la)
    gamma, beta = lam * dt, (1 - lam) * dt * alpha
    cs = torch.cumsum(la, 1)
    tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=u.device))[None, :, :, None]
    dec = torch.exp((cs[:, :, None] - cs[:, None, :]).masked_fill(~tri, float("-inf")))  # (B,t,s,H)
    band = torch.zeros(Bsz, T, T, H, dtype=torch.float64, device=u.device)
    i = torch.arange(T, device=u.device)
    band[:, i, i] = gamma
    band[:, i[1:], i[:-1]] = beta[:, 1:]
    Lm = torch.einsum('btkh,bksh->btsh', dec, band)
    G = torch.einsum('bthin,bshjn->bhtsij', Cb, Bb)
    y = torch.einsum('btsh,bhtsij,bshpj->bthip', Lm, G, x)
    return _finish(L, y, z)
