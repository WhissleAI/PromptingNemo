"""One runner for every Mamba-3 invariant. Exits non-zero on failure.

    python -m promptingnemo.models.mamba3.tests

Runs, at float64:
  - Mamba3Layer selftest (causality, step==forward, rotation orthogonality,
    trapezoidal two-tap, BCNorm)
  - chunked scan vs recurrence (fwd+bwd)
  - SSD reduction vs recurrence (fwd+bwd)
  - PAPER FAITHFULNESS: the recurrence, the scan and the SSD path all match a
    direct transcription of the paper's Proposition 4 / eq (11) to <1e-12.

The last check is the one that would have caught the beta-rotation bug.
Run it before every commit.
"""
from __future__ import annotations

import sys

import torch

from .layer import Mamba3Layer, rotate_pairs, selftest as layer_selftest
from .scan import verify as scan_verify, scan_forward
from .ssd import verify as ssd_verify, scan_forward_ssd


def paper_eq11_check() -> dict:
    """Transcribe paper eq (11) directly; require all paths to match it."""
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    lay = Mamba3Layer(d_model=16, d_state=8, rank=2, head_dim=8).eval()
    B, T = 2, 14
    u = torch.randn(B, T, 16)
    a, beta, gamma, Bm, Cm, x, theta = lay._coeffs(lay.norm(u))
    H, N, R, P = lay.H, lay.N, lay.R, lay.P
    Phi = torch.cumsum(theta, 1)
    Phi_prev = torch.cat([torch.zeros_like(Phi[:, :1]), Phi[:, :-1]], 1)
    from .scan import rotate_by
    # eq (11): h_t = a_t h_{t-1} + b_t (prod_{i<=t-1}R_i^T)B_{t-1}x_{t-1}
    #                            + g_t (prod_{i<=t}  R_i^T)B_t   x_t
    #          y_t = [(prod_{i<=t}R_i^T)C_t]^T h_t
    h = torch.zeros(B, H, N, P)
    yp = []
    for t in range(T):
        Scur = rotate_by(Bm[:, t], -Phi[:, t]) @ x[:, t].transpose(-1, -2)
        h = a[:, t, :, None, None] * h + gamma[:, t, :, None, None] * Scur
        if t > 0:
            Sprev = rotate_by(Bm[:, t - 1], -Phi_prev[:, t]) @ x[:, t - 1].transpose(-1, -2)
            h = h + beta[:, t, :, None, None] * Sprev
        yp.append((rotate_by(Cm[:, t], -Phi[:, t]).transpose(-1, -2) @ h)
                  .reshape(B, H * R * P))
    y_paper = lay.out_proj(torch.stack(yp, 1) * torch.nn.functional.silu(lay.in_proj(lay.norm(u))))
    with torch.no_grad():
        out = {
            "recurrence_vs_eq11": float((lay(u) - y_paper).abs().max()),
            "scan_vs_eq11": float((scan_forward(lay, u, 4) - y_paper).abs().max()),
            "ssd_vs_eq11": float((scan_forward_ssd(lay, u, 4) - y_paper).abs().max()),
            "ssd_vec_vs_eq11": float((scan_forward_ssd(lay, u, 4, vectorized=True) - y_paper).abs().max()),
        }
    torch.set_default_dtype(torch.float32)
    return out


def main() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("== Mamba3Layer selftest =="); L = layer_selftest(verbose=False)
    check("causality == 0", L["causality"] < 1e-12)
    check("step == forward", L["streaming"] < 1e-10)
    check("rotation orthogonal", L["orthogonal"] < 1e-10)
    check("trapezoidal two-tap", L["twotap_y4"] > 1e-6 and L["twotap_y2"] == 0.0)
    check("BCNorm unit RMS", L["bcnorm_rms"] < 1e-10)

    print("== chunked scan (fwd+bwd) =="); S = scan_verify(verbose=False)
    for k, v in S.items():
        check(f"scan {k}", v < 1e-9)

    print("== SSD reduction (fwd+bwd) =="); D = ssd_verify(verbose=False)
    for k, v in D.items():
        check(f"ssd {k}", v < 1e-9)

    print("== paper eq (11) faithfulness =="); P = paper_eq11_check()
    for k, v in P.items():
        print(f"    {k} = {v:.3e}")
        check(k, v < 1e-12)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
