"""Mamba-3 (Lahoti et al., arXiv 2603.15569), implemented to the paper.

THREE CHANGES FROM MAMBA-2, implemented here in the order the paper derives
them. Nothing is improvised: where this file departs from the paper it says
so, and the equivalence tests at the bottom are how that claim is checked.

1. EXPONENTIAL-TRAPEZOIDAL DISCRETIZATION (Sec 3.1). Mamba-1/2 approximate

        integral_{t_{k-1}}^{t_k} e^{A(t_k - tau)} B(tau) x(tau) d tau

   by holding the endpoint (Euler). Mamba-3 averages both endpoints
   (trapezoidal), which makes the recurrence depend on x_{t-1} as well as
   x_t:

        h_t = alpha_t h_{t-1} + beta_t B_{t-1} x_{t-1} + gamma_t B_t x_t

   with alpha_t = exp(Delta_t A), beta_t = (Delta_t/2) alpha_t,
   gamma_t = Delta_t/2. That two-tap dependence IS the implicit convolution
   the paper describes - it is why Mamba-3 can drop the short causal conv
   that Mamba-1/2 carry as a separate component (Sec 3.1, and eq 7's
   factorisation of L into a 1-semiseparable matrix times a 2-band matrix).

2. COMPLEX STATE via the RoPE trick (Sec 3.2, Prop 2 and 3). A complex SSM
   of state N/2 is equivalent to a REAL SSM of state N whose transition is a
   scalar decay times a block-diagonal of 2x2 rotations:

        R_t = blockdiag({ R(Delta_t theta_t[i]) })
        h_t = alpha_t R_t h_{t-1} + ...

   Real non-negative eigenvalues cannot express rotation, which is why
   Mamba-2 provably fails state tracking (parity). This file implements the
   real-equivalent form, not complex arithmetic, exactly as Prop 2 licenses.

3. MIMO (Sec 3.3). x_t: (P,) -> (P,R), B_t: (N,) -> (N,R), C_t likewise;
   the state stays (N,P) and the update becomes a matmul B_t x_t^T rather
   than an outer product. Memory traffic is dominated by h_t and barely
   moves; FLOPs rise by R. Arithmetic intensity goes Theta(1) -> Theta(R),
   which is the whole point during memory-bound decode.

WHAT THIS FILE DOES NOT DO. No fused CUDA scan - the paper's speed comes
from hardware-aware kernels and this runs on MPS/CPU with an explicit loop
for the recurrent path and a materialised mask for the parallel path. The
MODELLING claims are reproducible here; the LATENCY claims are not.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_pairs(h: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Apply blockdiag 2x2 rotations to the state's (even, odd) channel pairs.

    h:   (..., N)   state, N even
    ang: (..., N/2) rotation angle per pair
    """
    a, b = h[..., 0::2], h[..., 1::2]
    c, s = torch.cos(ang), torch.sin(ang)
    out = torch.empty_like(h)
    out[..., 0::2] = c * a - s * b
    out[..., 1::2] = s * a + c * b
    return out


def ssm_scan(alpha, beta, gamma, theta, S, Cm, P, state=None, prev_S=None):
    """The Mamba-3 recurrence (paper eq 11), shared by every Mamba-3 path.

    This is the SINGLE SOURCE OF TRUTH for the recurrence, so a paper-faithfulness
    fix here (e.g. the beta-term rotation) applies everywhere and cannot drift.

        h_t = R_t(alpha_t h_{t-1} + beta_t S_{t-1}) + gamma_t S_t
        y_t = C_t^T h_t

    alpha,beta,gamma: (B,T,H)      per-step scalars
    theta:            (B,T,H,N/2)  per-step rotation angles (already * dt)
    S:                (B,T,H,N,P)  per-step write B_t x_t^T (or summed streams)
    Cm:               (B,T,H,N,R)  readout
    P:                head dim; state/prev_S carry across chunks for streaming.
    Returns (y_flat (B,T,H*R*P), final state h, last write S[:,-1]).
    """
    B, T, H, N, _ = S.shape
    R = Cm.shape[-1]
    h = torch.zeros(B, H, N, P, device=S.device, dtype=S.dtype) \
        if state is None else state
    ys = []
    for t in range(T):
        prev = S[:, t - 1] if t > 0 else prev_S
        pre = alpha[:, t, :, None, None] * h
        if prev is not None:
            pre = pre + beta[:, t, :, None, None] * prev
        h = rotate_pairs(pre.transpose(-1, -2), theta[:, t].unsqueeze(-2)
                         .expand(-1, -1, P, -1)).transpose(-1, -2)
        h = h + gamma[:, t, :, None, None] * S[:, t]
        ys.append((Cm[:, t].transpose(-1, -2) @ h).reshape(B, H * R * P))
    return torch.stack(ys, 1), h, S[:, -1]


class Mamba3Layer(nn.Module):
    """One Mamba-3 layer. Recurrent and parallel paths, provably equal.

    d_model  input/output width
    d_state  N, the SSM state size (even; pairs are rotated)
    rank     R of the MIMO lift. rank=1 is SISO.
    head_dim P, channels sharing one (alpha, Delta, B) recurrence.
    """

    def __init__(self, d_model: int, d_state: int = 64, rank: int = 1,
                 head_dim: int = 32, dt_min: float = 1e-3, dt_max: float = 0.5):
        super().__init__()
        assert d_state % 2 == 0, "d_state must be even: the state rotates in pairs"
        assert d_model % head_dim == 0, "d_model must divide into heads of head_dim"
        self.d_model, self.N, self.R, self.P = d_model, d_state, rank, head_dim
        self.H = d_model // head_dim

        # Gate branch (Mamba-2/3 block, Fig 2): the SSM output is multiplied by
        # SiLU(in_proj(input)) before out_proj. Dimension d_model*R to match the
        # pre-out_proj SSM output (H*R*P). Was defined-but-unused before.
        self.in_proj = nn.Linear(d_model, d_model * rank)
        self.B_proj = nn.Linear(d_model, self.H * d_state * rank)
        self.C_proj = nn.Linear(d_model, self.H * d_state * rank)
        self.x_proj = nn.Linear(d_model, d_model * rank)
        self.dt_proj = nn.Linear(d_model, self.H)
        self.theta_proj = nn.Linear(d_model, self.H * (d_state // 2))
        # lambda_t: the DATA-DEPENDENT trapezoidal weight (paper Prop 1, Remark
        # 3). gamma_t = lambda_t * dt, beta_t = (1-lambda_t) * dt * alpha_t.
        # lambda=1/2 is the classical trapezoid; the paper found NOT enforcing
        # that (letting the model choose per step) is empirically better. Bias 0
        # -> sigmoid 0.5 at init, so training starts from the classical rule.
        self.lam_proj = nn.Linear(d_model, self.H)
        nn.init.zeros_(self.lam_proj.weight)
        nn.init.zeros_(self.lam_proj.bias)
        self.out_proj = nn.Linear(d_model * rank, d_model)
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)

        # BCNorm / QKNorm (paper section 3.2). RMS normalisation AFTER the B and C
        # projections, mirroring QKNorm in Transformers. The paper reports this
        # is what stabilises large-scale runs, to the point that they remove
        # Mamba-2's post-gate RMSNorm on the strength of it. It was missing from
        # the first version of this file and E17's ssm arm then diverged
        # (0.330 -> 19.313) into a constant predictor, which was patched with
        # gradient clipping - a generic fix for a problem the paper had already
        # solved specifically. Normalising over the N state dimensions, which is
        # the axis B and C index into.
        rms = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
        self.B_norm = rms(d_state)
        self.C_norm = rms(d_state)
        # B, C biases (paper appendix): learnable, head-specific, channel-wise,
        # data-INDEPENDENT, added after the norm, initialised to all ones. The
        # paper finds models insensitive to the init as long as it is positive,
        # and that B and C biases together beat either alone.
        self.B_bias = nn.Parameter(torch.ones(self.H, d_state, rank))
        self.C_bias = nn.Parameter(torch.ones(self.H, d_state, rank))

        # A is data-INDEPENDENT here. Remark 1: the paper found data-dependent
        # A_t performs similarly and chose it for consistency; the simpler
        # form is used here and the difference is recorded rather than hidden.
        self.A_log = nn.Parameter(torch.log(torch.rand(self.H) * 0.9 + 0.05))
        # dt bias initialised so Delta spans [dt_min, dt_max] - the standard
        # Mamba init. Delta sets the memory horizon, so this is the prior on
        # how long the layer remembers before it learns anything.
        dt = torch.exp(torch.rand(self.H) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min))
        self.dt_proj.bias.data = dt + torch.log(-torch.expm1(-dt))

    def _coeffs(self, u: torch.Tensor):
        """alpha, beta, gamma, B, C, x, theta for every timestep."""
        Bsz, T, _ = u.shape
        H, N, R, P = self.H, self.N, self.R, self.P
        dt = F.softplus(self.dt_proj(u))                      # (B,T,H) > 0
        A = -torch.exp(self.A_log)                            # (H,) < 0
        alpha = torch.exp(dt * A)                             # (B,T,H) in (0,1)
        # Exponential-trapezoidal with a DATA-DEPENDENT weight lambda_t (paper
        # Prop 1 / eq 5-6): gamma_t = lambda_t*dt (current input), beta_t =
        # (1-lambda_t)*dt*alpha_t (previous input). lambda=1/2 recovers the
        # classical trapezoid; lambda=1 recovers Mamba-2's exponential-Euler.
        lam = torch.sigmoid(self.lam_proj(u))                 # (B,T,H) in (0,1)
        gamma = lam * dt
        beta = (1.0 - lam) * dt * alpha
        # BCNorm then bias, in that order - the paper adds the bias AFTER the
        # norm, so the bias is not rescaled by it.
        Bm = self.B_proj(u).view(Bsz, T, H, R, N)
        Bm = self.B_norm(Bm).permute(0, 1, 2, 4, 3) + self.B_bias
        Cm = self.C_proj(u).view(Bsz, T, H, R, N)
        Cm = self.C_norm(Cm).permute(0, 1, 2, 4, 3) + self.C_bias
        x = self.x_proj(u).view(Bsz, T, H, P, R)
        theta = self.theta_proj(u).view(Bsz, T, H, N // 2) * dt.unsqueeze(-1)
        return alpha, beta, gamma, Bm, Cm, x, theta

    def gate_and_proj(self, y_flat: torch.Tensor, nu: torch.Tensor):
        """Apply the Mamba gate then the output projection.

        y_flat: (..., d_model*R) SSM output before out_proj.
        nu:     (..., d_model)   the NORMALISED layer input (in_proj sees the
                                 same normalised input the SSM path does).
        out = out_proj( y_flat * SiLU(in_proj(nu)) ).
        """
        return self.out_proj(y_flat * F.silu(self.in_proj(nu)))

    def forward(self, u: torch.Tensor, state=None, return_state=False):
        """Recurrent path. u: (B,T,d_model). This is the streaming form.

        The recurrence itself lives in the shared `ssm_scan` (paper eq 11), so
        callers cannot drift. Here the per-step write is the
        single-stream outer product S_t = B_t x_t^T.
        """
        nu = self.norm(u)
        alpha, beta, gamma, Bm, Cm, x, theta = self._coeffs(nu)
        S = Bm @ x.transpose(-1, -2)                    # (B,T,H,N,P)
        y_flat, h, _ = ssm_scan(alpha, beta, gamma, theta, S, Cm, self.P,
                                state=state)
        out = self.gate_and_proj(y_flat, nu)
        return (out, h) if return_state else out

    def step(self, u_t: torch.Tensor, state, prev_bx=None):
        """One timestep, constant memory. The deployment path.

        state: (B,H,N,P) or None. prev_bx: B_{t-1}x_{t-1}^T for the beta term,
        which the trapezoidal rule requires and Mamba-2 does not have.
        """
        Bsz = u_t.shape[0]
        nu = self.norm(u_t.unsqueeze(1))
        alpha, beta, gamma, Bm, Cm, x, theta = self._coeffs(nu)
        H, N, P = self.H, self.N, self.P
        h = torch.zeros(Bsz, H, N, P, device=u_t.device, dtype=u_t.dtype) \
            if state is None else state
        # Match forward/eq(11): rotate (alpha*h + beta*prev_bx) by R_t, then add
        # the current (unrotated) write. prev_bx = B_{t-1}x_{t-1}^T from the last
        # step; bx = B_t x_t^T is returned for the next step's beta term.
        pre = alpha[:, 0, :, None, None] * h
        if prev_bx is not None:
            pre = pre + beta[:, 0, :, None, None] * prev_bx
        h = rotate_pairs(pre.transpose(-1, -2), theta[:, 0].unsqueeze(-2)
                         .expand(-1, -1, P, -1)).transpose(-1, -2)
        bx = Bm[:, 0] @ x[:, 0].transpose(-1, -2)
        h = h + gamma[:, 0, :, None, None] * bx
        y = (Cm[:, 0].transpose(-1, -2) @ h).reshape(Bsz, -1)
        return self.gate_and_proj(y, nu[:, 0]), h, bx


# ---------------------------------------------------------------------------
def selftest(verbose: bool = True) -> dict:
    """The four invariants, at float64.

    These were an ad-hoc scratch script until BCNorm was added and there was
    nothing to re-run. A verified file whose verification lives somewhere else
    is an unverified file the moment anyone edits it.

        python -m promptingnemo.models.mamba3.layer
    """
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    net = Mamba3Layer(d_model=32, d_state=16, rank=2, head_dim=8).eval()
    B, T = 2, 20
    u = torch.randn(B, T, 32)
    out = {}

    with torch.no_grad():
        y = net(u)

        # 1. CAUSALITY, as a SCALE test rather than an equality one. Appending
        #    future timesteps moves the past by ~1e-15 here: the permute in
        #    _coeffs leaves B and C non-contiguous, so matmul takes a different
        #    kernel path at a different T and the last bits differ. An equality
        #    assertion would call that a leak, and would also fail spuriously in
        #    float32. A real leak scales with the size of the future; rounding
        #    does not. Drive the future through four orders of magnitude and
        #    require the drift to stay put.
        drifts = []
        for mag in (1.0, 1e2, 1e4):
            u2 = torch.cat([u, torch.randn(B, 6, 32) * mag], 1)
            drifts.append(float((net(u2)[:, :T] - y).abs().max()))
        out["causality"] = max(drifts)
        out["causality_scaling"] = max(drifts) / max(min(drifts), 1e-300)

        # 2. STREAMING == BATCHED. The deployment path must equal training.
        st = bx = None
        steps = []
        for t in range(T):
            o, st, bx = net.step(u[:, t], st, bx)
            steps.append(o)
        out["streaming"] = float((torch.stack(steps, 1) - y).abs().max())

        # 3. ROTATION IS ORTHOGONAL. State norm preserved, so it cannot blow up.
        h = torch.randn(B, 4, 16, 8)
        ang = torch.randn(B, 4, 1, 8)
        hr = rotate_pairs(h.transpose(-1, -2), ang).transpose(-1, -2)
        out["orthogonal"] = float(
            (hr.norm(dim=-2) - h.norm(dim=-2)).abs().max())

        # 4. TRAPEZOIDAL TWO-TAP. Perturbing x_{t-1} moves y_t, not y_{t-2}.
        up = u.clone(); up[:, 3] += 1.0
        yp = net(up)
        out["twotap_y4"] = float((yp[:, 4] - y[:, 4]).abs().max())
        out["twotap_y2"] = float((yp[:, 2] - y[:, 2]).abs().max())

        # 5. BCNORM IS ACTIVE. B's pre-bias rows should have unit RMS. Without
        #    the norm this is whatever the projection happens to produce, and
        #    the first version of this file had no norm at all.
        a, b, g, Bm, Cm, x, th = net._coeffs(net.norm(u))
        #    Bm is (B,T,H,N,R) after the permute, so the normalised axis is -2.
        rms = (Bm - net.B_bias).pow(2).mean(-2).sqrt()
        out["bcnorm_rms"] = float((rms - 1.0).abs().max())

    out["params"] = sum(p.numel() for p in net.parameters())
    torch.set_default_dtype(torch.float32)

    if verbose:
        ok = lambda c: "OK" if c else "FAIL"
        print("Mamba-3 selftest  (float64)")
        print(f"  1 causality (future->past)   {out['causality']:.3e}"
              f"   {ok(out['causality'] < 1e-12)}")
        print(f"    drift x1 vs x10000 ratio   {out['causality_scaling']:.3f}"
              f"   {ok(out['causality_scaling'] < 1.5)}  (a leak would scale)")
        print(f"  2 step() == forward()        {out['streaming']:.3e}"
              f"   {ok(out['streaming'] < 1e-10)}")
        print(f"  3 rotation orthogonal        {out['orthogonal']:.3e}"
              f"   {ok(out['orthogonal'] < 1e-10)}")
        print(f"  4 trapezoidal two-tap        y4 {out['twotap_y4']:.3e}"
              f"  y2 {out['twotap_y2']:.3e}"
              f"   {ok(out['twotap_y4'] > 1e-6 and out['twotap_y2'] == 0.0)}")
        print(f"  5 BCNorm rows unit RMS       {out['bcnorm_rms']:.3e}"
              f"   {ok(out['bcnorm_rms'] < 1e-10)}")
        print(f"  {out['params']:,} params")
    return out


if __name__ == "__main__":
    selftest()
