"""One runner for every Mamba-3 invariant. Exits non-zero on failure.

    python -m steertts.mamba3.tests          # ~1 min on CPU
    python -m steertts.mamba3.tests --full   # adds fp32/bf16 accuracy at T=3000

Checks, at float64 unless stated:
  1. layer selftest: causality, step == forward, rotation orthogonal, trapezoidal two-tap, BCNorm
  2. chunked scan and SSD reduction vs the recurrence, fwd + bwd (their own verify())
  3. PAPER FAITHFULNESS against reference.py, which recomputes every coefficient from raw parameters:
     two independent references (complex arithmetic, eq (7) parallel form) agree with each other and
     with every layer path (forward, step, scan, SSD loop, SSD vectorized) over ranks, lengths
     (including T=1 and T not a multiple of the chunk) and chunk sizes
  4. gradients: torch.autograd.gradcheck wrt the input AND every parameter on each path, and gradient
     agreement of each fast path with the reference
  5. causality: hard (perturb the future by 1e3, past outputs unchanged) and Jacobian (dy_t/du_s == 0
     for s > t, exactly)
  6. streaming: forward() in segments carrying (h, S_last) == one forward() over the whole sequence
  7. strong decay: finite outputs and gradients (incl. wrt alpha) on every scan path in float32
  8. (--full) float32 and bf16-autocast error vs the float64 reference at T = 100/1000/3000
"""
from __future__ import annotations

import argparse
import contextlib
import math
import sys

import torch

from .layer import Mamba3Layer, selftest as layer_selftest
from .reference import ref_complex, ref_parallel
from .scan import chunked_scan, scan_forward, verify as scan_verify
from .ssd import scan_forward_ssd, ssd_pytorch, ssd_vectorized, verify as ssd_verify


@contextlib.contextmanager
def default_dtype(dt):
    """set_default_dtype that is restored even when a check raises (the old runner leaked float64)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def make(d=16, N=8, R=1, P=8, seed=0, dtype=torch.float64, randomize=True):
    """A layer moved off its symmetric init, so lambda, the biases, the norms and the rotation all matter."""
    torch.manual_seed(seed)
    L = Mamba3Layer(d_model=d, d_state=N, rank=R, head_dim=P).to(dtype).eval()
    if randomize:
        with torch.no_grad():
            L.lam_proj.weight.normal_(0, 0.5); L.lam_proj.bias.normal_(0, 1)
            L.B_bias.normal_(1, 0.5); L.C_bias.normal_(1, 0.5)
            for n in (L.B_norm, L.C_norm, L.norm):
                n.weight.normal_(1, 0.2)
            L.theta_proj.weight.mul_(4)  # angles large enough that the rotation is far from identity
    return L


def paths(L, u, chunk):
    """Every way the package can compute a layer's output."""
    with torch.no_grad():
        st = bx = None
        steps = []
        for t in range(u.shape[1]):
            o, st, bx = L.step(u[:, t], st, bx)
            steps.append(o)
        return {"forward": L(u), "step": torch.stack(steps, 1), "scan": scan_forward(L, u, chunk),
                "ssd_loop": scan_forward_ssd(L, u, chunk, vectorized=False),
                "ssd_vec": scan_forward_ssd(L, u, chunk, vectorized=True)}


class _Path(torch.nn.Module):
    def __init__(self, L, path):
        super().__init__(); self.L, self.path = L, path

    def forward(self, u):
        if self.path == "forward":
            return self.L(u)
        if self.path == "scan":
            return scan_forward(self.L, u, 3)
        return scan_forward_ssd(self.L, u, 3, vectorized=self.path == "ssd_vec")


def run(full: bool) -> bool:
    ok = True

    def check(name, val, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name:<62} {val:.2e}")

    print("== 1. layer selftest ==")
    with default_dtype(torch.float64):
        L = layer_selftest(verbose=False)
    check("causality (future -> past)", L["causality"], L["causality"] < 1e-12)
    check("step() == forward()", L["streaming"], L["streaming"] < 1e-10)
    check("rotation orthogonal", L["orthogonal"], L["orthogonal"] < 1e-10)
    check("trapezoidal two-tap (y4 moves, y2 does not)", L["twotap_y4"],
          L["twotap_y4"] > 1e-6 and L["twotap_y2"] == 0.0)
    check("BCNorm rows unit RMS", L["bcnorm_rms"], L["bcnorm_rms"] < 1e-10)

    print("== 2. scan / SSD vs recurrence (fwd + bwd) ==")
    with default_dtype(torch.float64):
        for name, res in (("scan", scan_verify(verbose=False)), ("ssd", ssd_verify(verbose=False))):
            for k, v in res.items():
                check(f"{name} {k}", v, v < 1e-9)

    with default_dtype(torch.float64):
        print("== 3. paper faithfulness: independent references ==")
        worst = {}
        for d, N, R, P in [(16, 8, 1, 8), (16, 8, 2, 8), (24, 4, 3, 8), (8, 2, 1, 4)]:
            lay = make(d, N, R, P, seed=R)
            for T in (1, 2, 5, 16, 37):
                u = torch.randn(2, T, d)
                with torch.no_grad():
                    rc, rp = ref_complex(lay, u), ref_parallel(lay, u)
                worst["ref_complex vs ref_parallel"] = max(worst.get("ref_complex vs ref_parallel", 0),
                                                           float((rc - rp).abs().max()))
                for chunk in (1, 4, 16, 64):
                    for k, v in paths(lay, u, chunk).items():
                        worst[k] = max(worst.get(k, 0.0), float((v - rc).abs().max()))
        for k, v in worst.items():
            check(f"{k} (4 configs x 5 lengths x 4 chunks)", v, v < 1e-11)

        print("== 4. gradients ==")
        lay = make(8, 4, 2, 4, seed=5)
        names = [n for n, _ in lay.named_parameters()]
        params = tuple(p.detach().clone().requires_grad_(True) for p in lay.parameters())
        u = torch.randn(1, 7, 8, requires_grad=True)
        for path in ("forward", "scan", "ssd_loop", "ssd_vec"):
            mod = _Path(lay, path)
            fn = lambda u, *ps: torch.func.functional_call(mod, {"L." + n: p for n, p in zip(names, ps)}, (u,))
            try:
                good = torch.autograd.gradcheck(fn, (u,) + params, eps=1e-6, atol=1e-6, rtol=1e-5)
            except Exception as ex:  # noqa: BLE001 - report, do not crash the runner
                print("     ", str(ex).splitlines()[0][:120]); good = False
            check(f"gradcheck {path} wrt input + all {len(params)} params", 0.0, good)
        lay = make(16, 8, 2, 8, seed=7)
        u, w = torch.randn(2, 37, 16), torch.randn(2, 37, 16)

        def grads(f):
            lay.zero_grad(); ui = u.clone().requires_grad_(True)
            (f(ui) * w).sum().backward()
            return [ui.grad.clone()] + [p.grad.clone() for p in lay.parameters()]
        gr = grads(lambda v: ref_complex(lay, v))
        for name, f in (("forward", lay), ("scan", lambda v: scan_forward(lay, v, 8)),
                        ("ssd_loop", lambda v: scan_forward_ssd(lay, v, 8, vectorized=False)),
                        ("ssd_vec", lambda v: scan_forward_ssd(lay, v, 8, vectorized=True))):
            e = max(float((a - b).abs().max()) for a, b in zip(grads(f), gr))
            check(f"grad(input, params) {name} vs ref_complex", e, e < 1e-9)

        print("== 5. causality ==")
        lay = make(16, 8, 2, 8, seed=3)
        u = torch.randn(2, 40, 16)
        base = paths(lay, u, 16)
        for t0 in (15, 16, 39):  # inside a chunk, on a chunk boundary, last step
            u2 = u.clone(); u2[:, t0:] += 1e3 * torch.randn_like(u2[:, t0:])
            for k, v in paths(lay, u2, 16).items():
                e = float((v[:, :t0] - base[k][:, :t0]).abs().max())
                check(f"hard causality {k} t0={t0}", e, e < 1e-9)
        ug = torch.randn(1, 20, 16, requires_grad=True)
        for name, f in (("forward", lay), ("scan", lambda v: scan_forward(lay, v, 8)),
                        ("ssd_loop", lambda v: scan_forward_ssd(lay, v, 8, vectorized=False)),
                        ("ssd_vec", lambda v: scan_forward_ssd(lay, v, 8, vectorized=True))):
            g, = torch.autograd.grad(f(ug)[0, 9].sum(), ug)
            e = float(g[0, 10:].abs().max())
            check(f"Jacobian causality {name}: |dy_9/du_s>9|", e, e == 0.0)

        print("== 6. streaming in segments ==")
        lay = make(16, 8, 2, 8, seed=9)
        u = torch.randn(2, 30, 16)
        with torch.no_grad():
            whole = lay(u)
            for cuts in ((17,), (1, 2, 16, 29), tuple(range(1, 30))):
                outs, st, s0 = [], None, 0
                for c in cuts + (30,):
                    o, st = lay(u[:, s0:c], state=st, return_state=True)
                    outs.append(o); s0 = c
                e = float((torch.cat(outs, 1) - whole).abs().max())
                check(f"forward(state) over segments cut at {cuts if len(cuts) < 5 else 'every step'}", e, e < 1e-10)

    print("== 7. strong decay: finite fwd + bwd in float32 ==")
    with default_dtype(torch.float32):
        torch.manual_seed(0)
        for la_step in (-1.5, -3.0, -20.0):  # per-step log decay; a 64-chunk sums to -96 .. -1280
            al = torch.full((1, 64, 1), math.exp(la_step), requires_grad=True)
            Bf = torch.randn(1, 64, 1, 4, 2, requires_grad=True)
            xf, Cf = torch.randn(1, 64, 1, 2, 2), torch.randn(1, 64, 1, 4, 1)
            for name, fn in (("ssd_loop", ssd_pytorch), ("ssd_vec", ssd_vectorized)):
                al.grad = Bf.grad = None
                y = fn(al, Bf, xf, Cf, 64); y.sum().backward()
                bad = sum(int((~torch.isfinite(t)).sum()) for t in (y, al.grad, Bf.grad))
                check(f"{name} log-decay/step {la_step}: non-finite entries", float(bad), bad == 0)
            al.grad = None
            g, _ = chunked_scan(al, torch.randn(1, 64, 1, 4, 2), 64); g.sum().backward()
            bad = int((~torch.isfinite(al.grad)).sum()) + int((~torch.isfinite(g)).sum())
            check(f"chunked_scan log-decay/step {la_step}: non-finite entries", float(bad), bad == 0)

    if full:
        print("== 8. float32 / bf16 accuracy vs float64 (d=256, N=32, P=32) ==")
        for T in (100, 1000, 3000):
            with default_dtype(torch.float64):
                L64 = make(256, 32, 1, 32, seed=11, randomize=False)
                u = torch.randn(2, T, 256)
                with torch.no_grad():
                    ref = scan_forward_ssd(L64, u)
            L32 = make(256, 32, 1, 32, seed=11, dtype=torch.float32, randomize=False)
            L32.load_state_dict({k: v.float() for k, v in L64.state_dict().items()})
            with torch.no_grad():
                sc = float(ref.abs().max())
                e32 = float((scan_forward_ssd(L32, u.float()).double() - ref).abs().max()) / sc
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    ebf = float((scan_forward_ssd(L32, u.float()).double() - ref).abs().max()) / sc
            check(f"float32 relative error T={T}", e32, e32 < 1e-4)
            check(f"bf16-autocast relative error T={T} (informational; BiMamba3 runs fp32)", ebf, ebf < 1e-1)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="add the slow long-sequence precision checks")
    ok = run(ap.parse_args().full)
    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
