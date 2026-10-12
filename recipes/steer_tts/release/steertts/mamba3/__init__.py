"""Mamba-3: a paper-faithful, verified implementation of Lahoti et al. (arXiv 2603.15569).

Ported from the MMSE research repo (drowsiness-uta-rldd/mamba3, commit c5280e6+), without the MMSE
multi-stream extension or the LM tooling. Checked fwd+bwd at float64 against the recurrence:
  - exponential-trapezoidal discretization with DATA-DEPENDENT lambda_t (Prop 1)
  - complex state as real block-diagonal rotations / data-dependent RoPE (Prop 2-4, eq 11)
  - MIMO rank-R (Sec 3.3)
  - BCNorm/QKNorm on B,C + learnable data-independent B,C biases
  - output gating
Recorded deviations: data-independent A (Remark 1 says equivalent); MIMO input uses a full DPR
projection rather than the paper's DP+PR (R=1 identical).

Paths: `Mamba3Layer.forward` (O(T) recurrence, reference), `scan_forward` (chunked scan),
`scan_forward_ssd` (SSD reduction, the fast path for training). Run `python -m
steertts.mamba3.tests` after any change.
"""
from .layer import Mamba3Layer, rotate_pairs, ssm_scan
from .scan import scan_forward, chunked_scan, rotate_by, cumulative_angles
from .ssd import scan_forward_ssd, ssd_inputs, ssd_pytorch

__all__ = [
    "Mamba3Layer", "rotate_pairs", "ssm_scan",
    "scan_forward", "chunked_scan", "rotate_by", "cumulative_angles",
    "scan_forward_ssd", "ssd_inputs", "ssd_pytorch",
]
