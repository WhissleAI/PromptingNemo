"""DNSMOS P.835 (Reddy et al., ICASSP 2022; the Microsoft DNS-Challenge ONNX models as packaged by `speechmos`) on CPU.

~/anaconda3/envs/tb_score/bin/python -s steer/paper/dnsmos_score.py <list.txt> <cache.json> [procs]
list.txt: one wav path per line. cache.json: {path: {ovrl, sig, bak, p808}} (incremental; existing entries kept).
Audio is mixed to mono and resampled to 16 kHz (soxr); non-personalised model; clips < 9 s are handled by the
package's own looping (identical to the official dnsmos_local.py).
"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")


_M = None


def _model():
    """speechmos' DNSMOS class with single-threaded ONNX sessions (its default sessions spin up one thread per core in
    every worker, which took ~45 cores)."""
    global _M
    if _M is None:
        import onnxruntime as ort
        from speechmos import dnsmos
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        d = os.path.join(os.path.dirname(dnsmos.__file__), "dnsmos_models")
        m = dnsmos.DNSMOS.__new__(dnsmos.DNSMOS)
        m.primary_model_path = os.path.join(d, "sig_bak_ovr.onnx")
        m.onnx_sess = ort.InferenceSession(m.primary_model_path, so, providers=["CPUExecutionProvider"])
        m.p808_onnx_sess = ort.InferenceSession(os.path.join(d, "model_v8.onnx"), so, providers=["CPUExecutionProvider"])
        _M = m
    return _M


def one(p):
    import soundfile as sf
    import soxr
    y, sr = sf.read(p, dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    if sr != 16000:
        y = soxr.resample(y, sr, 16000)
    y = y.clip(-1.0, 1.0)  # resampling overshoot; the package rejects |x| > 1
    r = _model()(y, 16000, False)
    return p, dict(ovrl=float(r["ovrl_mos"]), sig=float(r["sig_mos"]), bak=float(r["bak_mos"]), p808=float(r["p808_mos"]))


def main():
    lst, cache = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    res = json.load(open(cache)) if os.path.exists(cache) else {}
    paths = [l.strip() for l in open(lst) if l.strip()]
    todo = [p for p in paths if p not in res and os.path.exists(p)]
    print(f"dnsmos: {len(todo)} new of {len(paths)}", flush=True)
    with Pool(procs) as pool:
        for i, (p, r) in enumerate(pool.imap_unordered(one, todo, chunksize=8)):
            res[p] = r
            if i % 1000 == 999:
                json.dump(res, open(cache, "w"))
                print(f"  {i + 1}/{len(todo)}", flush=True)
    json.dump(res, open(cache, "w"))
    print("dnsmos done", len(res), flush=True)


if __name__ == "__main__":
    main()
