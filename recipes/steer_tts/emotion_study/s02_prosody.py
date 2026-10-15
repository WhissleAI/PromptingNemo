"""Frame- and clip-level acoustic features for every CREMA-D clip (CPU, parallel).

Pitch uses the two-pass speaker-adaptive range of De Looze & Hirst: pass 1 with a
wide 50-700 Hz range gives per-actor quartiles, pass 2 uses
[0.75*q25, 1.5*q75] (widened a little because angry/fearful speech runs high).

Frame features (10 ms): f0, intensity, HNR, F1-F3, alpha ratio, Hammarberg index,
eGeMAPS spectral slopes (0-500, 500-1500 Hz), spectral centroid, H1-H2.
Clip features: summary stats of the above plus duration/rate, CPPS, jitter, shimmer.
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import soundfile as sf
from tqdm import tqdm

from common import WAV, OUT, FRAMES, SR, HOP

SYLLABLES = {"IEO": 6, "TIE": 8, "IOM": 8, "IWW": 8, "TAI": 7, "MTI": 9,
             "IWL": 8, "ITH": 10, "DFA": 6, "ITS": 7, "TSI": 5, "WSI": 9}

N_FFT, WIN = 1024, 640  # 40 ms window: resolves harmonics for H1-H2 down to ~60 Hz
FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)


def load(clip):
    y, sr = sf.read(WAV / f"{clip}.wav", dtype="float64")
    assert sr == SR
    if y.ndim > 1:
        y = y.mean(1)
    return y


def pitch_track(snd, lo, hi):
    p = snd.to_pitch_ac(time_step=HOP, pitch_floor=lo, pitch_ceiling=hi,
                        very_accurate=True, voicing_threshold=0.5)
    f0 = p.selected_array["frequency"]
    f0[f0 == 0] = np.nan
    return p.xs(), f0


def pass1(clip):
    snd = parselmouth.Sound(load(clip), SR)
    _, f0 = pitch_track(snd, 50, 700)
    v = f0[~np.isnan(f0)]
    return v.astype(np.float32)


def band_db(P, lo, hi):
    m = (FREQS >= lo) & (FREQS < hi)
    return 10 * np.log10(P[:, m].sum(1) + 1e-12)


def slope(P, lo, hi):
    """Linear-regression slope of the dB spectrum between lo and hi Hz (dB/Hz, eGeMAPS)."""
    m = (FREQS >= lo) & (FREQS <= hi)
    x = FREQS[m]
    Y = 10 * np.log10(P[:, m] + 1e-12)
    xc = x - x.mean()
    return (Y - Y.mean(1, keepdims=True)) @ xc / (xc @ xc)


def spectral(y, times, f0):
    """Spectral frame features centred on the pitch-track times."""
    centers = np.round(times * SR).astype(int)
    pad = np.pad(y, (WIN, WIN))
    idx = centers[:, None] + WIN + np.arange(-WIN // 2, WIN // 2)[None]
    frames = pad[idx] * np.hanning(WIN)[None]
    P = np.abs(np.fft.rfft(frames, N_FFT)) ** 2
    lo = band_db(P, 50, 1000)
    alpha = band_db(P, 1000, 5000) - lo
    Pdb = 10 * np.log10(P + 1e-12)
    hamm = Pdb[:, (FREQS >= 0) & (FREQS < 2000)].max(1) - Pdb[:, (FREQS >= 2000) & (FREQS < 5000)].max(1)
    cent = (P * FREQS).sum(1) / (P.sum(1) + 1e-12)
    s0 = slope(P, 0, 500)
    s1 = slope(P, 500, 1500)
    # H1-H2: harmonic amplitude peaks searched within +-10% of f0 and 2*f0
    h1h2 = np.full(len(times), np.nan)
    for i in np.where(~np.isnan(f0))[0]:
        def peak(f):
            m = (FREQS > 0.9 * f) & (FREQS < 1.1 * f)
            return Pdb[i, m].max() if m.any() else np.nan
        h1h2[i] = peak(f0[i]) - peak(2 * f0[i])
    return alpha, hamm, cent, s0, s1, h1h2


def st(x, ref):
    return 12 * np.log2(x / ref)


def features(args):
    clip, lo, hi, sex = args
    y = load(clip)
    snd = parselmouth.Sound(y, SR)
    t, f0 = pitch_track(snd, lo, hi)

    inten = snd.to_intensity(minimum_pitch=max(lo, 60), time_step=HOP)
    I = np.interp(t, inten.xs(), inten.values[0])
    hnr_o = snd.to_harmonicity_cc(time_step=HOP, minimum_pitch=lo)
    hv = hnr_o.values[0].copy()
    hv[hv < -100] = np.nan  # praat marks unvoiced with -200; mask before resampling
    H = np.interp(t, hnr_o.xs(), hv)
    fm = snd.to_formant_burg(time_step=HOP, max_number_of_formants=5,
                             maximum_formant=5000 if sex == "Male" else 5500)
    F = np.array([[fm.get_value_at_time(k, x) for k in (1, 2, 3)] for x in t])
    alpha, hamm, cent, s0, s1, h1h2 = spectral(y, t, f0)

    voiced = ~np.isnan(f0)
    F[~voiced] = np.nan  # formants only meaningful on voiced frames
    fr = dict(t=t, f0=f0, int=I, hnr=H, f1=F[:, 0], f2=F[:, 1], f3=F[:, 2],
              alpha=alpha, hamm=hamm, cent=cent, slope0_500=s0, slope500_1500=s1, h1h2=h1h2)
    np.savez_compressed(FRAMES / f"{clip}.npz", **{k: np.asarray(v, np.float32) for k, v in fr.items()})

    # speech region: intensity within 30 dB of the clip max, voiced span as the core
    sp = I > I.max() - 30
    if voiced.any():
        on, off = t[voiced][0], t[voiced][-1]
    else:
        on, off = t[sp][0], t[sp][-1]
    dur = off - on
    # pauses: >=150 ms runs of low intensity inside the speech span
    inside = (t >= on) & (t <= off)
    low = (~sp) & inside
    runs, n = [], 0
    for b in low:
        n = n + 1 if b else (runs.append(n) or 0)
    runs.append(n)
    pauses = [r * HOP for r in runs if r * HOP >= 0.15]

    v = f0[voiced]
    ref = np.nanmedian(v) if len(v) else np.nan
    sv = st(v, ref) if len(v) else np.array([np.nan])
    tv = t[voiced]
    f0_slope = np.polyfit(tv, sv, 1)[0] if len(v) > 5 else np.nan
    last = tv > tv[-1] - 0.3 if len(v) else np.array([], bool)
    f0_final = np.polyfit(tv[last], sv[last], 1)[0] if last.sum() > 5 else np.nan
    Isp = I[sp & inside]

    pp = call(snd, "To PointProcess (periodic, cc)", lo, hi)
    try:
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        jitter = shimmer = np.nan
    cep = call(snd, "To PowerCepstrogram", 60, 0.01, 5000, 50)
    cpps = call(cep, "Get CPPS", True, 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0.05,
                "Exponential decay", "Robust slow")

    def m(x, mask=voiced):
        x = x[mask]
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    sent = clip.split("_")[1]
    return dict(
        clip=clip,
        f0_med=float(ref), f0_mean=float(np.nanmean(v)) if len(v) else np.nan,
        f0_p5=float(np.nanpercentile(v, 5)) if len(v) else np.nan,
        f0_p95=float(np.nanpercentile(v, 95)) if len(v) else np.nan,
        f0_range_st=float(np.nanpercentile(sv, 95) - np.nanpercentile(sv, 5)),
        f0_sd_st=float(np.nanstd(sv)), f0_slope_st=float(f0_slope), f0_final_st=float(f0_final),
        int_mean=float(np.mean(Isp)), int_sd=float(np.std(Isp)),
        int_range=float(np.percentile(Isp, 95) - np.percentile(Isp, 5)),
        voiced_frac=float(voiced[inside].mean()),
        dur=float(dur), rate_syl=SYLLABLES[sent] / dur if dur > 0 else np.nan,
        n_pause=len(pauses), pause_tot=float(sum(pauses)),
        hnr=m(H), alpha=m(alpha), hamm=m(hamm), cent=m(cent),
        slope0_500=m(s0), slope500_1500=m(s1), h1h2=m(h1h2),
        f1=m(F[:, 0]), f2=m(F[:, 1]), f3=m(F[:, 2]),
        jitter=jitter, shimmer=shimmer, cpps=cpps,
    )


def main():
    clips = pd.read_parquet(OUT / "clips.parquet")
    names = clips["clip"].tolist()
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    with ProcessPoolExecutor(nproc) as ex:
        p1 = list(tqdm(ex.map(pass1, names, chunksize=16), total=len(names), desc="pitch pass1"))
    clips["_v"] = p1
    rng = {}
    for a, g in clips.groupby("actor"):
        v = np.concatenate(g._v.tolist())
        q25, q75 = np.percentile(v, [25, 75])
        rng[a] = (max(50.0, 0.75 * q25), min(800.0, 1.75 * q75))
    jobs = [(c, *rng[a], s) for c, a, s in zip(clips["clip"], clips.actor, clips.sex)]
    with ProcessPoolExecutor(nproc) as ex:
        rows = list(tqdm(ex.map(features, jobs, chunksize=8), total=len(jobs), desc="features"))
    feat = pd.DataFrame(rows)
    feat["pitch_lo"] = [j[1] for j in jobs]
    feat["pitch_hi"] = [j[2] for j in jobs]
    feat.to_parquet(OUT / "prosody.parquet", index=False)
    print(feat.describe().T.round(3))


if __name__ == "__main__":
    main()
