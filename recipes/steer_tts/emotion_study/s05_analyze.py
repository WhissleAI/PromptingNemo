"""Analysis of CREMA-D emotion acoustics -> out/report_data.json (+ printed summary).

Sections (each runs only if its inputs exist):
  labels     acted vs voice-only perceived labels, agreement, actor/sentence reliability
  deltas     Bou-Ghazale-style parallel deltas: each emotional clip minus the same
             actor's NEUTRAL take of the same sentence
  drivers    which deltas make listeners perceive the acted emotion (per-emotion logistic)
  classify   actor-disjoint UAR of interpretable features (raw vs speaker-normalised)
  contours   mean F0 / intensity contours per sentence x emotion on a word-normalised time grid
  words      per-word duration / F0 / intensity deltas vs neutral (where in the sentence emotion lives)
  ssl        AVD (arousal/valence/dominance) by emotion; WavLM layer-wise probing; 2-D map
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, FRAMES, EMOS, SENTENCES

warnings.filterwarnings("ignore")
EMO5 = ["ANG", "SAD", "HAP", "FEA", "DIS"]
R = {}


def r(x, k=3):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return json.loads(x.round(k).to_json())
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


clips = pd.read_parquet(OUT / "clips.parquet")

# ---------------------------------------------------------------- labels
conf = pd.crosstab(clips.emo, clips.perc_emo, normalize="index").reindex(index=EMOS, columns=EMOS + ["MIX"], fill_value=0)
emo_clips = clips[clips.emo != "NEU"]
actor = emo_clips.groupby(["actor", "sex", "age"]).p_acted.mean().reset_index().sort_values("p_acted", ascending=False)
R["labels"] = dict(
    n_clips=len(clips), n_actors=int(clips.actor.nunique()),
    raters_mean=r(clips.n_raters.mean(), 1),
    perceived_eq_acted=r((clips.perc_emo == clips.emo).mean()),
    confusion=r(conf),
    by_emo=r(clips.groupby("emo")[["p_acted", "agreement", "entropy", "perc_int"]].mean().reindex(EMOS)),
    actor=[dict(actor=int(a.actor), sex=a.sex, age=int(a.age), p=r(a.p_acted)) for a in actor.itertuples()],
    sentence=r(emo_clips.groupby("sent").p_acted.mean().sort_values()),
    level=r(clips[clips.level != "XX"].groupby(["emo", "level"])[["p_acted", "perc_int"]].mean().reset_index()
            .pivot(index="emo", columns="level", values="p_acted")[["LO", "MD", "HI"]]),
    level_int=r(clips[clips.level != "XX"].groupby(["emo", "level"]).perc_int.mean().unstack()[["LO", "MD", "HI"]]),
    sex=r(emo_clips.groupby(["sex", "emo"]).p_acted.mean().unstack()),
)
print("== labels: perceived==acted", R["labels"]["perceived_eq_acted"])
print(actor.head(5).to_string(), "\n...\n", actor.tail(5).to_string())

# ---------------------------------------------------------------- deltas
pros_path = OUT / "prosody.parquet"
if pros_path.exists():
    pros = pd.read_parquet(pros_path)
    d = clips.merge(pros, on="clip")
    FEAT = ["f0_med", "f0_range_st", "f0_sd_st", "f0_slope_st", "f0_final_st", "int_mean", "int_sd",
            "int_range", "voiced_frac", "dur", "rate_syl", "pause_tot", "hnr", "alpha", "hamm", "cent",
            "slope0_500", "slope500_1500", "h1h2", "f1", "f2", "jitter", "shimmer", "cpps"]
    ref = d[(d.emo == "NEU")].groupby(["actor", "sent"])[FEAT].mean()
    e = d[d.emo != "NEU"].join(ref, on=["actor", "sent"], rsuffix="_ref")
    D = pd.DataFrame({"clip": e["clip"], "emo": e.emo, "actor": e.actor, "sent": e.sent, "level": e.level,
                      "p_acted": e.p_acted, "perc_int": e.perc_int})
    # semitone / dB / log-ratio deltas where that is the perceptual scale
    D["dF0_st"] = 12 * np.log2(e.f0_med / e.f0_med_ref)
    D["dF0range_st"] = e.f0_range_st - e.f0_range_st_ref
    D["dF0slope"] = e.f0_slope_st - e.f0_slope_st_ref
    D["dF0final"] = e.f0_final_st - e.f0_final_st_ref
    D["dInt_dB"] = e.int_mean - e.int_mean_ref
    D["dIntRange_dB"] = e.int_range - e.int_range_ref
    D["dDur_pct"] = 100 * (e.dur / e.dur_ref - 1)
    D["dRate_pct"] = 100 * (e.rate_syl / e.rate_syl_ref - 1)
    D["dPause_s"] = e.pause_tot - e.pause_tot_ref
    D["dVoiced"] = e.voiced_frac - e.voiced_frac_ref
    D["dAlpha_dB"] = e.alpha - e.alpha_ref
    D["dHamm_dB"] = e.hamm - e.hamm_ref
    D["dSlope0_500"] = 1000 * (e.slope0_500 - e.slope0_500_ref)
    D["dSlope500_1500"] = 1000 * (e.slope500_1500 - e.slope500_1500_ref)
    D["dCent_pct"] = 100 * (e.cent / e.cent_ref - 1)
    D["dH1H2_dB"] = e.h1h2 - e.h1h2_ref
    D["dHNR_dB"] = e.hnr - e.hnr_ref
    D["dCPPS_dB"] = e.cpps - e.cpps_ref
    D["dF1_pct"] = 100 * (e.f1 / e.f1_ref - 1)
    D["dF2_pct"] = 100 * (e.f2 / e.f2_ref - 1)
    D["dJitter_pct"] = 100 * (e.jitter / e.jitter_ref - 1)
    D["dShimmer_pct"] = 100 * (e.shimmer / e.shimmer_ref - 1)
    DCOLS = [c for c in D.columns if c.startswith("d") and c != "dur"]
    D = D.replace([np.inf, -np.inf], np.nan)
    D.to_parquet(OUT / "deltas.parquet", index=False)

    D["convincing"] = D.p_acted >= 0.5
    stats = []
    for c in DCOLS:
        for emo in EMO5:
            for subset, g in [("all", D[D.emo == emo]), ("convincing", D[(D.emo == emo) & D.convincing]),
                              ("unconvincing", D[(D.emo == emo) & ~D.convincing])]:
                x = g[c].dropna()
                # clip robustly: per-feature outliers (octave errors etc.) shouldn't drive the mean
                x = x.clip(*x.quantile([0.02, 0.98])) if len(x) > 20 else x
                stats.append(dict(feat=c, emo=emo, subset=subset, n=len(x), mean=r(x.mean()),
                                  ci=r(1.96 * x.std() / np.sqrt(max(len(x), 1))), med=r(x.median()),
                                  d=r(x.mean() / x.std()) if x.std() > 0 else None))
    R["deltas"] = dict(features=DCOLS, stats=stats)
    piv = pd.DataFrame([s for s in stats if s["subset"] == "all"]).pivot(index="feat", columns="emo", values="d")
    print("\n== parallel deltas, effect size d (mean/sd) vs same-actor neutral\n", piv.reindex(DCOLS)[EMO5].round(2))

    # ------------------------------------------------------------ drivers
    KEY = ["dF0_st", "dF0range_st", "dInt_dB", "dRate_pct", "dAlpha_dB", "dH1H2_dB", "dCPPS_dB",
           "dHNR_dB", "dF1_pct", "dVoiced", "dPause_s", "dF0final"]
    drivers = {}
    for emo in EMO5:
        g = D[D.emo == emo].dropna(subset=KEY)
        X, y = g[KEY].to_numpy(float), (g.p_acted >= 0.5).astype(int).to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        p = cross_val_predict(model, X, y, groups=g.actor.to_numpy(), cv=GroupKFold(5), method="predict_proba")[:, 1]
        model.fit(X, y)
        drivers[emo] = dict(auc=r(roc_auc_score(y, p)), base_rate=r(y.mean()),
                            coef={k: r(v) for k, v in zip(KEY, model[-1].coef_[0])},
                            int_corr={k: r(g[k].corr(g.perc_int, method="spearman")) for k in KEY})
        top = sorted(drivers[emo]["coef"].items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"{emo}: AUC {drivers[emo]['auc']} (convincing rate {drivers[emo]['base_rate']}) top: {top}")
    R["drivers"] = dict(features=KEY, by_emo=drivers)

    # ------------------------------------------------------------ classify
    RAW = [f for f in FEAT if f not in ("f2",)]
    Z = d[RAW].copy()
    Z = (Z - d.groupby("actor")[RAW].transform("mean")) / d.groupby("actor")[RAW].transform("std")
    Zn = d[RAW] - d[d.emo == "NEU"].groupby("actor")[RAW].mean().reindex(d.actor).values
    Zn = Zn / d.groupby("actor")[RAW].transform("std")

    def uar(X, y, groups):
        X = np.nan_to_num(np.asarray(X, float))
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        p = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(5))
        return balanced_accuracy_score(y, p), p

    agreed = (d.perc_emo == d.emo).to_numpy(bool)
    res = {}
    for name, X in [("raw features", d[RAW]), ("actor z-norm", Z), ("actor neutral-referenced", Zn)]:
        u_all, _ = uar(X, d.emo.to_numpy(str), d.actor.to_numpy())
        u_agr, _ = uar(np.asarray(X)[agreed], d.emo.to_numpy(str)[agreed], d.actor.to_numpy()[agreed])
        res[name] = dict(acted_all=r(u_all), acted_on_agreed=r(u_agr))
        print(f"classify [{name}] UAR acted(all) {u_all:.3f} | on crowd-agreed clips {u_agr:.3f}")
    hum = d[d.perc_emo.isin(EMOS)]
    res["human crowd (voice only)"] = dict(acted_all=r(balanced_accuracy_score(hum.emo, hum.perc_emo)))
    R["classify"] = res

    # ------------------------------------------------------------ contours
    wpath = OUT / "align_words.parquet"
    words = pd.read_parquet(wpath) if wpath.exists() else None
    if words is not None:
        kc = pd.read_parquet(OUT / "align_clip.parquet")
        good = set(kc[(kc.cer <= 0.25)]["clip"])
    neu_f0 = d[d.emo == "NEU"].groupby("actor").f0_med.median()
    neu_int = d[d.emo == "NEU"].groupby("actor").int_mean.mean()
    PTS = 12  # samples per word
    contours, wordstats = {}, []
    for sent in SENTENCES:
        sub = d[d.sent == sent]
        acc = {}
        for row in sub.itertuples():
            fr = np.load(FRAMES / f"{row.clip}.npz")
            t, f0, I = fr["t"], fr["f0"], fr["int"]
            f0s = 12 * np.log2(f0 / neu_f0[row.actor])
            Ir = I - neu_int[row.actor]
            if words is not None and row.clip in good:
                wd = words[words["clip"] == row.clip].sort_values("w_idx")
                grid = np.concatenate([np.linspace(w.start, w.end, PTS, endpoint=False) + 0.5 * (w.end - w.start) / PTS
                                       for w in wd.itertuples()])
                if len(wd) != len(SENTENCES[sent].split()):
                    continue
                for w in wd.itertuples():
                    m = (t >= w.start) & (t < w.end)
                    wordstats.append(dict(clip=row.clip, emo=row.emo, actor=row.actor, sent=sent, w_idx=w.w_idx,
                                          word=w.word, wdur=w.end - w.start,
                                          wf0=np.nanmean(f0s[m]) if np.isfinite(f0s[m]).any() else np.nan,
                                          wint=np.nanmax(Ir[m]) if m.any() else np.nan))
            else:
                v = ~np.isnan(f0)
                if v.sum() < 5:
                    continue
                grid = np.linspace(t[v][0], t[v][-1], PTS * len(SENTENCES[sent].split()))
            # nearest-frame sampling keeps unvoiced gaps as NaN instead of interpolating across them
            idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
            acc.setdefault(row.emo, []).append(np.stack([f0s[idx], Ir[idx]]))
        contours[sent] = {}
        for emo, arr in acc.items():
            a = np.stack(arr)  # [n, 2, G]
            with np.errstate(all="ignore"):
                mu = np.nanmean(a, 0)
                se = np.nanstd(a, 0) / np.sqrt(np.sum(np.isfinite(a), 0))
            contours[sent][emo] = dict(n=len(arr), f0=[r(x, 2) for x in mu[0]], f0_se=[r(x, 2) for x in se[0]],
                                       int=[r(x, 2) for x in mu[1]], int_se=[r(x, 2) for x in se[1]])
    R["contours"] = dict(pts_per_word=PTS, word_aligned=words is not None, by_sent=contours,
                         words={s: SENTENCES[s].split() for s in SENTENCES})

    if wordstats:
        W = pd.DataFrame(wordstats)
        ref = W[W.emo == "NEU"].groupby(["actor", "sent", "w_idx"])[["wdur", "wf0", "wint"]].mean()
        W = W[W.emo != "NEU"].join(ref, on=["actor", "sent", "w_idx"], rsuffix="_ref")
        W["dDur_pct"] = 100 * (W.wdur / W.wdur_ref - 1)
        W["dF0_st"] = W.wf0 - W.wf0_ref
        W["dInt_dB"] = W.wint - W.wint_ref
        W = W.replace([np.inf, -np.inf], np.nan)
        W.to_parquet(OUT / "word_deltas.parquet", index=False)
        agg = W.groupby(["sent", "w_idx", "word", "emo"])[["dDur_pct", "dF0_st", "dInt_dB"]].median().reset_index()
        R["words"] = json.loads(agg.round(2).to_json(orient="records"))
        # position effect: first / middle / last word
        W["pos"] = np.where(W.w_idx == 0, "first",
                            np.where(W.w_idx == W.groupby("clip").w_idx.transform("max"), "last", "middle"))
        pos = W.groupby(["emo", "pos"])[["dDur_pct", "dF0_st", "dInt_dB"]].median().round(2)
        print("\n== word position effect (median delta vs neutral)\n", pos)
        R["word_pos"] = json.loads(pos.reset_index().to_json(orient="records"))

# ---------------------------------------------------------------- ssl
if (OUT / "avd.parquet").exists():
    avd = pd.read_parquet(OUT / "avd.parquet").merge(clips[["clip", "emo", "perc_emo", "p_acted", "perc_int", "actor"]])
    R["avd"] = dict(by_emo=r(avd.groupby("emo")[["arousal", "valence", "dominance"]].mean().reindex(EMOS)),
                    by_emo_agreed=r(avd[avd.perc_emo == avd.emo].groupby("emo")[["arousal", "valence", "dominance"]].mean().reindex(EMOS)),
                    corr_perc_int=r(avd[avd.emo != "NEU"][["arousal", "valence", "dominance"]].corrwith(avd.perc_int, method="spearman")))
    rng = np.random.default_rng(0)
    pts = avd.sample(1500, random_state=0)
    R["avd"]["points"] = [dict(a=r(p.arousal), v=r(p.valence), e=p.emo, ok=bool(p.perc_emo == p.emo)) for p in pts.itertuples()]
    print("\n== AVD by emotion\n", avd.groupby("emo")[["arousal", "valence", "dominance"]].mean().round(3))

if (OUT / "wavlm_layers.npy").exists():
    E = np.load(OUT / "wavlm_layers.npy").astype(np.float32)
    y_emo, y_sent, grp = clips.emo.to_numpy(str), clips.sent.to_numpy(str), clips.actor.to_numpy()
    agreed = (clips.perc_emo == clips.emo).to_numpy(bool)
    rows = []
    for L in range(E.shape[1]):
        X = E[:, L]
        Xn = X - pd.DataFrame(X).groupby(grp).transform("mean").values  # speaker mean normalisation
        out = {"layer": L}
        for name, XX, yy, gg in [("emotion", Xn, y_emo, grp), ("emotion_agreed", Xn[agreed], y_emo[agreed], grp[agreed]),
                                 ("sentence", X, y_sent, grp)]:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            p = cross_val_predict(m, XX, yy, groups=gg, cv=GroupKFold(5), n_jobs=5)
            out[name] = r(balanced_accuracy_score(yy, p))
        rows.append(out)
        print("wavlm layer", out)
    R["wavlm"] = rows
    best = max(rows, key=lambda x: x["emotion"])["layer"]
    import umap
    X = E[:, best] - pd.DataFrame(E[:, best]).groupby(grp).transform("mean").values
    xy = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    keep = np.random.default_rng(0).choice(len(xy), 2500, replace=False)
    R["umap"] = dict(layer=best, points=[dict(x=r(xy[i, 0], 2), y=r(xy[i, 1], 2), e=clips.emo.iat[i],
                                              s=clips.sent.iat[i], ok=bool(agreed[i])) for i in keep])

json.dump(R, open(OUT / "report_data.json", "w"))
print("\nwrote", OUT / "report_data.json")
