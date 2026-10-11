"""Dataset, length-bucketed batch sampler and collate for SteerTTS."""
import json
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from scipy.stats import betabinom
from torch.utils.data import Dataset, Sampler

from .config import FEATS, HOP, META, ROOT, CONTROL_DIMS

ZCOLS = [f"z_{k}" for k in CONTROL_DIMS]
# reference-conditioned (spk_enc) runs level-normalise each SPEAKER: log-mels are shifted so that speaker's neutral
# speech sits at LEVEL_DB (spk_loud units). A speaker-verification vector is level-invariant, so without this the
# model cannot know a speaker's recording level (EARS ~19 dB below LibriTTS) and rendered every voice at the loud
# corpora's level, clipping and leaving "loud" no headroom (spk_enc@130k, 2026-10-08). Within-speaker loud/whisper
# differences are untouched; log-mel + log(g) == mel of the waveform scaled by g.
LEVEL_DB = 0.0
PITCH_SCALE = 3.0  # frame pitch targets are semitones re speaker median / 3 (~unit variance)


# per-worker cache: entries are [T, N] float32 (up to a few MB), so keep it small -- maxsize=20000 grew to
# ~6 GB per DataLoader worker and OOM-froze the machine at 12 workers
@lru_cache(maxsize=256)
def beta_binomial_prior(n_text, n_mel, scaling=1.0):
    x = np.arange(0, n_text)[None, :]
    i = np.arange(1, n_mel + 1)[:, None]
    return betabinom.pmf(x, n_text - 1, scaling * i, scaling * (n_mel + 1 - i)).astype(np.float32)  # [T, N]


def load_meta():
    df = pd.read_parquet(META / "controls.parquet")
    stats = json.load(open(META / "control_stats.json"))
    return df, stats


def split(df, val_frac=0.02, seed=0):
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(df)) < val_frac
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


class TTSData(Dataset):
    def __init__(self, df, stats, train=True, p_dim_drop=0.35, p_all_drop=0.15, p_tag_drop=0.3, xvec=None, aug=None,
                 env_dim=0):
        """xvec: optional {id: speaker vector}. Then each item also carries spk_vec, the vector of a DIFFERENT
        utterance of the same speaker, drawn from its neutral speech (the reference the speaker-relative controls
        are defined against; LibriTTS-R counts as neutral). Validation uses a fixed reference per speaker."""
        self.df = df
        self.level_norm = xvec is not None
        self.xvec = None
        if xvec is not None:
            # one f16 matrix + a row index, not a dict of arrays: forked DataLoader workers share the matrix pages
            # instead of each copying ~170k refcounted objects (host-RAM blowups froze this machine before)
            keys = sorted(set(xvec) & set(df.id))
            self.xv_row = {k: i for i, k in enumerate(keys)}
            self.xv_mat = np.stack([xvec[k] for k in keys]).astype(np.float16)
            self.xvec = self.xv_row
            neu = df[((df.tag == "neutral") | (df.corpus == "libritts")) & df.id.isin(xvec)]
            if "speed" in df:  # cartesia neutral-emotion clips at altered speed/volume are not voice references
                neu = neu[neu.speed.fillna(1.0).eq(1.0) & neu.volume.fillna(1.0).eq(1.0)]
            pool = neu.groupby("speaker").id.apply(list).to_dict()
            for spk, ids in df[df.id.isin(xvec)].groupby("speaker").id:  # no neutral speech: any of theirs
                pool.setdefault(spk, list(ids))
            self.ref_pool = pool
        self.spk2i = {s: i for i, s in enumerate(stats["speakers"])}
        self.tag2i = {t: i for i, t in enumerate(stats["tags"])}
        self.train = train
        self.p_dim, self.p_all, self.p_tag = p_dim_drop, p_all_drop, p_tag_drop
        self.Z = df[ZCOLS].to_numpy(np.float32)
        # acoustic environment (steer.acoustic): aug = Augmenter applied to a share of TRAIN items; env_dim > 0 makes
        # every item carry an environment vector (all zeros = clean, which is what un-augmented items teach)
        self.aug, self.env_dim = aug, env_dim

    def _augment(self, r, item, z):
        """Re-render one item inside a sampled environment; None leaves it clean."""
        import soundfile as sf
        from .features import mel_extractor, stft_mag
        from .text import to_phonemes
        rng = np.random.default_rng()
        try:
            y = sf.read(ROOT / "data" / "wav24" / r.corpus / f"{r['id']}.wav", dtype="float32")[0]
        except Exception:  # noqa: BLE001 - missing wav: keep the clean item
            return None
        a = self.aug(y, r["text"], r.speaker, rng)
        if a is None:
            return None
        fe = mel_extractor()
        shift = np.float32((LEVEL_DB - r.spk_loud) * np.log(10.0) / 20.0) if self.level_norm else np.float32(0)
        mel = fe(torch.from_numpy(a["y"])[None])[0].numpy() + shift
        dry = fe(torch.from_numpy(a["dry"])[None])[0].numpy() + shift
        T = min(mel.shape[1], dry.shape[1])
        mel, dry = mel[:, :T], dry[:, :T]
        o = int(round(a["off"] / HOP))
        pitch = np.zeros(T, np.float32); voiced = np.zeros(T, bool)
        n = max(0, min(len(item["pitch"]), T - o))
        pitch[o:o + n], voiced[o:o + n] = item["pitch"][:n], item["voiced"][:n]
        P = stft_mag(a["dry"]) ** 2  # energy of the dry signal: speech + events, not the background
        e_db = 10 * np.log10(P.mean(0) + 1e-10)[:T]
        energy = np.zeros(T, np.float32); energy[: len(e_db)] = (e_db - r.spk_loud) / 10.0
        ids, widx, _ = to_phonemes(a["text"])
        item.update(text=a["text"], ids=np.asarray(ids, np.int64), widx=np.asarray(widx, np.int64), mel=mel,
                    mel_dry=dry, pitch=pitch, voiced=voiced, energy=energy, env=a["env"], aug=a["desc"])
        if a.get("ev_span"):
            # the aligner, trained on speech, gave inserted event tokens ~1 frame and their audio to neighbouring
            # <sp> / punctuation (v2: [laugh]=1 frame, "?"=47). We know the event's exact span: force it (collate)
            from .text import SYM2ID
            tok_id = SYM2ID.get(f"[{a['ev_tok']}]")
            if tok_id is not None and tok_id in ids:
                k = ids.index(tok_id)
                f0, f1 = int(a["ev_span"][0] / HOP), int(round(a["ev_span"][1] / HOP))
                f0 = max(f0, k)  # every token before it keeps >= 1 frame
                f1 = min(f1, T - (len(ids) - k))  # ... and every token after it
                if f1 - f0 >= 2:
                    item["force"] = (k, f0, f1)
        return item

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            z = dict(np.load(FEATS / f"{r['id']}.npz"))  # read every member now, inside the guard
            mel = z["mel"].astype(np.float32)
        except Exception as e:  # noqa: BLE001
            # this machine's non-ECC RAM has flipped a bit in a cached feature file (2026-10-08): log it and
            # train on another utterance instead of killing the run (the bad copy is only in the page cache)
            print(f"[data] unreadable {r['id']}: {type(e).__name__}: {e}; substituting", flush=True)
            return self[np.random.randint(len(self.df))]
        f0 = z["f0"][: mel.shape[1]]
        voiced = f0 > 0
        st = 12 * np.log2(np.maximum(f0, 1.0) / 100.0) - r.spk_f0_st
        # octave errors / whisper noise: drop implausible frames, and treat near-unvoiced utterances as unvoiced
        voiced &= (st > -12) & (st < 18)
        if voiced.mean() < 0.12:
            voiced[:] = False
        pitch = np.where(voiced, st / PITCH_SCALE, 0.0).astype(np.float32)
        energy = ((z["energy"][: mel.shape[1]] - r.spk_loud) / 10.0).astype(np.float32)
        if self.level_norm:
            mel = mel + np.float32((LEVEL_DB - r.spk_loud) * np.log(10.0) / 20.0)
        ctrl = self.Z[i].copy()
        cmask = (~np.isnan(ctrl)).astype(np.float32)
        tag = self.tag2i.get(r.tag, 0)
        if self.train:
            cmask *= (np.random.rand(len(ctrl)) > self.p_dim)
            if np.random.rand() < self.p_all:
                cmask[:] = 0
            if np.random.rand() < self.p_tag:
                tag = 0
        ids = z["ids"].astype(np.int64)
        item = dict(id=r["id"], text=r["text"], ids=ids, widx=z["widx"].astype(np.int64), mel=mel, pitch=pitch, voiced=voiced,
                    energy=energy, ctrl=np.nan_to_num(ctrl), cmask=cmask, spk=self.spk2i[r.speaker], tag=tag)
        if self.xvec is not None:
            pool = self.ref_pool[r.speaker]
            if self.train:
                others = [k for k in pool if k != r["id"]] or pool
                ref = others[np.random.randint(len(others))]
            else:
                ref = pool[0]
            item["spk_vec"] = self.xv_mat[self.xv_row[ref]].astype(np.float32)
        if self.env_dim:
            item["env"] = np.zeros(self.env_dim, np.float32)
            if self.train and self.aug is not None:
                self._augment(r, item, z)
        return item


def collate(items):
    B = len(items)
    N = max(len(x["ids"]) for x in items)
    T = max(x["mel"].shape[1] for x in items)
    ids = torch.zeros(B, N, dtype=torch.long)
    widx = torch.full((B, N), -1, dtype=torch.long)
    mel = torch.full((B, 100, T), -11.5)  # log(1e-5) ~ silence
    pitch = torch.zeros(B, T); energy = torch.zeros(B, T); voiced = torch.zeros(B, T, dtype=torch.bool)
    prior = torch.zeros(B, T, N)
    tl, ml = [], []
    for b, x in enumerate(items):
        n, t = len(x["ids"]), x["mel"].shape[1]
        ids[b, :n] = torch.from_numpy(x["ids"]); widx[b, :n] = torch.from_numpy(x["widx"])
        mel[b, :, :t] = torch.from_numpy(x["mel"])
        pitch[b, :t] = torch.from_numpy(x["pitch"]); energy[b, :t] = torch.from_numpy(x["energy"])
        voiced[b, :t] = torch.from_numpy(x["voiced"])
        prior[b, :t, :n] = torch.from_numpy(beta_binomial_prior(n, t))
        if "force" in x:  # known event span: those frames belong to the event token only, and it to them only
            k, f0, f1 = x["force"]
            prior[b, f0:f1, :] = 0.0
            prior[b, f0:f1, k] = 1.0
            prior[b, :f0, k] = 0.0
            prior[b, f1:, k] = 0.0
        tl.append(n); ml.append(t)
    return dict(ids=ids, widx=widx, text_lens=torch.tensor(tl), mel=mel, mel_lens=torch.tensor(ml), pitch=pitch,
                energy=energy, voiced=voiced, prior=prior,
                ctrl=torch.from_numpy(np.stack([x["ctrl"] for x in items])),
                cmask=torch.from_numpy(np.stack([x["cmask"] for x in items])),
                spk=torch.tensor([x["spk"] for x in items]), tag=torch.tensor([x["tag"] for x in items]),
                id=[x["id"] for x in items], text=[x.get("text", "") for x in items],
                **({"env": torch.from_numpy(np.stack([x["env"] for x in items]))} if "env" in items[0] else {}),
                **({"mel_align": _pad_mels([x.get("mel_dry", x["mel"]) for x in items], T)}
                   if any("mel_dry" in x for x in items) else {}),
                **({"spk_vec": torch.from_numpy(np.stack([x["spk_vec"] for x in items]))} if "spk_vec" in items[0] else {}))


def _pad_mels(mels, T):
    out = torch.full((len(mels), 100, T), -11.5)
    for b, m in enumerate(mels):
        out[b, :, :m.shape[1]] = torch.from_numpy(m)
    return out


def pair_pool(df, corpora=("cartesia", "ears", "expresso"), max_per_group=6, seed=0, max_frames=700,
              max_frames_ears=1100):
    """Index pairs (i, j) of df rows with the SAME speaker and text but different renders (EARS reading styles,
    Cartesia speed / volume / emotion variants, Expresso styles): interventional data, where the only difference
    between the two clips is how they were said."""
    rng = np.random.default_rng(seed)
    # pair batches are not frame-bucketed: cap lengths (EARS style readings are longer, and they are the only
    # same-sentence high / low pitch, loud, whisper, fast / slow pairs)
    cap = np.where(df.corpus == "ears", max_frames_ears, max_frames)
    sub = df[df.corpus.isin(corpora) & (df.n_frames <= cap)]
    key = sub.speaker + "\t" + sub.text.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
    pairs = []
    for _, idx in sub.groupby(key).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        cand = [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:]]
        for c in rng.permutation(len(cand))[:max_per_group]:
            pairs.append(cand[c])
    return pairs


class PairSampler(Sampler):
    """Endless batches of k pairs, flattened [a1, b1, a2, b2, ...]."""

    def __init__(self, pairs, k=8, seed=0, weights=None):
        self.pairs, self.k, self.rng = pairs, k, np.random.default_rng(seed)
        self.p = None if weights is None else np.asarray(weights, float) / np.sum(weights)

    def __iter__(self):
        while True:
            sel = self.rng.choice(len(self.pairs), self.k, replace=False, p=self.p)
            yield [int(x) for p in sel for x in self.pairs[p]]

    def __len__(self):
        return 10 ** 9


def pair_delta_loss(o, batch):
    """Paired-intervention loss: for two renders of the same text by the same voice, the CHANGE the model predicts
    between their conditions must equal the real change. Per phoneme (same ids): log-duration, pitch, energy; per
    utterance: the mean log-mel spectrum (loudness / tilt / brightness / breathiness). Supervising each clip alone
    weights this difference term by only 1/2 (|pa-ta|^2 + |pb-tb|^2 = 1/2|sum|^2 + 1/2|delta|^2), and label noise
    in the controls attenuates it further; this term targets it directly."""
    a, b = slice(0, None, 2), slice(1, None, 2)
    ids, tl = batch["ids"], batch["text_lens"]
    same = (ids[a] == ids[b]).all(1) & (tl[a] == tl[b])
    if not same.any():
        return None
    tm = o["tmask"][a].float() * same[:, None].float()
    out = {}
    for k_p, k_t in (("dur_p", "logdur_t"), ("pitch_p", "pitch_t"), ("energy_p", "energy_t")):
        dp = o[k_p][a].float() - o[k_p][b].float()
        dt = o[k_t][a].float() - o[k_t][b].float()
        out[f"pair_{k_t.split('_')[0]}"] = (((dp - dt) ** 2) * tm).sum() / tm.sum().clamp_min(1)
    mm = o["mmask"][:, None].float()
    mean = lambda x: (x.float() * mm).sum(2) / mm.sum(2).clamp_min(1)  # [B, 100]
    mp, mt = mean(o["mel_p"]), mean(batch["mel"])
    dm = (mp[a] - mp[b]) - (mt[a] - mt[b])
    out["pair_mel"] = ((dm ** 2).mean(1) * same.float()).sum() / same.float().sum().clamp_min(1)
    out["pair_total"] = out["pair_logdur"] + out["pair_pitch"] + out["pair_energy"] + out["pair_mel"]
    return out


class FrameBucketSampler(Sampler):
    """Batches of similar length capped by total mel frames (memory ~ frames)."""

    def __init__(self, n_frames, max_frames=12000, shuffle=True, seed=0, weights=None, max_items=64):
        self.n = np.asarray(n_frames)
        self.max_frames, self.shuffle, self.seed, self.epoch = max_frames, shuffle, seed, 0
        self.max_items = max_items  # caps padding cost on the text side for batches of very short clips
        self.weights = weights  # optional per-item sampling weights (oversampling)

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.weights is not None:
            p = np.array(self.weights, dtype=float); p /= p.sum()
            idx = rng.choice(len(self.n), size=len(self.n), replace=True, p=p)
        else:
            idx = rng.permutation(len(self.n)) if self.shuffle else np.arange(len(self.n))
        # sort within large chunks for bucketing, keep randomness across chunks
        chunks = [idx[i:i + 2000] for i in range(0, len(idx), 2000)]
        batches = []
        for ch in chunks:
            ch = ch[np.argsort(self.n[ch])]
            cur, mx = [], 0
            for j in ch:
                mx2 = max(mx, self.n[j])
                if cur and (mx2 * (len(cur) + 1) > self.max_frames or len(cur) >= self.max_items):
                    batches.append(cur); cur, mx2 = [], self.n[j]
                cur.append(int(j)); mx = mx2
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        b = self._batches()
        self.epoch += 1
        return iter(b)

    def __len__(self):
        return len(self._batches())
