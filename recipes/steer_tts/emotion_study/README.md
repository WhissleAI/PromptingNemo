# CREMA-D emotion study (motivation for the SteerTTS control space)

Parallel-delta analysis of CREMA-D (Cao et al. 2014; 91 actors, 12 sentences, 7,442 clips): every
emotional take is compared with **the same actor's neutral take of the same sentence**, so speaker and
text cancel and what is left is the emotion. It produced the claims in
`docs/prompt_steerable_small_tts.md` ("Whissle's CREMA-D findings") and the paper's Appendix
"CREMA-D Acoustic Analysis" (`app:cremad`).

## Pipeline

| step | what | device | output (in `$EMOTION_STUDY_OUT`) |
|---|---|---|---|
| `s01_build_table.py` | clip table + voice-only crowd soft labels (`finishedResponses.csv`, `queryType==1`) | CPU, seconds | `clips.parquet` |
| `s02_prosody.py [nproc]` | Praat frame/clip features (F0 two-pass, intensity, HNR, formants, alpha ratio, Hammarberg, slopes, H1-H2, CPPS, jitter, shimmer) | CPU, ~1 h at 12 procs | `prosody.parquet`, `frames/*.npz` |
| `s03_align.py` | wav2vec2 CTC word/char alignment to the script | **GPU** | `align_{words,chars,clip}.parquet` |
| `s04_embed.py` | audeering AVD + WavLM layer embeddings (only for the SSL section) | **GPU** | `avd.parquet`, `wavlm_layers.npy` |
| `s05_analyze.py` | labels, parallel deltas, perception drivers, classifier UAR, contours, word position, (SSL if present) | CPU, ~30 s without SSL | `deltas.parquet`, `word_deltas.parquet`, `report_data.json` |
| `s06/s07` | interactive HTML report (`viz/`) | CPU | `viz/` |
| `s08_steer_controls.py [nproc]` | the **12 SteerTTS controls** for every clip via `steer.prep.trim_resample` + `steer.features.measure` (same path as `steer.extract`) | CPU, ~1 min at 12 procs | `steer_controls.parquet` |
| `s09_paper_stats.py [B]` | every paper number with 95% bootstrap CIs over actors (B resamples) | CPU, ~15 min at B=2000 | `paper_stats.json` |
| `s10_paper_assets.py <stats.json> <paper_dir>` | copies the stats to `results/cremad.json`, draws `figures/F_cremad.pdf` (paper env, uses the paper's `scripts/style.py`) | CPU | `F_cremad.png` preview |

## Exact commands (paper numbers)

```bash
cd PromptingNemo/recipes/steer_tts/emotion_study
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=$PWD/..
export CREMA_D_DIR=/home/karan/Desktop/whissle/workspace/CREMA-D           # AudioWAV/, finishedResponses.csv, ...
export EMOTION_STUDY_OUT=/home/karan/Desktop/whissle/workspace/steer_tts/cremad_rerun
PY=~/anaconda3/envs/emotts/bin/python
$PY s01_build_table.py
$PY s02_prosody.py 12
# GPU alignment is not re-run: copy the cached alignment of the original study
cp /home/karan/Desktop/whissle/workspace/emotion_study/out/align_*.parquet $EMOTION_STUDY_OUT/
$PY s05_analyze.py          # SSL sections are skipped when avd.parquet / wavlm_layers.npy are absent
$PY s08_steer_controls.py 12
$PY s09_paper_stats.py 2000
~/anaconda3/envs/paper/bin/python s10_paper_assets.py $EMOTION_STUDY_OUT/paper_stats.json \
    /home/karan/Desktop/whissle/marketing_post_agent/paper/steertts
```

The original study's outputs live in `/home/karan/Desktop/whissle/workspace/emotion_study/out`
(same scripts, same CREMA-D copy). Pointing `EMOTION_STUDY_OUT` at a directory that symlinks its
`clips.parquet`, `prosody.parquet`, `align_*.parquet` and `frames/` and re-running `s05`/`s09`
reproduces its numbers exactly; the fresh `s01`/`s02` re-run above is the from-audio check.

## Definitions

* **Parallel delta**: emotional clip minus the mean of the same actor's NEU clip(s) of the same
  sentence. F0 deltas in semitones, levels in dB, durations/rates in percent.
* **Study effect size** (`study_delta_dz`, s05): d_z = mean/SD of the deltas, winsorised at the 2nd/98th
  percentile per emotion x feature.
* **SteerTTS effect size** (`steer_delta_d`, s09): mean delta / sigma_d, where sigma_d is the pooled
  within-actor SD of neutral clips -- i.e. the mean speaker-relative z of Eq. (z) in the paper.
  `steer_delta_dz` gives d_z for the same deltas.
* **Perception drivers**: per emotion, actor-grouped 5-fold logistic regression from 12 standardised
  deltas to "crowd chose the acted emotion on >= 50% of ratings"; AUC out of fold.
* **Word position**: per-word median of peak intensity (dB) and mean F0 (st) vs the same actor's neutral
  word, on clips whose alignment CER <= 0.25 and word count matches the script.
* **CIs**: 95% percentile bootstrap over actors (resampled actors are relabelled so duplicates are
  distinct speakers).

## Privacy

`clips.parquet` / `report_data.json` contain CREMA-D actor ids (local only). `paper_stats.json` and
the figure contain no actor or clip ids; actor spread is reported only as ratios/percentiles.
