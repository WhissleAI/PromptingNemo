# Reproducing the SteerTTS checkpoints

All commands assume

```bash
cd PromptingNemo/recipes/steer_tts
export STEER_TTS_ROOT=/path/to/steer_tts_root      # data/ and runs/ (about 1 TB with raw corpora and features)
export PY=/path/to/env/bin/python                   # torch 2.11 + cu128 tested; see requirements.txt
export MEMCAP=48G                                   # optional: run long jobs in a memory-capped systemd scope
```

The scripts in `scripts/` source `scripts/env.sh`, which sets `PYTHONPATH` (the `steer` package and the PromptingNemo
root for `promptingnemo.models.mamba3`) and checks `STEER_TTS_ROOT`. Hardware used: one RTX 5070 (12 GB), 78 GB RAM.

## 1. Data

```bash
bash scripts/01_download.sh ears expresso libritts libritts360 libritts500 nvtts paraspeechcaps
bash scripts/01_download.sh noise                   # only for --noise_mode all / steer.mixer scenes
python -m steer.cartesia_gen voices && python -m steer.cartesia_gen plan && \
  python -m steer.cartesia_gen plan_coverage && python -m steer.cartesia_gen run --conc 4 --max_chars 4.5e6   # optional, paid
bash scripts/02_prep.sh 12                          # prep/extract, EARS freeform, x-vectors, controls, voices, adapter, readback, ear
```

The emotion probe (`steer.ear train`) needs the CREMA-D study outputs first: run `emotion_study/s01_build_table.py`
and `s04_embed.py` with `CREMA_D_DIR` set.

## 2. Base model (`runs/spk_enc_v3/step312500.pt`)

```bash
for s in s0 s1 s2 s3 s4 s5 s6; do bash scripts/10_train_base.sh $s; done
```

| Stage | Run, step range | Output used downstream |
|---|---|---|
| s0 | `ears25_base` 0-50k | `runs/ears25_base/step50000_full.pt` |
| s1 | `mix_s1` 0-17k | `runs/mix_s1/init_for_m3.pt` |
| s2 | `mix_m3hyb` 17k-116k | `runs/mix_m3hyb/final_116000.pt` |
| s3 | `spk_enc` 116k-178k | `runs/spk_enc/phase1_last.pt` |
| s4 | `spk_enc_full` 178k-290k | `runs/spk_enc_full/final.pt` |
| s5 | `spk_enc_v2` 290k-297.5k | `runs/spk_enc_v2/step297500.pt` |
| s6 | `spk_enc_v3` 297.5k-317.5k | `runs/spk_enc_v3/step312500.pt` (paper base), `step305000.pt` (flow GTA source) |

Each stage was stopped by hand in the original runs; `10_train_base.sh` stops at the same steps automatically and keeps
the original `--steps` (cosine schedule length). Every trainer resumes from `runs/<run>/last.pt` after a crash
(`--resume`), so re-running a stage continues it.

**Not bit-identical.** The original lineage grew the data between stages (EARS 25 speakers -> + Expresso +
LibriTTS-R clean-100 -> + clean-360 + Cartesia -> + other-500 -> + EARS freeform + NonverbalTTS) and rebuilt
`controls.parquet` each time; the 2 % validation split is drawn over the current row order. To mirror the history,
run `steer.build_controls` with the stage's corpus list before each stage (s0: `ears` with 25 speakers; s1-s2:
`ears expresso libritts`; s3: `+ libritts360 cartesia`; s4: `+ libritts500`; s5-s6: `+ earsff nvtts`) and back up
`data/meta` in between. Other known differences: in s3 the per-speaker level normalisation was switched on at 134k
(restart), and s6 ran its first 7.5k steps with reward renders capped at 650 frames and `--max_frames 9000`
(500 / 8000 afterwards, after an OOM).

## 3. Flow decoders

```bash
bash scripts/20_flow.sh gta        # teacher-forced mels of base 305k -> data/gta/v3_305k
bash scripts/20_flow.sh refiner    # runs/flow_v3/step100000.pt
bash scripts/20_flow.sh p1         # runs/flow_v3p   (prompted, in-clip context), 60k
bash scripts/20_flow.sh p2         # runs/flow_v3pc  (+ mu_cmn), 40k
bash scripts/20_flow.sh p3         # runs/flow_v4pa  (+ attention at 1,3,5,7,9,11) -> step40000.pt is the cloning flow
bash scripts/20_flow.sh p5         # runs/flow_v5    (pflow5), 90k
```

## 4. Evaluation

```bash
B=runs/spk_enc_v3/step312500.pt F=runs/flow_v3/step100000.pt P=runs/flow_v4pa/step40000.pt
bash scripts/32_benchmark.sh v3_312k $B
OTL_REPO=/path/to/open-tts-leaderboard bash scripts/30_leaderboard.sh whissle/steer-tts-9m-flow "$B,flow=$F"
OTL_REPO=/path/to/open-tts-leaderboard bash scripts/30_leaderboard.sh whissle/steer-tts-9m-pflow4 "$B,flow=$F,pflow=$P" --voice_clone
SEEDTTS_DIR=/path/to/seedtts SEEDEVAL_PY=/path/to/seedeval/bin/python bash scripts/31_seedtts.sh v3_pflow4 "$B,flow=$F,pflow=$P"
```

## 5. Original launch scripts

The canonical scripts that produced the checkpoints live outside the repo in `$STEER_TTS_ROOT/logs/*.sh` (with their
logs). The `scripts/` copies are the same commands with paths parametrised; the mapping is:

| scripts/ | original |
|---|---|
| `01_download.sh` | `data/download.sh`, `logs/dl_360.sh`, `logs/dl_500.sh`, `logs/hf_wget_dl.sh`, `logs/noise_dl.sh` |
| `02_prep.sh` | `logs/prep_all.sh`, `logs/prep_expr.sh`, `logs/earsff.sh`, `logs/cartesia_gen.sh`, `logs/stage_spk.sh` (x-vectors, controls), `logs/morning_switch.sh` (adapter, readback), `logs/noise_phase.sh` (nvtts, acoustic, voices) |
| `10_train_base.sh` s0-s6 | `run.sh train`, `logs/mix_s1.sh`, `logs/mix_m3hyb.sh`, `logs/stage_spk.sh`, `logs/stage_phase2.sh`, `logs/spk_enc_full_loop.sh`, `logs/v2_run.sh`, `logs/v3_run.sh`, `logs/v3b_run.sh` |
| `20_flow.sh` | `logs/flow_pipeline.sh`, `logs/overnight.sh`, `logs/v3b_run.sh` (pre-stage), `logs/sim_attn.sh`, `logs/v5_run.sh` |
| `30_leaderboard.sh` | `logs/leaderboard.sh` and the `lb()` helper in `logs/sim_attn.sh` / `logs/v5_run.sh` |
| `31_seedtts.sh` | `logs/seedtts.sh` |
| `32_benchmark.sh` | `logs/bench.sh`, `logs/bench_all.sh`, `logs/queue_final.sh` |
