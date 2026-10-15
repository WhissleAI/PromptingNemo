# SteerTTS: a small, steerable, expressive TTS

SteerTTS is a non-autoregressive, explicit-variance TTS (FastSpeech 2 / FastPitch lineage) in which every layer is
conditioned on one global vector built from:

* a **voice**: a frozen WavLM-base-plus-sv x-vector (512-d) of a reference clip, mapped by a linear `spk_proj`
  (a 4,920-row speaker table is kept as a fallback for known voices);
* a **style tag** (51 entries: none + 50 canonicalised emotions / reading styles);
* an **interpretable control vector**: 12 utterance-level acoustic measurements (rate, loudness, F0 median / IQR /
  end slope, pause share, spectral tilt, voicing share, flatness, centroid, CPPS, H1-H2), expressed as z-scores
  relative to the speaker's own neutral speech, plus an explicit "specified" mask so any subset can be requested.

The controls are measured by the same DSP code that later reads them back from synthesized audio, so steering is
evaluated in the units it is requested in. Free-form directions ("say it slowly, almost whispering") are mapped to
(tag, which dims, z values) by a 0.5M adapter on a frozen MiniLM sentence encoder. Inline markup (emphasis, per-word
pitch / energy / duration shifts, pauses, style spans, sound events) and sentence-level streaming work on top without
retraining. An optional flow-matching refiner samples mel detail on top of the base mel; a prompted variant is used for
voice cloning. Mel -> wave is the pretrained `charactr/vocos-mel-24khz`.

Full technical description: [`docs/architecture.md`](docs/architecture.md). Evaluation tools and protocols:
[`docs/evaluation.md`](docs/evaluation.md). Exact commands to rebuild the checkpoints: [`docs/reproduce.md`](docs/reproduce.md).
The original design note (2026-10-07, pre-Mamba, pre-x-vector) is kept as
[`docs/prompt_steerable_small_tts.md`](docs/prompt_steerable_small_tts.md); literature notes are in
[`docs/research_notes/`](docs/research_notes/) and the benchmark survey in [`docs/benchmarks.md`](docs/benchmarks.md).

## Current checkpoints and parameter budget

| Component | Checkpoint (under `$STEER_TTS_ROOT`) | Params |
|---|---|---|
| Base SteerTTS (hybrid attention + BiMamba-3, x-vector conditioned) | `runs/spk_enc_v3/step312500.pt` | 9.80M total; 9.59M inference path (excl. 0.21M aligner); 9.27M excl. speaker table |
| Flow refiner (OT-CFM, ConvNeXt + AdaLN-zero) | `runs/flow_v3/step100000.pt` | 17.6M |
| Prompted flow for cloning (v4, attention at 6 layers) | `runs/flow_v4pa/step40000.pt` | 24.7M |
| Prompted flow v5 (in training) | `runs/flow_v5/` | 38.7M |
| Vocoder (frozen, pretrained) | `charactr/vocos-mel-24khz` | 13.5M |
| Prompt adapter | `data/meta/prompt_adapter.pt` | 0.50M (+ frozen MiniLM-L6, 22.7M) |
| Speaker encoder (frozen, external) | `microsoft/wavlm-base-plus-sv` | ~94M |

System totals at inference: L1 path 9.6M + Vocos 13.5M = 23.1M; with refiner 40.7M; cloning path (base + prompted
flow v4 + Vocos) 47.8M, plus the x-vector encoder. Base block budget: 4 encoder + 4 decoder hybrid blocks at d=192,
each MHA 148k + BiMamba-3 414k + conv FFN 295k + 3 AdaLN 49.5k; three variance predictors of 321k each.

## Layout

| Path | What |
|---|---|
| `steer/config.py` | paths (`STEER_TTS_ROOT`), audio constants (24 kHz, hop 256, 100-band Vocos mel), control dims, tags |
| `steer/text.py` | text normalisation + g2p_en -> ARPAbet ids with word indices; inline `[event]` tokens (118 symbols) |
| `steer/prep.py`, `steer/extract.py`, `steer/features.py` | corpus -> trimmed 24 kHz wavs + manifest; mel / F0 / energy + 12 raw control measurements |
| `steer/ears_freeform.py`, `steer/cartesia_gen.py` | extra data: transcribed EARS freeform monologues; Cartesia Sonic synthetic steering corpus |
| `steer/build_controls.py` | raw controls -> speaker-relative z-scores, speaker reference stats (`controls.parquet`, `control_stats.json`) |
| `steer/xvec.py`, `steer/voices.py` | WavLM-sv x-vectors per utterance; public voice names (dataset ids never leave the machine) |
| `steer/data.py` | dataset with control / tag dropout, pairs for the delta loss, level normalisation, frame-bucketed sampler |
| `steer/model.py` | SteerTTS: AdaLN-conditioned hybrid attention + BiMamba-3 stacks, variance predictors, RAD-TTS aligner, guidance |
| `steer/train.py` | training (supervised + paired-delta + reward fine-tuning with PCGrad), TensorBoard, in-loop harness |
| `steer/readback.py`, `steer/consistency.py`, `steer/rewards.py` | mel -> control readback net; differentiable consistency measurement; reward losses |
| `steer/ear.py`, `steer/ear2.py` | emotion probes (CREMA-D soft labels; EARS in-domain) on WavLM-base+ |
| `steer/acoustic.py`, `steer/mixer.py`, `steer/env_captions.py` | noise / room / event augmentation library; inference-time environment mixing |
| `steer/captions.py`, `steer/prompt.py` | acoustically anchored captions and the prompt -> control adapter |
| `steer/flow.py`, `steer/pflow5.py`, `steer/flow_sim.py`, `steer/vocoder_ft.py` | flow refiner and prompted flows; SIM fine-tune and vocoder GAN fine-tune (both negative results) |
| `steer/synth.py`, `steer/markup.py`, `steer/demo.py` | inference API, inline markup + streaming, local HTTP demo |
| `steer/harness.py`, `steer/evaluate.py`, `steer/tag_eval.py`, `steer/quick_eval.py`, `steer/benchmark.py`, `steer/baselines.py`, `steer/leaderboard_eval.py`, `steer/seedtts.py`, `steer/abtest.py`, `steer/samples.py`, `steer/tb_report.py`, `steer/asr.py` | evaluation, see [`docs/evaluation.md`](docs/evaluation.md) |
| `scripts/` | path-parametrised launch scripts (download, prep, training lineage, flow, leaderboard, Seed-TTS, benchmark) |
| `run.sh` | the original single-entry recipe (stage-1 / stage-2 era); still works for a from-scratch attention model |
| `emotion_study/` | CREMA-D acoustic emotion study that the ear probe and control design build on |
| `data/download.sh`, `data/hf_curl.sh`, `data/hf_files.tsv` | resumable corpus downloads |
| `../../promptingnemo/models/mamba3/` | Mamba-3 layer (SSD chunked scan, trapezoidal discretisation, complex state), reference impl + tests |

## Setup

```bash
pip install -r requirements.txt                 # tested: torch 2.11 + cu128, transformers 5.19, vocos 0.1.0
export STEER_TTS_ROOT=/big/disk/steer_tts       # data/ and runs/ (default: this directory)
export PY=~/anaconda3/envs/emotts/bin/python    # the scripts default to `python`
# Hugging Face token for gated / rate-limited downloads, kept off the command line:
mkdir -p $STEER_TTS_ROOT/data && echo 'header = "Authorization: Bearer hf_xxx"' > $STEER_TTS_ROOT/data/.hf_curlrc
chmod 600 $STEER_TTS_ROOT/data/.hf_curlrc
```

All `python -m steer.X` commands run with `PYTHONPATH=recipes/steer_tts` and `STEER_TTS_ROOT` set (`scripts/env.sh`
does both). Relative checkpoint paths in the flow / leaderboard tools are resolved against `$STEER_TTS_ROOT`.

## Data pipeline (end to end)

1. **Download** (`scripts/01_download.sh`): EARS (107 speakers), Expresso read, LibriTTS-R train-clean-100 / 360 /
   other-500, NonverbalTTS, ParaSpeechCaps (for the prompt adapter and the benchmark), optional noise sources.
2. **Prep + extract** (`steer.prep`, `steer.extract`): trim + resample to 24 kHz mono, manifest rows
   `{id, corpus, speaker, text, raw_style, tag, wav, dur}`; per-utterance `data/feats/<id>.npz` (Vocos log-mel, Praat
   F0 with a two-pass adaptive range, frame energy, phone ids) and the 12 raw control measurements.
3. **Extra corpora**: `steer.ears_freeform` (Whisper-turbo transcription of EARS freeform monologues, ParaSpeechCaps
   test/dev/holdout files excluded); `steer.cartesia_gen` (Cartesia renders of the same text and voice under labelled
   emotion / speed / volume, 33k clips / 43 h / 432 voices).
4. **Speaker vectors** (`steer.xvec`): frozen WavLM-base-plus-sv x-vector of every utterance (`xvec_<corpus>.npz`).
5. **Controls** (`steer.build_controls`): z = (x - speaker neutral mean) / pooled within-speaker SD, clipped to
   [-5, 5]; F0 dims NaN for whispered speech; duration / rate filters. Then `steer.voices assign` (public names).
6. **Auxiliary nets**: prompt adapter (`steer.prompt train`), level-normalised readback net
   (`steer.readback train 8 --level_norm --max 80000`), emotion probe (`steer.ear train`).

The v3 META is 410,566 utterances / 659 h / 4,919 speakers over 8 corpora (LibriTTS-R 516 h, EARS + freeform 77.6 h,
Expresso 10.7 h, Cartesia 43 h, NonverbalTTS 12.5 h). `scripts/02_prep.sh` runs steps 2-6.

## Training lineage

The step counter is cumulative across stages; each stage initialises from the previous one (`--init --init_continue`).

| Stage | Run | Steps | What changed |
|---|---|---|---|
| s0 | `ears25_base` | 0-50k | attention-only model on 25 EARS speakers |
| s1 | `mix_s1` | 0-17k | + Expresso + LibriTTS-R clean-100 (speaker / tag rows mapped by name) |
| s2 | `mix_m3hyb` | 17k-116k | zero-init bidirectional Mamba-3 branch grafted into every block (identity at init) |
| s3 | `spk_enc` | 116k-178k | x-vector conditioning (`--spk_enc`, ridge-initialised `spk_proj`), per-speaker level normalisation, + clean-360 + Cartesia |
| s4 | `spk_enc_full` | 178k-290k | + other-500, rebalanced corpus sampling |
| s5 | `spk_enc_v2` | 290k-297.5k | paired-intervention delta loss, reward fine-tuning (PCGrad + trust-region anchor), noise augmentation, + EARS freeform + NonverbalTTS |
| s6 | `spk_enc_v3` | 297.5k-317.5k | EARS style pairs restored, in-voice nonverbal splices only; **paper checkpoint: step 312.5k** |
| flow | `flow_v3` | 100k | OT-CFM refiner on teacher-forced mels of base step 305k |
| pflow | `flow_v3p` -> `flow_v3pc` -> `flow_v4pa` | 60k / 40k / 40k | prompted (infilling) flow, + base-mel CMN, + attention at 6 layers |
| pflow5 | `flow_v5` | 90k (running) | online mu over all data, cross-clip prompts, voice swap, RoPE + prompt pool + skips + alignment loss |

Losses (base): L1 mel + 0.1 x MSE on log-duration / token pitch / token energy + forward-sum CTC + ramped binarisation
KL; from s5 the paired delta loss (weight 1.0) and every 4th step a reward step (x0.25; control readback, leak,
emotion probe, wav2vec2 CTC, UTMOS, speaker SV, anchor) combined by PCGrad. Details in
[`docs/architecture.md`](docs/architecture.md).

## Command-line entry points

Every module below runs as `python -m steer.<module> ...` from `recipes/steer_tts` (or with `PYTHONPATH` set).

| Module | Example |
|---|---|
| `prep` | `python -m steer.prep ears 12` (corpora: ears, expresso, libritts, libritts360, libritts500, nvtts) |
| `extract` | `python -m steer.extract ears 12` |
| `ears_freeform` | `python -m steer.ears_freeform [--limit N]` |
| `cartesia_gen` | `python -m steer.cartesia_gen voices|plan|plan_coverage|run|check` (API key in `~/.config/cartesia/api_key`) |
| `xvec` | `XVEC_BATCH=8 python -m steer.xvec ears expresso libritts` |
| `build_controls` | `python -m steer.build_controls ears expresso libritts libritts360 libritts500 cartesia earsff nvtts` |
| `voices` | `python -m steer.voices assign`; `python -m steer.voices export --ckpt runs/X/final.pt --out release/steertts` |
| `acoustic` | `python -m steer.acoustic build` (noise library for `--noise_mode all`) |
| `prompt` | `python -m steer.prompt train`; `train_env`; `python -m steer.prompt try "speak sadly"` |
| `readback` | `python -m steer.readback train 8 --level_norm --max 80000` |
| `ear`, `ear2` | `python -m steer.ear train`; `python -m steer.ear2 embed && python -m steer.ear2 train` |
| `train` | see `scripts/10_train_base.sh`; e.g. `python -m steer.train --run spk_enc_v3 --resume --init runs/spk_enc_v2/step297500.pt --init_continue --spk_enc --enc_mixer hybrid --dec_mixer hybrid --compile ...` |
| `flow` | `python -m steer.flow gta --base runs/spk_enc_v3/step305000.pt --out data/gta/v3_305k`; `train --gta data/gta/v3_305k --run runs/flow_v3`; `eval --flow ...`; `ab --base ... --flows ...` |
| `pflow5` | `python -m steer.pflow5 --init runs/flow_v4pa/step40000.pt --run runs/flow_v5` |
| `flow_sim` | `python -m steer.flow_sim --init runs/flow_v3ps/step240000.pt --run runs/flow_sim_a --neg 0.25` |
| `vocoder_ft` | `python -m steer.vocoder_ft --gta data/gta/v3_305k --flow runs/flow_v3/last.pt --run runs/vocoder_v3` |
| `synth` | `python -m steer.synth --ckpt runs/spk_enc_v3/step312500.pt --text "Hello there." --speaker voice_4223 --tag anger --ctrl "loud=2,f0_med=1.5" --out out.wav` |
| `demo` | `python -m steer.demo --ckpt v3=runs/spk_enc_v3/step312500.pt [--port 7860]` |
| `harness` | `python -m steer.harness --ckpt runs/spk_enc_v3/step312500.pt --out runs/spk_enc_v3/harness_312500.json` |
| `evaluate` | `python -m steer.evaluate --ckpt runs/X/last.pt --out runs/X/eval_full.json [--voices 12]` |
| `tag_eval` | `python -m steer.tag_eval --ckpt runs/X/last.pt --out runs/X/tag_eval.json [--guidance 1.5]` |
| `quick_eval` | `python -m steer.quick_eval v3_312k=runs/spk_enc_v3/step312500.pt` |
| `benchmark` | `python -m steer.benchmark prep`; `run --ckpt name=ckpt [--gt]`; `report` |
| `baselines` | `STEER_TTS_ROOT=... <ttsbase-python> steer/baselines.py kokoro|piper|matcha|parler_mini` |
| `leaderboard_eval` | `python -m steer.leaderboard_eval --ckpt runs/spk_enc_v3/step312500.pt,flow=runs/flow_v3/step100000.pt --model_id whissle/steer-tts-9m-flow --set seed [--voice_clone]` |
| `seedtts` | `python -m steer.seedtts --ckpt runs/spk_enc_v3/step312500.pt --meta <seedtts>/en/meta.lst --out runs/seedtts/v3` |
| `abtest`, `samples` | `python -m steer.abtest --ckpt a=... --ckpt b=... --out runs/_ab`; `python -m steer.samples --ckpt ... --out runs/_listen` |
| `tb_report` | `python -m steer.tb_report [run]` (results tables -> TensorBoard TEXT tab) |

Python API:

```python
from steer.synth import Synth
from steer.markup import MarkupSynth
tts = Synth("runs/spk_enc_v3/step312500.pt,flow=runs/flow_v3/step100000.pt,pflow=runs/flow_v4pa/step40000.pt")
wav = tts("I can't believe it worked.", speaker="voice_4223", tag="happiness", ctrl={"rate": 1.0, "f0_iqr": 1.5})
wav = tts("Same words, my voice.", None, ref_wav="me.wav")    # cloning: x-vector + prompted flow
stream = MarkupSynth(tts).stream("Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow>", voice="voice_4223")
```

`Synth(..., guidance=s)` applies classifier-free-style guidance (s > 1 strengthens the requested change; default off).

## TensorBoard

`runs/<run>/tb`: `train/*` losses, LR, grad norm, PCGrad conflict and reward terms; `perf/*` throughput and memory;
`val/*`, `val_audio/*`, `val_mel/*`, `val_align/*` every `--eval_every`; `steer_ratio/*`, `steer_hit/*`,
`steer_leak/*`, `steer_summary/*`, `steer_audio/*` from the in-loop harness every `--harness_every`; results tables
(TEXT tab) written by `steer.tb_report`. Resuming purges events logged after the checkpoint step.

## Known caveats

* **Circular judges.** UTMOS22, WavLM-base-plus-sv and the CREMA-D ear probe are reward judges during s5/s6 *and*
  metrics in `quick_eval` / `benchmark` / `tag_eval`. Independent metrics: Whisper / Qwen3 WER, WavLM-large SIM
  (Seed-TTS / leaderboard), the DSP harness. Report those (or human ratings) for claims.
* **Test-set-informed development.** Text-normalisation rules came from CV3-Eval error analysis, the default voice
  (`voice_4223`) was the lowest-WER of 8 on the full Seed + CV3 sets, and flow design used subsets of those sets.
* **Unstable train / val split.** `data.split` draws 2 % of the current `controls.parquet` rows; META was rebuilt
  across stages, so earlier stages trained on some current val items.
* **Readback references**: `spk_ref_mel_ln.npy` lacks the 2,098 NonverbalTTS speakers (a constant fallback is used in
  the reward), and `readback_ln.pt` predates the final META rebuild.
* **Prompt-CFG in `Flow.sample`**: the "no prompt" branch still sees the prompt through mu on the prefix frames, so
  the observed SIM drop with prompt CFG is not evidence against prompt guidance. Keep `flow_cfg=0`.
* **Version skew**: refiner and prompted flows v3-v4 were trained on teacher-forced mu from base 305k and are used
  with free-running mu from 312.5k.
* **Emotion reward** stays near chance (r_emotion ~3 nats); the harness "tag hit" for anger / sadness mostly reflects
  loudness. The pooled-SD floor makes `pause` / `voiced` z-scores slightly compressed.
* **Hard-coded paths**: `prep.py` (NonverbalTTS) and `acoustic.py` read raw data from `/external1/karan/steer_tts/noise_raw`;
  `leaderboard_eval.py` reads Seed-TTS / CV3 data from `~/Desktop/whissle/workspace/{seedtts,lb_data}`. Symlink or edit.
* **Licensing**: EARS and Expresso are CC BY-NC, ParaSpeechCaps CC BY-NC-SA; Cartesia renders fall under Cartesia's
  terms (competing-model clause); NonverbalTTS includes VoxCeleb audio. Released weights are non-commercial.
* `steer.baselines` requires `STEER_TTS_ROOT` at import time and runs in its own env.
