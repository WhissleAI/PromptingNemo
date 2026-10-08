# SteerTTS: a small, prompt-steerable expressive TTS

A ~5.5M-parameter explicit-variance TTS (inference path; plus a 0.2M training-only aligner) whose voice is
rendered from an **interpretable control vector**: 12 speaker-relative acoustic dimensions (rate, loudness,
median F0, F0 range, end contour, pauses, spectral tilt, voicing, flatness, centroid, CPPS, H1-H2) plus a
style/emotion tag. A small adapter on a frozen sentence encoder maps free-form directions
("say it slowly, almost whispering") to that vector.

Design rationale, literature review and evaluation plan: [`docs/prompt_steerable_small_tts.md`](docs/prompt_steerable_small_tts.md)
(research notes in [`docs/research_notes/`](docs/research_notes/)).

## Layout

| Path | What |
|---|---|
| `steer/config.py` | paths (`STEER_TTS_ROOT`), audio constants (24 kHz, 100-band Vocos mel), control dims, tag vocabulary |
| `steer/prep.py` | corpus -> trimmed 24 kHz wavs + manifest (`ears`, `expresso`, `libritts`) |
| `steer/features.py`, `steer/extract.py` | mel / F0 / energy frames + the 12 raw control measurements per utterance |
| `steer/build_controls.py` | raw controls -> z-scores relative to each speaker's neutral speech |
| `steer/text.py` | text -> ARPAbet ids (g2p_en) with word indices |
| `steer/data.py` | dataset with control/tag dropout, frame-bucketed sampler |
| `steer/model.py` | SteerTTS: AdaLN-conditioned encoder/decoder, duration/pitch/energy predictors, RAD-TTS-style aligner |
| `steer/train.py` | stage-1 training; `--cons` turns on the stage-2 control-consistency loss |
| `steer/readback.py`, `steer/consistency.py` | mel -> control readback net and the cycle-consistency loss built on it |
| `steer/captions.py`, `steer/prompt.py` | acoustically anchored captions and the prompt -> control adapter |
| `steer/synth.py` | inference (`Synth` class + CLI) with Vocos vocoding |
| `steer/harness.py` | paired steerability evaluation (achieved / requested z, hit rate, leakage, Wilcoxon + Holm) |
| `steer/ear.py` | perceived-emotion probe (WavLM layer 6, CREMA-D crowd labels) |
| `steer/asr.py` | Whisper intelligibility guard (WER) |
| `emotion_study/` | CREMA-D acoustic emotion study (s01-s07) that the ear probe and control design build on |
| `data/download.sh`, `data/hf_curl.sh`, `data/hf_files.tsv` | resumable corpus downloads |

## Setup

```bash
pip install -r requirements.txt
export STEER_TTS_ROOT=/big/disk/steer_tts      # data + runs (default: this directory)
# Hugging Face token for gated/rate-limited downloads, kept off the command line:
mkdir -p $STEER_TTS_ROOT/data && echo 'header = "Authorization: Bearer hf_xxx"' > $STEER_TTS_ROOT/data/.hf_curlrc
chmod 600 $STEER_TTS_ROOT/data/.hf_curlrc
```

Corpora: EARS (CC BY-NC 4.0, 107 speakers, from GitHub releases), Expresso (CC BY-NC 4.0),
LibriTTS-R train-clean-100 (CC BY 4.0), ParaSpeechCaps (CC BY-NC-SA 4.0). Weights trained on these
should be released under a matching non-commercial (ShareAlike where ParaSpeechCaps is used) licence.

## Running

```bash
./run.sh download                                   # resumable; re-run until complete
CORPORA="ears" ./run.sh prep                        # or all three corpora (default)
RUN=base ./run.sh train --steps 150000 --max_frames 16000
RUN=base ./run.sh tensorboard                       # http://localhost:6006
./run.sh readback && RUN=base ./run.sh stage2 --steps 30000
RUN=base ./run.sh harness --voices ears_p001,ears_p004
./run.sh prompt                                     # prompt adapter; then: python -m steer.prompt try "speak sadly"
```

The emotion probe needs the CREMA-D study outputs first:
`CREMA_D_DIR=/path/to/CREMA-D python emotion_study/s01_build_table.py` and `s04_embed.py`
(outputs go to `$EMOTION_STUDY_OUT`, default `emotion_study/out`), then `./run.sh ear`.

## TensorBoard

`runs/<run>/tb` holds everything needed to judge a run without opening files:

| Group | Contents | When |
|---|---|---|
| `train/*` | each loss, total, lr, grad norm, alignment-binarisation weight | every `--log_every` |
| `perf/*` | steps/s, mel frames/s, utterances per batch, peak GPU memory, skipped non-finite steps | every `--log_every` |
| `val/*` | validation losses | every `--eval_every` |
| `val_audio/*` | free-running synthesis of 4 fixed validation utterances; vocoded ground truth once (vocoder ceiling) | every `--eval_every` |
| `val_mel/*`, `val_align/*` | mel images (ground truth / teacher-forced / free-running) and the soft text-to-frame alignment | every `--eval_every` |
| `steer_ratio/*`, `steer_hit/*`, `steer_leak/*` | per-direction harness results | every `--harness_every` |
| `steer_summary/*` | median ratio, mean hit, mean leakage, share of directions significant after Holm; tag hit rate | every `--harness_every` |
| `steer_audio/*` | one sentence and voice rendered under every direction, for listening | every `--harness_every` |
| Text tab | args, model size, data composition, full harness table | start / harness |

Resuming (`--resume`) purges events logged after the checkpoint step, so curves don't double back.
