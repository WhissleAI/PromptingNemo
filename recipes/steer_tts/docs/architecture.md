# SteerTTS architecture (current system, 2026-10-11)

Technical description of the system as checkpointed: base `runs/spk_enc_v3/step312500.pt`, refiner
`runs/flow_v3/step100000.pt`, prompted flow `runs/flow_v4pa/step40000.pt` (and `runs/flow_v5` = `steer/pflow5.py`,
in training). Architecture values were read from the checkpoints (instantiated on CPU, all keys matched) and the
run directories' `args.json`. Commands to rebuild them: [`reproduce.md`](reproduce.md); evaluation:
[`evaluation.md`](evaluation.md).

## 0. Summary

SteerTTS is a non-autoregressive, explicit-variance TTS (FastPitch/FastSpeech2 lineage) whose every layer is
conditioned, via adaptive LayerNorm, on one global vector g = MLP([speaker, style-tag embedding, masked 12-d control
vector, mask]). The 12 controls are utterance-level acoustic measurements (rate, loudness, F0 median/IQR/end-slope,
pause share, spectral tilt, voicing share, flatness, centroid, CPPS, H1-H2) expressed as speaker-relative z-scores
against each speaker's neutral speech, measured by the same DSP code that later reads them back from synthesized
audio. Voice comes from a frozen WavLM-base-plus-sv x-vector of a *different neutral* utterance of the speaker, mapped by a
linear `spk_proj` (initialised by ridge regression onto a previously trained speaker table). Encoder and decoder are
4+4 "hybrid" blocks: multi-head self-attention plus a zero-initialised bidirectional Mamba-3 branch, each with a conv FFN.
Durations come from a jointly trained RAD-TTS aligner (forward-sum CTC + MAS + binarisation). Free-form text prompts
are mapped to (tag, which-dims, z-values) by a 0.5M adapter on a frozen MiniLM sentence encoder, trained on
acoustically-anchored template captions + real instructions with measured targets. The base is fine-tuned with a
paired-intervention delta loss and differentiable reward losses (readback control, leak, emotion probe, CTC, UTMOS,
speaker-SV, trust-region anchor) through the frozen Vocos vocoder with PCGrad. An optional 17.6M OT-CFM flow refiner
(ConvNeXt + AdaLN-zero) samples mel detail on top of the base mel; a prompted variant (Voicebox-style infilling with
the reference mel as prefix context, base-mel CMN, 6 attention layers) is used for voice cloning. Mel->wave is the
pretrained `charactr/vocos-mel-24khz` (unchanged). Inline markup (emphasis, per-word pitch/energy/duration shifts,
pauses, style spans, events) and sentence-level streaming sit on top without retraining.

## 1. Audio representation (config.py, features.py)

* 24 kHz, n_fft 1024, hop 256 (93.75 frames/s), 100 mel bands, Vocos `MelSpectrogramFeatures` (log of magnitude mel,
  clamp 1e-5, `padding="center"`) - bit-compatible with `charactr/vocos-mel-24khz`.
* Frame features: pitch by Praat autocorrelation (`to_pitch_ac`), two-pass utterance-adaptive range (De Looze & Hirst:
  pass 1 60-700 Hz; pass 2 [max(50, 0.75 q25), min(900, 1.75 q75)]), sampled at mel-frame centres; frame energy
  = 10 log10 mean STFT power (dB).
* Adaptive VAD: speech frames = energy > p5 + 0.4 (p95 - p5).

## 2. Text front-end (text.py)

* g2p_en -> ARPAbet with stress (39 phones; vowels x {0,1,2}) = 69 phones + `<pad>`, `<sp>` + 7 punctuation tokens
  (`, . ? ! ; : -`) = 78 base symbols (`N_BASE`), plus 40 inline sound-event tokens (12 nonverbals: laugh, cry, scream,
  yell, cheer, cough, sneeze, yawn, throat, breath, sigh, sniff; 15 environmental events; 13 reserved) -> 118 symbols.
* Word boundaries become `<sp>` tokens; every token carries a word index (-1 for `<sp>`/punct/events), used for
  per-word prosody edits.
* Normalisation `_expand` (added 2026-10-09 after CV3-Eval error analysis): numeric ranges ("1990-1995" -> "to"),
  "4x4" -> "by", hyphen/slash compounds split, currency, percent, 15 units, ordinals, comma-grouped and decimal numbers,
  >=5-digit runs via num2words, unknown 2-5-letter all-caps acronyms spelled letter by letter (CMUdict entries kept).
  Em/en dashes -> commas, ellipsis -> period.
* Inline `[event]` markers are split out and become single tokens between `<sp>`; unknown markers dropped.

## 3. Control space (config.CONTROL_DIMS, features.measure, build_controls.py)

12 utterance-level measurements, computed by `features.measure` on speech frames inside the first..last speech frame:

| dim | definition |
|---|---|
| rate | phones (excl. `<sp>`/punct) / (speech span - pauses) [phones/s] |
| loud | mean frame energy of speech frames [dB] |
| f0_med | median F0 of voiced frames, semitones re 100 Hz |
| f0_iqr | F0 IQR [st] |
| f0_end | linear slope of F0 [st/s] over the last 300 ms of voicing (>=8 voiced frames) |
| pause | share of span in silent runs >= 100 ms |
| tilt | alpha ratio, 10log10(E[1-5 kHz]/E[50 Hz-1 kHz]) [dB] |
| voiced | voiced share of speech frames |
| flat | log of mean per-frame spectral flatness (50 Hz-8 kHz) |
| centroid | log spectral centroid of speech frames (pooled spectrum) |
| cpps | smoothed cepstral peak prominence (Hillenbrand-style 3x3 smoothing, 60-330 Hz quefrency band) [dB] |
| h1h2 | median uncorrected H1-H2 over every 2nd voiced frame [dB] |

Speaker-relative z-normalisation (`build_controls.py`):
* Reference = speaker's neutral speech: EARS `regular`, Expresso `default` (+essentials/longform/narration ->
  "neutral"), all LibriTTS-R, Cartesia only default speed/volume renders; speakers without neutral speech use their
  overall mean.
* z = (x - mu_speaker_neutral) / sigma_pooled, sigma_pooled = pooled within-speaker SD of neutral residuals, floored at
  0.05 x median(sigma over dims), z clipped to [-5, 5]. F0 dims and H1-H2 set NaN when voiced share < 0.2 (whisper).
* Filters: duplicate ids dropped, 0.8 s <= dur <= 20 s, 3-40 chars/s, <= 0.6 tokens/frame.
* Also stores per-utterance `spk_f0_st` and `spk_loud` (speaker neutral means) for frame-level normalisation.
* Pooled SDs of the current META (run dir `control_stats.json`): rate 1.02 ph/s, loud 1.93 dB, f0_med 2.09 st,
  f0_iqr 1.96 st, f0_end 21.7 st/s, pause 0.081, tilt 2.98 dB, voiced 0.081, flat 0.545, centroid 0.234, cpps 1.31 dB,
  h1h2 2.23 dB. NOTE: pause and voiced sit exactly on the floor (0.0812) - see risks.

Frame-level targets (data.py): pitch = (12 log2(f0/100) - spk_f0_st)/3 on voiced frames (implausible st outside
[-12, 18] unvoiced; utterances < 12% voiced treated as fully unvoiced); energy = (frame dB - spk_loud)/10.

Training data (current META, 410,566 utts / 659 h / 4,919 speakers): LibriTTS-R train-clean-100/360 + other-500
(337,510 / 516 h / 2,278 spk), EARS incl. freeform (23,108 / 77.6 h / 107), Expresso (11,539 / 10.7 h / 4),
Cartesia synthetic renders (33,076 / 43 h / 432 voices; emotion/speed/volume instructions), NonverbalTTS ("nvtts",
5,333 / 12.5 h / 2,098 spk; emoji nonverbal markers -> event tokens). 51 tags (none + 50 canonicalised styles/emotions).
v3 oversampling weights: ears 4, earsff 1, expresso 3, libritts 0.6, cartesia 2, nvtts 2 (sampling with replacement).

Conditioning dropout (TTSData, train): each control dim masked w.p. 0.35, all controls masked w.p. 0.15, tag -> none
w.p. 0.3. The mask is an explicit input to g, so "unspecified" is distinct from "z = 0" and partial specification is
in-distribution; the dropout also trains the unconditional branch used by guidance.

## 4. Style tags

51-entry tag embedding table (dim 32, index 0 = none). Canonicalised from EARS styles/emotions, Expresso styles,
Cartesia emotions (angry/outraged retagged `irritation`), NonverbalTTS emotions. LibriTTS-R = none.

## 5. Acoustic model (model.py) - base `spk_enc_v3/step312500`

Arch from checkpoint: `enc_mixer=hybrid, dec_mixer=hybrid, ssm_state=32, spk_enc_dim=512, n_symbols=118, n_env=30`;
defaults d=192, heads=2, ffn=384, conv k=3, 4 encoder + 4 decoder layers, g_dim=128, spk_dim=64, tag_dim=32, dropout 0.1.

* Conditioning: s = spk_proj(x-vector) (Linear 512->64) or table row (4,919+1 x 64, fallback/unknown row);
  g = Linear(256->128)(SiLU(Linear(64+32+12+12 -> 256)([s, tag, nan_to_num(z)*mask, mask]))) (+ zero-init env MLP
  30->128->128 for the acoustic-environment vector; unused/zero in v3, which trains with noise_mode=nv).
* Every norm is AdaLN: LN without affine, (1+scale, shift) = Linear(g) zero-initialised.
* FFT block (pre-norm): AdaLN -> MHA (2 heads, d 192) -> residual; [hybrid] AdaLN -> BiMamba3 (zero-init out_proj) ->
  residual; AdaLN -> Conv1d(192->384, k3) -> GELU -> Conv1d(384->192, k1) -> residual. Sinusoidal positions added at
  stack input; output AdaLN.
* BiMamba3: forward + backward `Mamba3Layer` (d_model 192, d_state 32, head_dim 32 -> 6 heads, rank 1 = SISO,
  trapezoidal discretisation with data-dependent lambda, rotary (complex) state via theta, BC RMSNorm + B/C biases,
  data-independent A), summed; backward pass reverses each sequence within its own length so padding never enters the
  state; SSD chunked scan (chunk 64) in fp32; optional torch.compile. Implementation:
  `promptingnemo/models/mamba3/` (`layer.py`, `ssd.py` chunked SSD with the decay mask applied before `exp`, `scan.py`,
  a naive `reference.py` and `tests.py`: `python -m promptingnemo.models.mamba3.tests`).
  The "hybrid" model was created by grafting zero-init SSM branches into a trained attention model (mix_s1 @17k) and
  fine-tuning (exact identity at init).
* Variance predictors (duration, pitch, energy; each 320.8k): Conv(k3)-ReLU-AdaLN-Dropout(0.15) x2 -> Linear(1), on
  encoder output, conditioned on g. Duration predicted as log(d+1).
* Pitch/energy embeddings: Conv1d(1->192, k3) of token-level values, added to encoder output (teacher-forced
  targets during training).
* Length regulator by integer durations; inference: round((exp(logd)-1)/pace), first token >= 1 frame.
* Aligner (training only, 214.6k): RAD-TTS/FastPitch-1.1 - key = 2 convs on text *embeddings*, query = 3 convs on mel,
  80-d, logits = -0.0005 ||q-k||^2, log-softmax over tokens + log beta-binomial prior; MAS (numba, parallel) on soft
  attention gives hard durations; token targets = duration-averaged frame pitch (voiced frames only) / energy.
  Noise-augmented items align on the dry mel; inserted event spans can be forced via the prior.
* Output: Linear(192->100) log-mel. No postnet.
* Level normalisation (`data.LEVEL_DB = 0`): in spk_enc runs every target mel is shifted by
  (0 - spk_loud) ln10/20, i.e. each speaker's neutral speech is put at the same level. Motivation: x-vectors are
  level-invariant, so without it the model cannot infer a speaker's recording level (EARS ~19 dB below LibriTTS) and
  rendered all voices at the loud corpora's level, clipping (observed spk_enc@130k).
* Guidance (`infer(guidance=s, base=(tag0, 0, 0))`): CFG-style extrapolation applied at every stage - log-duration,
  pitch, energy predictions, then the decoder mel (both passes share the guided timeline):
  out = out(base) + s (out(request) - out(base)). Base = same voice, no tag, all controls masked. Motivation: control
  labels are noisy measurements so the learned effect is attenuated (regression dilution, harness ratio ~0.5).
  Default s = 1 (off) in Synth and all evaluation tools.
* Per-word edits (`word_shift`): additive pitch (st/3), energy (dB/10), multiplicative duration on tokens of a word.

## 6. Losses and training schedule (train.py, data.py)

Supervised: L = 1.0 L1(mel) + 0.1 MSE(log dur) + 0.1 MSE(token pitch) + 0.1 MSE(token energy) + 1.0 forward-sum CTC
(blank logprob -1) + w_bin(step) * binarisation KL (w_bin ramps 0->1 over steps 4k-8k).
Optimiser: fused AdamW lr 1e-3, betas (0.9, 0.98), wd 1e-6, warmup 4k then cosine to 5% over `--steps`, grad clip 1.0,
bf16 autocast, non-finite steps skipped on device. Batches: length-bucketed by total frames (`max_frames` 8000 in v3;
16000 earlier), max 64 items.

Optional terms (all ON in spk_enc_v2/v3):
* Paired-intervention delta loss (`--pair_w 1.0, --pair_k 5`): pairs of real clips with the same speaker and same
  normalised text but different render conditions (EARS reading styles, Cartesia speed/volume/emotion variants,
  Expresso styles); for each pair the *predicted change* between the two conditions must match the *real change*:
  per-token log-duration, pitch, energy (same token ids), plus per-utterance mean log-mel spectrum. Pairs are also
  supervised normally at weight 0.5; EARS pairs drawn ~30% of the time.
* Reward fine-tuning (`--reward 0.25 --reward_every 4 --reward_k 4`, weights control 0.5, leak 4.0, emotion 0.3,
  ctc 0.3, utmos 0.3, spk 0.5, anchor 1.0): see section 9. Combined with the supervised gradient by PCGrad.
* Noise/nonverbal augmentation (`--noise 0.4 --noise_mode nv`): 40% of training items get a same-speaker EARS
  nonverbal (laugh/cough/...) spliced at an utterance edge with the event token inserted in the text.
* `--cons` (stage-2 control-consistency) exists but is OFF (0.0) in the whole spk_enc lineage.

Lineage of the base checkpoint (cumulative step counter):
ears25_base (attn, 50k) -> mix_s1 (attn, 0-17k) -> mix_m3hyb (graft hybrid, 17k-116k) -> spk_enc (x-vector
conditioning, LS-init spk_proj, LEVEL_DB from 134k, 116k-~178k) -> spk_enc_full (+LibriTTS other-500, rebalanced
oversampling, ->290k) -> spk_enc_v2 (reward+pairs+full noise env, 290k-297.5k) -> spk_enc_v3 (reward+pairs+nv noise,
297.5k-317.5k, best by eval = 312.5k). Training data grew at several stages (META rebuilt).

## 7. Speaker encoder / x-vectors (xvec.py, train.save_spk_refs, train.fit_spk_proj)

* Frozen `microsoft/wavlm-base-plus-sv` x-vector (512-d, L2-normalised; first 8 s; clips < 0.5 s tiled) of every
  training utterance, precomputed (`xvec_<corpus>.npz`, f16).
* Training: each item is conditioned on the vector of a DIFFERENT, neutral utterance of the same speaker (random per
  sample; validation fixed), so the vector must carry identity, not the target's style.
* Known voices at inference: mean of the speaker's neutral vectors, renormalised (`spk_ref.npz`); cloning: vector of
  the reference clip (Synth(ref_wav=...)).
* `spk_proj` init: ridge (lambda 0.1) least-squares map from mean-neutral vectors to the previous model's trained
  64-d speaker-table rows (R^2 printed), so known voices start where they were.

## 8. Prompt adapter (prompt.py, captions.py)

* Frozen `sentence-transformers/all-MiniLM-L6-v2` (22.7M, not counted in the TTS) -> 384-d normalised embedding.
* Adapter (498k): Linear 384->512, GELU, Dropout 0.1, Linear 512->512, GELU; heads: tag logits (51), "specified"
  logits per control dim (12, BCE), value per dim (12, smooth-L1 masked by specified). Env variant (557k) adds scene,
  room, phone, SNR, start/end event heads.
* Training sources:
  1. Template captions (`captions.caption`, 6 per utterance over all of META): a phrase for a dim is emitted only when
     that utterance's measured |z| is salient, adverb grade follows |z| (0.8/1.6/2.6 thresholds); tag phrases per
     style; targets = the clip's measured z and tag.
  2. Cartesia: the instruction each clip was rendered with + 2 regenerated paraphrases; specified dims = rate if speed
     != 1, loud if volume != 1; targets = MEASURED z (not the requested setting).
  3. ParaSpeechCaps `train_base` descriptions of EARS/Expresso clips present in META (dev/test/holdout excluded);
     specified = f0_med, rate.
  12 epochs, AdamW 2e-3, batch 1024, 2% random val split.
* Inference: argmax-ranked tag (first that the checkpoint knows), dims with sigmoid(spec) > 0.5 and their values.
  Note: the tag is used HARD (lookup), the distribution is not mixed.

## 9. Readback, consistency, rewards (readback.py, consistency.py, rewards.py)

Readback net (1.01M; `readback_ln.pt` for level-normalised models): input [mel - spk_ref_mel, spk_ref_mel] (200 ch,
spk_ref_mel = speaker's mean neutral speech-frame log-mel), Conv k5 -> 4 residual dilated convs (1,2,4,1) with
GroupNorm, attention pooling + max pooling -> MLP -> 9 dims (all except rate, pause, f0_end). Trained 8-15 epochs
(smooth-L1) on real mels with measured z targets (corpus-balanced cap 80k for `_ln`).

Consistency measurement (differentiable): acoustic dims from readback on the free-run mel; rate and pause from SOFT
predicted durations (pause = soft gate sigmoid((d-9)/2) on non-phone tokens inside the phone span), f0_end from the
predicted token pitch of the last 4 phone tokens; all z-normalised with the run's speaker means / pooled SDs.
Perturbation: 1-2 dims per item shifted by U(1, 2.5) x +/-1 from current value (clamped +/-4); perturbed dims weigh 3x.

Reward step (every 4th step, K=4 items with mel_len <= 650 frames; renders capped at 750 frames for the judges):
* request: 50% an emotion tag from {anger, sadness, happiness, fear, disgust, excited, anxiety} (half of these pure
  tag requests), plus 1-2 perturbed controls (voiced/cpps excluded from perturbation in rewards);
* free_run (durations rounded, no gradient through them) -> frozen Vocos decode (differentiable ISTFT) -> 16 kHz;
* r_control: smooth-L1(measured, requested) as above;
* r_leak: for pure control requests, non-requested dims of the steered render must match the unsteered render
  (no-grad) - prevents gaming by moving every dim;
* r_emotion: frozen "ear" probe (LayerNorm + Linear, 6.2k params, on WavLM-base-plus layer 6, trained on CREMA-D soft
  labels) applied to embedding(render) - embedding(same voice/text with no request); -log p(target class);
* r_ctc: CTC loss of the transcript under frozen wav2vec2-base-960h;
* r_utmos: -UTMOS22-strong;
* r_spk: 1 - cos(WavLM-base-plus-sv(render), conditioning x-vector);
* r_anchor (trust region): the same items under their own (unsteered) conditions free-run by the model and by a frozen
  copy of the init checkpoint must agree per token (log-dur, pitch, energy, MSE) and in mean spectrum.
* PCGrad: reward gradient projected to remove its component opposing the supervised gradient.
Observed at ~318k: pc_conflict ~0.7 (70% of reward steps conflict), grad norm 4-5 (always clipped at 1),
r_emotion ~3.0 nats (p(target) ~0.05, i.e. below chance 1/6: the emotion reward is not being satisfied).

## 10. Flow-matching refiner (flow.py) - `flow_v3/step100000`

* Purpose: the L1 decoder predicts a conditional median (hazy upper bands; val mel L1 flat ~0.645); a conditional flow
  samples p(real mel | base mel mu, g); everything that steers still comes from the base.
* OT-CFM (sigma_min 1e-4): x_t = (1-(1-s)t) x0 + t x1, target u = x1 - (1-s) x0; t sampled half U(0,1), half
  logit-normal; masked MSE. Mels normalised (mean -1.583, std 2.131 from 300 level-normalised training mels).
* Net (17.64M): input Conv1d(200->384, k3) on [x_t, mu]; 12 ConvNeXt blocks (depthwise k7, dilations
  1,2,4 repeated, LN, MLP x3, GELU) with AdaLN-zero (scale, shift, gate) from c = TimeEmb(t) (sinusoidal x1000, MLP) +
  MLP(g) (g = base model's 128-d conditioning vector); final LN + zero-init Conv1d(384->100) velocity.
* Training data: teacher-forced ("GTA") mu dumped from base spk_enc_v3/**step305000** (`data/gta/v3_305k`): all
  non-LibriTTS clips + 60k LibriTTS, 40-1600 frames; random 256-frame crops (flow_v3 default), batch 48, AdamW 2e-4,
  betas (0.9, 0.99), wd 0.01, 1k warmup + cosine to 10%, EMA 0.9995, 100k steps; corpus weights ears/expresso 2.
* Inference: x0 ~ N(0, temp^2 I), temp 0.4 (Synth default), 10 midpoint steps (20 NFE), mu = FREE-RUNNING base mel.
* Reported effect: +0.13-0.27 UTMOS (quick_eval tts 4.12 -> 4.27).

## 11. Prompted flow (flow.py `--prompt`, flow_v4pa; pflow5.py = v5)

flow_v4pa/step40000 (24.67M), lineage flow_v3 -> v3p (+prompt, 60k) -> v3pc (+mu_cmn, 40k) -> v4pa (+attention at
layers 1,3,5,7,9,11; crop 384, batch 48, lr 1.5e-4; step 40k used):
* Input [x_t, mu, ctx, cm] (301 ch): ctx = real mel on context frames, cm = mask; mu replaced by the real mel on
  context frames (Voicebox/F5-style infilling); loss only on non-context frames.
* Context encoder (Conv-SiLU-Conv) -> masked mean -> zero-init Linear added to the global conditioning c.
* Six zero-gated full self-attention blocks (MHA 6 heads, AdaLN-zero, no positional encoding) after odd ConvNeXt blocks.
* mu_cmn: per-band time-mean of the base mel over target frames removed (cepstral-mean-normalisation of mu), so the
  static spectral envelope (voice colouring guessed from the x-vector) is not available and must come from the prompt.
* Training context: in-clip spans (10-50% of the crop; prefix w.p. 0.7; none w.p. 0.1, min 24 frames).
* Inference (Synth, only when a reference clip is given): prompt = first 8 s of the reference, log-mel level-matched
  (mean of the louder half of frames on valid bands); for band-limited (<24 kHz) references the missing upper bands
  are filled from the render's mean spectral shape anchored at the top valid band (+4 SIM on CV3); [prompt | base mel]
  sampled jointly; prompt frames sliced off.
* pflow5 / flow_v5 (38.74M, training now from v4pa@40k): mu computed ONLINE by the frozen base (spk_enc_v3/312.5k) on
  the full 410k-clip corpus; prompts are 150-600-frame crops of ANOTHER clip of the same speaker (p_ref 0.7, else
  in-clip context as before); 30% of items get mu/g from a random other speaker's x-vector (voice-swap, so the flow
  must take identity from the prompt); new zero-init modules: RoPE self-attention at even layers (6), a 32-query
  2-layer Perceiver-resampler PromptPool cross-attended by the RoPE blocks, E2-TTS-style long skips (i -> n-1-i), and a
  TLA-SA-style alignment loss (w 0.2): target-frame mean features at layers 5, 8, 11 -> MLP -> 512-d, 1 - cos to the
  target clip's precomputed x-vector. lr 1e-4 (new params x2), crop 512, frame budget 16k, 90k steps.

## 12. Vocoder

Pretrained `charactr/vocos-mel-24khz` (13.53M: ConvNeXt backbone 13.01M + ISTFT head 0.53M), unchanged.
A GAN fine-tune on flow mels (`vocoder_ft.py`, `runs/vocoder_v3`) was abandoned (discriminators at chance). Copy-synthesis
SIM ceiling on Seed (68.4 vs 72.0 real) indicates the vocoder is not the cloning bottleneck.

## 13. Inline markup + streaming (markup.py)

* `*word*`/`<emph>` -> +2.5 st, +4 dB, x1.2 duration on those words; `<slow>` x1.4, `<fast>` x0.75, `<high>/<low>`
  +/-3 st, `<loud>/<soft>` +/-6 dB; nestable (pitch/energy add, duration multiplies) - implemented as `word_shift`
  inside ONE forward pass, so prosody is continuous.
* `[pause Nms|Ns]` (default 400 ms, max 5 s) inserts silence; `<tag>..</tag>` (any tag the checkpoint knows) and
  `<style prompt="...">` (via the adapter) create separate segments (concatenated with 10 ms fades; a documented
  boundary artefact); `[event]` tokens stay inline (or are mixed from real recordings by `mixer.EnvMixer`).
* Streaming: segments split at sentence ends, pauses and style boundaries; each is rendered and yielded in order
  (first audio ~61-84 ms warm on a shared GPU). Word-level streaming would need a causal model (not done).

## 14. Synthesis API (synth.py)

`Synth("base.pt[,flow=F][,pflow=P]", guidance=1, flow_steps=10, flow_temp=0.4, flow_cfg=0)`;
`__call__(text, speaker, tag, ctrl={dim: z}, pace, word_shift, ref_wav, env, guidance)`. Loads the run dir's
`control_stats.json` / `spk_ref.npz`; public voice names resolved via voices.py; drops event tokens an old checkpoint
lacks; prompted flow used only with `ref_wav`, else the unprompted flow, else the L1 mel; Vocos decode, no output
normalisation.

## 15. Parameter budget (instantiated from the checkpoints)

| component | params |
|---|---|
| Base SteerTTS total (spk_enc_v3/312.5k) | **9,800,243** |
|   encoder (4 hybrid blocks) / decoder (4 hybrid blocks) | 4,072,976 / 4,072,976 |
|   of which bidirectional Mamba-3 branches (8 x 413,540 + their AdaLN) | 3,704,608 |
|   per block: MHA 148k, BiMamba3 414k, conv FFN 295k, 3 AdaLN x 49.5k | |
|   duration / pitch / energy predictors | 3 x 320,833 |
|   speaker table (4,920 x 64) / spk_proj / tag / symbol emb | 314,880 / 32,832 / 1,632 / 22,656 |
|   cond MLP / env_proj / pitch+energy emb / to_mel | 63,872 / 20,480 / 1,536 / 19,300 |
|   aligner (training only) | 214,604 |
| Base inference path (excl. aligner) | 9,585,639 (9.27M excl. speaker table) |
| Flow refiner flow_v3 | 17,642,596 |
| Prompted flow flow_v3pc / flow_v4pa / flow_v5 | 21,569,764 / 24,674,404 / 38,740,324 |
| Vocos vocoder (frozen, pretrained) | 13,531,650 |
| Prompt adapter (/ env variant) | 498,251 (/ 556,733) |
| MiniLM-L6 sentence encoder (frozen, external) | ~22.7M |
| WavLM-base-plus-sv speaker encoder (frozen, external) | ~94M (base-plus) |
| Readback (training only) / ear probe (training/eval only) | 1,007,626 / 6,150 |

System totals at inference: L1 path 9.6M + Vocos 13.5M = 23.1M; +flow = 40.7M; cloning path base + pflow v4pa + Vocos
= 47.8M (+ x-vector encoder).
