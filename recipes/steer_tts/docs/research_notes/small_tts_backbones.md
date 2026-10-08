# Small / Efficient TTS Acoustic Models and Vocoders for a <=10M (stretch <=30M incl. vocoder) Expressive, Style-Controllable English TTS (2020-2026)

Verification legend used throughout:
- **[V]** = number confirmed this session from the primary source (paper table, abstract, or official model card/repo).
- **[V-2nd]** = confirmed only from a secondary/aggregator source this session.
- **[M]** = from the cited paper as recalled from prior knowledge; NOT re-fetched this session, so the report writer should treat it as "to double-check" (the arXiv ID/link is correct; exact decimals may differ).
- **[Vendor]** = vendor/model-card claim with no independent benchmark.

## Q1. Parameter counts and reported quality of candidate acoustic backbones / end-to-end models (<=100M)

### Takeaway
Only a few published systems have genuinely <=10M params with acceptable quality: EfficientSpeech (266k, mel only), LightSpeech (~1.8M, mel only), Piper x_low (5-7M, end-to-end VITS) and the Kitten TTS nano (15M, end-to-end, closed architecture). The best documented quality/size trade-off in the 15-20M band is Matcha-TTS (18.2M, MOS 3.84 vs VITS 3.71, WER 2.09%), plus a ~14M vocoder. StyleTTS 2 / Kokoro-82M are the strongest "style-vector" designs but are 4-8x over the budget; their style mechanism, not their size, is the thing to borrow.

### Cited findings

**Matcha-TTS benchmark table (LJSpeech, all with HiFi-GAN V1 except VITS; GPU RTX 3090 RTF incl. vocoder)** — [Matcha-TTS, arXiv 2309.03199 (ICASSP 2024), Table 1](https://arxiv.org/html/2309.03199v2) [V]

| System | Params (acoustic) | RTF (GPU) | WER % | MOS |
|---|---|---|---|---|
| Vocoded GT (HiFi-GAN V1, 13.9M) | — | 0.001 | 1.97 | 4.13 |
| FastSpeech 2 | 41.2M | 0.010 | 4.18 | 3.29 |
| VITS | 36.3M | 0.074 | 2.52 | 3.71 |
| Grad-TTS 10 steps | 14.8M | 0.049 | 3.44 | 3.49 |
| Grad-TTS 4 steps | 14.8M | 0.019 | 3.69 | 3.20 |
| Matcha-TTS 10 steps | 18.2M | 0.038 | 2.09 | 3.84 |
| Matcha-TTS 4 steps | 18.2M | 0.019 | 2.15 | 3.77 |
| Matcha-TTS 2 steps | 18.2M | 0.015 | 2.34 | 3.65 |

- Matcha-TTS uses OT-CFM, needs fewer ODE steps than score-matching (Grad-TTS), claims smallest memory footprint and highest MOS among the compared pre-trained baselines — [arXiv 2309.03199](https://arxiv.org/abs/2309.03199) [V]. Training memory 4.8 GiB vs 12.4 GiB for VITS — [Table 1](https://arxiv.org/html/2309.03199v2) [V]. No CPU RTF reported in the paper [V].
- Note: FastSpeech 2's low MOS (3.29) in Matcha's table is a re-implementation baseline; in the original FS2 paper it was closer to Tacotron 2 — [FastSpeech 2, arXiv 2006.04558](https://arxiv.org/abs/2006.04558) [M].

**EfficientSpeech (2023)** — 266k parameters, ~90 MFLOPs, ~1% the size/compute of Mixer-TTS, mel-generation RTF 104.3 (i.e., 104x faster than real time) on a Raspberry Pi 4 ARM CPU, "only slight degradation" vs FastSpeech 2 in human evaluation; no MOS number in abstract — [arXiv 2305.13905 (ICASSP 2023)](https://arxiv.org/abs/2305.13905) [V]. It still needs a vocoder (the paper pairs it with HiFi-GAN) [M].

**LightSpeech (2021)** — NAS-searched FastSpeech-style model; ~1.8M params, ~15x model compression and ~6.5x CPU inference speedup vs FastSpeech 2 with on-par CMOS — [arXiv 2102.04040 (ICASSP 2021)](https://arxiv.org/abs/2102.04040) [M].

**FastPitch (2020)** — FastSpeech-style with explicit per-symbol pitch predictor (later energy); ~45M params (base) — [arXiv 2006.06873](https://arxiv.org/abs/2006.06873) [M]. NVIDIA reports large GPU speedups; small variants are not formally published.

**Grad-TTS (2021)** — 14.8M params [V via Matcha Table 1](https://arxiv.org/html/2309.03199v2); diffusion decoder; original paper [arXiv 2105.06337](https://arxiv.org/abs/2105.06337).

**VITS / VITS2 / iSTFT-VITS family**
- VITS 36.3M params, MOS 3.71, WER 2.52% (LJSpeech) — [Matcha Table 1](https://arxiv.org/html/2309.03199v2) [V]. Original VITS: [arXiv 2106.06103](https://arxiv.org/abs/2106.06103).
- MB-iSTFT-VITS (2022/ICASSP 2023): replaces HiFi-GAN decoder with multi-band iSTFT; 4.1x faster than VITS, RTF 0.066 on Intel Core i7 CPU, naturalness "as natural as VITS"; a smaller version (MS/mini) significantly outperformed a lightweight baseline in naturalness and speed — [arXiv 2210.15975](https://arxiv.org/abs/2210.15975) [V]. Param counts not in abstract (see repo [MasayaKawamura/MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)) — gap.
- VITS2 (2023): transformer block in normalizing flow, adversarial duration predictor, reduced phoneme-conversion dependence — [arXiv 2307.16430](https://arxiv.org/abs/2307.16430) [M]; params similar to VITS (~tens of M) [M].

**Piper (rhasspy, VITS-based, ONNX)** — quality tiers: x_low 16 kHz 5-7M params; low 16 kHz 15-20M; medium 22.05 kHz 15-20M; high 22.05 kHz 28-32M — [Piper voice docs, reproduced by techfeed](https://techfeed.io/entries/6a2dc63cee3ca6ad897bc8f1) [V-2nd]. The official TRAINING.md lists only low/medium/high with sample rates and "smaller"/"larger voice model", no param counts — [rhasspy/piper TRAINING.md](https://github.com/rhasspy/piper/blob/master/TRAINING.md) [V]. No published MOS for Piper tiers (gap).

**StyleTTS 2 (2023)** — style diffusion + adversarial training with WavLM discriminator; style vector sampled by diffusion (acoustic + prosodic halves) conditions duration/pitch/energy predictors and decoder; reported to surpass human recordings on LJSpeech (single speaker) and match on VCTK — [arXiv 2306.07691 (NeurIPS 2023)](https://arxiv.org/abs/2306.07691) [M]. Repo: [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2).

**Kokoro-82M** — 82M params; StyleTTS 2 + ISTFTNet; "decoder only: no diffusion, no encoder release" (i.e., style vectors are precomputed voicepacks, no reference-audio style encoder shipped); trained on "few hundred hours" (v1.0; v0.19 <100 h) of permissive audio with IPA labels; ~1,000 A100-hours (~$1,000); Apache-2.0; v1.0 8 languages/54 voices; no MOS on card, entered TTS Arena — [hexgrad/Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M) [V/Vendor]. Voicepacks are 256-d style vectors (128 acoustic + 128 prosodic, indexed by token length) [M — from StyleTTS2 design; verify against voices/*.pt].

**Kitten TTS (KittenML)** — official repo lists: nano 15M (56 MB fp32 / 25 MB int8), micro 40M (41 MB), mini 80M (80 MB) "legacy ONNX" models (v0.8), plus KittenTTS 2 (1.7B, 1-bit speech LM, GGUF/llama.cpp, 47 voices, 10 emotion tags, 20 languages, 24 kHz). Architecture of the small ONNX models is not documented; license for small models not stated on the README; no MOS/WER for small models — [KittenML/KittenTTS GitHub](https://github.com/KittenML/KittenTTS) [V/Vendor]. Release timeline conflicts: a blog says nano-0.1/0.2 in Aug 2025 and v0.8 series on 2026-05-03; an NYU page says Feb 19, 2026 — [tanxy blog](https://tanxy.typlog.io/2025/Kittentts); [NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/kittentts-state-of-the-art-voice-synthesis-in-under-25-mb/) [V-2nd, conflicting]. Early nano reported audible soft distortion and hiss — [tanxy blog](https://tanxy.typlog.io/2025/Kittentts) [V-2nd]. (Community inspection widely reports the small Kitten models are StyleTTS2/Kokoro-like with style-vector voice embeddings — unverified, gap.)

**SupertonicTTS (Supertone, 2025)** — 44M params in the paper: speech autoencoder (continuous latents) + flow-matching text-to-latent + utterance-level duration predictor; raw character input with cross-attention (no G2P, no external aligner); "comparable to contemporary zero-shot TTS" — [arXiv 2503.23108](https://arxiv.org/abs/2503.23108) [V]. Product release Supertonic / Supertonic 2 is 66M params, on-device, Supertonic 2 adds EN/KO/ES/PT/FR — [Supertone/supertonic-2 HF card](https://huggingface.co/Supertone/supertonic-2) [V-2nd/Vendor]. A ~99M "Supertonic 3" claim appears only on an aggregator — unverified.

**Pocket TTS (Kyutai)** — ~100M params, transformer + Mimi codec, CPU-capable, zero-shot cloning from ~3 s, EN/FR, MIT — only an aggregator page found — [tts.ai Pocket TTS](https://tts.ai/voices/pocket-tts/?lang=or) [V-2nd/Vendor; unverified].

**NeuTTS Air / Nano (Neuphonic)** — Air built on Qwen 0.5B LLM backbone (~360M active, ~552M incl. embeddings) + NeuCodec (50 Hz single codebook), GGUF Q4/Q8 for phones/RPi, 3 s voice cloning, English only; NeuTTS-Nano ~120M active / ~229M total, 45 tokens/s on Galaxy A25 CPU — [Voice AI Space listing](https://www.voiceaispace.com/knowledge/repo/57090f14-a2f1-417b-a9a2-989a7576879b); [Spheron guide](https://spheron.network/blog/neutts-air-spheron-voice-ai) (lists a conflicting 748M total) [V-2nd/Vendor]. Far above budget; included for context.

### Inferences
- For a <=10M acoustic model, the realistic architecture class is a FastSpeech2/FastPitch-style NAR transformer/conv (LightSpeech/EfficientSpeech-scale) or a slimmed Matcha (fewer U-Net channels) feeding a small iSTFT vocoder; VITS-family end-to-end at 5-7M (Piper x_low) is possible but quality is noticeably lower and 16 kHz.
- Matcha-TTS at 18.2M + Vocos (13.5M) lands at ~32M, essentially the stretch budget; a halved Matcha decoder + a tiny iSTFT head is likely needed to hit <=30M.
- Kokoro/StyleTTS2 are the best existence proof that a global style vector can carry voice + prosody in a small model; Kitten nano (15M) suggests the StyleTTS2 recipe compresses to ~15M, but this is unverified.

### Gaps
- No MOS/UTMOS/WER published for Piper tiers, Kitten nano/micro/mini, Kokoro (beyond Arena), Pocket TTS, or Supertonic 2 in the sources fetched.
- MB-iSTFT-VITS/MS-iSTFT-VITS exact parameter counts and MOS not fetched (see paper tables).
- No independent CPU-RTF benchmark across all candidates on the same hardware was found.

## Q2. Vocoders: params, quality, CPU speed

### Takeaway
Vocos (13.5M) is ~29x faster than HiFi-GAN on CPU (169.6x vs 5.8x real time) with equal or better quality; iSTFT-style heads are the clear choice for CPU/edge. For a total <=10M budget, HiFi-GAN V2 (~0.9M) / V3 (~1.5M), MB-MelGAN (~1.9M) or a shrunken Vocos/iSTFT head are the candidates.

### Cited findings
- Vocos paper (LibriTTS, ISTFT/ Fourier-domain GAN vocoder) — Table 1/2 and Table 6 — [arXiv 2306.00814 (ICLR 2024)](https://arxiv.org/html/2306.00814v3) [V]:

| Vocoder | Params | xRT GPU | xRT CPU | UTMOS | PESQ | ViSQOL | MOS |
|---|---|---|---|---|---|---|---|
| Vocos | 13.5M | 6696.5 | 169.6 | 3.734 | 3.70 | 4.66 | 3.62 |
| HiFi-GAN (V1) | 14.0M | 495.5 | 5.84 | 3.669 | 3.093 | 4.57 | 3.54 |
| iSTFTNet | 13.3M | 1045.9 | 14.44 | 3.564 | 2.942 | 4.56 | 3.57 |
| BigVGAN (base) | 14.0M | 98.6 | 0.40 | 3.749 | 3.693 | 4.65 | 3.64 |

  (Column mapping of Table 6 inferred from the text "~13x faster than HiFi-GAN and ~70x faster than BigVGAN on GPU" — fetch summary; consistent with ratios.)
- HiFi-GAN V1 13.92M, V2 0.92M, V3 1.46M params; LJSpeech MOS V1 4.36, V2 4.23, V3 4.05 (GT 4.45); V2/V3 many times faster than real time on CPU — [arXiv 2010.05646 (NeurIPS 2020)](https://arxiv.org/abs/2010.05646) [M]. Matcha used HiFi-GAN V1 = 13.9M [V](https://arxiv.org/html/2309.03199v2).
- iSTFTNet: replaces last upsampling layers with inverse STFT; faster and lighter than HiFi-GAN with comparable quality — [arXiv 2203.02395 (ICASSP 2022)](https://arxiv.org/abs/2203.02395) [M]; Kokoro uses an ISTFTNet decoder [V](https://huggingface.co/hexgrad/Kokoro-82M).
- Multi-band MelGAN: ~1.91M params, multi-band PQMF, very fast on CPU — [arXiv 2005.05106](https://arxiv.org/abs/2005.05106) [M].
- LPCNet: WaveRNN + linear prediction, ~3 GFLOPS, real-time on a single phone CPU core, but autoregressive per sample — [arXiv 1810.11846](https://arxiv.org/abs/1810.11846) [M].
- BigVGAN: base 14M, large 112M; best OOD robustness but slowest on CPU (0.40x RT in Vocos table → slower than real time) — [arXiv 2206.04658](https://arxiv.org/abs/2206.04658) [M]; CPU number [V](https://arxiv.org/html/2306.00814v3).
- APNet2 (amplitude+phase spectra prediction, ConvNeXt v2 backbone), WaveNeXt (ConvNeXt + linear waveform head, ASRU 2023) — both Vocos-like Fourier/linear-head designs claimed faster than HiFi-GAN at similar quality — [APNet2 arXiv 2311.11545](https://arxiv.org/abs/2311.11545) [M]; WaveNeXt (Okamoto et al., ASRU 2023) [M, no link verified].
- MB-iSTFT-VITS demonstrates the same iSTFT trick inside an end-to-end model: 4.1x faster than VITS, RTF 0.066 on i7 CPU — [arXiv 2210.15975](https://arxiv.org/abs/2210.15975) [V].

### Inferences
- Vocos at 13.5M fits the "stretch" budget only if the acoustic model is <=15M; for the strict <=10M total, use HiFi-GAN V2/V3-sized or a channel-reduced Vocos (Vocos's speed comes from no upsampling, so it shrinks gracefully — design inference, not benchmarked).
- Training the vocoder on the student's own predicted mels (fine-tuning) is standard practice to recover quality lost by small acoustic models.

### Gaps
- No published "Vocos-small" (<5M) quality numbers found this session.
- CPU numbers for HiFi-GAN V2/V3, APNet2, WaveNeXt on the same hardware as Vocos not obtained.

## Q3. Explicit fine-grained prosody & emotion-intensity control at small scale

### Takeaway
Explicit variance-adaptor architectures (FastSpeech 2, FastPitch, Daft-Exprt) give direct per-phone pitch/energy/duration targets and remain the most controllable small backbones; emotion intensity is typically injected as a scalar (relative-attribute ranking) or a vector in an emotion space (EmoSphere's spherical AVD). StyleTTS 2's predictors are also explicit (duration, F0, energy) but conditioned on a global style vector, making it the best hybrid of "global style + local override".

### Cited findings
- FastSpeech 2: variance adaptor with explicit duration, pitch (CWT), energy predictors; values can be replaced at inference — [arXiv 2006.04558](https://arxiv.org/abs/2006.04558) [M].
- FastPitch: per-input-symbol pitch prediction enabling interactive pitch editing (shifting/flattening/modulating contours per phone) — [arXiv 2006.06873](https://arxiv.org/abs/2006.06873) [M].
- Daft-Exprt: phoneme-, word- and utterance-level prosody encoding with FiLM conditioning, adversarial speaker disentanglement for cross-speaker prosody transfer — [arXiv 2108.02271 (Interspeech 2022)](https://arxiv.org/abs/2108.02271) [M].
- EmoSphere-TTS: emotion style and intensity as a spherical vector from arousal/valence/dominance pseudo-labels (Cartesian-to-spherical), dual conditional adversarial network; controllable style+intensity — [arXiv 2406.07803 (Interspeech 2024)](https://arxiv.org/abs/2406.07803v1) [V]; follow-up EmoSphere++ with CFM decoder [V-2nd](https://awesomepapers.io/speech-audio/papers/2406.07803). UDDETTS (2025) unifies discrete and dimensional emotions — [arXiv 2505.10599](https://fugumt.com/fugumt/paper_check/2505.10599v2_enmode) [V-2nd].
- Relative-attribute ranking for emotion intensity in FastSpeech2-style models (e.g., Zhu et al. 2019; MsEmoTTS multi-scale emotion) — [MsEmoTTS arXiv 2201.06460](https://arxiv.org/abs/2201.06460) [M].
- StyleTTS 2: style vector conditions duration, F0 and energy predictors (all explicit, editable) and the decoder — [arXiv 2306.07691](https://arxiv.org/abs/2306.07691) [M].
- SupertonicTTS uses only an utterance-level duration predictor (no phone-level variance targets) — [arXiv 2503.23108](https://arxiv.org/abs/2503.23108) [V] → poor fit for per-word control.
- Matcha-TTS has an explicit duration predictor (MAS alignment) but no explicit pitch/energy — [arXiv 2309.03199](https://arxiv.org/abs/2309.03199) [M]; VITS/Piper similarly only expose duration (length_scale, noise scales) [M].

### Inferences
- Recommended conditioning design for Whissle: FastSpeech2/StyleTTS2-style variance adaptor (phone-level duration, F0, energy; all overridable by word/phone targets) + global style vector (StyleTTS2/Kokoro-like 128-256-d) + emotion soft-vector and scalar intensity (EmoSphere-style AVD/spherical or relative-attribute scalar) fed via FiLM (Daft-Exprt) into encoder and variance predictors.
- Flow-matching or VITS decoders reduce direct pitch controllability unless explicit F0 conditioning is added.

### Gaps
- No paper found this session that demonstrates EmoSphere-style intensity control in a <=10M model; the published systems are FastSpeech2-scale (~30-40M).
- No quantitative comparison of controllability (e.g., F0 RMSE vs target) across small backbones was found.

## Q4. Distillation / compression evidence (small student from large expressive teacher)

### Takeaway
Data-level distillation (training a small model purely on large-TTS output) is now documented: a ~20.9M Matcha-TTS student trained on synthetic data from XTTS-v2/CosyVoice2/ChatTTS can match or beat real-data training, with speaker/text diversity and cleanliness mattering most. Kokoro-82M is a practical existence proof that a high-ranked model can be trained on a few hundred hours (partly synthetic per community reports) for ~$1k.

### Cited findings
- "Training TTS with Purely Synthetic Data" (SJTU, Dec 2025): Matcha-TTS student (~20.9M params, 100 h); teachers XTTS-v2 (750M, 16k h), CosyVoice2 (0.5B, 200k h), ChatTTS (3B, 40k h); higher speaker and text diversity significantly improve quality/robustness; cleaner data helps; *standard* speaking styles are learned more effectively than expressive ones; synthetic-trained models "have great potential to outperform" real-data training — [arXiv 2512.17356](https://arxiv.org/abs/2512.17356) [V via abstract/search summary; results tables not fetched].
- Related prior work TTS-by-TTS (using a high-quality TTS to augment target TTS data) — cited in [arXiv 2512.17356](https://arxiv.org/html/2512.17356v1) [V-2nd].
- Kokoro-82M training cost ~1,000 A100-hours, few hundred hours data — [Kokoro card](https://huggingface.co/hexgrad/Kokoro-82M) [V].
- FastSpeech 1/2 themselves were trained with teacher-distilled targets (FS1 used autoregressive teacher mels/durations; FS2 moved to GT targets to avoid distillation information loss) — [FastSpeech 2 arXiv 2006.04558](https://arxiv.org/abs/2006.04558) [M]. LightSpeech compresses FastSpeech via NAS — [arXiv 2102.04040](https://arxiv.org/abs/2102.04040) [M].

### Inferences
- A viable Whissle path: use a large prompt-steerable TTS as teacher to generate a balanced corpus over (emotion soft-vector, intensity, style) grid with forced-aligned durations and extracted F0/energy, then train a small explicit-variance student; the 2512.17356 finding that expressive styles are harder suggests oversampling expressive conditions and checking intensity monotonicity.
- Because the student uses explicit F0/energy/duration targets extracted from teacher audio, the teacher's prompt-steerability transfers as labelled control signals rather than needing feature-level KD.

### Gaps
- No paper found that distills a prompt-steerable (text-description) TTS specifically into a <=10M NAR student with emotion-intensity evaluation.
- Exact quality deltas (MOS/WER) in 2512.17356 not retrieved.

## Q5. Streaming / latency considerations

### Takeaway
NAR mel models + iSTFT vocoders give the lowest CPU latency for full-sentence synthesis (Vocos 170x RT on CPU; EfficientSpeech 104x RT mel on RPi4), but they are not natively streaming; chunked vocoding works well, while VITS/flow models need full-utterance latent sampling. LLM-codec tiny models (NeuTTS, Pocket TTS, KittenTTS 2) stream natively but are 100M-1.7B.

### Cited findings
- Vocos 169.6x RT on CPU vs HiFi-GAN 5.8x, BigVGAN 0.4x — [arXiv 2306.00814](https://arxiv.org/html/2306.00814v3) [V].
- EfficientSpeech RTF 104.3 on RPi4 (mel only) — [arXiv 2305.13905](https://arxiv.org/abs/2305.13905) [V].
- MB-iSTFT-VITS RTF 0.066 on i7 CPU (end-to-end) — [arXiv 2210.15975](https://arxiv.org/abs/2210.15975) [V].
- Matcha-TTS 2-step RTF 0.015 GPU incl. vocoder; ODE step count is a direct latency/quality knob (MOS 3.65 @2 steps vs 3.84 @10) — [arXiv 2309.03199 Table 1](https://arxiv.org/html/2309.03199v2) [V].
- KittenTTS 2 streaming on A100: first 21 s of audio ready after 17 s for a 936-char passage — [KittenML GitHub](https://github.com/KittenML/KittenTTS) [Vendor] (i.e., the 1.7B model is not low-latency).
- NeuTTS-Nano 45 tokens/s on Galaxy A25 CPU (codec at 50 Hz → slower than real time) — [Voice AI Space](https://www.voiceaispace.com/knowledge/repo/57090f14-a2f1-417b-a9a2-989a7576879b) [Vendor/V-2nd].

### Inferences
- For a sub-100 ms time-to-first-audio on CPU, the NAR acoustic model can run on the full sentence (a few ms at <=10M params) and the vocoder can stream chunk-wise; per-word control is easiest in this setup because all variance targets are known before decoding.
- 45 tokens/s at 50 Hz codec implies ~0.9x real time for NeuTTS-Nano on a mid-range phone — LLM-codec tiny models are not yet competitive with NAR+iSTFT on edge CPUs (derived arithmetic).

### Gaps
- No same-hardware CPU latency comparison (time-to-first-audio) across Kokoro, Kitten nano, Piper, Matcha+Vocos found.
- Kokoro/Kitten official CPU RTF numbers not published in sources fetched.
