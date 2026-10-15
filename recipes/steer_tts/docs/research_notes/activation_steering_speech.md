# Activation Steering, Steering Vectors and Lightweight Inference-Time Control for TTS / Speech LMs (2023–2026)

Notes compiled 2026-10-07. Primary sources are arXiv HTML pages read in full or in part. Anything known only from an aggregator or abstract is flagged **[abstract-only]** or **[aggregator-only]**.

---

## Q1. How are steering vectors computed and applied in speech generation models (EmoSteer-TTS, CoCoEmo, follow-ups, speech LLMs)? Which layers/tokens, how is strength set, how are vectors composed, what are the failure modes?

### Takeaway
Every TTS steering paper so far builds its vector as a **mean difference of activations between emotional and neutral speech**, usually with same-speaker or same-text pairs. The vector is added as `h + α·v`, then the activation is **renormalized to its original L2 norm**. Composition is a weighted sum of per-emotion vectors, and erasure is projection removal. In hybrid LM+flow-matching TTS, steering the **LM attention output at mid layers** keeps speaker identity much better than steering the flow-matching DiT. Large α (about 3 for DiT steering, about 6 for LM steering) hurts intelligibility.

### Cited Findings

**EmoSteer-TTS (Xie, Yang, Li, Yu, Liu; HKUST-GZ + Tencent AI Lab; arXiv 2508.03543, Aug 2025, v3 Oct 2025)**: [arXiv](https://arxiv.org/abs/2508.03543), [HTML v3](https://arxiv.org/html/2508.03543v3), [demo](https://emosteer-tts-demo.pages.dev/)
- The method is training-free and plugs into flow-matching TTS (F5-TTS, E2-TTS, CosyVoice2). It supports emotion conversion, interpolation, erasure, and replacement. The pipeline has three stages: (1) activation extraction, (2) emotional token search, (3) inference-time steering. — [HTML v3](https://arxiv.org/html/2508.03543v3)
- Vector per layer l: `u^l = mean_j x^l_emotional,j − mean_i x^l_neutral,i`, then L2-normalized. Activations are resampled with nearest interpolation to the average sequence length, so the vector has shape [avg_seq_len, hidden_dim] and is **token/frame-specific**, not one broadcast vector. The steering site is "the first residual stream" of each DiT block. — [HTML v3](https://arxiv.org/html/2508.03543v3)
- Emotional token search: each token index is perturbed in turn, the sample is synthesized, and the target-emotion probability is read from **emotion2vec**. The top-k tokens are kept (k=200), all other tokens are zeroed, and the kept tokens are weighted by a softmax over their emotion probabilities. All steered layers share the same token indices, which reduces search cost to O(avg_seq_len). — [HTML v3](https://arxiv.org/html/2508.03543v3)
- Application formulas:
  - Steer: `x̂ = f_r(x + α·ŝ)` with `f_r = ‖x‖/‖x+ŝ‖` (norm-preserving).
  - Erase: `x̂ = f_r(x − β(ŝ·x)ŝ)`.
  - Replace: erase emo1, then add α·ŝ_emo2.
  - Multi-emotion: `x̂ = f_r(x + Σ_e α_e ŝ_e)`.
  - α<0 steers away from an emotion. The hooks are forward pre-hooks. — [HTML v3](https://arxiv.org/html/2508.03543v3)
- Layers and steps:
  - F5-TTS (22 DiT layers): layers 1, 6, 11, 16, 21.
  - E2-TTS (8 layers): every 3rd layer.
  - CosyVoice2 flow-matching DiT (56 layers): every 5th layer.
  - Steering is applied at all CFM steps (32 for F5/E2, 10 for CosyVoice2). — [HTML v3](https://arxiv.org/html/2508.03543v3)
- Ablations (qualitative, no numbers given):
  - Middle layers (8–14 of 22) are slightly stronger than shallow layers, and deep layers (15–22) decline. Spaced multi-layer steering is best.
  - Early flow steps (0–10) have little effect. Middle and late steps matter more. All steps together is strongest.
  - Effect plateaus at k≈200.
  - α=3 can produce unintelligible speech. — [HTML v3](https://arxiv.org/html/2508.03543v3)
- Steering data: 6,900 utterances from 11 corpora (MSP-Podcast, IEMOCAP, RAVDESS, CREMA-D, TESS, SAVEE, ASVP-ESD, CASIA, M3ED, ESD, Emo-Emilia). It covers 6 emotions plus neutral, about 1k samples per emotion, in English and Chinese. Utterances were filtered by duration, silence, SNR, and emotion2vec confidence ≥0.6. — [HTML v3](https://arxiv.org/html/2508.03543v3)

**CoCoEmo (Wang, Tan, Liu, Jia, Huang, Bailey, Dang; arXiv 2602.03420, Feb 2026; ICML 2026)**: [HTML](https://arxiv.org/html/2602.03420v1), [ICML](https://icml.cc/virtual/2026/poster/60804)
- Targets hybrid TTS (CosyVoice2: Qwen2-based 24-layer SLM, DiT flow-matching, HiFi-GAN; IndexTTS2: GPT2-style 24-layer SLM, DiT, BigVGANv2). A cross-conditioning test on CosyVoice2 (N=300) shows emotional prosody variation comes mainly from the **SLM**, not the flow decoder:
  - F0 CCC: 0.109 (SLM-driven) vs 0.305 (flow-driven).
  - Energy CCC: 0.308 vs 0.737. — [HTML](https://arxiv.org/html/2602.03420v1)
- Vector: mean difference of **last-token** activations, emotional minus neutral. Data is 20,691 utterances from ESD + RAVDESS + CREMA-D with speaker-independent splits. Pairs share speaker and, where possible, transcript. — [HTML](https://arxiv.org/html/2602.03420v1)
- Site selection by linear probes over layer × operation. **Attention output** was best.
  - CosyVoice2: layers 10–17 are most separable, and steering uses layers 17 and 14.
  - IndexTTS2: steering uses layers 6, 8, and 1. Steering layer 0 severely degraded intelligibility. — [HTML](https://arxiv.org/html/2602.03420v1)
- Composition: the mixed vector is a weighted sum of single-emotion vectors, with weights summing to 1 (taken from multi-rater label distributions). Injection is `h + α v`, renormalized. α∈[0,4.5] was stable, and α up to 6 may reduce intelligibility. — [HTML](https://arxiv.org/html/2602.03420v1)
- Failure modes:
  - On IndexTTS2, explicit emotion conditioning can override a last-token-only perturbation.
  - Combining steering with an instruction prompt dropped N-MOS to 2.18–3.00 in some settings.
  - On IEMOCAP, adding steering to emotion conditioning slightly lowered E-SIM. — [HTML](https://arxiv.org/html/2602.03420v1)

**A Geometric Perspective on Composable Emotion Steering in TTS (Wang, Bailey, Dang; arXiv 2607.00946, 2026)**: [HTML](https://arxiv.org/html/2607.00946v1)
- Setup: CosyVoice2. The SLM is 24 layers with d=896 and is steered at attention outputs, last token, layers 14 and 17. The CFM is a 56-layer DiT with d=256 and 10 steps, steered on the residual stream at every 5th layer (12 layers) with top-k frame masking.
- **Speaker entanglement**, from linear-probe accuracy (within-speaker / cross-speaker):
  - SLM: 0.80 / 0.71 (gap 0.08).
  - CFM: 0.89 / 0.62 (gap 0.32).
  - Local intrinsic dimensionality is about 28 for the SLM and about 13 for the CFM. ΔLID suggests distinct per-emotion subspaces in the SLM but a single shared acoustic manifold in the CFM.
- Steering speaker similarity on CREMA-D: CFM steering (α=2) drops S-SIM from 0.871 to 0.807, while SLM steering (α=5) keeps it at 0.870.
- Joint steering of SLM and CFM does **not** add up. It lowers rank correlation on CREMA-D (0.176 vs 0.209 for SLM alone) and raises WER compared with single-site steering.
- Recommendations: orthogonalize CFM vectors against speaker directions, tune α per site, and use frame-level adaptive α.

**EmoRES-TTS (Huang, Liu, Peng, Wu, Ni, Lee, Lee, Chachra; arXiv 2609.38157, Sep 2026)** **[aggregator-only; arXiv ID and details not confirmed on arxiv.org]**: [OpenTrain page](https://www.opentrain.ai/papers/emores-tts-residual-enhanced-vector-steering-for-emotional-speech-generation--arxiv-2609.38157/), [deeplearn.org](https://deeplearn.org/arxiv/834635/emores-tts:-residual-enhanced-vector-steering-for-emotional-speech-generation)
- Splits each emotion vector into a **shared component**, which moves speech "away from neutral" (common to all emotions), and an **emotion-specific residual**. The two are weighted separately.
- Reported gains over CoCoEmo on IEMOCAP: rank correlation +26.13 points on IndexTTS-2 and +12.97 points on CosyVoice2. Hit rate +12.95 / +6.92 points. Human identification of the dominant emotion up to 35% better in relative terms.
- Ablation: keep the shared component and amplify the residual. The exact decomposition formula is not stated on the page. One plausible reading is projection onto the mean of all emotion vectors, but **that is unverified**.

**EmoShift (Zhou, Jiang, Li, Wang, Haizhou Li; arXiv 2601.22873, Jan 2026)** **[abstract-only]**: [arXiv](https://arxiv.org/abs/2601.22873v1)
- Not training-free. An "EmoSteer layer" **learns** one steering vector per emotion in the output embedding space of an LLM-based TTS, with about 10M trainable parameters (under 1/30 of full fine-tuning).
- Claims to beat zero-shot and fully fine-tuned baselines while keeping naturalness and speaker similarity. Analysis suggests the vector scale can control intensity.

**Accent steering**
- Accent-neutralized zero-shot TTS (Mu Yang, John H. L. Hansen; arXiv 2603.05977, 2026) **[abstract-only]**: layer-specific vectors are computed offline from activation differences between accented and native speech, then added at inference. The authors report reduced accent, generalization to unseen accented speakers, and preserved timbre, measured by speaker cosine and by speaker-verification acceptance in an "accent-orthogonal subspace". No numbers or base model were visible in the abstract. — [arXiv](https://arxiv.org/abs/2603.05977)
- Accent adaptation in large audio LMs (ASR side): arXiv 2603.05813, title only. — [arXiv PDF](https://arxiv.org/pdf/2603.05813)

**Speech LLMs / audio-LMs**
- Neuron-Level Emotion Control in Speech-Generative LALMs (arXiv 2603.17231, 2026) **[abstract-only]**:
  - Models: Qwen2.5-Omni-7B, MiniCPM-o 4.5, Kimi-Audio.
  - A compact set of "emotion-sensitive neurons" is found by aggregating activations over *successful* generations only (target emotion realized and content preserved). These neurons are then intervened on at inference.
  - Controllability depends on mask sparsity, filtering criteria, and intervention strength. Gains generalize to unseen speakers.
  - Abstract motivation: LALM emotion conversion currently suffers refusals, hallucinations, and paraphrasing. — [arXiv](https://arxiv.org/abs/2603.17231)
- SALSA (arXiv 2606.00460) learns steering activation vectors for adapting speech-aware LLMs (Qwen2-Audio-7B-Instruct, Granite-Speech-3.3-8B). It is for understanding tasks, not generation. — [HTML](https://arxiv.org/html/2606.00460)

### Inferences
- The consistent recipe is: paired mean-difference → L2 normalize → add with α → renormalize to the original norm → apply at spaced mid layers. The norm-preservation step looks important for keeping quality at moderate α; every paper uses it.
- The useful α window is narrow and model-specific. The main failure mode is intelligibility collapse at high α, followed by speaker drift when the steering site carries acoustic or timbre information (DiT/CFM).
- Emotion appears to be linearly decodable and steerable in all architectures studied, from about 0.3B flow DiTs up to 7B omni-LLMs. No study reports results on a sub-100M model.
- Classifier-guided token selection (EmoSteer) adds an emotion2vec dependency and an expensive search. EmoSteer's E-SIM may be partly optimized toward emotion2vec; the authors check with SenseVoice for this reason.

### Gaps
- None of these papers tested steering on small (≤30M) TTS models. Whether linear emotion directions emerge in tiny models is unknown.
- EmoRES-TTS: the decomposition formula and the arXiv record were not verified on arxiv.org.
- Accent-steering and neuron-level LALM papers: numeric results were not retrieved (abstract only).
- I found no paper that steers speaker identity or age directly via activations in TTS. Gender steering exists only via SAE features (see Q3) and prompt directions (see Q4).

---

## Q2. Quantitative results: emotion accuracy, intensity monotonicity, WER, speaker similarity, MOS

### Takeaway
Training-free steering reaches emotion similarity comparable to trained label- or description-based emotional TTS, at a modest cost:
- WER rises by about 0.5–0.65 absolute points (F5-TTS: 2.14 → 2.79).
- Speaker similarity drops by about 0.02–0.05 (more for DiT/CFM steering).
- Naturalness MOS drops by about 0.5 on F5-TTS.

Intensity is monotone in α over a limited range. Proportional mixing is still weak: Spearman ρ is about 0.2 for CoCoEmo and higher with EmoRES.

### Cited Findings

EmoSteer-TTS, in-distribution (α=2.0, k=200), from [HTML v3](https://arxiv.org/html/2508.03543v3):

| System | WER | S-SIM | E-SIM | N-MOS | EI-MOS |
|---|---|---|---|---|---|
| F5-TTS unsteered | 2.14 | 0.66 | 0.055 | 3.79 | – |
| EmoSteer + F5-TTS | 2.79 | 0.64 | 0.275 | 3.29 | 4.00 |
| EmoSteer + E2-TTS | 3.28 | 0.59 | 0.280 | 3.31 | 3.38 |
| EmoSteer + CosyVoice2 | 2.83 | 0.65 | 0.275 | 3.65 | 3.56 |
| CosyVoice2 (description-prompted) | 2.53 | 0.73 | 0.255 | 3.69 | – |
| EmoSphere++ (label) | 16.25 | 0.44 | 0.245 | 3.23 | 3.50 |

- On OOD data (SeedTTS/EMNS), F5 + EmoSteer gives WER 2.65, S-SIM 0.65, and N-MOS 3.58. MOS standard deviations are about ±0.8–1.2.
- Composition shifts are small in absolute terms. Steering F→H moves happiness +0.28 and fear −0.33. A Sad+Surprise mix yields sadness 0.28 and surprise 0.42. — [HTML v3](https://arxiv.org/html/2508.03543v3)

CoCoEmo, mixed emotion, CREMA-D, CosyVoice2, from [HTML](https://arxiv.org/html/2602.03420v1):
- No steering: E-SIM 0.743, TEP 0.065, WER 1.07, N-MOS 4.11.
- CoCoEmo (α=5): E-SIM 0.779, TEP 0.149, ρ 0.209, H-Rate 0.724, WER 0.78, N-MOS 3.96.
- High text–emotion mismatch (IEMOCAP), CosyVoice2 CoCoEmo α=6: TEP 0.504 vs 0.436 for instruction prompting and 0.197 with no steering. WER 8.74 vs 4.01 for instruction prompting.
- So steering beats instruction prompting on emotion but costs intelligibility here.

EmoKnob (speaker-embedding steering on MetaVoice-1B, α=0.4), from [HTML](https://arxiv.org/html/2410.00316v1):
- Emotion Selection Accuracy 86% and Emotion Enhancement Accuracy 93%, against a 50% chance level.
- WER 0.085 vs 0.079 uncontrolled; SIM 0.705 vs 0.719.
- SIM and WER degrade as α grows. More reference pairs make the method robust to larger α.
- Weak cases: Contempt discrimination 13%; empathy enhancement vs commercial systems 22%.

SAE emotion steering on IndexTTS2, from [HTML](https://arxiv.org/html/2606.01479):
- SAE-Emotion vs global mean-difference steering, Emo-SIM: anger 0.912 vs 0.910, happiness 0.885 vs 0.879, sadness 0.880 vs 0.876.
- Human evaluation: EMOS 3.22 vs 3.10, NMOS 3.49 vs 3.38.
- Single-latent scaling moves emotion2vec similarity monotonically from about 0.77 (α=−60) to 0.86 (α=+60).

Geometric study (CosyVoice2), from [HTML](https://arxiv.org/html/2607.00946v1):
- CFM steering: S-SIM 0.807. SLM steering: S-SIM 0.870, against a 0.871 baseline.
- Rank correlation ρ for proportional mixing stays at 0.12–0.22 across configurations.

### Inferences
- E-SIM numbers are not comparable across papers. EmoSteer's E-SIM (~0.27) and CoCoEmo's (~0.78) use different definitions or normalizations.
- Steering raises emotion strongly from a near-neutral base (EmoSteer: 0.055 → 0.275). Gains are small when the base model is already emotion-conditioned (CoCoEmo: 0.743 → 0.779).
- Proportional or mixed control is still the weakest area: ρ ≈ 0.2 in CoCoEmo, and up to about 0.48 if EmoRES's +26 points are taken at face value.

### Gaps
- No paper reports a formal intensity-monotonicity statistic (e.g., Spearman of α vs perceived intensity across a sweep). Mostly figures are shown.
- EmoSteer's α-sweep values are given only as figures.

---

## Q3. Related latent-space control: speaker-embedding directions (EmoKnob), spherical AVD (EmoSphere), diffusion mixing (EmoMix), relative-attribute intensity, guidance-based control, and SAE interpretability

### Takeaway
Before activation steering, the main approach was to steer an **explicit conditioning embedding**: a speaker embedding (EmoKnob), an emotion embedding scaled by a scalar or ranked by relative attributes, or an AVD vector in spherical coordinates (EmoSphere). Other methods blend score or velocity predictions (EmoMix, CFG variants). SAEs are emerging as a way to get sparser, more separable steering directions than dense mean differences, but so far the gains over mean differences are marginal.

### Cited Findings
- **EmoKnob (Chen, Chen, Hirschberg; Columbia; EMNLP 2024; arXiv 2410.00316)**:
  - Direction: `v_e = mean_i (u_e^i − u_n^i)/‖u_e^i − u_n^i‖` in the speaker-embedding space of a pretrained voice-cloning model (MetaVoice-1B). A single same-speaker pair suffices for simple emotions.
  - Applied as `u_s + α v_e`, with α=0.4 to 0.5.
  - Open-ended emotions are handled by retrieval or by synthetic pairs from emotional-text TTS (10 pairs).
  - Framed as working for any embedding-conditioned cloning model. — [HTML](https://arxiv.org/html/2410.00316v1), [project](https://emoknob.cs.columbia.edu/)
- **EmoSphere-TTS (Cho et al.; Korea Univ. / Ajou; Interspeech 2024; arXiv 2406.07803)**: pseudo-labels arousal/valence/dominance with an SER model, then converts AVD from Cartesian to spherical coordinates. Angles encode emotional style and the radius encodes intensity. A dual conditional adversarial network improves quality. MOS figures (nMOS 3.88, sMOS 3.48) come from a secondary summary and are **unverified**. — [arXiv](https://arxiv.org/html/2406.07803v2), [alphaXiv overview](https://alphaxiv.org/overview/2406.07803v2)
- **EmoMix (arXiv 2306.00648, Interspeech 2023)**: a diffusion TTS with SER-derived emotion embeddings. Mixed emotion comes from combining noise predictions conditioned on different emotions in a single sampling run. Intensity comes from mixing Neutral with the target emotion in varying degrees. — [arXiv](https://arxiv.org/pdf/2306.00648)
- **Relative attribute ranking (RAR)** is the prevailing training-time intensity labeler. A max-margin (SVM) ranking over prosodic features from neutral/emotional pairs gives a scalar intensity that is fed as an auxiliary input. RAR-weighted autoregressive models have been reported to suffer quality degradation. Simpler alternatives scale the emotion embedding by a scalar at inference, or interpolate in emotion-embedding space (Um et al.). — [KazEmoTTS related-work summary](https://arxiv.org/pdf/2404.01033)
- **Guidance-based, inference-time control**:
  - "Rectifying the Emotional Flow" (ACL 2026): uniform CFG distorts the acoustic manifold on high-arousal speech, and a Gaussian prior biases flow-matching TTS toward neutral prosody. It proposes Likelihood-Inverse Guidance (dynamic λ(t), stronger when the trajectory drifts neutral) plus an emotion-rectified noise prior. It is described as pure inference-time and applicable to any flow-matching TTS. — [ACL Anthology](https://preview.aclanthology.org/ingest-acl/2026.acl-long.998/)
  - Mismatch-aware / cross-modal-consistency CFG (arXiv 2510.13293): scales CFG by text–style emotion mismatch on CosyVoice2, then distills it to avoid the cost of double decoding. — [arXiv](https://arxiv.org/pdf/2510.13293)
- **SAE work**:
  - Du et al. (William & Mary; ICML 2026; arXiv 2606.01479):
    - Top-k SAE (1280→4096 latents, k=32, about 10.5M params) on the layer-16 pre-LN residual of the IndexTTS2 semantic backbone, trained on 56k generations.
    - Selects the top 6 latents per emotion by paired emotion-minus-neutral firing frequency, with no overlap across emotions. Steers by additive offsets on those latents, decoded back to the residual.
    - Emotion is distributed across several sparse latents. One happiness latent raises F0 by about 23 Hz and RMS energy, with no change in duration.
    - A wider SAE (10,240) fragmented the emotion features.
    - Gains over dense mean-difference steering are small (often under 0.01 Emo-SIM).
    - [HTML](https://arxiv.org/html/2606.01479), [PMLR](https://proceedings.mlr.press/v306/du26i.html)
  - Koriagin, Aparin, Balagansky, Gavrilov (arXiv 2606.10029): SAEs on the CosyVoice3 LM backbone find features for phonemes, laughter, accent prompts, and speaker gender. Steering layer-20 features causally changes laughter, gender, and speech rate. **[abstract/snippet-only]** — [arXiv PDF](https://arxiv.org/pdf/2606.10029)
  - AudioSAE (arXiv 2602.05027): SAEs on Whisper/HuBERT, with steering used to reduce Whisper hallucination. Not TTS. — [HTML](https://arxiv.org/html/2602.05027)

### Inferences
- EmoKnob is the closest prior art to "steering in an explicit low-dimensional control space". It works with a single sample pair because the speaker embedding is already a compact bottleneck with emotion partly entangled. Its cost is a small drop in speaker similarity (0.719 → 0.705 at α=0.4).
- EmoSphere shows the alternative: a **designed** control space (AVD sphere), where intensity is the radius and is monotone by construction rather than discovered after the fact.
- SAE features give interpretable, sign-symmetric knobs, such as raising F0 without changing duration. They cost an extra trained module about the size of an entire small TTS (10.5M params), which matters at ≤30M.

### Gaps
- I did not retrieve a primary source on StyleTTS2 style-vector arithmetic. Community demos exist, but I found no peer-reviewed quantitative evaluation in this search.
- EmoDubber's positive/negative flow guidance was seen only in a secondary summary (emergentmind) and is not verified.

---

## Q4. Steering via text-prompt embeddings (prompt arithmetic, CLAP-style directions)

### Takeaway
Direction vectors between **contrasting style-prompt embeddings** work for inter-utterance interpolation in prompt-based TTS, with near-perfect gender flips and sizable pitch and rate shifts. Within-utterance changes need extra decoding tricks in autoregressive models because early tokens dominate. Calibrated "intensity" via prompt arithmetic has not been evaluated.

### Cited Findings
- **Kang, Lee, Park, Shim (arXiv 2605.27376, May 2026), "Unlocking Fine-Grained and Within-Utterance Speaking Style Control in Prompt-Based TTS"** **[abstract-only]**:
  - Builds direction vectors between contrastive style prompts in embedding space and interpolates along them.
  - Results: 99–100% gender conversion, up to 36 Hz pitch shift, up to 1.6 syllables/s speed change.
  - For within-utterance control, KV-cache swapping and sliding-window attention masking counter the early-token attention bias. Intra-utterance speaker similarity is 0.81–0.91 and smoothness 3.48–4.48. — [papers.cool](https://papers.cool/arxiv/2605.27376)
- **HiStyle (arXiv 2509.25842)**: in text-prompt TTS, predicted style embeddings cluster first by timbre and then by finer style. It uses a two-stage hierarchical predictor plus text–audio contrastive alignment. There is no arithmetic, but it shows the geometry of the prompt-derived style space. — [arXiv](https://arxiv.org/pdf/2509.25842)
- **EmoKnob's text-to-emotion extensions** take an open-ended text description and either retrieve sample pairs or synthesize them with emotional-text TTS, then build the direction in speaker-embedding space. — [HTML](https://arxiv.org/html/2410.00316v1)

### Inferences
- If a small TTS has a text-prompt encoder that projects into a style bottleneck, then `z = E(p_neutral) + α[E(p_target) − E(p_neutral)]` is the cheapest prompt-steering knob. Kang et al. suggest it transfers reasonably for global attributes (gender, pitch, rate).
- Contrastive text–audio alignment, as in HiStyle, probably makes prompt-direction arithmetic land in regions the decoder has seen. This is inferred, not tested.

### Gaps
- I found no paper evaluating CLAP-direction vectors specifically for TTS emotion intensity with SER/WER metrics.

---

## Q5. Practical recipe for a small (≤10–30M) NAR / flow-matching TTS with an explicit style bottleneck: is steering more robust in a low-dimensional control space than in hidden activations?

### Takeaway
The evidence points toward an explicit low-dimensional style/control space as the more robust steering site for a small model. Three findings support this:
- Steering in hidden flow/DiT activations entangles emotion with speaker and costs speaker similarity and naturalness.
- The cleanest steering sites in large models are compact, "semantic", pre-acoustic representations (SLM attention outputs, speaker or style embeddings).
- Dense mean-difference directions already capture most of the effect. Sparse features add little.

This recommendation is an inference. **No paper directly compares the two sites in a small model.**

### Cited Findings (supporting evidence)
- The flow-matching DiT shares one acoustic manifold across emotions and is speaker-entangled: the cross-speaker probe gap is 0.32 vs 0.08 for the SLM, and CFM steering drops S-SIM from 0.871 to 0.807. The SLM, a more abstract layer, keeps S-SIM at 0.870. — [2607.00946](https://arxiv.org/html/2607.00946v1)
- In CosyVoice2, emotional prosody is driven mainly by the SLM, not the flow decoder. — [CoCoEmo](https://arxiv.org/html/2602.03420v1)
- Steering a compact conditioning embedding (EmoKnob) works from 1–10 pairs, with SIM falling only 0.719 → 0.705 at α=0.4. — [EmoKnob](https://arxiv.org/html/2410.00316v1)
- Designed spherical AVD control gives intensity as an explicit radius. — [EmoSphere-TTS](https://arxiv.org/html/2406.07803v2)
- Learned per-emotion steering offsets (EmoShift, about 10M params) beat fine-tuning in an LLM-TTS. This suggests that "steering vectors" can also be *trained* as lightweight residual offsets. — [EmoShift](https://arxiv.org/abs/2601.22873v1)
- Separating a shared "away-from-neutral" (arousal-like) component from an emotion-specific residual improves mixing control. — [EmoRES, aggregator-only](https://www.opentrain.ai/papers/emores-tts-residual-enhanced-vector-steering-for-emotional-speech-generation--arxiv-2609.38157/)
- DiT steering: shallow layers are weak, middle layers strongest, deep layers decline. Early flow steps barely matter, and middle and late steps matter more. — [EmoSteer](https://arxiv.org/html/2508.03543v3)
- Uniform CFG on flow-matching distorts high-arousal speech, and adaptive time-dependent guidance helps. — [ACL 2026](https://preview.aclanthology.org/ingest-acl/2026.acl-long.998/)

### Inferences (design recipe, derived from the above, not tested at small scale)
1. **Primary knob is an explicit style bottleneck** z (e.g., 16–128-d) feeding a NAR or flow-matching decoder through FiLM/AdaLN. Compute emotion directions as same-speaker, same-text mean differences in z: `d_e = mean(z_emo) − mean(z_neu)`. Apply `z' = z + α·d_e`, optionally renormalizing ‖z'‖ to ‖z‖. This is EmoKnob/CoCoEmo transplanted to z.
2. **Split each direction** into a shared "arousal / away-from-neutral" axis (the mean over emotions of d_e) and per-emotion residuals, following EmoRES. Expose intensity as the coefficient on the shared axis and category as the residual. This resembles EmoSphere's radius/angle split.
3. **Orthogonalize emotion directions against speaker directions** in z, for example by projecting out the top PCs of the speaker-mean variance. This targets the speaker drift seen in CFM steering (recommended in 2607.00946).
4. **Make the prompt encoder land in the same z space** (contrastive text↔audio alignment, as in HiStyle). Text-prompt arithmetic `E(p_target) − E(p_neutral)` then gives the same kind of direction as audio-derived mean differences, and both can be mixed.
5. **Hidden-activation steering as a secondary knob**: if used, apply it at spaced mid layers and mid-to-late flow steps, with norm preservation, and sweep α conservatively. Expect intelligibility loss around α≈3 (in EmoSteer's scaling). Prefer steering the text/duration encoder side, which is closer to the "SLM" analogue, over late acoustic decoder layers.
6. **Optional guidance**: train with condition dropout so emotion CFG `v = v_uncond + w(v_cond − v_uncond)` is available. A time-dependent w(t) is likely better than a constant w for high-arousal speech. Note that CFG doubles inference cost unless distilled.
7. **Evaluation protocol** to mirror the literature:
   - Emotion: emotion2vec E-SIM and target-emotion probability, plus a second classifier (SenseVoice) to avoid metric overfitting.
   - Intensity: Spearman ρ between α and emotion probability or perceived intensity.
   - Intelligibility: WER with Whisper.
   - Speaker: S-SIM with WavLM-TDNN.
   - N-MOS / EI-MOS.
   - Sweep α ∈ {−2…+4}.

### Gaps
- No paper tests whether ≤30M-parameter TTS models develop linearly steerable emotion directions in their hidden activations. Small models may have more entangled, lower-dimensional manifolds, consistent with the low LID of CosyVoice2's 256-d CFM, but this is speculation.
- No direct head-to-head between bottleneck steering and activation steering on the same model was found.
- Background LLM-steering references (Turner et al. ActAdd, Rimsky et al. CAA, Zou et al. RepE) were not re-fetched in this session. The mean-difference + α formulation above matches what the speech papers describe.
