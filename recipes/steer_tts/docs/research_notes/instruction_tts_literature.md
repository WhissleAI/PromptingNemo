# Natural-language prompt / instruction-controlled TTS (2022–2026): architectures, prompt encoding, data, generalization, small models

Verification legend used throughout:
- **[V]** = verified this session against the primary source (arXiv abstract/HTML, vendor docs) or a search snippet quoting it.
- **[M]** = from the researcher's prior knowledge of the paper. The arXiv ID/link is given, but the specific number/detail was NOT re-checked this session. Treat it as likely correct but confirm before quoting in a design doc.
- **[U]** = uncertain or conflicting; flagged explicitly.

---

## Q1. Per-system survey: backbone, prompt encoder, injection point, params, data, attributes, metrics

### Takeaway
The field moved in three generations. (1) 2022–23: non-autoregressive (FastSpeech/VITS) or diffusion acoustic models conditioned on a BERT/RoBERTa-encoded style prompt, with a fixed attribute set of about 5 attributes (gender, pitch, speed, volume, emotion). (2) 2023–24: codec language models (VALL-E/MusicGen-style) with a frozen T5 cross-attention encoder or the prompt as a prefix (Parler-TTS, Salle, VoxInstruct). (3) 2024–26: LLM-backbone TTS (Qwen-0.5B and up) plus flow matching, where the instruction is simply LLM text tokens (CosyVoice 2, EmoVoice, Spark-TTS, Step-Audio-EditX, Inworld TTS-1/2, commercial). Only generation 1 is anywhere near the ≤30M budget.

### Cited Findings

**Academic systems**

- **PromptTTS (Guo et al., Microsoft, ICASSP 2023), arXiv 2211.12171** — https://arxiv.org/abs/2211.12171 [M]
  - Structure: a BERT-based *style encoder* turns the prompt into a style embedding, with auxiliary per-attribute classification heads. There is a separate content encoder and a FastSpeech-like NAR mel decoder plus vocoder. The style embedding is added to the content-encoder hidden states (global conditioning, no cross-attention).
  - Attributes: 5 (gender, pitch, speaking speed, volume, emotion).
  - Data: a new "PromptSpeech" dataset of prompts written for those attributes.
  - Reported metric: per-attribute accuracy measured by classifiers on the generated speech.
  - On InstructTTSEval (EN, Gemini judge) it scores APS 64.3 / DSD 47.2 / RP 31.4 / Avg 47.6 [V] (https://arxiv.org/html/2506.16381v1).
- **PromptTTS 2 (Leng et al., Microsoft, ICLR 2024), arXiv 2309.02285** — https://arxiv.org/abs/2309.02285
  - Adds a **variation network** that predicts the reference-speech representation from the text-prompt representation, to capture voice variability the prompt does not describe [V].
  - Adds a **prompt generation pipeline**: a speech language understanding (SLU) model recognises attributes (e.g. gender, speed) from speech, and an LLM writes text prompts from those labels [V].
  - Trained on 44K hours [V].
  - Backbone is a NaturalSpeech 2-style latent diffusion TTS. The variation network is itself a diffusion model over prompt-encoder query tokens. Attributes are gender, speed, volume and pitch [M].
  - The abstract gives no accuracy numbers, only "more consistent with text prompts" [V].
- **InstructTTS (Yang et al., Tencent, 2023, IEEE/ACM TASLP), arXiv 2301.13662** — https://arxiv.org/abs/2301.13662 [M]
  - Three-stage training: a RoBERTa-based style prompt encoder, aligned to a speech style encoder by cross-modal metric learning, conditions a **discrete diffusion** model over VQ acoustic tokens.
  - Data: the NLSpeech dataset, Mandarin, roughly 44 h, 1 speaker [U on exact hours].
  - Mainly Chinese, so of limited direct relevance to English.
- **PromptStyle (Liu et al., Interspeech 2023), arXiv 2305.19522** — https://arxiv.org/abs/2305.19522 [M]
  - **Two-stage, VITS backbone**:
    - Stage 1 trains a reference-speech style extractor, which defines a *style space*.
    - Stage 2 trains a text-prompt encoder (BERT-based) whose output is mapped *into that same style space* by cross-modal alignment.
  - At inference the prompt replaces the reference audio.
  - This is the closest published template to a small VITS-class model.
  - InstructTTSEval EN: APS 57.4 / DSD 46.4 / RP 30.9 [V]. The table lists Avg 38.2, which does not equal the mean of its three columns (≈44.9) [U: likely a typo in the paper or in the extraction] (https://arxiv.org/html/2506.16381v1).
- **Parler-TTS / Lyth & King 2024, "Natural language guidance of high-fidelity text-to-speech with synthetic annotations" (Stability AI + Edinburgh), arXiv 2402.01912** — https://arxiv.org/abs/2402.01912
  - Architecture: a decoder-only transformer over audio-codec tokens, following the MusicGen/AudioCraft recipe. The description goes through a **frozen T5-family encoder** that the decoder reads via **cross-attention**, and the transcript is prepended as a prefix [M].
  - Annotation: data-speech (https://github.com/huggingface/dataspeech) automatically labels gender, accent, pitch, speaking rate, SNR/noise, reverberation and monotony. It bins them into keywords, then an LLM rewrites them into natural sentences [M].
  - The open HF reproduction:
    - Parler-TTS Mini v1 is listed at **880M** params, trained on **45K hours** (described in the repo as audiobook data), with Descript DAC as the codec [V] (https://huggingface.co/collections/parler-tts/parler-tts-fully-open-source-high-quality-tts).
    - Licence is Apache-2.0 [V via secondary source].
    - Large v1 is about 2.2–2.3B [M].
    - Other reported counts for Mini are 878M and about 885M [U].
  - InstructTTSEval EN: Mini Avg 46.9, Large Avg 45.7. Large is *not* better than Mini, which suggests the ceiling comes from data and annotations rather than scale [V] (https://arxiv.org/html/2506.16381v1).
  - Fine-tuning Parler-TTS on ParaSpeechCaps gave +7.9% Consistency MOS and +15.5% Naturalness MOS over the best baseline [V] (https://arxiv.org/abs/2503.04713v1).
- **TextrolSpeech + Salle (Ji et al., ICASSP 2024), arXiv 2308.14430** — https://arxiv.org/abs/2308.14430 [M]
  - Salle is a VALL-E-style codec LM in which the style text is a prefix condition.
  - TextrolSpeech has about 236k utterance–description pairs, about 330 h, with gender, pitch, speed, volume and emotion [M; numbers not re-verified].
- **CosyVoice (Alibaba, arXiv 2407.05132) / CosyVoice 2 (arXiv 2412.10117)** — https://arxiv.org/abs/2407.05132, https://arxiv.org/abs/2412.10117 [M]
  - Pipeline: a text-speech LM over supervised semantic tokens, then conditional flow matching to mel, then a HiFi-GAN-class vocoder.
  - The CosyVoice 2 LM is initialised from **Qwen2.5-0.5B**, with chunk-aware causal flow matching for streaming.
  - "Instruct" models take a natural-language instruction (emotion, speaking style, role) as LLM text tokens before the content. They also take inline tags such as `[laughter]` and `<strong>`.
  - Training data for the instruction capability is not public in detail.
- **ControlSpeech (Ji et al., ACL 2025), arXiv 2406.01205** — https://arxiv.org/abs/2406.01205
  - Takes speech, content and style prompts. It uses bidirectional attention with mask-based parallel decoding (Conformer) over a decoupled codec (FACodec) [V] (https://arxiv.org/html/2406.01205).
  - The second stage fuses timbre embeddings through conditional normalization into a pre-trained decoder [V].
  - The style ground truth is a **512-dim global style vector** from the FACodec conv+LSTM style extractor [V] (https://en.papernotes.org/ACL2025/audio_speech/controlspeech_zero_shot/, secondary).
  - An SMSD (Style Mixture Semantic Density) module, a mixture density network, handles the one-to-many mapping from text to style [M].
  - Dataset: VccmDataset, built on TextrolSpeech [M].
  - No total parameter count was found [V gap].
- **FleSpeech (Li et al., arXiv 2501.04644, Jan 2025, revised Apr 2025)** — https://arxiv.org/abs/2501.04644
  - A multi-stage framework with a **multimodal prompt encoder** that unifies text, audio and visual (face) prompts into one representation [V].
  - No parameter count found.
- **EmoVoice (arXiv 2504.12867, 2025)** — https://arxiv.org/pdf/2504.12867
  - An LLM-TTS whose backbone is **Qwen2.5-0.5B** (24 layers, about 0.49B) [V].
  - Takes a free-form emotion description plus text and autoregressively generates audio tokens [V].
  - Uses EmoVoice-DB, an English emotional set with free-style descriptions, about 40 h [M/U].
- **UniStyle ("Unified style modeling for speaking style captioning and stylistic speech synthesis", ACM MM 2024)**
  - Only a reference listing was found. There is no arXiv ID, architecture or parameter count [V gap] (https://arxiv.org/pdf/2412.06602, survey reference list).
- **VoxInstruct (Zhou et al., ACM MM 2024), "Expressive Human Instruction-to-Speech Generation with Unified Multilingual Codec Language Modelling"**
  - A unified codec LM in which the instruction and the content are a single free-form text. It uses semantic speech tokens and classifier-free guidance [V-secondary].
  - The arXiv ID is believed to be 2408.15676 [U].
  - InstructTTSEval EN Avg 50.4, the best open model in that table. ZH Avg 47.5 [V] (https://arxiv.org/html/2506.16381v1).
- **Spark-TTS (SparkAudio, arXiv 2503.01710, 2025)** — https://arxiv.org/abs/2503.01710 [M]
  - Uses Qwen2.5-0.5B with a single-stream BiCodec (global speaker tokens plus semantic tokens).
  - **Attribute control is NOT free-form.** It takes discrete labels for gender, pitch (5 levels) and speed (5 levels), with coarse-to-fine generation: attribute labels → fine-grained values → tokens.
  - Data: VoxBox, about 100k h, with attribute annotations.
- **Step-Audio-EditX (StepFun, arXiv 2511.03601, Nov 2025)** — https://arxiv.org/abs/2511.03601v1
  - An open-source LLM-based audio model that uses RL (SFT + PPO) for iterative editing of emotion, speaking style and paralinguistics, plus zero-shot TTS [V].
  - It claims control comes from post-training on "large-margin" data alone, with no extra style adapters [V].
  - Model size is **3B** [V-secondary].
  - Uses a dual-codebook tokenizer, an LLM initialised from a text LLM, flow matching and a BigVGANv2 vocoder [V-secondary, liner.com review].
  - A third-party review reports iterative editing raising emotion accuracy from 55.0% to 71.0% and style accuracy from 44.6% to 66.2% (Iter0 → Iter3). This was **not checked against the paper's tables** [U] (https://liner.com/review/stepaudioeditx-technical-report).
  - It claims to beat MiniMax-2.6-hd and Doubao-Seed-TTS-2.0 on emotion editing [V].
- **ParaStyleTTS (Lou, Paik, Hu, Yao; arXiv 2510.18308, Oct 2025)** — https://arxiv.org/abs/2510.18308
  - A lightweight "two-level style adaptation architecture" that models prosodic and paralinguistic style separately, driven by text prompts only [V].
  - Attributes: emotion, gender, age [V].
  - Claims about **30× faster, 8× fewer params and 2.5× less CUDA memory** than SOTA LLM-based systems, at comparable quality [V].
  - Absolute parameter count not stated in the abstract [gap].

**Commercial / industrial systems**

- **Inworld TTS-1 / TTS-1-Max (arXiv 2507.21138, Jul 2025)** — https://arxiv.org/abs/2507.21138
  - Transformer autoregressive SpeechLM. **TTS-1 has 1.6B** params and **TTS-1-Max has 8.8B** [V].
  - Three-stage training: pre-train → SFT → RL alignment [V].
  - Output is 48 kHz, in 11 languages, with MIT-licensed training/modeling code [V].
  - "Audio markups" provide emotion and non-verbal control [V]. Per the Inworld engineer's blog, this used a **fixed instruction list with per-instruction datasets** [V] (https://altsoph.substack.com/p/wtf-is-voice-steering).
- **Inworld Realtime TTS-2 (`inworld-tts-2`, launched per release notes dated May 5, 2026)**
  - Free-form bracketed English steering, e.g. `[whisper in a hushed style]`, `[say sadly with deliberate pauses in a low voice]`, covering articulation, pitch, speed and volume, with no numeric parameters [V] (https://docs.inworld.ai/tts/capabilities/steering; https://replicate.com/inworld/realtime-tts-2/readme).
  - Delivery tags work best at the start of a segment. Non-verbals like `[laugh]` and `[sigh]` can go inline. On older models the instruction may be read aloud verbatim [V].
  - A tag applies until the next tag [V-secondary, Runware guide].
  - Architecture, parameter count and training data are **not disclosed**. The blog says only that "a more elaborate training scheme… let the model generalize" [V].
- **ElevenLabs Eleven v3 (alpha, 2025) audio tags**
  - Inline lowercase square-bracket tags. ElevenLabs gives examples in three categories: emotions (`[curious]`, `[crying]`, `[mischievously]`), delivery (`[whispers]`, `[shouts]`) and reactions (`[laughs]`, `[clears throat]`, `[sighs]`). It also gives SFX-like examples (`[gunshot]`, `[clapping]`) [V] (https://elevenlabs.io/blog/v3-audiotags; https://help.elevenlabs.io/hc/en-us/articles/35869142561297-How-do-audio-tags-work-with-Eleven-v3).
  - Tags can be stacked [V].
  - Synthflow says each tag affects about 4–5 words [V-secondary] (https://docs.synthflow.ai/changelog/2026/4/23).
  - Architecture and size are undisclosed.
- **xAI Grok TTS**
  - Two kinds of tag: *inline* (laugh, pause, whisper) and *wrapping* span tags for volume/intensity, pitch/speed and vocal style (whisper, slow, soft). Tags can be nested [V-secondary] (https://picsart.com/blog/all-about-grok-tts/; https://developers.telnyx.com/docs/voice/tts/providers/telnyx/grok).
  - Official docs: https://docs.x.ai/developers/model-capabilities/audio/text-to-speech (not fetched).
  - Architecture undisclosed.
- **OpenAI gpt-4o-mini-tts (Mar 2025)** — https://openai.com/index/introducing-our-next-generation-audio-models/ [M]
  - A separate free-form `instructions` field (accent, emotion, intonation, tone, whispering, etc.), based on GPT-4o-mini.
  - InstructTTSEval EN: APS 76.4 / DSD 74.3 / RP 54.8 / Avg 68.5. ZH Avg 51.1 [V] (https://arxiv.org/html/2506.16381v1).
- **Google Gemini 2.5 Flash/Pro TTS (2025)** — https://ai.google.dev/gemini-api/docs/speech-generation [M]
  - Style is controlled by a natural-language prompt prepended to the text, with multi-speaker support.
  - InstructTTSEval EN: gemini-flash Avg **88.7** (APS 92.3 / DSD 93.8 / RP 80.1), *above* the reference audio (84.3). gemini-pro Avg 80.3 [V].
  - Caveat: Gemini is also the judge, so self-preference bias is possible [inference].
- **Hume (Octave)**: InstructTTSEval EN Avg 71.1 [V].

### Inferences
- Prompt encoders fall into three families:
  - (a) BERT/RoBERTa, fine-tuned or frozen, with global additive or AdaLN-style injection (PromptTTS, PromptStyle, InstructTTS).
  - (b) Frozen T5 with cross-attention (Parler-TTS).
  - (c) Instruction as raw LLM text tokens in a shared sequence (CosyVoice 2, EmoVoice, VoxInstruct, Spark-TTS, Step-Audio-EditX, likely Inworld, OpenAI and Gemini).
- For a ≤30M model, family (a) is the only one that fits. The text encoder can be removed at inference by precomputing or caching prompt embeddings for a controlled vocabulary.
- Open models cluster at Avg ≈ 45–50 on InstructTTSEval EN, while commercial LLM-based ones reach 68–89. The gap is mostly on Descriptive-Style and Role-Play, not on acoustic-parameter specification, where PromptTTS (64.3) is comparable to Parler. A small model aimed at APS-type control (pitch, rate, volume, emotion class) is therefore competitive with much larger open systems on that axis.

### Gaps
- Exact parameter counts for PromptTTS, PromptTTS 2, PromptStyle, ControlSpeech, FleSpeech, VoxInstruct and ParaStyleTTS were not retrieved this session.
- No architecture details were found for UniStyle.
- InstructTTSEval scores for ElevenLabs, CosyVoice 2 and MiniMax are absent; the authors excluded ElevenLabs over subscription limits [V].
- TextrolSpeech hours and pairs, and EmoVoice-DB hours, were not re-verified.

---

## Q2. Compact intermediate style space vs direct conditioning on text embeddings

### Takeaway
Several systems deliberately map the prompt into a compact *speech-derived* style latent (PromptStyle, PromptTTS 2, InstructTTS, ControlSpeech). The motivation is that text prompts under-specify voice (one-to-many). Parler-TTS and the LLM-token systems condition directly on the text representation. For a small model, the two-stage "learn style space from audio, then map prompt into it" design is the best-documented route.

### Cited Findings
- **PromptTTS 2**: the variation network predicts the *reference-speech* representation from the text-prompt representation, to supply variability the prompt does not capture [V] (https://arxiv.org/abs/2309.02285).
- **ControlSpeech**: the style target is a 512-dim global vector from FACodec's style extractor [V-secondary]. The SMSD mixture density module samples from a distribution over style vectors given the text, to model many-to-many text↔style [M] (https://arxiv.org/html/2406.01205).
- **PromptStyle**: stage 1 learns the style space from reference audio, and stage 2 aligns BERT text-prompt embeddings to it [M] (https://arxiv.org/abs/2305.19522).
- **InstructTTS**: cross-modal metric learning aligns RoBERTa prompt embeddings with a speech style encoder [M] (https://arxiv.org/abs/2301.13662).
- **ParaStyleTTS**: a two-level (prosodic vs paralinguistic) style adaptation [V] (https://arxiv.org/abs/2510.18308).
- **Spark-TTS**: an intermediate *discrete* attribute space (gender plus 5-level pitch and speed) predicted before codec tokens (coarse-to-fine) [M] (https://arxiv.org/abs/2503.01710).
- **Direct conditioning**: Parler-TTS (T5 cross-attention), Salle/VoxInstruct (prefix), and CosyVoice 2/EmoVoice (LLM tokens) [M].

### Inferences
- For a controlled-vocabulary product, the intermediate space can be a small vector of continuous plus categorical attributes: pitch level, rate, energy, emotion class and intensity, plus a residual style vector. A tiny prompt-to-latent mapper (or a lookup table) feeds this into a FastSpeech2/VITS/Matcha-class model via AdaLN/FiLM. This also makes control accuracy directly measurable per attribute.

### Gaps
- No head-to-head ablation of "intermediate latent vs direct text conditioning" at matched size was found.

---

## Q3. Generalization to unseen free-form instructions vs fixed tag lists

### Takeaway
Evidence is thin and mostly qualitative. Generalization appears to come from LLM-scale text understanding plus diverse LLM-paraphrased training captions, not from small encoders. Vendors that used fixed lists (Inworld TTS-1, reportedly ElevenLabs v3 and Grok) trained per-instruction data. Inworld claims TTS-2 generalizes but publishes no numbers.

### Cited Findings
- Inworld TTS-1 used a fixed instruction list with per-instruction datasets. TTS-2 used "a more elaborate training scheme" so it "could try to follow instructions it hadn't seen" and "somehow it worked". No quantitative generalization results are given [V] (https://altsoph.substack.com/p/wtf-is-voice-steering).
- The same blog says ElevenLabs v3 and Grok use fixed lists with dedicated data. This is explicitly the author's understanding, not verified [V as claim; U as fact].
- ElevenLabs marketing shows open-ended-looking tags (e.g. SFX), but no official full list was confirmed [V/U] (https://elevenlabs.io/blog/v3-audiotags).
- Inworld's evaluation of steering:
  - LLM judges (Qwen-family and others) agreed with humans only about 60% on pairwise judgments, and confused low pitch with low volume [V].
  - Acoustic heuristics were cut from about 25 directions to 16 in 7 groups (rate, loudness, pitch, range, contour, articulation, speech mode), e.g. characters per second for rate, RMS for loudness, median F0 for pitch [V].
  - Steerability varies by voice (100+ English voices) [V] (https://altsoph.substack.com/p/wtf-is-voice-steering).
- InstructTTSEval (Huang et al., OpenMOSS, arXiv 2506.16381, Jun 2025):
  - Three tasks: Acoustic-Parameter Specification, Descriptive-Style Directive and Role-Play. Each language subset (EN, ZH) has 1k cases per task, 6k in total, with a Gemini judge [V] (https://arxiv.org/abs/2506.16381).
  - The authors find fine-grained acoustic control remains a major open challenge [V].
  - Open models score far below commercial LLM-based models on DSD and RP (free-form) [V].
- Inworld TTS-2 docs warn that contradictory instructions give unpredictable results, and that text/instruction mismatch (e.g. `[say sadly]` on happy text) degrades output [V] (https://replicate.com/inworld/realtime-tts-2/readme).
- PromptTTS 2 and Parler/data-speech use LLM-generated paraphrases of attribute labels to diversify captions [V for PromptTTS 2; M for Parler] (https://arxiv.org/abs/2309.02285).

### Inferences
- For a ≤30M model with a controlled vocabulary, free-form generalization is better handled *outside* the TTS. An LLM or a small text classifier normalizes arbitrary instructions into the fixed attribute and tag space. This is the same "tag list in system prompt" pattern Inworld recommends for LLM-driven apps [V] (https://docs.inworld.ai/tts/best-practices/prompting-for-tts-2).
- Evaluate with acoustic heuristics (F0, RMS, characters per second) plus an emotion classifier rather than LLM judges, given the roughly 60% judge–human agreement.

### Gaps
- No published quantitative unseen-instruction study for any commercial system.
- No paper isolates the effect of "number of paraphrases per attribute" on generalization.

---

## Q4. Datasets

### Takeaway
English caption datasets range from small human-labelled sets (Expresso, ESD, NonverbalTTS, ParaSpeechCaps-Base) to large auto-annotated ones (Parler/data-speech MLS + LibriTTS-R, ParaSpeechCaps-Scaled, SpeechCraft, CapSpeech). Licences are frequently non-commercial, which is a key constraint for Whissle.

### Cited Findings
- **ParaSpeechCaps (Diwan et al., EMNLP 2025, arXiv 2503.04713)**:
  - 59 style tags (33 intrinsic speaker-level and 26 situational utterance-level, e.g. happy, whispered, guttural, nasal) [V].
  - PSC-Base is human-labelled: **282 h** in EMNLP/v1 and **342 h** in the latest arXiv revision [U conflict].
  - PSC-Scaled is auto-annotated: **2427 h** (one lecture page says 2450 h) [V].
  - Licence: **CC BY-NC-SA 4.0** [V] (https://aclanthology.org/2025.emnlp-main.180; https://huggingface.co/datasets/ajd12342/paraspeechcaps).
- **TextrolSpeech (arXiv 2308.14430)**: about 236k pairs, about 330 h; gender, pitch, speed, volume, emotion [M].
- **LibriTTS-P (Kawamura et al., Interspeech 2024, arXiv 2406.07969)**: adds human-annotated speaker-characteristic prompts and style prompts on top of LibriTTS-R [M] (https://arxiv.org/abs/2406.07969). Exact counts not verified.
- **SpeechCraft (Jin et al., ACM MM 2024, arXiv 2408.13608)**: large-scale automatically captioned bilingual (EN/ZH) speech, about 2k+ hours [M/U].
- **CapSpeech (arXiv 2506.02863, 2025)**: a benchmark and dataset for style-captioned TTS and downstream tasks (sound effects, accent, emotion, chat agent) [V title] (https://arxiv.org/pdf/2506.02863). Sizes not verified.
- **Parler/data-speech annotations**: MLS-English about 45k h and LibriTTS-R, with gender, accent, pitch, speaking rate, SNR, reverb and monotony [V for 45k h; M for attribute list] (https://github.com/huggingface/dataspeech).
- **Expresso (Nguyen et al., Interspeech 2023, arXiv 2308.05725)**: 4 speakers, about 40+ h, 26 expressive styles including whisper, laughing and improvised dialogue; CC BY-NC 4.0 [M].
- **ESD (Zhou et al., 2021)**: 10 English and 10 Mandarin speakers × 350 utterances × 5 emotions (neutral, happy, angry, sad, surprise) [M].
- **EmoV-DB**: 4 English speakers (plus 1 Belgian French?), emotions amused, angry, sleepy, disgust, neutral [M/U].
- **NonverbalTTS (arXiv 2507.13155, 2025)**: about 17 h from VoxCeleb and Expresso, with 10 non-verbal types and 8 emotions [M].
- **MSP-Podcast**: naturalistic English podcast speech with categorical and dimensional emotion labels; academic licence [M].
- **InstructTTSEval**: evaluation only, 6k cases (3 tasks × 2 languages × 1k) [V].
- **EmoVoice-DB**: English, emotion-description captions [M/U].
- **VoxBox (Spark-TTS)**: about 100k h with gender, pitch and speed annotations [M].

### Inferences
- For a commercial English model, Expresso and ParaSpeechCaps are NC-licensed. The likely route is a self-annotated corpus using the data-speech pipeline (Apache-2.0 code) on commercially usable audio, or synthetic data distilled from a large instruct-TTS whose terms permit it. A third option is your own recordings in a fixed tag vocabulary.

### Gaps
- Exact licences for TextrolSpeech, SpeechCraft, LibriTTS-P, NonverbalTTS and CapSpeech were not verified this session. Check the HF/GitHub cards.

---

## Q5. Smallest published prompt-controlled TTS and its quality

### Takeaway
No peer-reviewed natural-language prompt TTS ≤30M including vocoder was found. The smallest documented designs are the 2022–23 NAR systems (PromptTTS on FastSpeech, PromptStyle on VITS). Their acoustic models are likely tens of millions of params, but the BERT prompt encoder (~110M) must be run or cached. ParaStyleTTS (2025) is the most recent "lightweight" claim but gives only relative size. The smallest *open* checkpoint verified is Parler-TTS Mini at 880M.

### Cited Findings
- Parler-TTS Mini v1 has 880M params and is trained on 45K h [V] (https://huggingface.co/collections/parler-tts/parler-tts-fully-open-source-high-quality-tts). It scores InstructTTSEval EN Avg 46.9, essentially matching Large (45.7) [V].
- ParaStyleTTS uses 8× fewer params than SOTA LLM-based systems and is 30× faster [V] (https://arxiv.org/abs/2510.18308). If the comparison point is a ~0.5B Qwen-based model, this implies roughly 60–100M, but that is an inference, not stated.
- PromptTTS (FastSpeech-style plus BERT) scores InstructTTSEval EN APS 64.3. This is the **highest APS among open models**, above Parler-Mini (63.4) and VoxInstruct (54.9) [V] (https://arxiv.org/html/2506.16381v1).
- LLM-based small open systems (CosyVoice 2, EmoVoice, Spark-TTS) all sit at about 0.5B [V for EmoVoice; M for others].

### Inferences
- A ≤30M design is plausible by combining:
  - a Matcha-TTS/FastSpeech2/VITS-class acoustic model (~10–20M);
  - a small vocoder such as a HiFi-GAN V2/Vocos-class model (~1–14M) [M sizes];
  - AdaLN/FiLM conditioning on a compact style vector;
  - prompt embeddings from a frozen sentence encoder, *precomputed offline* for the controlled vocabulary, or replaced by a learned embedding table plus an external LLM or classifier normaliser.
- PromptTTS's strong APS score suggests old-style NAR models are fine for acoustic-parameter control. Their weakness is open-ended description and role-play.

### Gaps
- No verified absolute parameter counts for PromptTTS, PromptStyle or ParaStyleTTS.
- A "parler-tts-tiny" checkpoint may exist on HF, but it was not verified [U].
- No small-model MOS numbers were collected this session.
