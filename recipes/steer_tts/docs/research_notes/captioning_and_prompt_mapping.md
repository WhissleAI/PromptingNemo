# Speech style captioning pipelines and mapping prompts to compact control spaces (2023–2026)

Research cutoff: 2026-10-07. About 17 tool calls. Claims marked **[unverified]** come from memory or secondary sources and were not checked against a primary source in this session.

## Q1. Parler-TTS / huggingface data-speech: features, binning, LLM rewriting, dataset sizes

### Takeaway
Data-Speech measures a small set of acoustic features: speaking rate (phonemes/s), utterance pitch mean/std, SNR, C50, SI-SDR/PESQ, plus gender and accent. Each continuous feature goes into one of 7 keyword bins based on its distribution over the whole dataset. Pitch is the only speaker-relative feature: a speaker's mean pitch is compared against same-gender speakers. An instruct LLM (Stable Beluga 2.5 in the paper; Mistral-7B-Instruct / Llama-3-8B-Instruct in the repo) turns the keyword set into a sentence. Parler-TTS v1 was trained on about 45k hours (English MLS plus LibriTTS-R).

### Cited Findings
- Data-Speech computes these features: speaking rate (nb_phonemes / utterance length), SI-SDR and PESQ (an intelligibility/noise proxy), reverberation `c50`, `snr`, `utterance_pitch_mean`, `utterance_pitch_std`, and "speech monotony". Gender and speaker ID are required inputs for the pitch computation. In the repo, accent appears only as a TODO — [huggingface/dataspeech](https://github.com/huggingface/dataspeech)
- Speaking rate has 7 bins: "very slowly", "quite slowly", "slightly slowly", "moderate speed", "slightly fast", "quite fast", "very fast". A speaker's pitch is the average of their clip pitches, compared against speakers of the same gender, on a scale from "very high-pitched" to "very low-pitched". Other features are utterance-level. Their bins come from histograms over the training samples with extreme values removed. Bin edges can be reused through `--path_to_bin_edges` / `--path_to_text_bins` (`v02_bin_edges.json`, `v02_text_bins.json`) and `--leading_split_for_bins` — [dataspeech README](https://github.com/huggingface/dataspeech)
- The repo examples use `mistralai/Mistral-7B-Instruct-v0.2` (Jenny, TGI) and `meta-llama/Meta-Llama-3-8B-Instruct` (LibriTTS-R) to write descriptions. Example output: "In a very expressive voice, Jenny pronounces her words incredibly slowly…". Speaker-naming flags include `--speaker_id_column`, `--speaker_ids_to_name_json`, `--is_single_speaker`, `--speaker_name` and `--is_new_speaker_prompt`. Squim quality estimation is enabled with `--apply_squim_quality_estimation` — [dataspeech](https://github.com/huggingface/dataspeech)
- The underlying paper (Lyth & King, "Natural language guidance of high-fidelity TTS with synthetic annotations", arXiv:2402.01912, 2024) labels gender, accent, pitch (speaker-level mean via PENN, relative to gender), monotony (utterance pitch std), speaking rate (g2p phonemes / duration), and SNR and C50 (both from Brouhaha). Pitch, rate, SNR and C50 each get **7 bins** over the full-dataset distribution. If SNR and C50 both fall in an extreme bin, the clip is labelled "very good recording" or "very bad recording" — [arXiv:2402.01912](https://arxiv.org/html/2402.01912)
- The paper's accent classifier is a linear classifier on language-ID embeddings. It was trained on EdAcc, VCTK and accented VoxPopuli (53 accents) and reached 86% on held-out data. Descriptions were written by **Stable Beluga 2.5**, but the prompts were not published. Data was about 45k hours of English MLS plus 585 hours of LibriTTS-R. Relevance MOS was 3.88 vs 3.19 for Audiobox and 3.62 for ground truth. C50 control correlated poorly with the prompt — [arXiv:2402.01912](https://arxiv.org/html/2402.01912)
- Parler-TTS Large v1 (about 2.2B parameters) was trained on about 45k hours of audiobook data from LibriTTS-R and English MLS. It controls gender, background noise, speaking rate, pitch and reverberation — [HF model card](https://huggingface.co/parler-tts/parler-tts-large-v1); [Gender-bias paper arXiv:2502.05649](https://arxiv.org/pdf/2502.05649)
- ParaSpeechCaps reuses the same "basic tag" approach but with absolute thresholds. Speaker-level mean pitch (PENN) uses gender-specific cut-offs, e.g. male low <115.7 Hz and high >149.7 Hz. Speed uses phonemes/s, with slow <11.5 and fast >19.1. SNR (Brouhaha) is split into 8 levels between 17.1 and 75.0 dB — [arXiv:2503.04713v2](https://arxiv.org/html/2503.04713v2)

### Inferences
- Data-Speech bins are global (dataset-relative) for everything except pitch level, which is speaker mean vs same-gender population. Nothing in it describes how an utterance deviates from the speaker's own baseline, such as "higher than usual" or "rising at the end". Whissle's per-utterance prosody deltas against a speaker-neutral reference are the missing piece. Two families of bins make sense:
  - Absolute or speaker-level bins for voice identity: register, gender, accent, recording quality.
  - Speaker-relative delta bins for style: Δpitch level/range/slope, Δloudness, Δrate, Δtilt, CPPS (breathiness), H1-H2 (breathy vs pressed).
- Seven quantile bins with graded adverbs ("slightly / quite / very") is a well-tested vocabulary. Delta features are signed around zero, so symmetric bins centred on 0 fit them better, for example ±0.5σ, ±1σ, ±2σ, giving 5 or 7 levels.

### Gaps
- Exact hour split between MLS and LibriTTS-R inside the 45k hours, and whether Parler Mini v1 used the same set: not confirmed here. The HF annotated datasets (`parler-tts/mls-eng-speaker-descriptions`, `parler-tts/libritts-r-filtered-speaker-descriptions`) are referenced in search snippets but their cards were not fetched **[unverified]**.
- The exact Data-Speech LLM prompt template was not retrieved. It lives in `scripts/run_prompt_creation.py` in the repo.
- Licenses of the annotated MLS/LibriTTS-R description datasets were not verified. The source corpora are CC BY 4.0 **[unverified]**.

## Q2. Style-caption datasets: annotation methods, vocabularies, verification, sizes, licenses

### Takeaway
The field has moved through three stages:
1. Template or tag captions built from basic attributes (PromptSpeech, TextrolSpeech, LibriTTS-P, Parler).
2. LLM-rewritten captions from classifier outputs (SpeechCraft).
3. Rich, human-seeded tag sets scaled automatically with embedders and an audio LLM judge (ParaSpeechCaps, CapSpeech).

Most rich datasets are non-commercial (CC BY-NC / NC-SA).

### Cited Findings
- **ParaSpeechCaps** (Diwan, Zheng, Harwath, Choi; arXiv:2503.04713; EMNLP 2025):
  - Tags: 59 in total. 28 rich intrinsic tags cover pitch (shrill, nasal, deep), texture (silky, husky, raspy, guttural, vocal-fry), clarity, volume, rhythm (flowing, monotonous, punctuated, hesitant, singsong) and 8 accents. 23 rich situational tags are 18 emotions (incl. sarcastic) plus 5 expressiveness tags (animated, laughing, passive, whispered, enunciated). There are 8 basic tags.
  - Scaling pipeline:
    - Intrinsic tags spread from a speaker to Emilia speakers whose VoxSim speaker-embedding cosine is at least 0.80.
    - Situational tags go through three steps: (1) expressivity filter with a dimensional VAD classifier; (2) transcript semantic matching with SFR-Embedding-Mistral; (3) Gemini 1.5 Flash rates emotion strength from 1–5, and only 5s are kept.
  - Captions are written by Mistral-7B-Instruct-v0.2 from the tag lists.
  - Human labels come from MTurk Masters with a qualification test, 2–3 agreeing annotators and 5 annotations per speaker.
  - Data: about 282 h human-labelled (PSC-Base) and about 2,427 h auto-scaled (PSC-Scaled; a conference listing says 2,450 h).
  - Results: fine-tuned Parler-TTS gained +7.9% consistency MOS and +15.5% naturalness MOS — [arXiv:2503.04713](https://arxiv.org/html/2503.04713v2); [ACL Anthology](https://aclanthology.org/2025.emnlp-main.180)
- ParaSpeechCaps HF dataset:
  - License **CC BY-NC-SA 4.0**; about 1.07M rows (train_scaled 925k, train_base 117k, dev 12k, holdout 14.8k, test 246).
  - Columns include intrinsic/situational/basic tags, utterance_pitch_mean, snr, speaking_rate and phonemes.
  - Example: "A female American's voice is flowing and nasal, high-pitched with a measured speed, recorded in a very clean environment." Audio stays under the source corpora licenses — [HF card](https://huggingface.co/datasets/ajd12342/paraspeechcaps)
- **SpeechCraft** (arXiv:2408.13608; ACM MM 2024):
  - Attributes: pitch, energy and speed (3 levels each, pitch gender-relative), age (5 groups), gender, emotion, word emphasis, topic summary.
  - Classifiers: wav2vec2 for age/gender, emotion2vec for English emotion, SECap captions for Chinese.
  - Captions: GPT-4 Turbo seed captions plus a fine-tuned LLaMA-2 (Baichuan2-7B for Chinese) as rewriter. Augmentation used attribute reordering, synonym substitution and back-translation.
  - Emphasis data was synthesised by scaling FastSpeech2 energy, pitch and duration on TextRank/Gensim keywords: 63k AISHELL-3 clips and 75k LibriTTS clips. The emphasis detector reached 66.9% word-level accuracy.
  - Size: about 2,381 h and 2.25M clips (LibriTTS-R, GigaSpeech-m, AISHELL-3, Zhvoice). The paper's figures are internally inconsistent (2,000 h / 2.1M descriptions) — [arXiv:2408.13608](https://arxiv.org/html/2408.13608v1). Dataset license not stated on the page.
- **TextrolSpeech** (arXiv:2308.14430): 330 h, 236,220 text descriptions, about 500 prompt variants ("prompt diversity"), with emotion labels. Its comparison table describes PromptSpeech as 27,893 items with diversity 5, partly open source. The table was extracted from a PDF, so the column alignment may be off — [ar5iv 2308.14430](https://ar5iv.labs.arxiv.org/html/2308.14430)
- **LibriTTS-P** (LINE; arXiv:2406.07969; Interspeech 2024):
  - Built on LibriTTS-R. Speaker-identity prompts are human-annotated for all speakers. Style prompts are per-utterance and synthetic, following PromptSpeech's scheme (gender, speed, pitch, loudness).
  - A style captioner trained on it produced 2.5× more accurate words than one trained on PromptSpeech — [arXiv:2406.07969](https://arxiv.org/html/2406.07969v1)
- **CapSpeech** (arXiv:2506.02863, 2025):
  - Over 10M machine-annotated plus about 0.36M (358,783) human-annotated caption pairs.
  - Tasks: CapTTS, CapTTS-SE (sound events), AccCapTTS, EmoCapTTS, AgentTTS.
  - Sources: Emilia, GigaSpeech, CommonVoice, MLS, LibriTTS-R, VoxCeleb, EARS, Expresso, VCTK, VGGSound, etc. Includes new voice-actor recordings.
  - HF release CapTTS-SFT is **CC BY-NC 4.0** — [arXiv:2506.02863](https://arxiv.org/html/2506.02863v1); [HF CapTTS-SFT card](https://huggingface.co/datasets/OpenSound/CapTTS-SFT/blob/refs%2Fpr%2F2/README.md)
- **InstructTTSEval** (arXiv:2506.16381, 2025):
  - Three tasks of increasing abstraction: Acoustic-Parameter Specification (APS), Descriptive-Style Directive (DSD) and Role-Play (RP). 1,000 cases per task per language (EN/ZH), 6,000 in total, each with reference audio. Gemini is the judge — [arXiv:2506.16381](https://arxiv.org/pdf/2506.16381)
  - Gemini-TTS-Pro scores 87.6 / 86.0 / 67.2 (APS / DSD / RP) on EN, as reported in MOSS-VoiceGenerator — [arXiv:2603.28086](https://arxiv.org/pdf/2603.28086)
- **PromptTTS 2** (ICLR 2024, arXiv:2309.02285): a speech-understanding model recognises attributes (gender, speed, …) and an LLM writes the prompts, applied to a 44k-hour dataset — [arXiv:2309.02285](https://arxiv.org/abs/2309.02285v1)

### Inferences
- Typical coverage:
  - Basic datasets cover about 4–7 attributes: gender, pitch, rate, energy/loudness, plus noise/reverb or age/emotion.
  - Rich datasets cover about 50–60 tags (ParaSpeechCaps 59).
  - Benchmarks separate three prompt types: explicit acoustic parameters, descriptive style and role-play/situational. Whissle's prompt-mapping eval should have the same three-tier split.
- The ParaSpeechCaps "audio-LLM judge, keep only top-rated clips" step is a cheap way to make crowd soft emotion labels high-precision before captioning.
- For commercial use, ParaSpeechCaps and CapSpeech are NC-licensed. Use them only as research references or eval sets, and caption Whissle's own data with its own pipeline.

### Gaps
- TextrolSpeech, SpeechCraft and LibriTTS-P licenses: not verified **[unverified]**. LibriTTS-P GitHub (line/LibriTTS-P) is believed to be CC BY 4.0, and TextrolSpeech's license is unknown.
- "EmoCaps" could not be located under that exact name. Possibly related: SECap, EmoCaps-style emotion captions in CapSpeech (EmoCapTTS), or EmoVoice / EmoSphere. Not searched further.
- LibriTTS-P utterance and speaker counts were not retrieved.

## Q3. Speech captioning models and style-text contrastive encoders

### Takeaway
Contrastive speech–style-text encoders have progressed in three steps:
1. ParaCLAP (2024): emotion, gender and eGeMAPS-bin templates on MSP-Podcast.
2. ParaSpeechCLAP (2026): rich intrinsic and situational captions; usable for retrieval, attribute classification and as a best-of-N TTS reward.
3. Generative TTS-side encoders that handle "one prompt → many voices": PromptTTS 2 diffusion variation network, ControlSpeech SMSD mixture density.

### Cited Findings
- **ParaCLAP** (arXiv:2406.07203, Interspeech 2024):
  - Queries come from labels via templates: "this is a [EMOTION] instance", "the speaker is [GENDER]". Arousal, valence and dominance are binned at 30/40/30 percentiles into low/mid/high.
  - Pseudo-captions come from openSMILE eGeMAPS features (pitch mean/std, intensity, jitter, shimmer, duration) binned to low/normal/high. Several are joined with "and" during training.
  - Encoders: pruned 12-layer wav2vec2-large (fine-tuned for dimensional SER) and BERT-base, projected to a shared 768-d space. Trained on MSP-Podcast 1.9 (45.6k utterances).
  - Beats CLAP and Pengi on zero-shot SER (e.g., IEMOCAP UAR .567) — [arXiv:2406.07203](https://arxiv.org/html/2406.07203v1)
  - Checkpoint on HF — [KeiKinn/paraclap](https://huggingface.co/KeiKinn/paraclap)
- **ParaSpeechCLAP** (arXiv:2603.28737, Mar 2026):
  - A dual encoder trained on ParaSpeechCaps, in Intrinsic, Situational and Combined variants. Specialised models win on individual dimensions; the Combined model is better on compositional prompts.
  - The Intrinsic model adds a classification loss and class balancing.
  - Applications: caption retrieval, attribute classification, and an inference-time best-of-N reward for style-prompted TTS. Code at github.com/ajd12342/paraspeechclap — [arXiv:2603.28737](https://arxiv.org/abs/2603.28737)
- **PromptTTS 2**: a diffusion "variation network" predicts the voice variability that the prompt cannot describe. It uses reference speech in training only, then samples from noise conditioned on the prompt at inference — [arXiv:2309.02285](https://arxiv.org/abs/2309.02285v1); [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/fed1ea8dcc2a13f3835cc854e8c8294c-Paper-Conference.pdf)
- **ControlSpeech** (arXiv:2406.01205):
  - The Style Mixture Semantic Density (SMSD) module is a Gaussian mixture density network over style-description semantics, plus noise perturbation.
  - It addresses many-to-many mappings: paraphrases like "very rapid pace" / "considerable speed" map to one style, which maps to many speaking-rate realisations — [arXiv:2406.01205](https://arxiv.org/html/2406.01205v3)
- **CALM** (arXiv:2308.16021): contrastive cross-modal speaking-style modelling for expressive TTS. It retrieves style references through text–speech contrastive learning — [arXiv:2308.16021](https://arxiv.org/pdf/2308.16021) (only the title was seen in search; details not read)
- **Captioners:** SpeechCraft's pipeline combines emotion2vec, SECap and attribute classifiers, then an LLM — [arXiv:2408.13608](https://arxiv.org/html/2408.13608v1). LibriTTS-P trains and evaluates a style-captioning model — [arXiv:2406.07969](https://arxiv.org/html/2406.07969v1)

### Inferences
- ParaCLAP's template recipe is almost exactly the recipe Whissle needs for a contrastive pretraining corpus:
  - categorical and VAD bins → templates;
  - acoustic-feature bins → templates;
  - random "and"-composition.
  - Whissle can swap in its own deltas (Δpitch slope, CPPS, H1-H2, tilt) and WavLM as the audio tower.
- ParaSpeechCLAP-style scoring can serve as an automatic metric and a best-of-N reranker for a tiny TTS without retraining.

### Gaps
- No paper found here that distils a speech-style CLAP text tower into a tiny (<20M) prompt encoder emitting control vectors. This appears to be an open or unpublished design point.
- CLAPSpeech (2023) targets prosody from text context, not style prompts **[unverified, not fetched]**.
- emotion2vec+ captioning variants were not investigated.

## Q4. Mapping free-form prompts to compact control vectors

### Takeaway
Four patterns appear in the literature:
- (a) An LLM planner that emits explicit acoustic or prosodic parameters: Sigurgeirsson & King 2023 (F0/energy scaling coefficients), BatonVoice 2025 (JSON plan of pitch/energy per segment), cascaded LLM-to-SSML regression 2025, WordVoice 2026 (word-level attributes).
- (b) A frozen text encoder plus a learned style head (Parler: T5 encoder; PromptTTS 2 / ControlSpeech: prompt encoder plus a sampling module to handle one-to-many).
- (c) Contrastive speech–text space (ParaCLAP, ParaSpeechCLAP) for retrieval or reranking.
- (d) Retrieval over captioned references (implied by CLAP retrieval).

For a tiny TTS, the practical route is a hybrid: an LLM teacher produces structured controls, and a small sentence-embedding regressor or classifier is distilled from it.

### Cited Findings
- **Sigurgeirsson & King, "Controllable speaking styles using a large language model"** (arXiv:2305.10321, 2023): the LLM is prompted to suggest local and global coefficients that scale the acoustic model's predicted F0 and energy. It was rated most appropriate in 50% of cases vs 31% for the baseline — [arXiv:2305.10321](https://arxiv.org/html/2305.10321v2)
- **BatonVoice** (arXiv:2509.26514, 2025):
  - An LLM "conductor" turns instructions into a textual plan of explicit vocal features (e.g., pitch, energy). The plan is consumed by a purpose-built BatonTTS "orchestra".
  - The prompt template returns a JSON list of segments of about 1 s or longer; word count per segment implicitly encodes local rate.
  - Claims zero-shot cross-lingual transfer of feature control — [arXiv:2509.26514](https://arxiv.org/abs/2509.26514)
  - JSON details come from search snippets, not a full read — [search snippet via arXiv PDF](https://arxiv.org/pdf/2509.26514)
- **Cascaded LLM → SSML for French TTS** (arXiv:2508.17494): predicts tags and regresses pitch/rate/volume. Cuts MAE by 25–40% vs prompting-only LLMs and a BiLSTM. MOS rose from 3.20 to 3.87 on a 14 h podcast corpus — [arXiv:2508.17494](https://arxiv.org/pdf/2508.17494v1.pdf) (snippet-level)
- **WordVoice** (arXiv:2607.06461, 2026): explicit, decoupled, multi-dimensional word-level control for LLM-based TTS.
  - "Free mode": the model predicts per-word attributes as a prosodic planner.
  - "Control mode": users override per-word values, e.g. high pitch on a word.
  - A flow-matching stage restores acoustic detail — [arXiv:2607.06461](https://arxiv.org/pdf/2607.06461) (snippet-level)
- **SpeechCraft** handles per-word emphasis by synthesising emphasised training data. FastSpeech2 pitch, energy and duration are scaled on chosen keywords, and the result is paired with descriptions naming the word — [arXiv:2408.13608](https://arxiv.org/html/2408.13608v1)
- **Handling one-to-many and paraphrase**: ControlSpeech's SMSD mixture density over style semantics plus noise perturbation — [arXiv:2406.01205](https://arxiv.org/html/2406.01205v3). PromptTTS 2 diffusion sampling of residual variability — [arXiv:2309.02285](https://arxiv.org/abs/2309.02285v1)
- **Compositionality**:
  - ParaSpeechCLAP's Combined model beats the specialised ones on compositional evaluation — [arXiv:2603.28737](https://arxiv.org/abs/2603.28737)
  - ParaCLAP trains on "and"-joined multi-attribute queries — [arXiv:2406.07203](https://arxiv.org/html/2406.07203v1)
  - InstructTTSEval shows performance drops sharply from APS to RP (abstract/situational prompts), e.g. Gemini-TTS-Pro EN 87.6 → 67.2 — [arXiv:2603.28086](https://arxiv.org/pdf/2603.28086)

### Inferences (proposed recipe for Whissle; not from a single source)
1. **Control space.** About 12–20 dimensions:
   - Speaker-relative continuous deltas in z-units: Δf0 level, Δf0 range, f0 slope (whole utterance plus final-region slope for "rising ending"), Δloudness, Δrate, Δtilt, CPPS, H1-H2.
   - A soft emotion distribution (about 8 classes) plus intensity, arousal and dominance.
   - A few discrete flags: whisper, laugh, voice quality.
   - Per-word emphasis as a separate sparse channel: a word-index mask plus a Δ-vector.
2. **Captioning (forward direction).** Bin each delta symmetrically around 0 into 5–7 graded phrases (dataspeech / ParaCLAP style).
   - Add emotion and intensity tags from crowd soft labels, filtered for high confidence (ParaSpeechCaps style).
   - Have an instruct LLM write 3–5 paraphrased captions per clip: plain descriptive, acoustic-parameter, situational/role-play ("as if something bad just happened"), and imperative ("speak sadly").
   - This matches InstructTTSEval's APS / DSD / RP split.
3. **Inverse mapping (prompt → vector).** The teacher is an LLM planner that outputs JSON in the control schema. It is trained or few-shot prompted on (caption → measured vector) pairs from step 2, so ground truth comes from measured acoustics, not LLM guesses.
   - Distil into a frozen small sentence encoder (MiniLM/BGE/E5) plus an MLP. Heads: regression for continuous dimensions, KL for the emotion distribution, BCE for flags. Optionally a mixture-density head (ControlSpeech-style) to keep one-to-many variance.
   - Fall back to kNN over a bank of captioned reference clips when encoder confidence is low.
4. **Per-word emphasis.** Parse quoted or targeted words in the prompt with a rule or LLM step. Emit a word-index mask, as in WordVoice's control mode. Synthesise emphasis training pairs by manipulating per-word deltas (SpeechCraft method) or by mining Whissle's per-word deltas.
5. **Evaluation.** Use a ParaSpeechCLAP-like scorer or Whissle's own WavLM-based attribute regressors, re-measuring control dimensions on the output (as Lyth & King did). Split the test set by APS, DSD and RP prompt types and include held-out compositions.

### Gaps
- No published head-to-head comparison was found between "frozen sentence embedding + regressor" and "contrastive prompt encoder" for predicting continuous prosody targets. This needs internal ablation.
- BatonVoice, WordVoice and the SSML paper were read only at abstract or snippet level. Exact schemas and training data are unverified.

## Q5. Data augmentation for instruction diversity and attribute coverage

### Takeaway
Diversity comes from LLM rewriting of tag sets, with attribute reordering, synonyms, back-translation and multiple captions per clip. The scale is large: TextrolSpeech reports about 500 prompt variants vs PromptSpeech's 5. Most datasets still cover only about 5–10 controllable dimensions; rich ones reach about 59 tags.

### Cited Findings
- SpeechCraft augments with input-attribute reordering, synonym substitution and multi-round translation. GPT-4 Turbo seeds are distilled into a fine-tuned LLaMA rewriter, and low-quality outputs are filtered. Rewriting is evaluated by omission and distortion rates — [arXiv:2408.13608](https://arxiv.org/html/2408.13608v1)
- TextrolSpeech: prompt diversity about 500 vs 5 for PromptSpeech — [ar5iv 2308.14430](https://ar5iv.labs.arxiv.org/html/2308.14430)
- ParaSpeechCaps stores `text_description` as a list, i.e. multiple captions per clip — [HF card](https://huggingface.co/datasets/ajd12342/paraspeechcaps)
- Lyth & King used LLM rewriting specifically so the model generalises and accepts users' own terminology — [arXiv:2402.01912](https://arxiv.org/html/2402.01912)
- Number of attributes per dataset:
  - Parler/Data-Speech: about 7 (gender, accent, pitch, monotony, rate, SNR, C50).
  - SpeechCraft: about 8 (pitch, energy, speed, age, gender, emotion, emphasis, topic).
  - ParaSpeechCaps: 59 tags across 11 factors.
  - ParaCLAP: emotion, gender, VAD, plus 6 eGeMAPS features.
  - Sources: [arXiv:2402.01912](https://arxiv.org/html/2402.01912); [arXiv:2408.13608](https://arxiv.org/html/2408.13608v1); [arXiv:2503.04713](https://arxiv.org/html/2503.04713v2); [arXiv:2406.07203](https://arxiv.org/html/2406.07203v1)

### Inferences
- Negative or contrast prompts are not well documented in the sources read here. Natural forms for Whissle:
  - Contrastive batches in CLAP-style training.
  - Relative prompts ("less sad", "slightly faster than before"), which map naturally onto signed delta control vectors.
  - Negation prompts ("not whispering") as explicit hard negatives.
- Bin-centre jitter plus multiple paraphrases per vector reduces the regressor's over-reliance on specific adverbs.

### Gaps
- No source found quantifying how many paraphrases per clip are optimal, or measuring the gain from negation/contrast prompts.
- No source here systematically covers "relative" or "edit" instructions ("a bit more excited") for TTS. This needs a further search (e.g., speech editing / EmoSphere-style intensity control).
