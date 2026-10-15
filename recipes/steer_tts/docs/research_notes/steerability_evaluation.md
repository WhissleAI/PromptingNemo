# Evaluating Steerability / Instruction-Following and Emotional Expressiveness of TTS (2023–2026)

Notes for designing an evaluation harness for a small prompt-steerable expressive TTS (Whissle.ai). Researched 2026-10-07. Every claim has an inline source. Items marked [UNVERIFIED] come from the researcher's background knowledge and were not confirmed in this session.

## 1. Inworld / altsoph starting point: acoustic-heuristic steerability

### Takeaway
Inworld (Realtime TTS-2) dropped audio-LLM judges after about 60% pairwise agreement with humans. Instead they score steerability per direction with 16 independent acoustic heuristics, each comparing instructed and uninstructed generations over a batch. The approach is deliberately batch-level and uncalibrated across directions. They also use it for per-voice and per-checkpoint comparisons and as the objective for an evolutionary prompt search.

### Cited Findings
- **Setup.** Realtime TTS-2 accepts inline directions such as "[whisper softly and intimate]". It was trained with a generalization scheme for unseen instructions. The 16 evaluation directions are used only for evaluation, not training. — [altsoph, "WTF is voice steering"](https://altsoph.substack.com/p/wtf-is-voice-steering)
- **Why LLM judges were dropped.** Judges from the Qwen and "Inkling" families reached about 60% agreement with humans on pairwise judgments, and fine-tuning "didn't help enough." Failure modes:
  - confusing low pitch with low volume;
  - confidently hallucinating properties they could not hear. — [altsoph](https://altsoph.substack.com/p/wtf-is-voice-steering)
- **Pruned directions.** About 25 directions were tried. Robotic delivery, stuttering and intimate voice were dropped for lack of a simple, reliable heuristic. — [altsoph](https://altsoph.substack.com/p/wtf-is-voice-steering)
- **The heuristics**, all computed as a change versus the no-instruction baseline. — [altsoph](https://altsoph.substack.com/p/wtf-is-voice-steering)
  - Rate (slow/fast): characters per second.
  - Loudness (loud/quiet): RMS.
  - Pitch (high/low): median F0 over voiced frames.
  - Range (wider/monotone): interquartile range of F0.
  - Contour (rising/falling ending): linear-regression slope of voiced F0 over the final 300 ms.
  - More pauses: retained silence divided by clip duration.
  - Clear articulation: spectral-centroid p90 − p10 dispersion over voiced frames, on sentences rich in /i a u/. The post gives no threshold value.
  - Emphasize: peak F0 and peak RMS in the target window. The post calls it "tricky" and gives no formula.
  - Whisper: high- and low-frequency band energy, voiced-frame fraction and spectral flatness.
  - Nasal: peak energy in 500–900 Hz minus peak energy in 150–350 Hz.
  - Melodic: voiced-frame ratio plus a vibrato proxy.
- **Aggregation.**
  - Heuristics are not calibrated against each other, so scores are not comparable across directions.
  - Per-example accuracy is not the goal. What matters is detecting a "statistically significant shift across a batch."
  - The post names no specific test, threshold or sample size. — [altsoph](https://altsoph.substack.com/p/wtf-is-voice-steering)
- **Evolutionary prompt search.** An overnight search "along the lines of" Karpathy's autoresearch looked for instruction phrasings that maximize each heuristic. It was also used on an "expressiveness" metric. — [altsoph](https://altsoph.substack.com/p/wtf-is-voice-steering); [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- **Per-voice findings (100+ English voices).**
  - Steerability differs by voice and by direction within a voice.
  - The most steerable voice is not necessarily the most desirable, since it may sound neutral without instructions.
  - The least steerable voice may suit stability needs.
  - Checkpoint comparisons appear only as charts, with no numbers in the text. — [altsoph](https://altsoph.substack.com/p/wtf-is-voice-steering)
- **"Voice direction resets the whole problem" (earlier post).** Inline direction forces you to redefine the problem space, datasets and metrics. Voices can respond on one axis (affect or intensity) and resist another (speed or pitch), and the full set of directions worth testing is unknown. The same post also critiques standard metrics:
  - WER is unbounded and overweights short texts; at under ~1.5% WER on SeedTTS-eval, ASR noise dominates.
  - MOS requires blinding, matched text, randomized order, CIs and enough raters.
  - In the author's tests NISQAv2 "Coloration" correlated with human preference better than expected, and ECAPA-TDNN worked well for speaker similarity in European languages.
  - Pushing speaker similarity degrades expressiveness.
  - Inworld's internal suite has 2 speaker-similarity scores, 8 alignment metrics, 12 hallucination detectors, 5 NISQAv2 metrics, plus expressiveness, arousal and WER-family metrics.
  - Recommended process: task → dataset → metrics → calibration → automatic evaluation → human review → decision. — [altsoph, "Voice direction resets the whole problem"](https://altsoph.substack.com/i/196912463/voice-direction-resets-the-whole-problem)
- **Inworld expressiveness metrics.**
  - An in-house "arousal" metric measures emotionality regardless of emotion category. They built it because most of 20+ Hugging Face SER models were domain-overfit or unstable.
  - An "expressivity" metric measures variation and dynamic range across prosodic features.
  - Both are used as RL rewards. Inworld claims TTS 1.5 companion voices scored 30% higher on expressivity.
  - No features, model or human validation are disclosed. — [Inworld blog](https://inworld.ai/blog/beyond-quality-emotionality-and-expressiveness)
- Inworld TTS-1 technical report: arXiv 2507.21138 (2025). — [arXiv](https://arxiv.org/abs/2507.21138)

### Inferences
- The heuristics design is cheap, transparent and reproducible, and it suits a small model where per-voice checkpoint tracking matters. Its weaknesses are no cross-direction calibration and weak heuristics for emphasis, articulation and voice quality.
- The missing statistics can be filled with standard paired tests (see section 5).

### Gaps
- Inworld has not published heuristic thresholds, the statistical test, sample sizes or checkpoint numbers.
- I could not confirm what "Inkling" judges are.

## 2. Benchmarks and metrics for instruction-following TTS

### Takeaway
The 2025–2026 benchmarks (InstructTTSEval, EmergentTTS-Eval) rely on Gemini judges, binary or pairwise. Earlier work (PromptTTS, TextrolSpeech, ParaSpeechCaps) uses per-attribute classifiers and binned acoustic labels plus consistency MOS. Embedding similarity (CLAP-style, emotion2vec) is a third, weaker family. A good harness combines all three plus quality guards (WER, UTMOS/NISQA, speaker similarity).

### Cited Findings
- **InstructTTSEval** (arXiv 2506.16381, Jun 2025, Fudan/OpenMOSS). — [arXiv](https://arxiv.org/html/2506.16381); [GitHub](https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/README.md)
  - Tasks: Acoustic-Parameter Specification (APS), Descriptive-Style Directive (DSD) and Role-Play (RP). EN and ZH, 1k items each, 6k total.
  - 12 dimensions: gender, pitch, texture, clarity, fluency, speed, accent, age, volume, emotion, tone, personality.
  - Scoring: Gemini judge with a binary per-item score, macro-averaged.
  - Gemini agrees with majority-vote human labels 79.0% overall: APS 87%, DSD 79%, RP 71%.
  - EN averages: gemini-flash 88.7, gemini-pro 80.3, Hume 71.1, gpt-4o-mini-tts 68.5, reference audio 84.3. Open-source systems are much lower (PromptTTS 47.6, VoxInstruct 50.4).
- **EmergentTTS-Eval** (arXiv 2505.23009, NeurIPS 2025 Datasets & Benchmarks, Boson AI). — [arXiv](https://arxiv.org/html/2505.23009)
  - 1,645 cases in 6 categories: emotions, paralinguistics, foreign words, syntactic complexity, complex pronunciation, questions.
  - Metric: win rate versus a gpt-4o-mini-tts (Alloy) baseline, (wins + 0.5 × ties) / N, with judge temperature 0 and randomized order.
  - Cost: about $50 per system with Gemini 2.5 Pro.
- **TextrolSpeech / PromptTTS protocol** (arXiv 2308.14430, 2023). — [arXiv](https://arxiv.org/pdf/2308.14430)
  - Data: 236,220 prompt–speech pairs with 5 style factors (gender, pitch, speed, volume, emotion).
  - Accuracy uses trained classifiers for gender (99.2%) and emotion (85%). Pitch, rate and energy use binned labels built the same way as the dataset.
  - Human ratings: MOS-Q and a style-consistency MOS.
  - Salle averages 87.6% attribute accuracy. Emotion is the weakest attribute, because of neutral confusion and limited data.
- **ParaSpeechCaps** (arXiv 2503.04713, EMNLP 2025). — [arXiv](https://arxiv.org/html/2503.04713v2); [GitHub](https://github.com/ajd12342/paraspeechcaps)
  - 59 tags: 33 intrinsic and 26 situational.
  - Evaluated with Consistency MOS and Naturalness MOS. Parler-TTS fine-tuned on it gained +7.9% CMOS-consistency and +15.5% naturalness MOS over the best baseline.
- **ParaSpeechCLAP** (arXiv 2603.28737, 2026). A CLAP-style dual encoder for rich speaking styles, evaluated on Intrinsic, Situational and Combined sets. It is used for best-of-N (N=10) inference-time reward guidance to raise style consistency. — [arXiv](https://arxiv.org/html/2603.28737v1)
- **Embedding metrics against human MOS** (arXiv 2601.05554). On TextrolSpeech, a WavLM-based metric (SPAM) reached LCC 0.584 with human MOS, versus 0.520 for RA-CLAP. Both correlations are moderate. — [alphaXiv summary](https://alphaxiv.org/overview/2601.05554v1)
- **emotion2vec caveat.** It is used either as an SER classifier on synthesized speech or as embedding similarity to a reference. Its resolution caps fine-grained emotion evaluation, and its reliability there is "unproven." — [EmoVoice, arXiv 2504.12867](https://arxiv.org/html/2504.12867v1)
- **EmoKnob** (EMNLP 2024, arXiv 2410.00316) defines paired subjective tests for emotion control. — [arXiv](https://arxiv.org/pdf/2410.00316)
  - Emotion Selection Accuracy: controlled vs. uncontrolled audio on neutral text; which conveys emotion X better?
  - Emotion Enhancement Accuracy: the same comparison on emotion-matching text.
  - Emotion Discrimination Test and Emotion Identification Test.
- **SpeechJudge** (arXiv 2511.07931, ICLR 2026).
  - About 99k human-labelled pairs for naturalness and intelligibility.
  - Gemini-2.5-Flash agrees with humans under 70% of the time.
  - SpeechJudge-GRM (Qwen2.5-Omni-7B, SFT then GRPO) reaches 77.2%, or 79.4% with 10-sample inference scaling, versus 72.7% for a Bradley-Terry reward model. — [arXiv](https://arxiv.org/html/2511.07931v2)
- **SpeechParaling-Bench** is a newer generation-side benchmark covering 100+ fine-grained paralinguistic features. I did not retrieve its details. — [alphaXiv (via search)](https://www.alphaxiv.org/overview/2604.20842) [link mapping uncertain]

### Inferences
- **Recommended metric stack.**
  1. Per-direction acoustic delta versus baseline (Inworld style).
  2. Attribute classifiers or bins for pitch, rate and energy against target bins, as in TextrolSpeech.
  3. SER (emotion2vec) accuracy and arousal/valence regression for emotion.
  4. A CLAP-style text–audio style score (ParaSpeechCLAP) as a soft holistic score.
  5. An LLM-judge pass (Gemini 2.5) on InstructTTSEval-style binary items, reported separately.
  6. Guards: WER/CER (Whisper-large-v3), UTMOS or NISQAv2, and ECAPA/WavLM speaker similarity to the voice's uninstructed output.
- For a small model, run InstructTTSEval EN (APS and DSD subsets at least) and EmergentTTS-Eval Emotions and Paralinguistics as external, comparable numbers.

### Gaps
- PromptTTS per-attribute numbers were not retrieved.
- [UNVERIFIED] TTS Arena (Hugging Face TTS-AGI/TTS-Arena, Elo from crowd pairwise votes) and Artificial Analysis' arena measure overall preference, not instruction-following. I did not fetch them in this session.
- [UNVERIFIED] The original PromptTTS (Guo et al., ICASSP 2023, arXiv 2211.12171) and PromptTTS 2 (arXiv 2309.02285) evaluation details were not fetched. As I recall, they report per-attribute classifier accuracy for gender, pitch, speed, volume and emotion.

## 3. Reliability of audio-LLM judges for paralinguistics

### Takeaway
Agreement depends heavily on the granularity of the judgment.
- **System ranking:** Gemini 2.5 Pro and GPT-4o-audio track human rankings well (Spearman around 0.9).
- **Per-item binary instruction checks:** Gemini agrees with humans about 70–87% of the time.
- **Per-item pairwise judgments of subtle attributes:** about 60–70% (Inworld 60%, SpeechJudge under 70%).
- **Pure acoustic attributes (pitch, volume, speed, intonation) when the text conflicts:** near chance, because models "transcribe rather than listen."

Use LLM judges only for system- or checkpoint-level aggregates and holistic or semantic items. Never use them as the sole detector for pitch, loudness or rate.

### Cited Findings
- **EmergentTTS-Eval, human study.** 512 pairs, 8 judges. — [arXiv 2505.23009](https://arxiv.org/html/2505.23009)
  - Spearman correlation of system rankings with humans: Gemini 2.5 Pro, 2.0 Flash, 2.5 Flash and GPT-4o-audio all 0.905; Qwen2.5-Omni 0.881; GPT-4o-mini-audio 0.762.
  - Kendall's W = 0.97 across five judges.
  - Qwen2.5-Omni win rates hovered near 50%, with 63–82 parsing failures per system.
  - Biases: preference for literary phrasing, and hallucinated pronunciation errors on emotion and prosody items. Gemini overestimated Hume and Sesame-1B.
- **Chiang et al., "Audio-Aware LLMs as Judges for Speaking Styles"** (arXiv 2506.05984, Findings of EMNLP 2025). — [arXiv](https://arxiv.org/html/2506.05984v1)
  - Voice-style instruction following: Pearson r human–human 0.596, human–Gemini-2.5-Pro 0.640, human–GPT-4o-audio 0.355.
  - Role-play style: r = 0.253, 0.319 and 0.305 respectively.
  - Setup: 4 human raters; 5 judge samples majority-voted; 20 instances × 4 SLMs.
  - Gemini results were insensitive to temperature (r 0.640–0.649).
  - Judges struggle to rank several weak systems against each other.
- **InstructTTSEval.** Gemini agreement with majority human labels: APS 87%, DSD 79%, RP 71%. — [arXiv 2506.16381](https://arxiv.org/html/2506.16381)
- **Inworld.** About 60% pairwise agreement for Qwen-family and "Inkling" judges. They confuse low pitch with low volume and hallucinate inaudible properties. — [altsoph](https://altsoph.substack.com/p/wtf-is-voice-steering)
- **SpeechJudge.** Best AudioLLM (Gemini-2.5-Flash) under 70% on pairwise naturalness. — [arXiv 2511.07931](https://arxiv.org/html/2511.07931v2)
- **LISTEN** (arXiv 2510.10444). Six LALMs show "consistent lexical dominance," default to "neutral" when the text is neutral, and fail when lexical and acoustic cues conflict. Paralinguistic-only performance approaches chance. — [arXiv](https://arxiv.org/abs/2510.10444v2)
- **VoxParadox** (USC, ICML 2026).
  - 2,000 adversarial items in which transcript claims conflict with speaking style, across 10 tasks including pitch, volume, speed, range and intonation.
  - Reported leaderboard numbers (third-party aggregator, treat as indicative): one model at 17.4% overall with Pitch 11.0%, Volume 11.5% and Speed 11.5%; another at 79% Emotion but 7.5% Pitch and 5% Intonation.
  - ICML slides: Qwen2-Audio adversarial label agreement drops from 70.2% to 15.9%.
  - A fix raises Audio Flamingo 3 from 17.4% to 65.2%. — [ICML slides](https://icml.cc/media/icml-2026/Slides/60946.pdf); [benchmarklist](https://benchmarklist.com/benchmarks/voxparadox/) [primary arXiv not located]

### Inferences
- Your test texts are often emotionally loaded ("I'm so excited!"). Include lexically neutral or contradicting texts so a judge, or an SER model, cannot score from the words alone. This also tests whether the TTS steers from the instruction rather than the text.
- If you use an LLM judge:
  - use Gemini 2.5 Pro or Flash rather than open Qwen2.5-Omni;
  - majority-vote 3–5 samples and randomize A/B order;
  - ask about one attribute per query;
  - calibrate on a 100–200-item human-labelled set per direction before trusting it;
  - report per-direction judge accuracy and exclude directions below roughly 75%.

### Gaps
- No source gives per-attribute confusion matrices for judges on TTS output, beyond Inworld's pitch/volume anecdote and VoxParadox per-task numbers.
- No reliable numbers for the GPT-4o-audio successors or Gemini 3 as judges were found.

## 4. Acoustic heuristics and voice-quality detectors; thresholds and significance

### Takeaway
Rate, loudness, pitch, range, contour and pauses are robust with standard tools (Parselmouth/Praat F0, RMS, VAD, forced alignment). Whisper and breathiness are detectable with CPPS, HNR, voiced fraction, spectral flatness and high-band energy ratios. Emphasis, creak, tense voice and "intimate" remain hard. Inworld dropped several directions for this reason. Thresholds should come from the baseline distribution, not absolute values.

### Cited Findings
- **Breathiness.** In a clinical validation study (VOXplot, 2023), CPPS and GNE were highly valid for perceived breathiness, while HNR and PPQ5 were highly valid for hoarseness. CPPS and GNE separate breathiness from hoarseness. — [PMC10380658](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10380658/)
- **Whisper vs. modal.**
  - Whispery voice has significantly lower HNR and CPP and lower energy.
  - High-frequency harmonic differences (H4–H2k, H2k–H5k) discriminate better than H1–H2 and H2–H4.
  - Whisper has no F0, which motivates a voiced-fraction feature. — [SPIE 2021](https://nanophotonics.spiedigitallibrary.org/conference-proceedings-of-spie/11933/119331H/An-acoustic-study-on-the-whispery-voice/10.1117/12.2615299.full)
- Spectral flatness is a baseline whisper-detection feature, evaluated with ROC curves under noise against LP-MVDR methods (EURASIP JASP 2012). — [EURASIP](https://asp-eurasipjournals.springeropen.com/articles/10.1186/1687-6180-2012-157/tables/3)
- **Inworld's choices.** Whisper uses band-energy ratios, voiced-frame fraction and spectral flatness. Contour uses the slope over the final 300 ms. Clear articulation uses spectral-centroid dispersion. — [altsoph](https://altsoph.substack.com/p/wtf-is-voice-steering)

### Inferences (proposed harness design)
- **Paired design.** For each (voice, text, direction), generate the instructed clip and the uninstructed baseline with K seeds each. Compute Δ = feature(instructed) − feature(baseline) on the same text.
- **Per-direction tests.**
  - One-sided Wilcoxon signed-rank on Δ, with sign set by the direction (fast should give Δ > 0 in chars/sec), across N texts.
  - Report effect size: median Δ normalized by the baseline's across-seed SD, giving a Cohen's d or Cliff's delta.
  - Also report the "hit rate": the share of pairs with Δ beyond the baseline seed-to-seed noise. A natural noise band is the 95th percentile of |baseline_seed_i − baseline_seed_j|.
  - Use Holm or Benjamini–Hochberg correction across 16 directions × V voices.
- **Thresholds.** Set them from baseline distributions, for example "whisper" when voiced fraction falls below the 1st percentile of the voice's baseline and flatness rises above the 99th percentile. Validate each detector on 50–100 human-labelled clips per direction for precision and recall before using it.
- **Other features to add.** The literature above did not verify these choices; they are standard tools.
  - Breathy: CPPS decrease, H1–H2 increase and HNR decrease.
  - Creaky: low-F0 or irregular-period fraction, jitter, subharmonics.
  - Tense or pressed: H1–H2 decrease, spectral tilt flattening, alpha ratio.
  - Emphasis: in the target word versus its neighbours, z-scored F0 peak, RMS and duration ratio, using a forced aligner (e.g. MFA or Whisper word timestamps).
  - Pauses: VAD with silences of at least 150–200 ms inside the speech span. The cutoff is a convention, not sourced.
  - Rising/falling contour: the final-300 ms slope in semitones per second, avoiding absolute Hz. Use a voiced-frame guard such as at least 5 voiced frames.
- **Cross-direction comparability.** Use a standardized effect size per direction (Δ / baseline SD). This partially fixes Inworld's admitted lack of calibration.

### Gaps
- No published validated whisper/breathy/creak classifier tailored to TTS output was found.
- No source gave recommended CPPS or H1–H2 cutoffs for TTS output.
- Emphasis detection lacks a validated recipe in the sources fetched.

## 5. Human evaluation designs

### Takeaway
Paired designs dominate for steerability:
- instructed vs. uninstructed, "which is more X?" (EmoKnob ESA/EEA);
- best-worst scaling across intensity levels (low/mid/high) to test monotonicity;
- MOS or MUSHRA for naturalness;
- binary or Likert "does it follow the instruction?" (InstructTTSEval, Chiang et al.).

Human–human agreement on style is itself modest (r about 0.25–0.6), so plan for several raters per item.

### Cited Findings
- **EmoKnob paired tests.** — [arXiv 2410.00316](https://arxiv.org/pdf/2410.00316)
  - ESA: listeners choose between the controlled and uncontrolled clip on neutral text; the score is the share choosing the controlled clip.
  - EEA: the same on emotion-congruent text.
  - EDT: discriminate between two emotions.
  - EIT: a 2-way label choice.
- **Best-worst scaling for intensity.** Synthesize at intensities 0.0, 0.5 and 1.0. Listeners pick the least and most expressive clip, and success means the lowest setting is chosen as least and the highest as most. Controllability is reported lower at utterance level than at word or phoneme level. BWS is often paired with MUSHRA. One study used 20 listeners. — [arXiv 2405.09171](https://arxiv.org/pdf/2405.09171); [arXiv 2403.02002](https://arxiv.org/pdf/2403.02002); [arXiv 2201.03967](https://arxiv.org/pdf/2201.03967)
- **Human–human agreement on style.** Pearson r = 0.596 for voice-style instruction following and 0.253 for role-play style, with 4 raters and 80 items. — [arXiv 2506.05984](https://arxiv.org/html/2506.05984v1)
- **Consistency MOS.** ParaSpeechCaps uses Consistency MOS alongside naturalness MOS. A related study uses 20 raters per item scoring caption–audio consistency on a 0–5 slider. — [arXiv 2503.04713](https://arxiv.org/html/2503.04713v2); [arXiv 2601.03065](https://arxiv.org/pdf/2601.03065)
- **Human study sizes.** EmergentTTS-Eval used 512 pairs and 8 judges and reported 95% CIs for human win rates. — [arXiv 2505.23009](https://arxiv.org/html/2505.23009)
- **MOS validity.** Requires blinding, matched text, randomized order, enough raters and CIs. — [altsoph](https://altsoph.substack.com/i/196912463/voice-direction-resets-the-whole-problem)

### Inferences
- **Recommended human protocol per direction.** A 3-way forced choice: "Which clip is more X? A / B / no difference." This is the paired design attributed to Bou-Ghazale & Hansen (1995) in the brief. Use instructed vs. baseline on the same voice and text.
- **Sample size for a binomial test against 50%.** Detecting a 65% preference with α = 0.05 (two-sided) and power 0.8 needs about 85 non-tie judgments; 60% needs about 200. This is a standard power calculation, not sourced. Use about 3 raters × 30–40 items per direction-voice cell, or pool voices.
- **Intensity.** Generate "slightly / moderately / very X" and run BWS or ranking. Test monotonicity with Spearman ρ between commanded level and both acoustic Δ and human rank, or with a Jonckheere–Terpstra trend test.
- **Naturalness guard.** A MUSHRA-style test with the uninstructed clip as a hidden reference, or MOS. Pair it with UTMOS or NISQA automatic proxies.

### Gaps
- [UNVERIFIED] Bou-Ghazale & Hansen (1995) was not retrieved. I could not confirm the exact "both/neither" response design attributed to it.
- [UNVERIFIED] The Kiritchenko & Mohammad BWS scoring formula (count-based (best − worst)/n) was not fetched.

## 6. Steerability as a scalar per direction and voice, tracked across checkpoints; trade-off with stability and naturalness

### Takeaway
No standard scalar exists. Inworld reports batch-level shifts per direction, per voice and averaged, and explicitly notes that the most steerable voice isn't the best. Both Inworld and other work report a trade-off: more speaker similarity costs expressiveness, and over-steering risks artifacts or WER rises. A scalar should therefore be reported alongside guard metrics, not alone.

### Cited Findings
- Inworld compares checkpoints by average, per-voice and per-direction steerability. Voices differ in steerability, and per-voice steerability is in tension with uninstructed expressiveness and stability. — [altsoph](https://altsoph.substack.com/p/wtf-is-voice-steering)
- Pushing speaker similarity past a point degrades expressiveness. — [altsoph](https://altsoph.substack.com/i/196912463/voice-direction-resets-the-whole-problem)
- Inworld uses the expressivity and arousal metrics as RL rewards. — [Inworld blog](https://inworld.ai/blog/beyond-quality-emotionality-and-expressiveness)
- EmergentTTS-Eval: win rate and MOS capture different aspects, so systems can rank differently on each. Emotions and Paralinguistics show the highest win-rate variance across voices. — [arXiv 2505.23009](https://arxiv.org/html/2505.23009)
- Best-of-N reward guidance with a style CLAP raises style consistency without training. This is a cheap way to probe a checkpoint's steerability ceiling. — [ParaSpeechCLAP arXiv 2603.28737](https://arxiv.org/html/2603.28737v1)

### Inferences (proposed scalar definitions)
- **Effect-size steerability.** S_dir,voice = median over texts of (Δ / σ_baseline). Report a 95% bootstrap CI over texts. Sign-correct so that a positive value means the requested direction.
- **Hit-rate steerability.** H_dir,voice = fraction of (text, seed) pairs where Δ exceeds the baseline noise band in the right direction. The value is bounded in [0, 1] and interpretable, and it is comparable across directions because the noise band is self-normalizing.
- **Leakage / specificity.** Off-target movement: for direction "loud," does F0 or rate also change? Report a direction × feature confusion matrix of standardized Δs. A good model has a dominant diagonal, analogous to InstructTTSEval APS's "independent manipulation."
- **Checkpoint score.**
  - Mean over directions of H (or of clipped S), with per-direction regressions flagged by a paired test between checkpoints on the same text, seed and voice grid.
  - Guards plotted alongside: ΔWER, ΔUTMOS or NISQA, and the speaker-similarity drop of instructed vs. uninstructed output.
  - View it as a Pareto front (steerability vs. naturalness or stability) rather than a single number.
- **Uninstructed stability.** Track the variance of features across seeds without instructions. Inworld notes that low steerability can be a desirable stability property for some voices.

### Gaps
- No published standard steerability scalar or checkpoint-tracking protocol with numbers was found. Inworld's charts are not quantified in text.
- No quantitative steerability–naturalness trade-off curves were found in the sources fetched.
