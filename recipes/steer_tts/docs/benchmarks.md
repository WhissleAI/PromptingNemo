# SteerTTS: external benchmarks (literature survey, 2026-10-08)

Few published numbers are directly comparable: controllable-TTS papers each use their own test set and
classifier. Apples-to-apples (same test set + same metric model) is only possible on Seed-TTS-eval test-en,
LibriSpeech-PC test-clean (F5-TTS protocol), ParaSpeechCaps holdout (WER), and InstructTTSEval (LLM judge).
Everything else is a rough reference unless we re-run baselines (Parler-TTS-mini) through our own pipeline.

## Plan

Metric models (all fit in 12 GB):
- WER: Whisper-large-v3 + Whisper normaliser (Seed/CapSpeech convention); also HuBERT-large-ft CTC (F5, ControlSpeech).
- SIM: WavLM-large-SV (`wavlm_large_finetune.pth` from seed-tts-eval), against a held-out clip of the same speaker.
- Quality: UTMOS22-strong (most reported) + UTMOSv2 (CapSpeech).
- Emotion: emotion2vec_plus_large: per-emotion accuracy + UAR, also on text-neutral sentences.
- Attributes: binned (low/normal/high) pitch/rate/energy accuracy (PromptTTS2/ControlSpeech style) and
  correlation with the requested control (Parler style); gender via a WavLM / Vox-Profile classifier.

Test sets:
1. LibriTTS-R test-clean, fixed 500-utt subset, 4-10 s (CC BY 4.0, openslr 141): WER / UTMOS / SIM, in-domain.
2. LJSpeech test (Matcha split, 100 sentences, public domain): text-only WER vs Matcha table (Whisper-medium there).
3. Seed-TTS-eval test-en transcripts (1k): WER only (we don't clone).
4. ParaSpeechCaps holdout (Expresso/EARS/LibriTTS-R; https://github.com/ajd12342/paraspeechcaps): closest to our
   data. Check its holdout utterances are not in our train split first.
5. InstructTTSEval-en APS subset (1k, CC BY 4.0; https://github.com/KexinHUANG19/InstructTTSEval): needs Gemini.
6. Emotion: ESD English text (academic licence) or EmoVoice-DB test (HF yhaha/EmoVoice-DB), scored with emotion2vec+.

## Reference numbers

Key: ✔ means same test set and same metric model. ~ means same metric model but different text, speakers or protocol.

| Metric / test set | Metric model | Literature | Comparable |
|---|---|---|---|
| WER, LJSpeech test | Whisper-medium | FastSpeech2 (41M) 4.18, VITS 2.52, Matcha (18M) 2.09, vocoded GT 1.97 | ~ |
| WER, Seed test-en | Whisper-large-v3 | F5 1.83, Spark 1.98, CosyVoice2 2.57 | ~ (not cloning) |
| WER, LibriSpeech-PC test-clean | HuBERT / Whisper-v3 | GT 2.23, F5 2.42 | ✔ |
| UTMOS, LibriSpeech test-clean | UTMOS22 | GT 4.08, CosyVoice2 4.23, Spark 4.35 | ✔ |
| Gender acc | WavLM / Vox-Profile | Parler 98.1, Spark 99.8 | ~ |
| Pitch / speed / volume bin acc | signal processing | ControlSpeech .83/.83/.89, InstructTTS .85/.76/.82, GT codec .95/.89/.98 | ~ (bin edges differ) |
| Emotion acc | emotion2vec+ large | CosyVoice3 happy .92 / sad .70 / angry .72 (text-matched; .64/.42/.58 text-unrelated); EmoSpeech (FS2) mean .72 | ~ |
| Style-ACC / UTMOSv2 / WER | Vox-Profile / UTMOSv2 / whisper-v3-turbo | CapSpeech-NAR 66.0 / 3.37 / 9.2 | ~ |
| WER, ParaSpeechCaps holdout | distil-whisper-large-v2 | Parler 4.62, PSC-Scaled 8.63, GT 7.93 | ✔ |
| InstructTTSEval-en APS / DSD | Gemini judge | Parler-mini 63.4 / 48.7, PromptTTS 64.3 / 47.2, gpt-4o-mini-tts avg 68.5 | ✔ (judge drift) |

Sources: InstructTTSEval arXiv 2506.16381; ControlSpeech 2406.01205; PromptTTS 2 2309.02285; Parler 2402.01912;
ParaSpeechCaps 2503.04713; CapSpeech 2506.02863; EmoVoice 2504.12867; CosyVoice 2407.05407 / 3 2505.17589;
EME-TTS 2507.12015; Spark-TTS 2503.01710; Matcha 2309.03199; StyleTTS 2 2306.07691; F5-TTS 2410.06885;
seed-tts-eval github.com/BytedanceSpeech/seed-tts-eval.

Uncertain: EmoVoice's emotion classifier; ControlSpeech's "PromptTTS 2" row label; Kokoro numbers are third-party only.
