# SteerTTS evaluation

What each tool measures, which judge it uses, and which numbers are independent of training. Launch wrappers:
`scripts/30_leaderboard.sh`, `scripts/31_seedtts.sh`, `scripts/32_benchmark.sh`.

## Tools

| Tool | Measures | Judges / protocol |
|---|---|---|
| `steer.harness` | **Paired steerability** on training voices (default: first 3 EARS speakers; in-loop during training: `--harness_voices` x `--harness_texts`). For each (voice, text): an uninstructed baseline render + 14 control directions (+/-2 z; whisper -4) + 7 tag directions. Per direction: *steer ratio* = median achieved / requested z, *hit* = share moving > 0.5 z the right way, one-sided Wilcoxon p (Holm across directions), *leak* = mean abs change on the other dims, plus a direction x dim leakage matrix. | The same DSP `features.measure` used to build the training labels, converted to the voice's speaker z. Independent of any learned judge. |
| `steer.evaluate` | Round-A evaluation of a finished checkpoint: harness on more voices / texts, Whisper WER per direction, ear-probe emotion on tag directions. | Whisper (`steer.asr`), ear probe. |
| `steer.tag_eval` | Tag accuracy beyond the harness: 13 emotion tags (paired delta vs the no-request render judged by the ear probe), inline events and scenes (AudioSet AST class score rises by > 0.1 and ends > 0.2), clean-render false-positive rate. | Ear probe (**also a training reward**), AST. |
| `steer.quick_eval` | Fixed subsets, minutes per checkpoint, runs beside training: *tts* 120 LibriTTS-R test-clean sentences x 4 training voices (WER, UTMOS); *clone* 80 zero-shot items from unseen LibriTTS-R speakers (WER, UTMOS, SIM-o); *prompt* 64 ParaSpeechCaps test prompts through the adapter (WER, UTMOS). | distil-whisper-large-v3 WER; UTMOS22 and WavLM-base-plus-sv SIM-o (**both training rewards / conditioning**). |
| `steer.benchmark` | LibriTTS-R test-clean 500 utterances (WER whisper-large-v3, UTMOS, SIM to the voice used = consistency), LJSpeech test (Matcha protocol, WER whisper-medium), zero-shot clone set (WER, SIM-o), ParaSpeechCaps test 246 prompts (WER with distil-large-v2 = the paper's ASR, UTMOS, objective gender / pitch-level / rate accuracy with thresholds fitted on real audio against PSC labels; rows whose audio is in training are flagged). Baselines (Kokoro, Piper, Matcha, Parler-mini) rendered by `steer.baselines` in a separate env. | As listed; report in `runs/bench/report.md`. |
| `steer.leaderboard_eval` | Stage 1 of the Hugging Face Open TTS Leaderboard protocol, run locally: Seed-TTS test-en (1,088) and CV3-Eval zero_shot en (500), default voice `voice_4223` or `--voice_clone`; RTFx at batch 1 on the local GPU. | The leaderboard repo's unmodified Qwen3-ASR-1.7B WER and seed-tts-eval WavLM-large SIM scorers. Independent. |
| `steer.seedtts` | Seed-TTS-eval test-en zero-shot cloning renders. | The benchmark's own `run_wer.py` (Whisper-large-v3) and UniSpeech WavLM-large SV (comparable to F5-TTS / CosyVoice numbers). Independent. |
| `steer.flow eval` / `ab` | UTMOS of L1 vs refiner renders; listening A/B pages of flows. | UTMOS (reward judge). |
| `steer.abtest`, `steer.samples` | Listening pages (index.html + TensorBoard audio) across checkpoints: voices, prompt+text pairs, directions, cloning, inline markup. | Human listening. |
| `steer.tb_report` | Writes progress / Seed-TTS / benchmark tables to TensorBoard's TEXT tab, per run and across runs. | - |

## Headline numbers (from run logs; see caveats)

| Setting | Result |
|---|---|
| Harness, base 312.5k | continuous ratio median 0.56, hit 0.90, leak 0.34; weakest: monotone 0.30, more_pauses 0.27 (hit 0.56), falling_end 0.38 |
| Guidance s = 2 (harness, earlier checkpoint) | ratio 0.41 -> 0.85, leak 0.30 -> 0.55 |
| Open TTS Leaderboard protocol, English avg WER, no clone | 1.80 / 1.81 % (base 305k / 312.5k; ~12th of 32), 1.69 % with `voice_4223` |
| Cloning SIM (WavLM-large, leaderboard scorer) | base 27.3 -> 32.5-35.5 with prompted flows (board minimum 37.2); real Seed targets 72.0, Vocos copy-synthesis 68.4 |
| ParaSpeechCaps test WER (distil-large-v2) | 3.07 vs Parler-TTS 4.62 (paper); attribute pitch / rate accuracy ~0.6 vs ~0.9 on real speech |
| Refiner effect (quick_eval tts UTMOS) | 4.12 -> 4.27 |

## Judge independence

* **Independent of training**: Whisper / Qwen3-ASR WER, WavLM-large SIM (Seed-TTS / leaderboard), the DSP harness
  (the DSP defines the labels, so the harness measures consistency with the label definition, not perception).
* **Circular**: UTMOS22 (reward w 0.3 and quality metric), WavLM-base-plus-sv (conditioning encoder, `r_spk` reward and
  the SIM-o metric in quick_eval / benchmark: quick_eval clone SIM 0.925 vs 0.27-0.35 with WavLM-large), the CREMA-D
  ear probe (`r_emotion` reward and `tag_eval` judge, identical pairing). Use independent judges (e.g. DNSMOS / NISQA,
  emotion2vec, ECAPA SIM) or human ratings for paper claims.
* The ear probe is weak on real EARS speech (anger 0.37, sadness 0.11, disgust 0.0 recall); `steer.ear2` is an
  in-domain alternative with speaker-disjoint CV.

## Protocol caveats

* Text-normalisation rules (`text._expand`) were derived from CV3-Eval error analysis; the default voice was chosen as
  the lowest-WER of 8 on the full Seed + CV3 sets; flow design decisions used 250 + 250 subsets of those sets (the
  first 250 CV3 rows are the hard half). A held-out dev split is needed for clean claims.
* Harness / tag_eval texts are also the in-training model-selection texts.
* quick_eval rows rendered at different dates used the then-current `text.py`, so WER differences across dates can
  be front-end changes rather than checkpoint changes.
* Leaderboard scorers must run one at a time (they hold a GPU flock): concurrent runs OOM and silently leave
  `pred_text` empty (WER 100 %).
* RTFx is measured on the local RTX 5070 at batch 1; the public leaderboard uses an H200.
* `seedtts` and cloning use only the prompt's speaker vector in the base (not its text or prosody) and render at the
  model's level-normalised loudness; the prompted flow additionally sees up to 8 s of the prompt mel.
