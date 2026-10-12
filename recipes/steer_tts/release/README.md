---
language:
- en
license: cc-by-nc-4.0  # placeholder: "CC BY-NC 4.0 (TBD by authors)"
library_name: steertts
pipeline_tag: text-to-speech
tags:
- text-to-speech
- tts
- controllable-tts
- prosody-control
- mamba
- state-space-model
- flow-matching
- voice-cloning
- small-model
---

# SteerTTS v1

A small English text-to-speech model that you steer with numbers. Each request can set any subset of **12 acoustic
controls**, for example speaking rate, loudness, median pitch, pitch range, final rise or fall, pauses, spectral tilt
and breathiness. Each control is a z-score relative to the chosen voice's own neutral speech. You can also pick a
**style tag**, give a **free-text direction** (which a small adapter turns into a request you can inspect), or use
**inline markup**. The model can **clone** a voice from a short reference clip.

| | |
|---|---|
| Acoustic model | **9.59M** parameters (encoder/decoder: attention + bidirectional Mamba-3, AdaLN conditioning) |
| + Vocos vocoder | 13.53M (`charactr/vocos-mel-24khz`, frozen), **23.1M** total |
| Flow-matching refiner (optional, on by default) | 17.64M → 40.8M total |
| Prompted flow (cloning only) | 24.67M, used in place of the refiner when `ref_wav` is given → 47.8M total |
| Frozen external encoders | WavLM-base-plus-sv 101M (cloning only), all-MiniLM-L6-v2 22.7M (free-text prompts only) |
| Output | 24 kHz mono, fp32, CPU or CUDA |
| Voices | 2,825 built-in voices under anonymised names (`voice_0001` …), default `voice_1361` |

> **Licence: CC BY-NC 4.0 (TBD by authors).** The weights were trained partly on non-commercial data (see
> [Training data](#training-data)). This is a placeholder until the authors decide the final terms. The package code
> is MIT.

## Quick start

```bash
git clone <this repo> steertts-v1 && cd steertts-v1      # or download it with huggingface_hub.snapshot_download
pip install ".[all]"          # core: torch, vocos, g2p_en …; [all] adds transformers (prompts, cloning) + parselmouth (measuring)
steertts say "Hello there, nice to meet you." -o hello.wav
steertts say "I can't believe it worked." --tag happiness --ctrl rate=1,f0_iqr=1.5 -o happy.wav
steertts say "Please keep this quiet." --prompt "whispering, slow and secretive" -o secret.wav   # prints the resolved request
steertts stream "Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow>" -o markup.wav
steertts say "Now in your voice." --ref-wav consenting_speaker.wav -o clone.wav
steertts voices --gender female | head
```

```python
from steertts import SteerTTS

tts = SteerTTS.from_pretrained(".")               # local dir or an HF repo id; device: cuda if available, else cpu
wav = tts("The meeting moved to Thursday.")       # float32 numpy, 24 kHz, default voice voice_1361
wav = tts("Are you serious?", voice="voice_0042", ctrl={"f0_end": 2.0, "f0_iqr": 1.5})   # rising, lively
wav = tts("It is over.", tag="sadness", ctrl={"loud": -1}, guidance=1.5)

req = tts.resolve({"text": "See you soon.", "prompt": "excited, a bit faster"})
print(req)    # {'text': ..., 'voice': 'voice_1361', 'tag': 'excited', 'ctrl': {'rate': 0.29}, 'prompt_mapping': {...}, ...}
wav = tts.synthesize(req)                         # a resolved request renders exactly as the prompt would

for chunk in tts.stream("First sentence. Second sentence."):   # sentence-level streaming
    play(chunk.audio)
tts.save(wav, "out.wav")
```

`inference.py` renders one example of each feature. `python -m steertts.cli --help` lists the CLI.

## The request

Every synthesis is one JSON object. It is documented field by field in [`request.schema.json`](request.schema.json)
(JSON Schema 2020-12; `steertts schema` prints it). Only `text` is required.

| Field | Meaning |
|---|---|
| `text` | English text. Numbers, units, currency, ordinals, ranges and unknown acronyms are spelled out. |
| `voice` / `ref_wav` | A built-in voice, **or** a reference clip to clone (first 8 s, at least 0.5 s). |
| `tag` | One of 51 style tags (`none`, `whisper`, `sadness`, `anger`, `fast`, `high_pitch`, …). |
| `ctrl` | `{control: z}` over the 12 controls below, each in [-5, 5]. |
| `prompt` | Free-text direction. `resolve()` maps it to `tag` + `ctrl`; an explicit `tag` or `ctrl` dim wins. |
| `guidance` | Steering scale s (default 1). See [Guidance](#how-well-it-steers). |
| `pace` | Global tempo divisor (>1 is faster). It does not change the style. |
| `word_shift` | `{word_index: {pitch: st, energy: dB, dur: ×}}`, per-word prosody edits. |
| `markup` | Parse inline markup in `text` (see below). `stream()` always parses it. |
| `flow`, `seed` | Use the flow refiner (default true) and set the sampler seed. Renders are deterministic per device. |

**Controls and the mask.** The 12 controls are utterance-level acoustic measurements:

- `rate`: articulation rate
- `loud`: speech level
- `f0_med`: median pitch
- `f0_iqr`: pitch range
- `f0_end`: final pitch slope
- `pause`: pause share
- `tilt`: spectral tilt
- `voiced`: voiced share
- `flat`: spectral flatness
- `centroid`: spectral centroid
- `cpps`: cepstral peak prominence
- `h1h2`: H1–H2

Each value is a z-score against the chosen voice's own neutral speech, in units of the pooled within-speaker SD
(`control_stats.json`). The model also receives an availability mask:

- A dim named in `ctrl` is requested.
- A dim left out is unspecified, and the model chooses it from voice, tag and text.
- `{"rate": 0}` asks for this voice's neutral rate. Omitting `rate` lets a `fast` tag or the text decide.

`steertts measure out.wav --text "…" --voice voice_1361` reads the controls back with the same measurement code
that labelled the training data.

**Markup** (`markup: true` or `stream()`):

| Markup | Effect |
|---|---|
| `*word*` or `<emph>…</emph>` | Emphasis: +2.5 st, +4 dB, ×1.2 duration |
| `<slow> <fast> <high> <low> <loud> <soft>` | Per-word changes; the sentence is still rendered in one pass |
| `[pause]`, `[pause 300ms]`, `[pause 1.5s]` | Exact silence |
| `<whisper>…</whisper>` (any tag), `<style prompt="…">…</style>` | The span is rendered as its own segment |

Text is rendered sentence by sentence, so the first sentence can play while the rest renders.

## Architecture

FastPitch-style explicit-variance TTS:

1. **Front end.** g2p_en ARPAbet phonemes with word indices.
2. **Encoder.** 4 blocks.
3. **Variance predictors.** Per-phone duration, pitch and energy.
4. **Length regulator.**
5. **Decoder.** 4 blocks.
6. **Output.** A 100-band log-mel, rendered by Vocos.

Each encoder/decoder block is a **hybrid**: multi-head self-attention, then a **bidirectional Mamba-3** branch (two
Mamba-3 layers run forward and backward and summed; complex-valued state, trapezoidal discretisation; 3.70M
parameters in total), then a conv FFN.

Every LayerNorm in every block and predictor is an **adaptive LayerNorm**. Its scale and shift are predicted from
g = MLP([speaker vector, tag embedding, ctrl ⊙ mask, mask]). The speaker slot is a linear projection of a frozen
512-d WavLM-base-plus-sv x-vector. Built-in voices use their stored mean vector; cloning uses the vector of the
reference clip.

**Guidance** renders the request and the uninstructed request, then extrapolates durations, pitch, energy and mel:
out = out₀ + s·(out − out₀).

Optional decoders:

- **Flow refiner.** A conditional flow-matching ConvNeXt network (OT-CFM, 10 midpoint steps, temperature 0.4). It
  samples mel detail given the acoustic model's mel and g, so every control still comes from the acoustic model.
- **Prompted flow.** The same network plus attention, given the reference clip's own mel as context
  (Voicebox-style infilling). Used for cloning.

The `steertts` package here is self-contained. It is the inference subset of the training code, operation for
operation, and its outputs are bit-identical to the training repository's synthesis on the same device (see
[Release verification](#release-verification)).

## How well it steers

The numbers below are **held-out**: 16 LibriTTS-R test-clean speakers who share no speaker, chapter or recording
with the training data (8 female, 8 male by a median-F0 rule; the corpus has no gender labels on disk), each
conditioned on the speaker vector of one recording, × 8 texts. They are for the acoustic model at guidance 1 **without the flow refiner**.
Steering through the refiner was not measured in the paper; the release check below confirms only the direction.

Columns:

- **Hit**: share of renders that move more than 0.5 SD the requested way.
- **Δz**: achieved change for a request of ±2.
- **ρ**: Spearman correlation between request and readback (by our extractor / by Praat).

| Tier | Control | Hit at −2 / +2 | Δz at −2 / +2 | ρ (ours / Praat) |
|---|---|---|---|---|
| **Reliable** | loud | 1.00 / 1.00 | −1.57 / +1.81 | .97 / .97 |
| | f0_med | .98 / 1.00 | −1.15 / +1.18 | .95 / .95 |
| | h1h2 | 1.00 / 1.00 | −1.59 / +1.49 | .96 / .96 |
| | tilt | 1.00 / .99 | −1.39 / +0.97 | .96 / .87 |
| | centroid | .78 / .94 | −0.79 / +1.20 | .92 / .92 |
| | f0_iqr | .66 / .91 | −0.63 / +1.09 | .88 / .88 |
| **Usable** | rate | 1.00 / 1.00 | −1.61 / +1.30 | .96 / .66 |
| | f0_end | .96 / .99 | −0.85 / +1.31 | .94 / .83 |
| | flat | .62 / .41 | −0.64 / +0.51 | .85 / .68 |
| **Weak** | pause | .38 / .64 | −0.37 / +0.71 | .76 / .70 |
| | voiced | .39 / .12 | −0.44 / +0.26 | .79 / .60 |
| | cpps | .56 / .35 | −0.45 / +0.30 | .58 / .46 |

- **Tiers.** Reliable: ρ ≥ 0.85 under both extractors. Usable: under one. Weak: under neither.
- **Gain.** The mean gain (achieved / requested) is about 0.5. `rate` and `f0_med` at +1 hit only .48 / .42.
- **Cross-talk.** Mean leakage into the other 11 dims is 0.15–0.30 SD. The off-diagonal / diagonal ratio is 0.10.
- **CPPS** is not a working control: a request moves pitch and level about as much as CPPS itself.
- **Intelligibility.** WER (Whisper-large-v3) is 0.94 % uninstructed and 0.84–1.01 % across all requests.

**Guidance** (pooled over loud, rate, f0_med, f0_iqr and tilt; ±2 requests; n = 1,280):

| s | Steer ratio (Δz / requested) | Hit | Leak | WER |
|---|---|---|---|---|
| 1 | 0.65 | .95 | .20 | 0.9 % |
| 2 | 1.35 | .99 | .34 | 1.0 % |
| 3 | 1.97 | .98 | .48 | 1.0 % |

- s ≈ 1.5 roughly delivers the requested size, at the cost of more leakage.
- Guidance on tags is riskier: `whisper` at s = 3 raises WER to 16 %.

**Markup** (8 held-out voices × 6 sentences × 2 words):

- Emphasis achieved +2.58 st and +3.92 dB, against +2.5 st and +4 dB requested.
- `<high>` / `<low>` gave +3.18 / −2.96 st for ±3.
- `<loud>` / `<soft>` gave +4.84 / −5.54 dB for ±6. The decoder attenuates level by 10–20 %.
- `<slow>` / `<fast>` and pauses are exact by construction.
- Neighbouring words move by at most 0.19 st and 0.16 dB.

**Free-text prompts** (161 hand-written prompts):

- Which-dims precision / recall: .85 / .66. Sign accuracy: .88.
- Tag top-1 / top-3: .56 / .69.
- Composition (.23) and negation (.22) mostly fail. "Not fast" can come out fast.
- Absolute-register prompts ("a deep male voice") are not supported. Pick a voice instead (`voices.json`).
- **Always look at the resolved request.**

## Quality

Results under the Open TTS Leaderboard protocol (run locally on an RTX 5070, not an official submission;
Qwen3-ASR-1.7B WER, WavLM-large SIM):

| Setting | Seed-TTS en | CV3-Eval en | Avg |
|---|---|---|---|
| Default voice `voice_1361`, acoustic model + Vocos, WER | 1.00 % | 2.63 % | **1.81 %** |
| Cloning (prompted flow), WER / SIM ×100 | 1.01 % / 37.4 | 2.73 % / 28.7 | 1.87 % / 33.1 |

- The flow refiner was scored with an earlier acoustic checkpoint: 1.04 % / 2.70 % WER (no change).
- **Cloning similarity is low.** The board's lowest cloning SIM is about 37, and its median is about 65. Do not
  expect a convincing likeness.

## Speed

Measured on an RTX 5070 and on a shared, heavily loaded 8-thread CPU, so treat CPU figures as lower bounds:

- **GPU.** Batch-1 RTFx 61–81 for the acoustic model + Vocos (leaderboard runs).
- **CPU, 8 threads.** RTFx 3.6–7.9 for the acoustic model + Vocos. The flow refiner adds 12× the compute and
  drops RTFx from 6.7 to 0.8 (0.4 with 1 thread). Use `flow=False` / `--no-flow` for real-time CPU use.
- **Cloning on CPU.** RTFx 0.2 with the prompted flow, including the WavLM speaker encoder.
- **Streaming on CPU.** First audio after about 1 s.
- See [Release verification](#release-verification) for the numbers measured on this exact package.

## Limitations

- **English only.** The front end is ARPAbet. Single speaker per request, and the controls are utterance-level.
- **Emotion tags are style switches, not emotions.** An independent emotion classifier (emotion2vec+ large)
  recognises only `sadness` (0.16 top-1 vs 0.01 uninstructed). Every other emotion tag scores at most 0.02. The same
  classifier recognises 7 of 10 of these emotions on real EARS recordings. 27 of 49 tags are acoustically within 0.5
  of `neutral`. Voice-quality tags such as `whisper` act strongly (voiced share −9.9 SD).
- **Event tokens only reserve time.** `[laugh]`, `[cough]`, `[dog]`, … add about 0.12 s of near-silence. An audio
  tagger detects 0 of 1,728 renders. The project's demo mixed recorded sounds in, but that mixer is not part of this
  release.
- **The acoustic-environment input is untrained.** The weights contain an environment projection, but the final
  training stage never saw a non-zero environment vector, so this package does not expose it.
- **Cloning similarity is low** (see above), and cloning speed on CPU is slow with the prompted flow.
- **The controls are acoustic measurements, not percepts.** `voiced`, `flat` and `cpps` barely move. `rate` and
  `tilt` depend on how they are measured. z = 0 is the voice's typical render, not its real recordings.
- **Test-set-informed choices.** The prompted-flow checkpoint was chosen on subsets of the Seed-TTS / CV3 test
  sets, and the text-normalisation rules came from CV3 error analysis. The default voice `voice_1361` is the
  pre-registered one: a sweep picked a lower-WER voice on the test sets, but it is not the default here.
- **No human evaluation.** Every number above is from automatic metrics, with one training seed.

## Intended use and misuse

**Intended uses:**

- Research on controllable / interpretable speech synthesis.
- Accessibility and prototyping, where explicit, inspectable control matters more than peak naturalness.
- On-device experiments with a ~23M-parameter TTS.

**Voice cloning requires consent.** Clone only a voice whose owner has agreed, and do not use this model to:

- impersonate real people;
- produce deceptive or fraudulent audio;
- bypass voice authentication;
- create non-consensual content.

Disclose synthetic speech to listeners. There is no watermark yet.

**Built-in voices are anonymised.** Built-in voices are training speakers under opaque names. Speaker IDs from the
source corpora are not included. Speakers from a celebrity-interview corpus (VoxCeleb, via NonverbalTTS) were
excluded from the voice list.

## Training data

410,566 utterances (659 h) from 4,919 speakers:

| Source | Hours | Licence |
|---|---|---|
| LibriTTS-R (clean-100, clean-360, other-500) | 516 | CC BY 4.0 |
| EARS (read + freeform) | 78 | CC BY-NC 4.0 |
| Expresso | 11 | CC BY-NC 4.0 |
| NonverbalTTS (VoxCeleb and Expresso sources) | 12.5 | Source terms apply |
| Synthetic speech from a commercial TTS API | 43 | The provider's terms of service, incl. a competing-model clause |

Control labels are measured from the audio. Style tags come from the corpora's own style/emotion labels.

**Licence: CC BY-NC 4.0 (TBD by authors). Non-commercial use only until stated otherwise.**

## Files

| File | Contents |
|---|---|
| `model.safetensors` | Acoustic model (fp32) |
| `flow.safetensors`, `pflow.safetensors` | Flow refiner and prompted flow |
| `prompt_adapter.safetensors` | Free-text → request adapter (0.5M) |
| `speaker_vectors.safetensors` | 512-d vector per built-in voice, in `config.json:voices_order` |
| `voices.json` | Measured descriptions (see below) |
| `control_stats.json` | Pooled SDs, tag list, and each voice's neutral control means (for `measure_z`) |
| `config.json` | Architecture and file map |
| `request.schema.json` | The request schema |
| `steertts/` | The package (incl. `steertts/mamba3` with its own invariant tests) |
| `tests/` | pytest suite |
| `leaderboard/` | Open TTS Leaderboard backend |
| `TEST_LOG.md` | Release verification log |
| `LICENSE.md` | Code (MIT) and weights (CC BY-NC 4.0, placeholder) |

`voices.json` descriptions are measured, not copied from corpus metadata:

- **Gender.** A probe on the speaker vector; 99.3 % 5-fold accuracy on 534 self-labelled speakers.
- **Pitch register.** Median F0 within that gender.
- **Pace.** Articulation rate.

## Release verification

The package was verified in a fresh CPU environment; `TEST_LOG.md` has the full log.

- **Tests.** The pytest suite passes (16 tests).
- **Parity.** Seven renders are bit-identical to the training code's synthesis on the same device: acoustic
  only, controls, tag + guidance, per-word shifts, the flow refiner, and cloning through the prompted flow.
- **Speed.** On this package, CPU with 4 threads on a loaded machine: RTFx 4.9 without the refiner, 0.7 with it.
  First streamed audio arrives after 0.5 s.

## Citation

```bibtex
@misc{steertts2026,
  title  = {SteerTTS: (title TBD)},
  author = {(authors TBD)},
  year   = {2026},
  note   = {Model card placeholder; citation to be added by the authors}
}
```
