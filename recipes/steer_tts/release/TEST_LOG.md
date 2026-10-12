# Release verification log: steertts-v1

Date: 2026-10-11. Everything ran on the CPU (4 threads, `nice -n 19`) of a shared 48-core workstation, with a 1-min
load average of 20 to 40 from other jobs. The timings are lower bounds.

## Weights

| File | Source checkpoint | sha256 (first 16) |
|---|---|---|
| `model.safetensors` | acoustic model, step 312,500 | `adc11912871113e3` |
| `flow.safetensors` | flow refiner (EMA), step 100,000 | `585ff2c931811c30` |
| `pflow.safetensors` | prompted flow with prompt attention (EMA), step 40,000 | `877f81523dfaba50` |
| `prompt_adapter.safetensors` | free-text → request adapter | `aca34bf753456e46` |
| `speaker_vectors.safetensors` | 2,825 built-in voices (public names only) | `edcabfd02dd4a520` |

These are the checkpoints behind the model card's numbers. The prompted flow is the one in the leaderboard cloning row
(1.01 % / 37.4 and 2.73 % / 28.7).

## 1. Fresh-environment install

- A new Python 3.11 venv (uv) with the CPU torch wheel. Then `pip install ".[test]"` from a copy of the package
  code, holding no weights.
- Resolved versions: torch 2.11.0+cpu, torchaudio 2.11.0+cpu, numpy 2.4.6, safetensors 0.8.0, vocos 0.1.0,
  g2p_en 2.1.0, nltk 3.10.3, num2words 0.5.14, soundfile 0.14.0, soxr 1.1.0, transformers 5.19.0,
  praat-parselmouth 0.4.7, pytest 9.1.1, jsonschema 4.26.0.
- Not fresh: the Hugging Face hub cache and `~/nltk_data` came from the host. A first run on a new machine
  downloads `charactr/vocos-mel-24khz`, plus `microsoft/wavlm-base-plus-sv` for cloning and
  `sentence-transformers/all-MiniLM-L6-v2` for free-text prompts, and the NLTK tagger and CMUdict.

## 2. Test suite

`STEERTTS_MODEL=<this dir> pytest` from the fresh env gave **16 passed** in 6 min 11 s. The suite covers:

- the Mamba-3 invariant suite;
- text normalisation and phonemes;
- the request schema and its examples;
- request validation errors;
- deterministic renders, both acoustic-only and through the flow refiner;
- pace and rate;
- Praat readback direction for `loud`, `f0_med` and `h1h2`;
- markup and streaming;
- cloning;
- prompt → request mapping.

## 3. Parity with the training code

The same inputs were rendered on the same device (CPU) by this package and by the training repository's
`steer.synth.Synth`, given the source checkpoints above. Every case was **bit-identical** (max |Δ| = 0):

| Case | Samples | Identical |
|---|---|---|
| default voice, acoustic only | 73,472 | yes |
| `ctrl` rate/f0_med/f0_end, acoustic only | 68,096 | yes |
| tag `sadness` + `loud` −1, guidance 1.5 | 85,760 | yes |
| `word_shift` (pitch/energy/dur on one word) | 37,376 | yes |
| another built-in voice, flow refiner | 73,216 | yes |
| `ctrl` loud/f0_iqr, flow refiner | 75,008 | yes |
| cloning from a reference clip, prompted flow | 51,456 | yes |

Cross-environment check: two renders (acoustic only, and flow + ctrl) were bit-identical between the fresh venv
(torch 2.11.0+cpu) and the training environment (torch 2.11.0+cu128, run on the CPU). GPU renders are deterministic
per device but are not bit-identical to CPU renders.

## 4. Speed on this package (CPU, 4 threads, load average about 22)

| Path | RTFx (3 texts, 1.7–5 s) |
|---|---|
| Acoustic model + Vocos (`flow=False`) | 4.35–5.59 (median 4.89) |
| + flow refiner (default) | 0.62–0.79 (median 0.71) |
| Streaming, first chunk (`flow=False`) | 0.49 s |

## 5. Hygiene

- No source-corpus speaker or clip identifiers appear in any file. The build fails on the pattern check in
  `steer.release`, and the check was re-run here with grep.
- VoxCeleb-derived speakers are excluded from the voice list.
- The schema `$id` is a URN (`urn:steertts:request-schema:v1`), not a URL.
- `leaderboard/` names a placeholder Hugging Face repo id. It is not referenced by the package code.
