# Licence (DRAFT, final terms to be decided by the authors)

This repository holds two kinds of material, each under its own licence.

## 1. Code: MIT License

This covers `steertts/`, `tests/`, `inference.py` and `leaderboard/`.

Copyright (c) 2026 The SteerTTS authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 2. Model weights and voice data: CC BY-NC 4.0 (placeholder)

This covers `*.safetensors`, `voices.json`, `control_stats.json` and `config.json`.

- **Licence.** Placeholder: Creative Commons Attribution-NonCommercial 4.0 International,
  https://creativecommons.org/licenses/by-nc/4.0/. Non-commercial use only.
- **Why.** The weights were trained partly on CC BY-NC 4.0 corpora (EARS, Expresso), on NonverbalTTS (its source
  terms apply), and on synthetic speech from a commercial TTS API, whose terms of service include a
  competing-model clause. Use of the weights must also respect the most restrictive of these terms. The model card
  (`README.md`, "Training data") lists every source.
- **Voice cloning.** Clone only voices whose owners have consented. See "Intended use and misuse" in the model card.
