"""Extended baselines for the NAACL SteerTTS comparison: registry, in-process loaders, and a subprocess batch renderer.

Standalone on purpose (stdlib only at module level, nothing from `steer`), so it imports in ANY env:
    PYTHONNOUSERSITE=1 PYTHONPATH=<recipes/steer_tts> <env python> -m steer.paper.baselines_ext ...
Every system lives in its own env (created by workspace/steer_tts/logs/baselines_ext/env_create.sh; conda gives python,
uv installs packages with hardlinks so torch 2.8.0+cu128 is stored once). Model code that is not on PyPI is cloned
under $BASELINES_SRC (default workspace/steer_tts/baselines_src). PYTHONNOUSERSITE=1 is REQUIRED: ~/.local holds a
numpy 2 for python3.10 that breaks compiled extensions in these envs.

Registry: SYSTEMS[name] = dict(env, group ("closest"|"advanced"), params_M (acoustic model, counted from the loaded
checkpoints 2026-10-11; vocoder listed separately; StyleTTS2 = inference modules only (no discriminators, aligner,
pitch extractor); ParaStyleTTS excludes its 109M all-mpnet-base-v2 prompt encoder; CosyVoice2 = LLM 505.8 + flow
112.5; XTTS = GPT 441.0; Piper from its ONNX size; Kokoro / Parler-mini / Paradee / Kitten from their model cards),
vocoder, controls (native knobs; "prompt" = free-text description), voice ("default"|"clone"|"prompt"),
checkpoint, load).  `load(device)` loads the model ONCE in-process and returns
    fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0) -> (wav float32 np.ndarray, sr)
`fn(text)` alone works for every system (cloning systems fall back to a fixed held-out reference clip), which is
what steer.paper.speed times.  rate = speed factor (>1 faster; mapped to the system's own length/pace/speed knob);
pitch = native pitch shift (FastPitch only: shift in units of its normalised-pitch SD); unsupported knobs raise.

Batch rendering (eval sets) goes through `synth()` -> subprocess in the system's env:
    synth(name, items, out_dir)  items: list of dicts {id, text, [gender, desc, ref, ref_text, rate, pitch]}
    -> out_dir/<id>.wav (24 kHz mono, resampled), skipping files that exist (resumable); returns #rendered.
CLI (inside the env): python -m steer.paper.baselines_ext render --system NAME --jobs jobs.jsonl --out DIR [--device]
                      python -m steer.paper.baselines_ext list
Eval-set planning, scoring (Whisper-large-v3 WER, UTMOS22, DNSMOS, WavLM-large SIM) and tables live in
steer.paper.baselines_ext_eval (emotts env).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("STEER_TTS_ROOT", "/home/karan/Desktop/whissle/workspace/steer_tts"))
SRC = Path(os.environ.get("BASELINES_SRC", ROOT / "baselines_src"))
RECIPE = Path(__file__).resolve().parents[2]
SR_OUT = 24000
ENVS = Path.home() / "anaconda3" / "envs"
THREADS = int(os.environ.get("BASELINE_THREADS", "4"))

# fixed held-out references (LibriTTS-R test-clean, data/bench) used when a cloning system is called without `ref`
_DEFAULT_REF = {}


def default_ref(gender="female"):
    """First data/bench/libritts.jsonl clip (4-10 s) of the requested gender (benchmark F0 rule) -> (path, text)."""
    if gender not in _DEFAULT_REF:
        import math
        cut = json.load(open(ROOT / "data/bench/calibration.json"))["gender_cut_st"]
        import soundfile as sf
        for l in open(ROOT / "data/bench/libritts.jsonl"):
            r = json.loads(l)
            g = "female" if 12 * math.log2(r["spk_f0"] / 100.0) > cut else "male"
            if g == gender and 4.0 <= sf.info(r["gt_path"]).duration <= 10.0:
                _DEFAULT_REF[gender] = (r["gt_path"], r["text"])
                break
    return _DEFAULT_REF[gender]


def _np(x):
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, np.float32).squeeze()


def _knobs(name, rate, pitch, allow_rate=True, allow_pitch=False):
    if rate != 1.0 and not allow_rate:
        raise ValueError(f"{name}: no native rate knob")
    if pitch != 0.0 and not allow_pitch:
        raise ValueError(f"{name}: no native pitch knob")


def _torch_threads():
    import torch
    torch.set_num_threads(THREADS)
    return torch


# ------------------------------------------------------------------------------------------------------- loaders
def load_matcha(device="cuda"):
    """Matcha-TTS (Mehta et al., ICASSP 2024), official matcha_ljspeech + HiFi-GAN (LJ, T2_v1), 10 ODE steps,
    temperature 0.667 (the package defaults). Knob: length_scale = 1/rate."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # official GitHub-release checkpoints (omegaconf)
    torch = _torch_threads()
    from matcha.cli import (MATCHA_URLS, VOCODER_URLS, assert_model_downloaded, get_user_data_dir, load_matcha as lm,
                            load_vocoder, process_text, to_waveform)
    d = get_user_data_dir()
    assert_model_downloaded(d / "matcha_ljspeech.ckpt", MATCHA_URLS["matcha_ljspeech"])
    assert_model_downloaded(d / "hifigan_T2_v1", VOCODER_URLS["hifigan_T2_v1"])
    dev = torch.device(device)
    model = lm("matcha_ljspeech", d / "matcha_ljspeech.ckpt", dev)
    voc, den = load_vocoder("hifigan_T2_v1", d / "hifigan_T2_v1", dev)

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("matcha", rate, pitch)
        with torch.inference_mode():
            t = process_text(0, text, dev)
            torch.manual_seed(0)
            o = model.synthesise(t["x"], t["x_lengths"], n_timesteps=10, temperature=0.667, spks=None,
                                 length_scale=1.0 / rate)
            return _np(to_waveform(o["mel"], voc, den)), 22050
    return fn


def load_fastpitch(device="cuda"):
    """FastPitch (Lancucki, ICASSP 2021), NVIDIA NeMo `tts_en_fastpitch` (LJSpeech, IPA) + `tts_en_hifigan`.
    Knobs: pace (= rate) and an additive shift of the predicted normalised pitch (`pitch` in SD units of the
    model's pitch normalisation), applied through FastPitchModule.infer(pitch=shift), which adds it to the pitch predictor output."""
    torch = _torch_threads()
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel
    fp = FastPitchModel.from_pretrained("tts_en_fastpitch").to(device).eval()
    voc = HifiGanModel.from_pretrained("tts_en_hifigan").to(device).eval()

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        with torch.inference_mode():
            toks = fp.parse(text, normalize=True)
            shift = torch.full(toks.shape, float(pitch), device=toks.device)  # added to the predicted normalised pitch
            out = fp.fastpitch.infer(text=toks, pitch=shift, pace=rate)
            spec = out[0]
            return _np(voc.convert_spectrogram_to_audio(spec=spec)), 22050
    return fn


def load_styletts2(device="cuda"):
    """StyleTTS 2 (Li et al., NeurIPS 2023), official LibriTTS checkpoint (yl4579/StyleTTS2-LibriTTS, epochs_2nd_00020)
    via the `styletts2` pip wrapper; zero-shot style from a reference clip; diffusion_steps=5, alpha=0.3, beta=0.7,
    embedding_scale=1 (the official LibriTTS demo settings). No native rate/pitch knob."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # official checkpoints pickle config objects
    torch = _torch_threads()
    from styletts2 import tts as st
    m = st.StyleTTS2()  # downloads the LibriTTS checkpoint + config from the HF hub on first use
    cache = {}

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("styletts2", rate, pitch, allow_rate=False)
        ref = ref or default_ref(gender)[0]
        with torch.inference_mode():
            if ref not in cache:
                cache[ref] = m.compute_style(ref)
            torch.manual_seed(0)
            y = m.inference(text, ref_s=cache[ref], alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1,
                            output_sample_rate=24000)
        return _np(y), 24000
    return fn


def load_parastyle(device="cuda"):
    """ParaStyleTTS (Lou et al., CIKM 2025), official HF weights haoweilou/ParaStyleTTS + all-mpnet-base-v2 style
    encoder; deterministic decoding (noise_scale=noise_scale_w=0) as in the official generate.py.
    Prompt: `desc` if given, else "A adult {gender} speaking English with neutral emotion" (its training template).
    Knob: length_scale = 1/rate."""
    torch = _torch_threads()
    repo = str(SRC / "ParaStyleTTS")
    sys.path.insert(0, repo)
    cwd = os.getcwd(); os.chdir(repo)
    try:
        from tts.model import ParaStyleTTS
        from g2p import mix_to_ipa, ipa_to_idx
        from sentence_transformers import SentenceTransformer
        tts = ParaStyleTTS.from_pretrained("haoweilou/ParaStyleTTS").to(device).eval()
        enc = SentenceTransformer("all-mpnet-base-v2", device=device)
    finally:
        os.chdir(cwd)

    import re
    from num2words import num2words

    def norm(t):
        """The released g2p has no text normaliser (digits/symbols crash it): spell out numbers and common symbols,
        drop other non-letters. Applied to ParaStyleTTS only (every other system has its own frontend)."""
        t = re.sub(r"\$(\d+(?:\.\d+)?)", lambda m: m.group(1) + " dollars", t)
        t = t.replace("%", " percent").replace("&", " and ")
        t = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), t)
        t = re.sub(r"\d+(?:\.\d+)?", lambda m: num2words(float(m.group(0)) if "." in m.group(0) else int(m.group(0))), t)
        t = t.replace("-", " ")
        return re.sub(r"[^a-zA-Z' ,.?!]", " ", t)

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("parastyle", rate, pitch)
        prompt = desc or f"A adult {gender} speaking English with neutral emotion"
        with torch.inference_mode():
            phon, tone = mix_to_ipa(norm(text).lower())
            x = torch.tensor([ipa_to_idx(phon)], device=device)
            t = torch.tensor([tone], device=device)
            s = enc.encode([prompt], convert_to_tensor=True, show_progress_bar=False).to(device)
            w = tts.infer(x, t, x_lengths=torch.tensor([x.shape[-1]], device=device), length_scale=1.0 / rate,
                          noise_scale=0, noise_scale_w=0, ab_style=s)[0]
        return _np(w[0]), 22050
    return fn


def load_kokoro(device="cuda"):
    """Kokoro-82M v1.0 (hexgrad), voices af_heart (F) / am_michael (M). Knob: speed (= rate)."""
    torch = _torch_threads()
    import numpy as np
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a", device=device)
    voice = dict(female="af_heart", male="am_michael")

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("kokoro", rate, pitch)
        return np.concatenate([_np(a) for _, _, a in pipe(text, voice=voice[gender], speed=rate)]), 24000
    return fn


def load_piper(device="cpu"):
    """Piper (VITS, ONNX), en_US-lessac-medium (F) / en_US-ryan-medium (M); CPU onnxruntime (Piper's deployment
    target; `device` is ignored). Knob: length_scale = 1/rate."""
    import numpy as np
    from piper import PiperVoice, SynthesisConfig
    from piper.download_voices import download_voice
    d = ROOT / "runs/bench/_ext/_piper_models"
    v = {}
    for g, n in dict(female="en_US-lessac-medium", male="en_US-ryan-medium").items():
        if not (d / f"{n}.onnx").exists():
            d.mkdir(parents=True, exist_ok=True); download_voice(n, d)
        v[g] = PiperVoice.load(str(d / f"{n}.onnx"))

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("piper", rate, pitch)
        cfg = SynthesisConfig(length_scale=1.0 / rate)
        ch = list(v[gender].synthesize(text, syn_config=cfg))
        return np.concatenate([c.audio_float_array for c in ch]), v[gender].config.sample_rate
    return fn


def load_parler(device="cuda", name="parler-tts/parler-tts-mini-v1"):
    """Parler-TTS (Lyth & King 2024; Lacombe et al.) mini-v1 / large-v1. Prompt = `desc` or a neutral gendered
    description (as in steer/baselines.py); greedy-free sampling with a fixed seed. No numeric knobs."""
    torch = _torch_threads()
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    m = ParlerTTSForConditionalGeneration.from_pretrained(name).to(device).eval()
    tok = AutoTokenizer.from_pretrained(name)

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("parler", rate, pitch, allow_rate=False)
        d = desc or (f"A {gender} speaker delivers clear, natural speech at a moderate speed and pitch in a very "
                     f"clean, close-sounding recording.")
        with torch.inference_mode():
            di = tok(d, return_tensors="pt").to(device)
            p = tok(text, return_tensors="pt").to(device)
            torch.manual_seed(0)
            a = m.generate(input_ids=di.input_ids, attention_mask=di.attention_mask, prompt_input_ids=p.input_ids,
                           prompt_attention_mask=p.attention_mask)
        return _np(a), m.config.sampling_rate
    return fn


def load_f5(device="cuda"):
    """F5-TTS v1 Base (Chen et al., ACL 2025) via f5_tts.api.F5TTS defaults (Vocos, NFE 32, CFG 2, sway -1).
    Zero-shot from (ref, ref_text). Knob: speed (= rate)."""
    torch = _torch_threads()
    from f5_tts.api import F5TTS
    m = F5TTS(device=device)

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("f5", rate, pitch)
        if ref is None:
            ref, ref_text = default_ref(gender)
        wav, sr, _ = m.infer(ref_file=ref, ref_text=ref_text, gen_text=text, speed=rate, seed=0,
                             show_info=lambda *a, **k: None, progress=None)
        return _np(wav), sr
    return fn


def load_cosyvoice2(device="cuda"):
    """CosyVoice 2 0.5B (Du et al. 2024), official FunAudioLLM/CosyVoice2-0.5B, fp32, no JIT/TRT/vLLM.
    ref given: zero-shot (prompt text + wav); desc given: inference_instruct2(desc + '<|endofprompt|>', ref wav) with
    a gender-matched held-out reference (it has no voice-free mode). Knob: speed (= rate; flow-level time scaling)."""
    torch = _torch_threads()
    import numpy as np
    repo = SRC / "CosyVoice"
    sys.path[:0] = [str(repo), str(repo / "third_party/Matcha-TTS")]
    from cosyvoice.cli.cosyvoice import CosyVoice2
    m = CosyVoice2(str(SRC / "models/CosyVoice2-0.5B"), load_jit=False, load_trt=False, fp16=False)

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("cosyvoice2", rate, pitch)
        if ref is None:
            ref, ref_text = default_ref(gender)
        torch.manual_seed(0)
        if desc:
            gen = m.inference_instruct2(text, desc + "<|endofprompt|>", ref, stream=False, speed=rate)
        else:
            gen = m.inference_zero_shot(text, ref_text, ref, stream=False, speed=rate)
        return np.concatenate([_np(o["tts_speech"]) for o in gen]), m.sample_rate
    return fn


def load_xtts(device="cuda"):
    """XTTS-v2 (Casanova et al., Interspeech 2024), coqui `tts_models/multilingual/multi-dataset/xtts_v2` through
    the maintained idiap coqui-tts fork; default sampling (temperature 0.75 ... from the model config), English,
    fixed seed. Zero-shot from a reference clip. Knob: speed (= rate)."""
    os.environ.setdefault("COQUI_TOS_AGREED", "1")  # CPML: non-commercial research evaluation
    torch = _torch_threads()
    from TTS.api import TTS
    m = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("xtts", rate, pitch)
        ref = ref or default_ref(gender)[0]
        torch.manual_seed(0)
        return _np(m.tts(text=text, speaker_wav=ref, language="en", speed=rate)), m.synthesizer.output_sample_rate
    return fn


def load_paradee(device="cpu"):
    """Paradee-8M v1.0 (Mahendrakar 2026, arXiv 2610.06817): 8.07M-param single-voice (af_heart) distillation of
    Kokoro-82M; official `paradee` package (misaki g2p + int8 ONNX, CPU onnxruntime; `device` ignored).
    Knob: speed (= rate)."""
    from paradee import Paradee, SAMPLE_RATE
    m = Paradee()

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("paradee", rate, pitch)
        return _np(m(text, speed=rate)), SAMPLE_RATE
    return fn


def load_kitten(device="cpu"):
    """Kitten TTS nano 0.8 (KittenML, fp32 ONNX, 15M) via `kittenml` installed --no-deps (no PyTorch); voices Bella (F)
    / Jasper (M); its own text cleaner on (numbers, currencies). CPU onnxruntime. Knob: speed (= rate)."""
    from kittenml import KittenTTS
    m = KittenTTS("KittenML/kitten-tts-nano-0.8-fp32")
    voice = dict(female="Bella", male="Jasper")

    def fn(text, gender="female", desc=None, ref=None, ref_text=None, rate=1.0, pitch=0.0):
        _knobs("kitten", rate, pitch)
        return _np(m.generate(text, voice=voice[gender], speed=rate, clean_text=True)), 24000
    return fn


# ------------------------------------------------------------------------------------------------------ registry
def _py(env):
    return str(ENVS / env / "bin/python")


SYSTEMS = {
    # -------- closest methods: small non-autoregressive / explicit-duration TTS
    "fastpitch": dict(env="tb_nemo", group="closest", params_M=45.8, vocoder="HiFi-GAN (NeMo, LJ) 13.9M",
                      controls=["pace", "pitch_shift"], voice="default", load=load_fastpitch,
                      checkpoint="nvidia/tts_en_fastpitch (NGC/HF, LJSpeech) + nvidia/tts_hifigan"),
    "matcha": dict(env="tb_matcha", group="closest", params_M=18.2, vocoder="HiFi-GAN (LJ, T2_v1) 13.9M",
                   controls=["length_scale"], voice="default", load=load_matcha,
                   checkpoint="matcha_ljspeech.ckpt (official GitHub release)"),
    "piper": dict(env="ttsbase", group="closest", params_M=15.8, vocoder="end-to-end (VITS)",
                  controls=["length_scale"], voice="default", load=load_piper,
                  checkpoint="rhasspy/piper-voices en_US-lessac-medium / en_US-ryan-medium"),
    "kokoro": dict(env="ttsbase", group="closest", params_M=82.0, vocoder="iSTFTNet (built in)",
                   controls=["speed"], voice="default", load=load_kokoro, checkpoint="hexgrad/Kokoro-82M v1.0"),
    "paradee": dict(env="tb_onnxtts", group="closest", params_M=8.07, vocoder="iSTFTNet (built in, Kokoro-style)",
                    controls=["speed"], voice="default", load=load_paradee,
                    checkpoint="sahilmahendrakar/Paradee-8M-v1.0 (int8 ONNX; single voice af_heart)"),
    "kitten": dict(env="tb_onnxtts", group="closest", params_M=15.0, vocoder="built in (end-to-end ONNX)",
                   controls=["speed"], voice="default", load=load_kitten,
                   checkpoint="KittenML/kitten-tts-nano-0.8-fp32 (voices Bella / Jasper)"),
    "styletts2": dict(env="tb_styletts2", group="closest", params_M=135.8, vocoder="iSTFTNet/HiFi-GAN (built in)",
                      controls=["reference_style"], voice="clone", load=load_styletts2,
                      checkpoint="yl4579/StyleTTS2-LibriTTS epochs_2nd_00020.pth"),
    "parastyle": dict(env="tb_parastyle", group="closest", params_M=52.5, vocoder="end-to-end (VITS)",
                      controls=["prompt", "length_scale"], voice="prompt", load=load_parastyle,
                      checkpoint="haoweilou/ParaStyleTTS (HF) + all-mpnet-base-v2"),
    # -------- widely used advanced systems
    "parler_mini": dict(env="ttsbase", group="advanced", params_M=880.0, vocoder="DAC (44.1 kHz codec)",
                        controls=["prompt"], voice="prompt", load=load_parler,
                        checkpoint="parler-tts/parler-tts-mini-v1"),
    "f5": dict(env="tb_f5", group="advanced", params_M=337.1, vocoder="Vocos 13.5M", controls=["speed"],
               voice="clone", load=load_f5, checkpoint="SWivid/F5-TTS F5TTS_v1_Base"),
    "cosyvoice2": dict(env="tb_cosyvoice", group="advanced", params_M=618.4, vocoder="HiFT 20.8M",
                       controls=["instruct", "speed"], voice="clone", load=load_cosyvoice2,
                       checkpoint="FunAudioLLM/CosyVoice2-0.5B"),
    "xtts": dict(env="tb_coqui", group="advanced", params_M=441.0, vocoder="HiFi-GAN decoder 25.9M",
                 controls=["speed"], voice="clone", load=load_xtts, checkpoint="coqui/XTTS-v2"),
}


def synth(name, items, out_dir, device="cuda", timeout=None):
    """Render `items` (dicts with id, text and optional gender/desc/ref/ref_text/rate/pitch) with system `name` in
    its own env (subprocess); writes out_dir/<id>.wav (24 kHz). Existing files are skipped. Returns #rendered."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jobs = out_dir / "_jobs.jsonl"
    with open(jobs, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=str(RECIPE), PIP_CONFIG_FILE="/dev/null",
               STEER_TTS_ROOT=str(ROOT))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(v, str(THREADS))
    e = SYSTEMS[name]["env"]
    py = e if e.startswith("/") else _py(e)
    r = subprocess.run([py, "-m", "steer.paper.baselines_ext", "render", "--system", name, "--jobs", str(jobs),
                        "--out", str(out_dir), "--device", device], env=env, timeout=timeout, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-4000:]); sys.stderr.write(r.stderr[-4000:])
    if r.returncode:
        raise RuntimeError(f"{name} render failed rc={r.returncode}")
    n = [l for l in r.stdout.splitlines() if l.startswith("RENDERED")]
    return int(n[-1].split()[1]) if n else 0


def _render(system, jobs, out, device):
    import numpy as np
    import soundfile as sf
    import soxr
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(jobs)]
    todo = [it for it in items if not (out / f"{it['id']}.wav").exists()]
    print(f"{system}: {len(todo)}/{len(items)} to render", flush=True)
    if not todo:
        print("RENDERED 0"); return
    fn = SYSTEMS[system]["load"](device)
    n = fails = 0
    for it in todo:
        try:
            y, sr = fn(it["text"], gender=it.get("gender", "female"), desc=it.get("desc"), ref=it.get("ref"),
                       ref_text=it.get("ref_text"), rate=float(it.get("rate", 1.0)), pitch=float(it.get("pitch", 0.0)))
            y = np.asarray(y, np.float32).squeeze()
            if sr != SR_OUT:
                y = soxr.resample(y, sr, SR_OUT)
            if not len(y) or not np.isfinite(y).all():
                raise ValueError("empty or non-finite audio")
            sf.write(out / f"{it['id']}.wav", y, SR_OUT); n += 1
        except Exception as e:  # noqa: BLE001 - one bad item must not stop a long render; failures are counted
            fails += 1
            print(f"{system} {it['id']} FAILED: {type(e).__name__}: {e}", flush=True)
        if n % 100 == 0 and n:
            print(f"{system} {n}/{len(todo)}", flush=True)
    print(f"RENDERED {n} FAILED {fails}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "list"])
    ap.add_argument("--system")
    ap.add_argument("--jobs")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd == "list":
        for k, v in SYSTEMS.items():
            print(k, {x: y for x, y in v.items() if x != "load"})
    else:
        _render(a.system, a.jobs, a.out, a.device)


if __name__ == "__main__":
    main()
