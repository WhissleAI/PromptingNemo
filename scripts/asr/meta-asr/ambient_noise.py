"""Procedural ambient noise augmentation for ASR training.

Inspired by Ambigram's physical modeling synthesis, implemented as NeMo-compatible
Perturbation subclasses. Every noise is synthesized from first principles — no
external audio files needed.

Noise types:
  - PinkNoisePerturbation: 1/f noise (Paul Kellet IIR filter)
  - BrownNoisePerturbation: Brownian noise (leaky integrator)
  - WindNoisePerturbation: Subtractive synthesis wind (cascaded LPF + LFO)
  - RainNoisePerturbation: Pink noise bed + Karplus-Strong raindrop transients
  - CrowdNoisePerturbation: Bandpass-filtered babble with amplitude modulation
  - MusicNoisePerturbation: Random harmonic tone clusters with vibrato
  - RoomReverbPerturbation: Synthetic exponential-decay impulse response
  - CodecNoisePerturbation: Telephone-band degradation (bandpass + quantization)
  - AmbientScenePerturbation: Compound meta-perturbation composing scene templates
"""

import random
import numpy as np
from scipy import signal as scipy_signal

from nemo.collections.asr.parts.preprocessing.perturb import Perturbation


def _rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def _mix_at_snr(clean, noise, snr_db):
    """Mix noise into clean signal at the specified SNR (dB)."""
    clean_rms = _rms(clean)
    noise_rms = _rms(noise)
    if noise_rms < 1e-10:
        return clean
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    return clean + noise * (target_noise_rms / noise_rms)


def _generate_pink_noise(n_samples):
    """Paul Kellet's refined 1/f pink noise algorithm (7-coefficient IIR)."""
    out = np.empty(n_samples, dtype=np.float64)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    for i in range(n_samples):
        w = random.random() * 2 - 1
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        b3 = 0.86650 * b3 + w * 0.3104856
        b4 = 0.55000 * b4 + w * 0.5329522
        b5 = -0.7616 * b5 - w * 0.0168980
        out[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11
        b6 = w * 0.115926
    return out.astype(np.float32)


def _generate_brown_noise(n_samples):
    """Brownian noise via leaky integrator."""
    out = np.empty(n_samples, dtype=np.float64)
    last = 0.0
    for i in range(n_samples):
        w = random.random() * 2 - 1
        last = (last + 0.02 * w) / 1.02
        out[i] = last * 3.5
    return out.astype(np.float32)


def _karplus_strong(sr, freq, duration_sec=0.05, decay=0.996):
    """Physical model of a plucked string / raindrop using Karplus-Strong."""
    n_samples = int(sr * duration_sec)
    delay_len = max(int(sr / freq), 2)
    buf = np.random.uniform(-1, 1, delay_len).astype(np.float32)
    out = np.zeros(n_samples, dtype=np.float32)
    idx = 0
    for i in range(n_samples):
        out[i] = buf[idx]
        avg = 0.5 * (buf[idx] + buf[(idx + 1) % delay_len])
        buf[idx] = avg * decay
        idx = (idx + 1) % delay_len
    return out


class PinkNoisePerturbation(Perturbation):
    """1/f pink noise — natural-sounding ambient noise."""

    def __init__(self, min_snr_db=10, max_snr_db=30, rng=None):
        super().__init__()
        self._min_snr = min_snr_db
        self._max_snr = max_snr_db
        if rng is not None:
            random.seed(rng)

    def perturb(self, data):
        snr = random.uniform(self._min_snr, self._max_snr)
        noise = _generate_pink_noise(len(data._samples))
        data._samples = _mix_at_snr(data._samples, noise, snr)


class BrownNoisePerturbation(Perturbation):
    """Brownian/red noise — low-frequency rumble (HVAC, traffic, room tone)."""

    def __init__(self, min_snr_db=10, max_snr_db=30, rng=None):
        super().__init__()
        self._min_snr = min_snr_db
        self._max_snr = max_snr_db
        if rng is not None:
            random.seed(rng)

    def perturb(self, data):
        snr = random.uniform(self._min_snr, self._max_snr)
        noise = _generate_brown_noise(len(data._samples))
        data._samples = _mix_at_snr(data._samples, noise, snr)


class WindNoisePerturbation(Perturbation):
    """Subtractive synthesis wind: white noise through cascaded LPF with LFO modulation."""

    def __init__(self, min_snr_db=8, max_snr_db=25, min_cutoff_hz=400,
                 max_cutoff_hz=1200, lfo_rate_hz=0.08, rng=None):
        super().__init__()
        self._min_snr = min_snr_db
        self._max_snr = max_snr_db
        self._min_cutoff = min_cutoff_hz
        self._max_cutoff = max_cutoff_hz
        self._lfo_rate = lfo_rate_hz
        if rng is not None:
            random.seed(rng)

    def perturb(self, data):
        sr = data.sample_rate
        n = len(data._samples)
        snr = random.uniform(self._min_snr, self._max_snr)

        white = np.random.randn(n).astype(np.float32)

        cutoff = random.uniform(self._min_cutoff, self._max_cutoff)
        nyq = sr / 2
        wn = min(cutoff / nyq, 0.99)
        b, a = scipy_signal.butter(4, wn, btype='low')
        noise = scipy_signal.lfilter(b, a, white).astype(np.float32)

        t = np.arange(n, dtype=np.float32) / sr
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * self._lfo_rate * t + random.uniform(0, 2 * np.pi))
        noise *= lfo

        data._samples = _mix_at_snr(data._samples, noise, snr)


class RainNoisePerturbation(Perturbation):
    """Rain: pink noise bed + Karplus-Strong raindrop transients."""

    def __init__(self, min_snr_db=8, max_snr_db=25, min_drop_rate=5,
                 max_drop_rate=40, min_freq=200, max_freq=3000, rng=None):
        super().__init__()
        self._min_snr = min_snr_db
        self._max_snr = max_snr_db
        self._min_drop_rate = min_drop_rate
        self._max_drop_rate = max_drop_rate
        self._min_freq = min_freq
        self._max_freq = max_freq
        if rng is not None:
            random.seed(rng)

    def perturb(self, data):
        sr = data.sample_rate
        n = len(data._samples)
        snr = random.uniform(self._min_snr, self._max_snr)

        bed = _generate_pink_noise(n) * 0.3

        drop_rate = random.uniform(self._min_drop_rate, self._max_drop_rate)
        duration_sec = n / sr
        n_drops = int(drop_rate * duration_sec)

        drops = np.zeros(n, dtype=np.float32)
        for _ in range(n_drops):
            freq = random.uniform(self._min_freq, self._max_freq)
            drop = _karplus_strong(sr, freq, duration_sec=0.03, decay=0.992)
            pos = random.randint(0, max(n - len(drop), 0))
            end = min(pos + len(drop), n)
            drops[pos:end] += drop[:end - pos] * random.uniform(0.3, 1.0)

        noise = bed + drops
        data._samples = _mix_at_snr(data._samples, noise, snr)


class CrowdNoisePerturbation(Perturbation):
    """Crowd babble: bandpass-filtered noise with AM simulating speech-band chatter."""

    def __init__(self, min_snr_db=5, max_snr_db=25, min_voices=3,
                 max_voices=8, rng=None):
        super().__init__()
        self._min_snr = min_snr_db
        self._max_snr = max_snr_db
        self._min_voices = min_voices
        self._max_voices = max_voices
        if rng is not None:
            random.seed(rng)

    def perturb(self, data):
        sr = data.sample_rate
        n = len(data._samples)
        snr = random.uniform(self._min_snr, self._max_snr)
        n_voices = random.randint(self._min_voices, self._max_voices)

        nyq = sr / 2
        noise = np.zeros(n, dtype=np.float32)
        for _ in range(n_voices):
            voice = np.random.randn(n).astype(np.float32)
            lo = random.uniform(200, 400) / nyq
            hi = random.uniform(2800, 3800) / nyq
            lo, hi = min(lo, 0.99), min(hi, 0.99)
            if lo < hi:
                b, a = scipy_signal.butter(2, [lo, hi], btype='band')
                voice = scipy_signal.lfilter(b, a, voice).astype(np.float32)

            t = np.arange(n, dtype=np.float32) / sr
            am_rate = random.uniform(2, 6)
            am = 0.5 + 0.5 * np.sin(2 * np.pi * am_rate * t + random.uniform(0, 2 * np.pi))
            voice *= am
            noise += voice

        noise /= max(n_voices, 1)
        data._samples = _mix_at_snr(data._samples, noise, snr)


class MusicNoisePerturbation(Perturbation):
    """Harmonic tone clusters with vibrato — simulates background music/TV."""

    def __init__(self, min_snr_db=10, max_snr_db=30, min_harmonics=3,
                 max_harmonics=8, fundamental_range=(100, 500), rng=None):
        super().__init__()
        self._min_snr = min_snr_db
        self._max_snr = max_snr_db
        self._min_harm = min_harmonics
        self._max_harm = max_harmonics
        self._fund_range = fundamental_range
        if rng is not None:
            random.seed(rng)

    def perturb(self, data):
        sr = data.sample_rate
        n = len(data._samples)
        snr = random.uniform(self._min_snr, self._max_snr)

        t = np.arange(n, dtype=np.float64) / sr
        fundamental = random.uniform(*self._fund_range)
        n_harmonics = random.randint(self._min_harm, self._max_harm)

        noise = np.zeros(n, dtype=np.float64)
        for h in range(1, n_harmonics + 1):
            freq = fundamental * h
            if freq > sr / 2:
                break
            amp = 1.0 / h
            vibrato = random.uniform(4, 8)
            vibrato_depth = random.uniform(0.5, 3)
            phase = random.uniform(0, 2 * np.pi)
            inst_freq = freq + vibrato_depth * np.sin(2 * np.pi * vibrato * t)
            noise += amp * np.sin(2 * np.pi * np.cumsum(inst_freq) / sr + phase)

        noise = noise.astype(np.float32)
        data._samples = _mix_at_snr(data._samples, noise, snr)


class RoomReverbPerturbation(Perturbation):
    """Synthetic room reverb via exponential-decay impulse response convolution."""

    def __init__(self, min_decay_sec=0.3, max_decay_sec=1.5,
                 min_predelay_ms=5, max_predelay_ms=30, rng=None):
        super().__init__()
        self._min_decay = min_decay_sec
        self._max_decay = max_decay_sec
        self._min_predelay = min_predelay_ms
        self._max_predelay = max_predelay_ms
        if rng is not None:
            random.seed(rng)

    def perturb(self, data):
        sr = data.sample_rate
        decay = random.uniform(self._min_decay, self._max_decay)
        predelay = random.uniform(self._min_predelay, self._max_predelay)

        ir_len = int(sr * (decay + predelay / 1000))
        pre_samples = int(predelay / 1000 * sr)

        ir = np.zeros(ir_len, dtype=np.float32)
        for i in range(pre_samples, ir_len):
            t = (i - pre_samples) / max(ir_len - pre_samples, 1)
            ir[i] = np.random.randn() * ((1 - t) ** 2.2)

        ir /= (np.max(np.abs(ir)) + 1e-10)

        wet = scipy_signal.fftconvolve(data._samples, ir, mode='full')[:len(data._samples)]
        wet = wet.astype(np.float32)

        wet_ratio = random.uniform(0.1, 0.4)
        data._samples = (1 - wet_ratio) * data._samples + wet_ratio * wet


class CodecNoisePerturbation(Perturbation):
    """Telephone/VoIP quality degradation: bandpass + quantization noise."""

    def __init__(self, low_hz=300, high_hz=3400, quantize_bits=8, rng=None):
        super().__init__()
        self._low_hz = low_hz
        self._high_hz = high_hz
        self._quantize_bits = quantize_bits
        if rng is not None:
            random.seed(rng)

    def perturb(self, data):
        sr = data.sample_rate
        nyq = sr / 2
        lo = self._low_hz / nyq
        hi = min(self._high_hz / nyq, 0.99)

        if lo < hi:
            b, a = scipy_signal.butter(4, [lo, hi], btype='band')
            data._samples = scipy_signal.lfilter(b, a, data._samples).astype(np.float32)

        bits = random.randint(self._quantize_bits, 16)
        if bits < 16:
            levels = 2 ** bits
            mx = np.max(np.abs(data._samples)) + 1e-10
            normalized = data._samples / mx
            quantized = np.round(normalized * levels) / levels
            data._samples = (quantized * mx).astype(np.float32)


# Scene templates for AmbientScenePerturbation
SCENE_TEMPLATES = {
    'outdoor': [
        ('wind', {'min_snr_db': 10, 'max_snr_db': 25}),
        ('brown_noise', {'min_snr_db': 15, 'max_snr_db': 30}),
    ],
    'indoor': [
        ('room_reverb', {'min_decay_sec': 0.4, 'max_decay_sec': 1.2}),
        ('brown_noise', {'min_snr_db': 20, 'max_snr_db': 35}),
    ],
    'rainy': [
        ('rain', {'min_snr_db': 8, 'max_snr_db': 20}),
        ('wind', {'min_snr_db': 15, 'max_snr_db': 25}),
    ],
    'crowd': [
        ('crowd', {'min_snr_db': 5, 'max_snr_db': 20}),
        ('room_reverb', {'min_decay_sec': 0.3, 'max_decay_sec': 0.8}),
    ],
    'music': [
        ('music', {'min_snr_db': 10, 'max_snr_db': 25}),
        ('room_reverb', {'min_decay_sec': 0.2, 'max_decay_sec': 0.6}),
    ],
    'street': [
        ('crowd', {'min_snr_db': 10, 'max_snr_db': 25}),
        ('wind', {'min_snr_db': 15, 'max_snr_db': 30}),
        ('brown_noise', {'min_snr_db': 12, 'max_snr_db': 25}),
    ],
    'call_center': [
        ('codec', {}),
        ('crowd', {'min_snr_db': 15, 'max_snr_db': 30}),
    ],
}

_NOISE_FACTORIES = {
    'pink_noise': PinkNoisePerturbation,
    'brown_noise': BrownNoisePerturbation,
    'wind': WindNoisePerturbation,
    'rain': RainNoisePerturbation,
    'crowd': CrowdNoisePerturbation,
    'music': MusicNoisePerturbation,
    'room_reverb': RoomReverbPerturbation,
    'codec': CodecNoisePerturbation,
}


class AmbientScenePerturbation(Perturbation):
    """Compound perturbation that applies a random acoustic scene.

    Randomly selects a scene template (outdoor, indoor, rainy, crowd, music,
    street, call_center) and applies its constituent noise layers.
    """

    def __init__(self, min_snr_db=5, max_snr_db=25, scenes=None, rng=None):
        super().__init__()
        self._min_snr = min_snr_db
        self._max_snr = max_snr_db
        self._scene_names = scenes or list(SCENE_TEMPLATES.keys())
        if rng is not None:
            random.seed(rng)

    def perturb(self, data):
        scene_name = random.choice(self._scene_names)
        layers = SCENE_TEMPLATES[scene_name]

        for noise_type, kwargs in layers:
            factory = _NOISE_FACTORIES.get(noise_type)
            if factory:
                perturb_instance = factory(**kwargs)
                perturb_instance.perturb(data)


def build_augmentor_from_config(aug_cfg):
    """Build an AudioAugmentor from a YAML augmentation config dict.

    Config format:
        augmentation:
          enabled: true
          prob_clean: 0.15
          perturbations:
            - type: ambient_scene
              prob: 0.5
              min_snr_db: 5
              max_snr_db: 25
            - type: pink_noise
              prob: 0.15
            ...

    Returns:
        AudioAugmentor with the configured perturbation pipeline
    """
    from nemo.collections.asr.parts.preprocessing.features import AudioAugmentor
    from nemo.collections.asr.parts.preprocessing.perturb import (
        WhiteNoisePerturbation, ShiftPerturbation, SpeedPerturbation,
        GainPerturbation,
    )

    TYPE_MAP = {
        'ambient_scene': AmbientScenePerturbation,
        'pink_noise': PinkNoisePerturbation,
        'brown_noise': BrownNoisePerturbation,
        'wind': WindNoisePerturbation,
        'rain': RainNoisePerturbation,
        'crowd': CrowdNoisePerturbation,
        'music': MusicNoisePerturbation,
        'room_reverb': RoomReverbPerturbation,
        'codec': CodecNoisePerturbation,
        'white_noise': WhiteNoisePerturbation,
        'shift': ShiftPerturbation,
        'speed': SpeedPerturbation,
        'gain': GainPerturbation,
    }

    perturbations = []
    prob_clean = aug_cfg.get('prob_clean', 0.0)

    for p_cfg in aug_cfg.get('perturbations', []):
        p_type = p_cfg['type']
        prob = p_cfg.get('prob', 1.0)
        kwargs = {k: v for k, v in p_cfg.items() if k not in ('type', 'prob')}

        factory = TYPE_MAP.get(p_type)
        if factory is None:
            raise ValueError(f"Unknown perturbation type: {p_type}")

        if p_type == 'white_noise':
            kwargs_mapped = {}
            if 'min_snr_db' in kwargs:
                kwargs_mapped['min_level'] = -kwargs['min_snr_db'] - 46
            if 'max_snr_db' in kwargs:
                kwargs_mapped['max_level'] = -kwargs['max_snr_db'] - 46
            kwargs = kwargs_mapped if kwargs_mapped else {'min_level': -90, 'max_level': -46}
        elif p_type == 'shift':
            kwargs.setdefault('min_shift_ms', 50)
            kwargs.setdefault('max_shift_ms', 300)
        elif p_type == 'speed':
            kwargs.setdefault('sr', 16000)
            kwargs.setdefault('resample_type', 'kaiser_fast')
            kwargs.setdefault('min_speed_rate', 0.9)
            kwargs.setdefault('max_speed_rate', 1.1)

        perturbations.append((prob, factory(**kwargs)))

    if prob_clean > 0:
        class CleanPassthrough(Perturbation):
            def perturb(self, data):
                pass
        effective_perturbations = [(1.0 - prob_clean, AmbientScenePerturbation())]
        for prob, p in perturbations:
            effective_perturbations.append((prob * (1.0 - prob_clean), p))
        return AudioAugmentor(perturbations=effective_perturbations)

    return AudioAugmentor(perturbations=perturbations)
