
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor

allvars = {'sample_rate': 16000, 'normalize': 'per_feature', 'window_size': 0.025, 'window_stride': 0.01, 'window': 'hann', 'features': 80, 'n_fft': 512, 'log': True, 'frame_splicing': 1, 'dither': 1e-05, 'pad_to': 0, 'pad_value': 0.0}
audio2melspectrogram = AudioToMelSpectrogramPreprocessor(**allvars)

