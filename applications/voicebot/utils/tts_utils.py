import io, os
from gtts import gTTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import torch
import numpy as np
import base64
import wave

def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav

def encode_audio_common(
    frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1
):
    """Return base64 encoded audio"""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    if encode_base64:
        b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
        return b64_encoded
    else:
        return wav_buf.read()


class TextToSpeech:
    def __init__(self, model_name=None, custom_model_path=None, device="cpu"):
        if custom_model_path and os.path.exists(custom_model_path) and os.path.isfile(custom_model_path + "/config.json"):
            model_path = custom_model_path
            print("Loading custom model from", model_path, flush=True)
        else:
            print("Downloading XTTS Model:", model_name, flush=True)
            ModelManager().download_model(model_name)
            model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
            print("XTTS Model downloaded", flush=True)
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.tts_model = Xtts.init_from_config(config)
        self.tts_model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=True if device == "cuda" else False)
        self.tts_model.to(device)

    def infer(self, text, language, file_path, speaker_wav_file_path=None):
        if speaker_wav_file_path:
            gpt_cond_latent, speaker_embedding = self.tts_model.get_conditioning_latents(
                speaker_wav_file_path
            )
            out = self.tts_model.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
            )
            wav = postprocess(torch.tensor(out["wav"]))

            wav = encode_audio_common(wav.tobytes(), encode_base64=False)
            
            with open(file_path, 'wb') as f:
                f.write(wav)
        else:
            tts = gTTS(text=text, lang=language)
            tts.save(file_path)

        return file_path
