import io
import os
from gtts import gTTS
import wave
import base64
import tempfile
from pydub import AudioSegment

def encode_audio_common(frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1):
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
        """Initialize TTS - parameters kept for compatibility"""
        self.model_name = model_name
        print("Initialized simplified TTS using gTTS")

    def infer(self, text, language='en', file_path=None, speaker_wav_file_path=None):
        """
        Generate speech from text and save to file
        Args:
            text (str): Text to convert to speech
            language (str): Language code (e.g., 'en' for English)
            file_path (str): Output file path
            speaker_wav_file_path (str): Ignored in gTTS implementation
        """
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            # If no file path is provided, use a temporary file
            if not file_path:
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                file_path = temp_file.name
                temp_file.close()

            # Save to MP3 first
            mp3_path = file_path if file_path.endswith('.mp3') else file_path + '.mp3'
            tts.save(mp3_path)

            # If the requested output is not MP3, convert it
            if not file_path.endswith('.mp3'):
                # Convert to WAV or other format if needed
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(file_path, format=file_path.split('.')[-1])
                # Clean up temporary MP3
                os.remove(mp3_path)

            return file_path

        except Exception as e:
            print(f"Error in TTS generation: {str(e)}")
            return None

    def tts_to_file_cloning(self, text, speaker_wav_folder, language, output_path):
        """Maintained for compatibility, falls back to regular TTS"""
        return self.infer(text, language, output_path)

# Optional: Add these utility functions if needed by your application
def postprocess(wav):
    """Dummy function maintained for compatibility"""
    return wav

def convert_wav_to_mp3(wav_path, mp3_path):
    """Convert WAV to MP3 if needed"""
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format='mp3')