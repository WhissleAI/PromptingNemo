import re
import numpy as np
from typing import Optional, List

def clean_text_for_piper(text: str) -> str:
    """Clean text for TTS processing"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    return text.strip()

class PiperSynthesizer:
    def __init__(self, model_path: str, config_path: str, length_scale: float = 1.0):
        """Initialize with simplified settings"""
        self.model_path = model_path
        self.config_path = config_path
        self.length_scale = length_scale
        print(f"Initialized simplified TTS system with model: {model_path}")

    def synthesize(self, text: str) -> bytes:
        """
        Simplified synthesis that falls back to gTTS
        Returns audio as bytes
        """
        from gtts import gTTS
        import io
        
        # Clean the text
        text = clean_text_for_piper(text)
        
        # Use gTTS as a fallback
        tts = gTTS(text=text, lang='en')
        
        # Save to BytesIO object
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        return fp.read()

    def synthesize_to_file(self, text: str, output_path: str) -> None:
        """
        Synthesize text and save directly to file
        """
        # Clean the text
        text = clean_text_for_piper(text)
        
        # Use gTTS
        tts = gTTS(text=text, lang='en')
        tts.save(output_path)

# Optional: Add these utility functions if needed
def convert_audio(audio_data: bytes, target_sample_rate: int = 22050) -> np.ndarray:
    """Convert audio data to numpy array with specified sample rate"""
    from pydub import AudioSegment
    import io
    
    # Load audio from bytes
    audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
    
    # Convert to target sample rate
    if audio.frame_rate != target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Convert to float32 and normalize
    samples = samples.astype(np.float32) / np.iinfo(np.int16).max
    
    return samples