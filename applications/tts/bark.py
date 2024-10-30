import sys
from transformers import AutoProcessor, BarkModel
import numpy as np
import io
import torch
import soundfile as sf

class TextToSpeech:
    def __init__(self, model_name="suno/bark", voice_preset="v2/en_speaker_6"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: " + str(self.device))
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = BarkModel.from_pretrained(model_name).to(self.device)
        self.voice_preset = voice_preset

    def generate_audio(self, text):
        inputs = self.processor(text, voice_preset=self.voice_preset)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        audio_array = self.model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        return audio_array

    def save_audio(self, audio_array, output_file):
        sf.write(output_file, audio_array, 22050)  # 22050 is the sample rate (adjust if needed)

if __name__ == "__main__":
    tts = TextToSpeech()
    
    while True:
        try:
            input_text = input("Enter text (or 'exit' to quit): ")
            if input_text.lower() == "exit":
                break
            audio_array = tts.generate_audio(input_text)
            output_file = input("Enter output file name (e.g., output.wav): ")
            tts.save_audio(audio_array, output_file)
            print(f"Audio saved to {output_file}")
        except KeyboardInterrupt:
            break
