import librosa
import speech_recognition as sr
import nltk
import re
import numpy as np
import pandas as pd
import noisereduce as nr
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize

class SpeechAnalyzer:
    def __init__(self):
        # Initialize the speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize the emotion recognition pipeline
        self.emotion_pipeline = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion", return_all_scores=True)
        
        # Define fillers
        self.fillers = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right', 'okay']
        
        # Initialize NLTK tokenizer
        nltk.download('punkt')
        nltk.download('punkt_tab')
    
    def analyze(self, audio_path):
        # Load and preprocess audio
        y, sr_rate = librosa.load(audio_path, sr=None)
        
        # Noise Reduction
        y = self.reduce_noise(y, sr_rate)
        
        # Transcription
        transcript = self.transcribe_audio(audio_path)
        
        # Calculate WPM
        wpm = self.calculate_wpm(transcript, y, sr_rate)
        
        # Calculate Valence/Arousal
        valence_arousal = self.calculate_valence_arousal(transcript)
        
        # Calculate Disfluency Score
        disfluency = self.calculate_disfluency(transcript, y, sr_rate)
        
        return {
            'wpm': round(wpm, 2),
            'valence': round(valence_arousal, 2),
            'disfluency': round(disfluency, 2)
        }
    
    def reduce_noise(self, y, sr_rate):
        # Assume the first 0.5 seconds is noise
        noise_clip = y[:int(0.5 * sr_rate)]
        y_reduced = nr.reduce_noise(y=y, sr=sr_rate, y_noise=noise_clip, prop_decrease=1.0)
        return y_reduced
    
    def transcribe_audio(self, audio_path):
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            # Use Google's speech recognition
            transcript = self.recognizer.recognize_google(audio)
            return transcript
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""
    
    def calculate_wpm(self, transcript, y, sr_rate):
        words = word_tokenize(transcript)
        num_words = len(words)
        duration_minutes = librosa.get_duration(y=y, sr=sr_rate) / 60
        wpm = num_words / duration_minutes if duration_minutes > 0 else 0
        return wpm
    
    def calculate_valence_arousal(self, transcript):
        if not transcript.strip():
            return 5.0  # Neutral if no transcript
        
        # Get emotion scores
        emotions = self.emotion_pipeline(transcript)[0]
        
        # Extract scores
        scores = {emotion['label']: emotion['score'] for emotion in emotions}
        
        # Valence: positive emotions vs negative emotions
        positive = scores.get('joy', 0) + scores.get('love', 0) + scores.get('surprise', 0)
        negative = scores.get('anger', 0) + scores.get('fear', 0) + scores.get('sadness', 0)
        valence = positive - negative  # Range approximately -1 to +1
        
        # Arousal is not directly provided, so we'll approximate it based on the intensity of emotions
        arousal = (positive + negative) / 2  # Range 0 to 1
        
        # Normalize to 1-10 scale
        scaler = MinMaxScaler(feature_range=(1, 10))
        valence_normalized = scaler.fit_transform(np.array([[valence]]))[0][0]
        arousal_normalized = scaler.fit_transform(np.array([[arousal]]))[0][0]
        
        # Combine valence and arousal into a single score (you can customize this)
        combined_score = (valence_normalized + arousal_normalized) / 2
        return combined_score
    
    def calculate_disfluency(self, transcript, y, sr_rate):
        words = nltk.word_tokenize(transcript.lower())
        num_fillers = sum(word in self.fillers for word in words)
        
        # Identify silent intervals
        silent_intervals = librosa.effects.split(y, top_db=20)
        num_pauses = len(silent_intervals)
        total_duration = librosa.get_duration(y=y, sr=sr_rate)
        average_pause_duration = librosa.get_duration(y=y[silent_intervals], sr=sr_rate) / num_pauses if num_pauses > 0 else 0
        
        # Calculate repetitions (simple heuristic: consecutive duplicate words)
        repetitions = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repetitions += 1
        
        # Normalize disfluency metrics
        # Assuming typical ranges:
        # - Fillers: 0-20
        # - Pauses: 0-30 seconds
        # - Repetitions: 0-10
        filler_score = min(num_fillers, 20) / 20  # 0 to 1
        pause_score = min(average_pause_duration, 30) / 30  # 0 to 1
        repetition_score = min(repetitions, 10) / 10  # 0 to 1
        
        # Weighted sum (you can adjust weights as needed)
        disfluency = (filler_score * 0.5 + pause_score * 0.3 + repetition_score * 0.2) * 10
        return disfluency

# Example Usage
if __name__ == "__main__":
    analyzer = SpeechAnalyzer()
    audio_file = '/home/ubuntu/workspace/PromptingNemo/applications/voicebot/demo_audio/EN_karan_angry.wav'  # Replace with your audio file path
    results = analyzer.analyze(audio_file)
    print("=== Analysis Results ===")
    print(f"Words Per Minute (WPM): {results['wpm']}")
    print(f"Valence/Arousal Score (1-10): {results['valence']}")
    print(f"Disfluency Score (1-10): {results['disfluency']}")

