import glob
import os
import subprocess
from tqdm import tqdm
import torch
import librosa
import soundfile as sf
from fastlang import fastlang
import nemo.collections.asr as nemo_asr
import json
from pathlib import Path

import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.interfaces import foreign_class

class VideoProcessor:
    def __init__(self, output_dir="processed_data"):
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, "audio")
        self.transcripts_dir = os.path.join(output_dir, "transcripts")
        
        # Create output directories
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.transcripts_dir, exist_ok=True)
        
        # Initialize models
        print("Loading ASR models...")
        self.asr_model_en = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_large")
        self.asr_model_es = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_es_conformer_ctc_large")
        self.asr_model_fr = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_fr_conformer_ctc_large")
        self.asr_model_de = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_de_conformer_ctc_large")
        self.asr_model_it = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_it_conformer_ctc_large")
        self.asr_model_ru = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name="nvidia/stt_ru_fastconformer_hybrid_large_pc")
        
        print("Loading language identification model...")
        self.language_id = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-commonlanguage_ecapa", 
            savedir="pretrained_models/lang-id-commonlanguage_ecapa"
        )
        
        print("Loading emotion recognition model...")
        self.emotion_classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier"
        )

    def extract_audio(self, video_path):
        """Extract 16kHz PCM WAV audio from video file"""
        audio_filename = os.path.splitext(os.path.basename(video_path))[0] + ".wav"
        audio_path = os.path.join(self.audio_dir, audio_filename)
        
        if os.path.exists(audio_path):
            print(f"Audio file already exists: {audio_path}")
            return audio_path
        
        try:
            # Use FFmpeg to extract audio
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn",  # Disable video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz sampling rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                audio_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return audio_path
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from {video_path}: {e}")
            return None

    def detect_language(self, audio_path):
        """Detect language of audio file"""
        out_prob, score, index, text_lab = self.language_id.classify_file(audio_path)
        return text_lab[0]

    def detect_emotion(self, audio_path):
        """Detect emotion in audio file"""
        out_prob, score, index, text_lab = self.emotion_classifier.classify_file(audio_path)
        
        return text_lab[0]



    def transcribe_audio(self, audio_path, language):
        """Transcribe audio using appropriate NeMo model"""
        try:
            # Select appropriate model based on language
            if language == 'English':
                model = self.asr_model_en
            elif language == 'Spanish':
                model = self.asr_model_es
            elif language == 'French':
                model = self.asr_model_fr
            elif language == 'German':
                model = self.asr_model_de
            elif language == 'Italian':
                model = self.asr_model_it
            elif language == 'Russian':
                model = self.asr_model_ru
            else:
                print(f"Unsupported language: {language}")
                return None
            
            # Transcribe
            transcription = model.transcribe([audio_path])[0]
            return transcription
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return None

    def process_video(self, video_path):
        """Process a single video file"""
        print(f"\nProcessing: {video_path}")
        
        # Generate output paths
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        transcript_path = os.path.join(self.transcripts_dir, f"{video_id}.json")
        
        # Skip if already processed
        if os.path.exists(transcript_path):
            print(f"Already processed: {video_path}")
            return
        
        # Extract audio
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return
        
        # Detect language
        language = self.detect_language(audio_path)
        if not language:
            return
        
        # Detect emotion
        emotion_result = self.detect_emotion(audio_path)
        if not emotion_result:
            return
        
        # Transcribe
        transcription = self.transcribe_audio(audio_path, language)
        if not transcription:
            return
        
        # Save results
        result = {
            "video_id": video_id,
            "video_path": video_path,
            "audio_path": audio_path,
            "language": language,
            "transcription": transcription,
            "emotion_analysis": emotion_result
        }
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully processed: {video_id}")
        print(f"Detected language: {language}")
        #print(f"Detected emotion: {emotion_result['emotion']} (confidence: {emotion_result['confidence']:.2f})")
        print(f"Transcription: {transcription}")

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process video files for transcription and emotion analysis')
    parser.add_argument('--data_path', help='Path to directory containing video files')
    parser.add_argument('--output_dir', default='processed_data', help='Output directory for processed files')
    args = parser.parse_args()

    # Initialize processor
    processor = VideoProcessor(output_dir=args.output_dir)
    
    # Get all video files
    print(args.data_path)
    all_videos = glob.glob(args.data_path + "/*.mp4")
    print(f"Found {len(all_videos)} video files")
    
    # Process each video
    for video_path in tqdm(all_videos, desc="Processing videos"):
        processor.process_video(video_path)
        
    print("\nProcessing complete!")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()