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
import numpy as np

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
        self.asr_model_ru = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_ru_fastconformer_hybrid_large_pc")
        
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

    def load_audio(self, audio_path):
        """Load audio file and return signal"""
        signal, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)
        return signal

    def process_audio_chunk(self, chunk, model):
        """Process a single audio chunk"""
        with torch.no_grad():
            logits = model.forward(
                input_signal=torch.tensor(chunk).unsqueeze(0),
                input_signal_length=torch.tensor([len(chunk)])
            )
            current_hypotheses = model.decoding.ctc_decoder_predictions_tensor(
                logits, predictions_len=None, return_hypotheses=True
            )
            
        return current_hypotheses[0]

    def merge_word_timestamps(self, chunks_with_timestamps, chunk_duration):
        """Merge word timestamps from multiple chunks"""
        merged_words = []
        time_offset = 0
        
        for chunk_idx, chunk_result in enumerate(chunks_with_timestamps):
            for word_info in chunk_result:
                # Adjust timestamp with chunk offset
                start_time = word_info['start_time'] + time_offset
                end_time = word_info['end_time'] + time_offset
                
                merged_words.append({
                    'word': word_info['word'],
                    'start_time': start_time,
                    'end_time': end_time
                })
            
            time_offset += chunk_duration
            
        return merged_words

    def transcribe_audio(self, audio_path, language, chunk_duration_ms=8000):
        """Transcribe audio using chunked inference and get word timestamps"""
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
            
            # Load audio
            signal = self.load_audio(audio_path)
            
            # Calculate chunk size in samples
            chunk_size = int(chunk_duration_ms * 16)  # 16 samples per ms at 16kHz
            
            # Split audio into chunks
            chunks = [signal[i:i + chunk_size] for i in range(0, len(signal), chunk_size)]
            
            # Process each chunk
            chunk_results = []
            for chunk in tqdm(chunks, desc="Processing chunks"):
                # Pad chunk if necessary
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                # Get word timestamps for chunk
                hypothesis = self.process_audio_chunk(chunk, model)
                
                # Extract word timing info
                words_with_timing = []
                for word, timing in zip(hypothesis.words, hypothesis.word_timestamps):
                    words_with_timing.append({
                        'word': word,
                        'start_time': timing[0],
                        'end_time': timing[1]
                    })
                
                chunk_results.append(words_with_timing)
            
            # Merge results from all chunks
            merged_results = self.merge_word_timestamps(chunk_results, chunk_duration_ms / 1000)
            
            # Create final result
            full_text = ' '.join(word_info['word'] for word_info in merged_results)
            
            return {
                'text': full_text,
                'word_timestamps': merged_results
            }
            
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
        
        # Transcribe with timestamps
        transcription_result = self.transcribe_audio(audio_path, language)
        if not transcription_result:
            return
        
        # Save results
        result = {
            "video_id": video_id,
            "video_path": video_path,
            "audio_path": audio_path,
            "language": language,
            "transcription": transcription_result['text'],
            "word_timestamps": transcription_result['word_timestamps'],
            "emotion_analysis": emotion_result
        }
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully processed: {video_id}")
        print(f"Detected language: {language}")
        print(f"Transcription: {transcription_result['text']}")
        print(f"Number of words with timestamps: {len(transcription_result['word_timestamps'])}")

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