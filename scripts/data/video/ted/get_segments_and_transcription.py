import os
import subprocess
import wave
import contextlib
import webrtcvad
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp

def convert_mp4_to_wav(mp4_folder, wav_folder):
    os.makedirs(wav_folder, exist_ok=True)
    for file in os.listdir(mp4_folder):
        if file.endswith(".mp4"):
            mp4_path = os.path.join(mp4_folder, file)
            wav_path = os.path.join(wav_folder, file.replace(".mp4", ".wav"))
            subprocess.call(['ffmpeg', '-i', mp4_path, '-ar', '16000', '-ac', '1', wav_path])

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate == 16000
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * frame_duration_ms / 1000.0 * 2)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n], offset
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = []
    triggered = False
    voiced_frames = []
    start_time = 0
    last_end_time = 0

    for frame, offset in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * num_padding_frames:
                triggered = True
                start_time = (offset - len(ring_buffer) * len(frame)) / (sample_rate * 2.0)
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer = []
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * num_padding_frames:
                triggered = False
                end_time = offset / (sample_rate * 2.0)
                yield b''.join([f for f in voiced_frames]), start_time, end_time
                ring_buffer = []
                voiced_frames = []

    if voiced_frames:
        end_time = offset / (sample_rate * 2.0)
        yield b''.join([f for f in voiced_frames]), start_time, end_time

def segment_wav_files(mp4_folder, wav_folder, segments_folder):
    os.makedirs(segments_folder, exist_ok=True)
    vad = webrtcvad.Vad(3)  # Aggressiveness mode from 0 to 3
    frame_duration_ms = 30
    padding_duration_ms = 1000  # Increase padding duration for longer segments
    for wav_file in os.listdir(wav_folder):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(wav_folder, wav_file)
            audio, sample_rate = read_wave(wav_path)
            frames = frame_generator(frame_duration_ms, audio, sample_rate)
            segments = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)
            segments_path = os.path.join(segments_folder, wav_file.replace(".wav", ".segments.txt"))
            mp4_file = wav_file.replace(".wav", ".mp4")
            mp4_path = os.path.join(mp4_folder, mp4_file)
            with open(segments_path, "w") as f:
                for i, (segment, start_time, end_time) in enumerate(segments):
                    segment_file = os.path.join(wav_folder, f"{wav_file[:-4]}_segment_{i}.wav")
                    video_segment_file = os.path.join(segments_folder, f"{mp4_file[:-4]}_segment_{i}.mp4")
                    with wave.open(segment_file, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(segment)
                    # Save the video segment
                    subprocess.call(['ffmpeg', '-i', mp4_path, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', video_segment_file])
                    f.write(f"{start_time} {end_time} {i}\n")

def transcribe_segments(wav_folder, segments_folder, transcription_folder):
    os.makedirs(transcription_folder, exist_ok=True)
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
    punct_cap_model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")

    for wav_file in os.listdir(wav_folder):
        if wav_file.endswith(".wav"):
            segments_path = os.path.join(segments_folder, wav_file.replace(".wav", ".segments.txt"))
            if not os.path.exists(segments_path):
                continue
            with open(segments_path, "r") as f:
                segments = f.readlines()
            transcription_path = os.path.join(transcription_folder, wav_file.replace(".wav", ".transcription.txt"))
            with open(transcription_path, "w") as f:
                for segment in segments:
                    start_time, end_time, segment_idx = segment.split()
                    segment_file = os.path.join(wav_folder, f"{wav_file[:-4]}_segment_{segment_idx.strip()}.wav")
                    transcription = asr_model.transcribe([segment_file])
                    punctuated_transcription = punct_cap_model.add_punctuation_capitalization(transcription)
                    f.write(f"{start_time}-{end_time}: {punctuated_transcription[0]}\n")

def main(mp4_folder, wav_folder, segments_folder, transcription_folder):
    convert_mp4_to_wav(mp4_folder, wav_folder)
    segment_wav_files(mp4_folder, wav_folder, segments_folder)
    transcribe_segments(wav_folder, segments_folder, transcription_folder)

if __name__ == "__main__":
    mp4_folder = "/external2/datasets/ted/visual"
    wav_folder = "/external2/datasets/ted/visual_wav"
    segments_folder = "/external2/datasets/ted/visual_segments"
    transcription_folder = "/external2/datasets/ted/visual_transcriptions"
    main(mp4_folder, wav_folder, segments_folder, transcription_folder)
