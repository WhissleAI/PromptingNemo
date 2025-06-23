import riva.client
from google.protobuf.json_format import MessageToDict


def get_transcript(audio_file, model_info, boosted_lm_words, boosted_lm_score, word_timestamps, enable_diarization=False, max_speakers=2):
    riva_config = riva.client.RecognitionConfig()
    auth = riva.client.Auth(uri=model_info['uri'])
    riva_asr = riva.client.ASRService(auth)
    if boosted_lm_words:
        riva.client.add_word_boosting_to_config(riva_config, boosted_lm_words, boosted_lm_score)

    riva_config.max_alternatives = 1
    riva_config.enable_automatic_punctuation = True
    riva_config.audio_channel_count = 1
    riva_config.enable_word_time_offsets = word_timestamps
    riva_config.model = model_info['model']
    if 'language_code' in model_info: riva_config.language_code = model_info["language_code"]
    
    # Add speaker diarization if enabled
    if enable_diarization:
        riva.client.add_speaker_diarization_to_config(riva_config, True, diarization_max_speakers=max_speakers)
    
    response = riva_asr.offline_recognize(audio_file, riva_config)
    transcripts = [result.alternatives[0].transcript for result in response.results]
    duration_seconds = sum([result.audio_processed for result in response.results] )
    final_transcript = " ".join(transcripts)
    
    timestamps = []
    if word_timestamps:
        for result in response.results:
            timestamps += list(result.alternatives[0].words) 
        timestamps = [MessageToDict(timestamp) for timestamp in timestamps]

    return final_transcript, timestamps, duration_seconds    


def transform_riva_output(timestamps):
    """
    Transform Riva diarization output into a more usable format.
    
    Args:
        timestamps: List of word timestamps with speaker information
        
    Returns:
        Dictionary containing diarization information
    """
    if not timestamps:
        return {"segments": [], "speakers": []}
    
    segments = []
    speakers = set()
    
    current_speaker = None
    current_segment = {
        "speaker": None,
        "start_time": None,
        "end_time": None,
        "text": ""
    }
    
    for word_info in timestamps:
        speaker = word_info.get('speaker', 'unknown')
        start_time = word_info.get('start_time', 0)
        end_time = word_info.get('end_time', 0)
        word = word_info.get('word', '')
        
        speakers.add(speaker)
        
        # Start new segment if speaker changes
        if current_speaker != speaker:
            # Save previous segment if it exists
            if current_segment["speaker"] is not None:
                segments.append(current_segment.copy())
            
            # Start new segment
            current_segment = {
                "speaker": speaker,
                "start_time": start_time,
                "end_time": end_time,
                "text": word
            }
            current_speaker = speaker
        else:
            # Continue current segment
            current_segment["end_time"] = end_time
            current_segment["text"] += " " + word
    
    # Add the last segment
    if current_segment["speaker"] is not None:
        segments.append(current_segment)
    
    return {
        "segments": segments,
        "speakers": list(speakers)
    }    
