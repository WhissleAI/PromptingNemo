import riva.client
from google.protobuf.json_format import MessageToDict


def get_transcript(audio_file, model_info, boosted_lm_words, boosted_lm_score, word_timestamps, diarize=False, max_speakers=2):
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
    
    if diarize:
        riva.client.asr.add_speaker_diarization_to_config(riva_config, diarization_enable=True, diarization_max_speakers=max_speakers)
        riva_config.enable_word_time_offsets = True

    response = riva_asr.offline_recognize(audio_file, riva_config)
    transcripts = [result.alternatives[0].transcript for result in response.results]
    duration_seconds = [result.audio_processed for result in response.results][-1]
    final_transcript = " ".join(transcripts)
    timestamps = []
    if word_timestamps or diarize:
        for result in response.results:
            timestamps += [MessageToDict(word) for word in result.alternatives[0].words]

    return final_transcript, timestamps, duration_seconds    

def transform_riva_output(words):
    output = []
    current_speaker = None
    current_text = []
    start_timestamp = 0
    end_timestamp = 0

    for word_data in words:
        speaker = word_data.get("speakerTag", current_speaker)  # Default to 1 if not present
        word = word_data["word"]
        start_time = word_data.get("startTime", end_timestamp)
        end_time = word_data.get("endTime", start_timestamp)
        
        if current_speaker is None:
            current_speaker = 1
            start_timestamp = start_time
            
        if speaker != current_speaker:
            output.append({
                "text": " ".join(current_text),
                "speaker_id": current_speaker,
                "start_timestamp": start_timestamp/1000,
                "end_timestamp": end_timestamp/1000
            })
            
            current_speaker = speaker
            current_text = []
            start_timestamp = start_time
        
        current_text.append(word)
        end_timestamp = end_time
    
    if current_text:
        output.append({
            "text": " ".join(current_text),
            "speaker_id": current_speaker,
            "start_timestamp": start_timestamp/1000,
            "end_timestamp": end_timestamp/1000
        })
    
    return output
