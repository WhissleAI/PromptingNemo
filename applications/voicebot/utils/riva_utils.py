import riva.client
from google.protobuf.json_format import MessageToDict


def get_transcript(audio_file, model_info, boosted_lm_words, boosted_lm_score, word_timestamps):
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
